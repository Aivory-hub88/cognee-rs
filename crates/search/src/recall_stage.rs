//! Per-stage timing for graph recall.
//!
//! Cerveau gives a graph recall a hard time budget (about 1.5 s) and drops the
//! request when it runs out. From the outside that only shows up as "timeout",
//! with no way to tell whether the query embedding, the vector search or the
//! graph expansion ate the budget. Each stage wrapped here emits one
//! structured event when it ends:
//!
//! ```text
//! target = "cognee::recall_stage"  stage = "vector_search"  elapsed_ms = 212.4  outcome = "ok"
//! ```
//!
//! * `stage`: one of [`EMBED_QUERY`], [`VECTOR_SEARCH`], [`GRAPH_EXPANSION`],
//!   [`TOTAL`].
//! * `elapsed_ms`: wall time of the stage, in milliseconds with a fractional part.
//! * `outcome`: `ok`, `error`, or `cancelled`.
//!
//! `cancelled` is the interesting one. When the caller's timeout fires, the
//! recall future is dropped mid-flight, so no stage ever reaches its own end.
//! [`StageTimer`] logs from `Drop` in that case, which pins the timeout on the
//! stage that was running at the time (and `total` is logged as `cancelled` too).
//!
//! Only the stage name, the duration and the outcome are recorded. Query text,
//! tenant or dataset identifiers, result contents and error messages are never
//! passed to the log, so these events are safe to ship to shared log storage.
//! Enable them with `RUST_LOG=cognee::recall_stage=info`.

use std::future::Future;
use std::time::Instant;

/// `tracing` target of every stage event.
pub const TARGET: &str = "cognee::recall_stage";

/// Checking that the vector collection exists (one backend round trip).
pub const COLLECTION_CHECK: &str = "collection_check";
/// Embedding the query text.
pub const EMBED_QUERY: &str = "embed_query";
/// Similarity search across the vector collections.
pub const VECTOR_SEARCH: &str = "vector_search";
/// Loading the graph neighbourhood (or node-set subgraph) around the seeds.
pub const GRAPH_EXPANSION: &str = "graph_expansion";
/// The whole `brute_force_triplet_search` call, including ranking.
pub const TOTAL: &str = "total";

/// Outcome label: the stage finished successfully.
pub const OUTCOME_OK: &str = "ok";
/// Outcome label: the stage finished with an error.
pub const OUTCOME_ERROR: &str = "error";
/// Outcome label: the stage was dropped before it finished (caller timeout or abort).
pub const OUTCOME_CANCELLED: &str = "cancelled";

/// Logs one stage event when finished, or `cancelled` when dropped first.
#[derive(Debug)]
pub struct StageTimer {
    stage: &'static str,
    started: Instant,
    done: bool,
}

impl StageTimer {
    /// Start timing `stage`.
    pub fn start(stage: &'static str) -> Self {
        Self {
            stage,
            started: Instant::now(),
            done: false,
        }
    }

    /// Emit the stage event with an explicit outcome label.
    pub fn finish(mut self, outcome: &'static str) {
        self.emit(outcome);
    }

    fn emit(&mut self, outcome: &'static str) {
        if self.done {
            return;
        }
        self.done = true;
        // Rounded to 0.1 ms so the field stays readable.
        let elapsed_ms = (self.started.elapsed().as_secs_f64() * 10_000.0).round() / 10.0;
        tracing::info!(
            target: "cognee::recall_stage",
            stage = self.stage,
            elapsed_ms,
            outcome,
            "recall stage"
        );
    }
}

impl Drop for StageTimer {
    fn drop(&mut self) {
        self.emit(OUTCOME_CANCELLED);
    }
}

/// Run `fut` as the named stage and log its duration and outcome.
///
/// The error value is only inspected for `is_err`; it is not logged.
pub async fn timed<T, E, F>(stage: &'static str, fut: F) -> Result<T, E>
where
    F: Future<Output = Result<T, E>>,
{
    let timer = StageTimer::start(stage);
    let result = fut.await;
    timer.finish(if result.is_ok() {
        OUTCOME_OK
    } else {
        OUTCOME_ERROR
    });
    result
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code — panics are acceptable failures"
)]
pub(crate) mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    #[derive(Clone, Default)]
    pub(crate) struct Buf(pub(crate) Arc<Mutex<Vec<u8>>>);

    impl std::io::Write for Buf {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(b);
            Ok(b.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for Buf {
        type Writer = Buf;
        fn make_writer(&'a self) -> Buf {
            self.clone()
        }
    }

    pub(crate) fn capture() -> (Buf, tracing::subscriber::DefaultGuard) {
        let buf = Buf::default();
        let sub = tracing_subscriber::fmt()
            .json()
            .with_writer(buf.clone())
            .with_max_level(tracing::Level::INFO)
            .finish();
        (buf, tracing::subscriber::set_default(sub))
    }

    pub(crate) fn events(buf: &Buf) -> Vec<serde_json::Value> {
        let raw = String::from_utf8(buf.0.lock().unwrap().clone()).unwrap();
        raw.lines()
            .map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap())
            .collect()
    }

    #[tokio::test(flavor = "current_thread")]
    async fn ok_stage_logs_stage_elapsed_and_outcome() {
        let (buf, _g) = capture();
        let r: Result<u8, ()> = timed(EMBED_QUERY, async { Ok(7) }).await;
        assert_eq!(r, Ok(7));
        let ev = events(&buf);
        assert_eq!(ev.len(), 1);
        assert_eq!(ev[0]["target"], TARGET);
        assert_eq!(ev[0]["fields"]["stage"], "embed_query");
        assert_eq!(ev[0]["fields"]["outcome"], "ok");
        assert!(ev[0]["fields"]["elapsed_ms"].as_f64().unwrap() >= 0.0);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn error_stage_logs_error_without_the_error_text() {
        let (buf, _g) = capture();
        let r: Result<(), String> = timed(VECTOR_SEARCH, async {
            Err("secret tenant detail".to_string())
        })
        .await;
        assert!(r.is_err());
        let raw = String::from_utf8(buf.0.lock().unwrap().clone()).unwrap();
        assert!(!raw.contains("secret tenant detail"));
        let ev = events(&buf);
        assert_eq!(ev[0]["fields"]["stage"], "vector_search");
        assert_eq!(ev[0]["fields"]["outcome"], "error");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn timeout_marks_the_running_stage_cancelled() {
        let (buf, _g) = capture();
        let outer = tokio::time::timeout(Duration::from_millis(20), async {
            let _total = StageTimer::start(TOTAL);
            let _: Result<(), ()> = timed(GRAPH_EXPANSION, async {
                tokio::time::sleep(Duration::from_secs(5)).await;
                Ok(())
            })
            .await;
        })
        .await;
        assert!(outer.is_err(), "the outer future must time out");
        let ev = events(&buf);
        let by_stage = |s: &str| {
            ev.iter()
                .find(|e| e["fields"]["stage"] == s)
                .unwrap_or_else(|| panic!("no event for stage {s}"))
        };
        assert_eq!(
            by_stage("graph_expansion")["fields"]["outcome"],
            "cancelled"
        );
        assert_eq!(by_stage("total")["fields"]["outcome"], "cancelled");
        assert!(
            by_stage("graph_expansion")["fields"]["elapsed_ms"]
                .as_f64()
                .unwrap()
                >= 15.0
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn finished_timer_does_not_also_log_cancelled() {
        let (buf, _g) = capture();
        StageTimer::start(TOTAL).finish(OUTCOME_OK);
        assert_eq!(events(&buf).len(), 1);
    }
}
