//! Cerveau fork: a durable-fact gate in front of `POST /api/v1/add`.
//!
//! Cerveau's auto-ingest sends every "durable" memory update to the graph,
//! and a measured 7.5% of what landed there was noise (probe facts, task
//! ledger states, "all emails have been sent"), each one paying for LLM
//! extraction in cognify and cluttering recall. One TypeSafe Jev Noul
//! ("does this state a durable business fact?") screens the text first.
//!
//! Modes (`COGNEE_INGEST_GATE`):
//! - `off` (default): no call.
//! - `shadow`: store as usual, judge in the background, log the verdict
//!   (target `ingest_gate`) so a threshold can be chosen from real traffic.
//! - `enforce`: judge before storing and skip the add only when every text
//!   part scores below `COGNEE_INGEST_GATE_THRESHOLD` (default 0.10: on 185
//!   live memories everything under it was noise, while real preferences
//!   such as "prefers short answers" scored ~0.2).
//!
//! Always fail-open: a timeout, HTTP error or unparsable answer keeps the
//! memory. The gate can only drop what Jev confidently calls chatter.

use std::time::Duration;

use serde_json::{Value, json};

const DEFAULT_URL: &str = "https://openrouter.ai/api/v1/systemone";
const DEFAULT_MODEL: &str = "typesafe/jev-1.13";
const DEFAULT_THRESHOLD: f64 = 0.10;
/// Longest text sent for judgment; the start of a memory carries its point.
const MAX_JUDGED_CHARS: usize = 6000;
/// Text parts larger than this are documents, not memory facts: never gated.
pub const MAX_GATED_BYTES: u64 = 64 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateMode {
    Off,
    Shadow,
    Enforce,
}

pub fn mode() -> GateMode {
    parse_mode(std::env::var("COGNEE_INGEST_GATE").ok().as_deref())
}

fn parse_mode(raw: Option<&str>) -> GateMode {
    match raw.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        Some("shadow") => GateMode::Shadow,
        Some("enforce") => GateMode::Enforce,
        _ => GateMode::Off,
    }
}

pub fn threshold() -> f64 {
    std::env::var("COGNEE_INGEST_GATE_THRESHOLD")
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|t| (0.0..=1.0).contains(t))
        .unwrap_or(DEFAULT_THRESHOLD)
}

/// The System One request for one memory text.
pub fn request_body(model: &str, text: &str) -> Value {
    let clipped: String = text.chars().take(MAX_JUDGED_CHARS).collect();
    json!({
        "model": model,
        "state": { "memory": { "text": clipped } },
        "questions": {
            "durable": {
                "type": "noul",
                "instructions": "Does `memory.text` state a durable business fact worth keeping in a long-term knowledge graph?",
                "criteria": {
                    "true": "A concrete, lasting fact about people, organisations, deals, products, policies, preferences, commitments or relationships that will still matter later.",
                    "false": "Small talk, greetings, acknowledgements, transient status updates, or chatter with no lasting fact."
                }
            }
        }
    })
}

/// `answers.durable.noul` from a System One response, if well-formed.
pub fn parse_noul(body: &Value) -> Option<f64> {
    body.get("answers")?
        .get("durable")?
        .get("noul")?
        .as_f64()
        .filter(|p| (0.0..=1.0).contains(p))
}

/// Judge one text. `None` on any failure (the caller keeps the memory).
pub async fn judge(text: &str) -> Option<f64> {
    let key = std::env::var("COGNEE_INGEST_GATE_API_KEY")
        .ok()
        .filter(|k| !k.trim().is_empty())
        .or_else(|| std::env::var("LLM_API_KEY").ok())
        .filter(|k| !k.trim().is_empty())?;
    let url = std::env::var("COGNEE_INGEST_GATE_URL").unwrap_or_else(|_| DEFAULT_URL.to_string());
    let model = std::env::var("COGNEE_INGEST_GATE_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .ok()?;
    let resp = client
        .post(url)
        .bearer_auth(key.trim())
        .json(&request_body(&model, text))
        .send()
        .await
        .map_err(|e| tracing::warn!(target: "ingest_gate", error = %e, "judge request failed; keeping memory"))
        .ok()?;
    if !resp.status().is_success() {
        tracing::warn!(target: "ingest_gate", status = %resp.status(), "judge returned an error; keeping memory");
        return None;
    }
    let body: Value = resp.json().await.ok()?;
    parse_noul(&body)
}

/// Enforce-mode decision for one add: drop only when there was at least one
/// judged text and every judged text scored below the threshold. Unjudged
/// (failed) parts count as keep.
pub fn should_drop(scores: &[Option<f64>], threshold: f64) -> bool {
    !scores.is_empty() && scores.iter().all(|s| matches!(s, Some(p) if *p < threshold))
}

/// First ~120 chars on one line, for the shadow log.
pub fn preview(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ").chars().take(120).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modes_default_off() {
        assert_eq!(parse_mode(None), GateMode::Off);
        assert_eq!(parse_mode(Some("nonsense")), GateMode::Off);
        assert_eq!(parse_mode(Some(" Shadow ")), GateMode::Shadow);
        assert_eq!(parse_mode(Some("enforce")), GateMode::Enforce);
    }

    #[test]
    fn parses_system_one_answers() {
        let ok = json!({"answers": {"durable": {"type": "noul", "noul": 0.02}}});
        assert_eq!(parse_noul(&ok), Some(0.02));
        assert_eq!(parse_noul(&json!({"answers": {}})), None);
        assert_eq!(parse_noul(&json!({"answers": {"durable": {"noul": 7.0}}})), None);
    }

    #[test]
    fn drops_only_when_every_judged_part_is_chatter() {
        assert!(should_drop(&[Some(0.02)], 0.10));
        assert!(!should_drop(&[Some(0.02), Some(0.6)], 0.10));
        assert!(!should_drop(&[Some(0.02), None], 0.10), "a failed judgment keeps the add");
        assert!(!should_drop(&[], 0.10));
        assert!(!should_drop(&[Some(0.10)], 0.10), "at the threshold is kept");
    }

    #[test]
    fn request_clips_long_text() {
        let long = "x".repeat(MAX_JUDGED_CHARS + 50);
        let body = request_body(DEFAULT_MODEL, &long);
        let sent = body["state"]["memory"]["text"].as_str().unwrap_or_default();
        assert_eq!(sent.chars().count(), MAX_JUDGED_CHARS);
        assert_eq!(body["questions"]["durable"]["type"], "noul");
    }
}
