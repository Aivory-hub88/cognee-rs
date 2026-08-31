//! Aivory Cerveau patch: `cerveau-server` — the same `cognee-http-server`
//! binary, with one addition the OSS `bin` target deliberately doesn't have:
//! a real per-request `AuthResolver` (crates/http-server/src/auth_resolver.rs
//! is the exact injection seam this exists for; the OSS binary always builds
//! with `state.auth_resolver = None`, so `require_authentication=true` there
//! just 401s every request with no way to authenticate at all).
//!
//! `TenantHeaderResolver` maps Cerveau's own tenant identity straight into
//! cognee-rs's isolation boundary: `AuthenticatedUser.id` (a UUID) is what
//! every dataset/data row's owner_id is actually scoped by (see the UUID5
//! invariant comment on `default_user_from_state`). Two different Cerveau
//! tenant aliases (`t_<user_id>.<agent_type>`) hash to two different UUIDs
//! via the same `Uuid::new_v5(NAMESPACE_OID, ..)` scheme this codebase
//! already uses for its own default-user identity — so isolation here is
//! the same "structurally separate, not configured" shape as
//! `create_memory_for_tenant`'s empty cross-agent allowlist on the Cerveau
//! side, not a new pattern invented for this sidecar.
//!
//! Trust boundary: `X-Tenant-Id` / `X-Agent-Type` are only ever meaningful
//! when `X-Cerveau-Internal-Secret` matches `CERVEAU_INTERNAL_SECRET` —
//! this sidecar has no other caller-identity check, so without the shared
//! secret those two headers would let literally anyone impersonate any
//! tenant. The comparison is constant-time on purpose (this is the one
//! check standing between "isolated" and "not", so it gets the same rigor
//! a webhook secret would).

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::Context as _;
use async_trait::async_trait;
use axum::http::request::Parts;
use clap::Parser;
use cognee_http_server::auth::{AuthMethod, AuthenticatedUser};
use cognee_http_server::auth_resolver::AuthResolver;
use cognee_http_server::observability::{BufferConfig, SpanBuffer, SpanBufferLayer};
use cognee_http_server::{AppState, HttpServerConfig, RouterBuilder, wiring};
use uuid::Uuid;

#[derive(Parser, Debug)]
#[command(
    name = "cerveau-server",
    about = "cognee-http-server, with Cerveau's tenant-header AuthResolver wired in",
    version
)]
struct Args {
    #[arg(long, env = "HTTP_API_HOST", default_value = "0.0.0.0")]
    host: String,
    #[arg(long, env = "HTTP_API_PORT", default_value_t = 8000)]
    port: u16,
    #[arg(long, env = "CORS_ALLOWED_ORIGINS")]
    cors_allowed_origins: Option<String>,
    #[arg(long, env = "ENV", default_value = "prod")]
    env: String,
}

/// Constant-time byte compare -- deliberately not `==`, for the reason in
/// the module doc: this is the one gate standing between "tenant-isolated"
/// and "wide open to header spoofing".
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

struct TenantHeaderResolver {
    shared_secret: String,
}

#[async_trait]
impl AuthResolver for TenantHeaderResolver {
    async fn resolve(&self, parts: &mut Parts) -> Option<AuthenticatedUser> {
        let headers = &parts.headers;

        let secret = headers.get("x-cerveau-internal-secret")?.to_str().ok()?;
        if !constant_time_eq(secret.as_bytes(), self.shared_secret.as_bytes()) {
            return None;
        }

        let tenant_id = headers.get("x-tenant-id")?.to_str().ok()?.trim();
        let agent_type = headers.get("x-agent-type")?.to_str().ok()?.trim();
        if tenant_id.is_empty() || agent_type.is_empty() {
            return None;
        }

        // Mirrors Cerveau's own alias format exactly (crates/zeroclaw-runtime
        // agent/tenant.rs on the AVRY-Cerveau side): `t_<user_id>.<agent_type>`.
        let alias = format!("t_{tenant_id}.{agent_type}");
        let id = Uuid::new_v5(&Uuid::NAMESPACE_OID, alias.as_bytes());

        Some(AuthenticatedUser {
            id,
            email: format!("{alias}@cerveau.internal"),
            is_superuser: false,
            is_verified: true,
            is_active: true,
            tenant_id: None,
            auth_method: AuthMethod::ApiKey,
        })
    }
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> anyhow::Result<()> {
    let _ = dotenv::dotenv();

    let spans = Arc::new(SpanBuffer::new(BufferConfig::from_env()));
    let logging_cfg = match cognee_logging::LoggingConfig::from_env() {
        Ok(cfg) => cfg,
        Err(err) => {
            eprintln!("warning: invalid logging env var: {err}; falling back to defaults");
            cognee_logging::LoggingConfig::defaults()
        }
    };
    let span_buffer_layer: cognee_logging::BoxedLayer =
        Box::new(SpanBufferLayer::new((*spans).clone()));
    let _log_guards = cognee_logging::init_logging(logging_cfg, std::iter::once(span_buffer_layer));

    let args = Args::parse();

    let mut cfg = HttpServerConfig::from_env().context("failed to load config from environment")?;
    cfg.host = args.host;
    cfg.port = args.port;
    if let Some(origins) = args.cors_allowed_origins {
        cfg.cors_allowed_origins = origins
            .split(',')
            .map(|s| s.trim().to_owned())
            .filter(|s| !s.is_empty())
            .collect();
    }
    if let Ok(env_val) = args.env.parse() {
        cfg.env = env_val;
    }

    let shared_secret = std::env::var("CERVEAU_INTERNAL_SECRET")
        .context("CERVEAU_INTERNAL_SECRET must be set -- this binary's whole point is per-tenant \
                  isolation, and without this secret the tenant headers are unauthenticated")?;
    if shared_secret.trim().is_empty() {
        anyhow::bail!("CERVEAU_INTERNAL_SECRET is set but empty");
    }
    if !cfg.require_authentication {
        tracing::warn!(
            "require_authentication=false with cerveau-server -- the tenant resolver still runs, \
             but any request that fails it (wrong/missing secret or headers) falls through to the \
             single shared default user instead of 401. Set REQUIRE_AUTHENTICATION=true."
        );
    }

    let handles = wiring::wire_default_backends(&cfg)
        .await
        .context("failed to wire default backend handles")?;

    let mut state = AppState::build_with_db(cfg.clone(), handles.database.clone())
        .await
        .context("failed to build AppState with database")?;
    state.lib = Some(Arc::new(handles));
    state.install_real_health_checker();
    state.spans = spans;

    let app = RouterBuilder::new(state)
        .with_auth_resolver(Arc::new(TenantHeaderResolver { shared_secret }))
        .build()
        .await
        .context("failed to build router")?;

    let addr: SocketAddr = format!("{}:{}", cfg.host, cfg.port)
        .parse()
        .context("invalid bind address")?;
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("failed to bind {addr}"))?;

    tracing::info!("cerveau-server listening on {addr}, tenant isolation enforced");
    axum::serve(listener, app)
        .await
        .context("server error")?;

    Ok(())
}
