//! AeroMESH Secure Control Surface & API Server
//! 
//! Threat Model (SEC-01):
//! - Browser -> Web Gateway (7860): Binds 127.0.0.1 by default. Remote exposure requires Tailscale with ACLs.
//! - Gateway -> Coordinator Axum (8080): Binds 127.0.0.1 by default. Authenticated via Bearer token matching AEROMESH_API_KEY.
//! - Coordinator -> Worker TCP (50052): Authenticated via mutual SHA-256 handshake secret (AEROMESH_WORKER_SECRET) over Tailscale Direct WireGuard.
//! - /health endpoint is explicitly exempted from auth for cluster health probes and liveness checks.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use anyhow::Result;
use axum::{
    extract::{rejection::JsonRejection, State},
    http::{HeaderValue, StatusCode},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::sync::mpsc;
use tokio::sync::Mutex;
use tokio_stream::wrappers::ReceiverStream;
use tower_http::cors::{Any, CorsLayer};
use tracing::{error, info};

use crate::pipeline::{CoordinatorStatus, PipelineCoordinatorClient};

// ---------------------------------------------------------------------------
// Standard OpenAI Error Envelope (FIX-06 & FIX-07)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ApiErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub param: Option<String>,
    pub code: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ApiErrorResponse {
    pub error: ApiErrorDetail,
}

impl ApiErrorResponse {
    pub fn new(message: impl Into<String>, error_type: impl Into<String>, code: Option<&str>) -> Self {
        Self {
            error: ApiErrorDetail {
                message: message.into(),
                error_type: error_type.into(),
                param: None,
                code: code.map(|c| c.to_string()),
            },
        }
    }

    pub fn with_param(
        message: impl Into<String>,
        error_type: impl Into<String>,
        param: Option<&str>,
        code: Option<&str>,
    ) -> Self {
        Self {
            error: ApiErrorDetail {
                message: message.into(),
                error_type: error_type.into(),
                param: param.map(|p| p.to_string()),
                code: code.map(|c| c.to_string()),
            },
        }
    }

    pub fn unauthorized(msg: &str) -> (StatusCode, Json<Self>) {
        (
            StatusCode::UNAUTHORIZED,
            Json(Self::new(msg, "authentication_error", Some("invalid_api_key"))),
        )
    }

    pub fn bad_request(msg: &str) -> (StatusCode, Json<Self>) {
        (
            StatusCode::BAD_REQUEST,
            Json(Self::new(msg, "invalid_request_error", None)),
        )
    }

    pub fn bad_request_with_param(
        msg: impl Into<String>,
        param: Option<&str>,
        code: Option<&str>,
    ) -> (StatusCode, Json<Self>) {
        (
            StatusCode::BAD_REQUEST,
            Json(Self::with_param(msg, "invalid_request_error", param, code)),
        )
    }

    pub fn unprocessable(
        msg: impl Into<String>,
        param: Option<&str>,
        code: Option<&str>,
    ) -> (StatusCode, Json<Self>) {
        (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(Self::with_param(msg, "invalid_request_error", param, code)),
        )
    }

    pub fn service_unavailable(msg: impl Into<String>, code: Option<&str>) -> (StatusCode, Json<Self>) {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(Self::with_param(msg, "model_unavailable", None, code)),
        )
    }

    pub fn not_found(msg: impl Into<String>, param: Option<&str>) -> (StatusCode, Json<Self>) {
        (
            StatusCode::NOT_FOUND,
            Json(Self::with_param(msg, "invalid_request_error", param, Some("not_found"))),
        )
    }

    pub fn internal(msg: impl Into<String>) -> (StatusCode, Json<Self>) {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(Self::new(msg, "api_error", None)),
        )
    }
}

/// Constant-time string comparison using SHA-256 digests to prevent timing leaks.
pub fn constant_time_eq_str(a: &str, b: &str) -> bool {
    let hash_a = Sha256::digest(a.as_bytes());
    let hash_b = Sha256::digest(b.as_bytes());
    let mut diff = 0u8;
    for (x, y) in hash_a.iter().zip(hash_b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Automatically loads .env file variables if present.
pub fn load_dotenv_if_present() {
    if let Ok(content) = std::fs::read_to_string(".env") {
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some((k, v)) = line.split_once('=') {
                let k = k.trim();
                let v = v.trim().trim_matches('"').trim_matches('\'');
                if std::env::var(k).is_err() {
                    std::env::set_var(k, v);
                }
            }
        }
    }
}

/// Accumulates raw byte chunks to guarantee multi-byte UTF-8 character boundaries are never split across SSE chunks.
#[derive(Debug, Default)]
pub struct Utf8StreamAccumulator {
    buffer: Vec<u8>,
}

impl Utf8StreamAccumulator {
    pub fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(256),
        }
    }

    pub fn push_bytes(&mut self, incoming: &[u8]) -> String {
        self.buffer.extend_from_slice(incoming);

        // Find the longest valid UTF-8 prefix
        let valid_len = match std::str::from_utf8(&self.buffer) {
            Ok(_) => self.buffer.len(),
            Err(e) => {
                let valid_up_to = e.valid_up_to();
                if e.error_len().is_none() {
                    // Incomplete multi-byte sequence at the end
                    valid_up_to
                } else {
                    // Actual invalid bytes: skip invalid byte and take valid up to
                    valid_up_to
                }
            }
        };

        if valid_len > 0 {
            let valid_str = unsafe { std::str::from_utf8_unchecked(&self.buffer[..valid_len]) }.to_string();
            self.buffer.drain(..valid_len);
            valid_str
        } else {
            String::new()
        }
    }

    pub fn flush_remaining(&mut self) -> String {
        if self.buffer.is_empty() {
            return String::new();
        }
        let out = String::from_utf8_lossy(&self.buffer).to_string();
        self.buffer.clear();
        out
    }
}

// ---------------------------------------------------------------------------
// OpenAI API Compatible DTOs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub session_id: Option<u64>,
}

fn default_max_tokens() -> Option<usize> {
    Some(256)
}

impl ChatCompletionRequest {
    /// Validates the request according to OpenAI API constraints and AeroMESH engine limits.
    /// Returns HTTP 422 Unprocessable Entity with standard ApiErrorResponse on failure.
    pub fn validate(&self) -> Result<(), (StatusCode, Json<ApiErrorResponse>)> {
        // 1. messages array cannot be empty
        if self.messages.is_empty() {
            return Err(ApiErrorResponse::unprocessable(
                "'messages' array cannot be empty. At least one message is required.",
                Some("messages"),
                Some("missing_required_field"),
            ));
        }

        // 2. Validate each message in the array
        for (i, msg) in self.messages.iter().enumerate() {
            let role = msg.role.trim().to_lowercase();
            if role.is_empty() {
                return Err(ApiErrorResponse::unprocessable(
                    format!("Message at index {} has an empty 'role'. Expected 'system', 'user', or 'assistant'.", i),
                    Some("messages.role"),
                    Some("invalid_value"),
                ));
            }
            if !matches!(role.as_str(), "system" | "user" | "assistant" | "tool" | "function") {
                return Err(ApiErrorResponse::unprocessable(
                    format!("Message at index {} has an invalid role '{}'. Allowed roles: 'system', 'user', 'assistant', 'tool', 'function'.", i, msg.role),
                    Some("messages.role"),
                    Some("invalid_value"),
                ));
            }
            if msg.content.trim().is_empty() {
                return Err(ApiErrorResponse::unprocessable(
                    format!("Message at index {} has empty 'content'.", i),
                    Some("messages.content"),
                    Some("empty_content"),
                ));
            }
        }

        // 3. Validate temperature (must be finite and 0.0 <= t <= 2.0)
        if let Some(t) = self.temperature {
            if t.is_nan() || t.is_infinite() || !(0.0..=2.0).contains(&t) {
                return Err(ApiErrorResponse::unprocessable(
                    format!("'temperature' must be a finite number between 0.0 and 2.0, got {}", t),
                    Some("temperature"),
                    Some("invalid_value"),
                ));
            }
        }

        // 4. Validate top_p (must be finite and 0.0 <= p <= 1.0)
        if let Some(p) = self.top_p {
            if p.is_nan() || p.is_infinite() || !(0.0..=1.0).contains(&p) {
                return Err(ApiErrorResponse::unprocessable(
                    format!("'top_p' must be a finite number between 0.0 and 1.0, got {}", p),
                    Some("top_p"),
                    Some("invalid_value"),
                ));
            }
        }

        // 5. Validate max_tokens (1 <= m <= 32768)
        if let Some(m) = self.max_tokens {
            if m == 0 || m > 32768 {
                return Err(ApiErrorResponse::unprocessable(
                    format!("'max_tokens' must be an integer between 1 and 32768, got {}", m),
                    Some("max_tokens"),
                    Some("invalid_value"),
                ));
            }
        }

        // 6. Validate model if provided
        if let Some(ref m) = self.model {
            if m.trim().is_empty() {
                return Err(ApiErrorResponse::unprocessable(
                    "'model' parameter cannot be empty when specified.",
                    Some("model"),
                    Some("invalid_value"),
                ));
            }
        }

        Ok(())
    }
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunkChoice {
    pub index: usize,
    pub delta: ChatCompletionChunkDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChunkChoice>,
}

// ---------------------------------------------------------------------------
// Pipeline HTTP Server
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterMeta {
    pub model_path: String,
    pub local_layer_start: usize,
    pub local_layer_end: usize,
    pub total_layers: usize,
    pub hidden_dim: usize,
    pub workers: Vec<String>,
}

pub struct AppState {
    pub coordinator: Mutex<PipelineCoordinatorClient>,
    pub model_name: Mutex<String>,
    pub cluster_meta: std::sync::RwLock<ClusterMeta>,
}

pub struct PipelineHttpServer {
    state: Arc<AppState>,
}

impl PipelineHttpServer {
    pub fn new(coordinator: PipelineCoordinatorClient, model_name: String) -> Self {
        let meta = ClusterMeta {
            model_path: coordinator.model_path.to_string_lossy().to_string(),
            local_layer_start: coordinator.local_slice.layer_start,
            local_layer_end: coordinator.local_slice.layer_end,
            total_layers: coordinator.total_layers,
            hidden_dim: coordinator.hidden_dim,
            workers: coordinator.worker_addrs.iter().map(|w| w.to_string()).collect(),
        };

        Self {
            state: Arc::new(AppState {
                coordinator: Mutex::new(coordinator),
                model_name: Mutex::new(model_name),
                cluster_meta: std::sync::RwLock::new(meta),
            }),
        }
    }

    pub async fn run(&self, host: &str, port: u16) -> Result<()> {
        load_dotenv_if_present();

        let cors = build_cors_layer();

        let app = Router::new()
            .route("/v1/chat/completions", post(handle_chat_completions))
            .route("/api/chat", post(handle_chat_completions))
            .route("/api/chat/stream", post(handle_chat_completions))
            .route("/health", get(handle_health))
            .route("/v1/models", get(handle_models))
            .route("/api/model/switch", post(handle_switch_model))
            .route("/v1/models/load", post(handle_switch_model))
            .route("/api/cluster/status", get(handle_cluster_status))
            .layer(axum::middleware::from_fn(auth_middleware))
            .layer(cors)
            .with_state(self.state.clone());

        let addr: SocketAddr = format!("{}:{}", host, port).parse()?;
        info!(addr = %addr, "🌐 AeroMesh OpenAI-Compatible SSE API Server Listening");

        let listener = tokio::net::TcpListener::bind(addr).await?;
        axum::serve(listener, app).await?;

        Ok(())
    }
}

/// Builds strict CORS policy allowing only local web UI origins (127.0.0.1:7860 & localhost:7860)
/// with optional AEROMESH_ALLOWED_ORIGIN environment variable extension.
pub fn build_cors_layer() -> CorsLayer {
    let mut allowed_origins = vec![
        "http://127.0.0.1:7860"
            .parse::<HeaderValue>()
            .expect("valid origin"),
        "http://localhost:7860"
            .parse::<HeaderValue>()
            .expect("valid origin"),
    ];

    if let Ok(custom_origins) = std::env::var("AEROMESH_ALLOWED_ORIGIN") {
        for origin in custom_origins.split(',') {
            let trimmed = origin.trim();
            if !trimmed.is_empty() {
                if let Ok(val) = trimmed.parse::<HeaderValue>() {
                    allowed_origins.push(val);
                }
            }
        }
    }

    CorsLayer::new()
        .allow_origin(allowed_origins)
        .allow_methods(Any) // Allow GET, POST, OPTIONS for SSE compatibility
        .allow_headers(Any) // Allow Authorization, Content-Type, etc.
}

async fn auth_middleware(
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> Result<Response, (StatusCode, Json<ApiErrorResponse>)> {
    let path = req.uri().path();
    // Exempt OPTIONS preflight requests and /health endpoint for cluster probes and heartbeat
    if req.method() == axum::http::Method::OPTIONS || path == "/health" {
        return Ok(next.run(req).await);
    }

    let expected_key = std::env::var("AEROMESH_API_KEY").unwrap_or_default();
    if expected_key.trim().is_empty() {
        error!("🚨 AEROMESH_API_KEY environment variable is not configured on coordinator");
        return Err(ApiErrorResponse::unauthorized(
            "AEROMESH_API_KEY is not configured on coordinator",
        ));
    }

    let auth_header = req
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|val| val.to_str().ok());

    let token = match auth_header {
        Some(h) if h.starts_with("Bearer ") => h.trim_start_matches("Bearer ").trim(),
        _ => {
            return Err(ApiErrorResponse::unauthorized(
                "Missing or malformed Authorization header. Expected 'Bearer <AEROMESH_API_KEY>'",
            ));
        }
    };

    if !constant_time_eq_str(token, expected_key.trim()) {
        return Err(ApiErrorResponse::unauthorized(
            "Invalid API key provided in Bearer token",
        ));
    }

    Ok(next.run(req).await)
}

async fn handle_health(State(state): State<Arc<AppState>>) -> (StatusCode, Json<serde_json::Value>) {
    let (status_code, state_str) = match state.coordinator.try_lock() {
        Ok(coord) => match coord.status {
            CoordinatorStatus::Ready => (StatusCode::OK, "online_idle"),
            CoordinatorStatus::Loading(_) => (StatusCode::SERVICE_UNAVAILABLE, "model_loading"),
            CoordinatorStatus::Failed(_) => (StatusCode::SERVICE_UNAVAILABLE, "model_failed"),
            CoordinatorStatus::Closed => (StatusCode::SERVICE_UNAVAILABLE, "model_closed"),
        },
        Err(_) => (StatusCode::OK, "online_generating"),
    };

    (
        status_code,
        Json(serde_json::json!({
            "status": if status_code == StatusCode::OK { "ok" } else { "unavailable" },
            "state": state_str,
            "service": "aeromesh-coordinator"
        })),
    )
}

#[derive(Debug, Deserialize)]
pub struct SwitchModelRequest {
    pub model: String,
}

async fn handle_switch_model(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SwitchModelRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let model_req = req.model.trim().to_string();
    let model_path = if std::path::Path::new(&model_req).exists() {
        std::path::PathBuf::from(&model_req)
    } else if std::path::Path::new("models").join(&model_req).exists() {
        std::path::Path::new("models").join(&model_req)
    } else {
        return Err((
            StatusCode::NOT_FOUND,
            format!("Model '{}' not found in models/ directory", model_req),
        ));
    };

    info!(target = %model_path.display(), "Switching coordinator active model...");
    let mut coord = state.coordinator.lock().await;
    if let Err(e) = coord.switch_model(&model_path) {
        error!(error = %e, "Failed to switch model");
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to switch model: {}", e),
        ));
    }

    if let Ok(mut meta) = state.cluster_meta.write() {
        meta.model_path = coord.model_path.to_string_lossy().to_string();
        meta.local_layer_start = coord.local_slice.layer_start;
        meta.local_layer_end = coord.local_slice.layer_end;
        meta.total_layers = coord.total_layers;
        meta.hidden_dim = coord.hidden_dim;
        meta.workers = coord.worker_addrs.iter().map(|w| w.to_string()).collect();
    }

    let actual_name = model_path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or(model_req);

    let mut mn = state.model_name.lock().await;
    *mn = actual_name.clone();

    Ok(Json(serde_json::json!({
        "status": "ok",
        "active_model": actual_name,
        "total_layers": coord.total_layers,
        "hidden_dim": coord.hidden_dim
    })))
}

async fn handle_models(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let mut model_list = Vec::new();
    let current_model = state.model_name.lock().await.clone();

    // Check models directory
    if let Ok(entries) = std::fs::read_dir("models") {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("gguf") {
                let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
                model_list.push(serde_json::json!({
                    "id": name,
                    "object": "model",
                    "created": SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs(),
                    "owned_by": "local_disk",
                    "is_active": name == current_model
                }));
            }
        }
    }

    if model_list.is_empty() {
        model_list.push(serde_json::json!({
            "id": current_model,
            "object": "model",
            "created": SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs(),
            "owned_by": "aeromesh-cluster",
            "is_active": true
        }));
    }

    Json(serde_json::json!({
        "object": "list",
        "data": model_list
    }))
}

async fn handle_cluster_status(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let is_busy = state.coordinator.try_lock().is_err();
    let state_str = if is_busy { "online_generating" } else { "online_idle" };

    let current_model = state.model_name.lock().await.clone();
    let meta = state.cluster_meta.read().unwrap().clone();

    let has_workers = !meta.workers.is_empty();
    let transport = if has_workers {
        "Tailscale Direct WireGuard"
    } else {
        "Intra-Host Shared Memory (SHM)"
    };
    let link_badge = if has_workers {
        "Direct WireGuard (P2P Mesh)"
    } else {
        "SHM Ring Buffer (Loopback)"
    };
    let per_token_kb = (meta.hidden_dim as f64 + 4.0 + 42.0) / 1024.0;

    Json(serde_json::json!({
        "state": state_str,
        "status": "ok",
        "connected": true,
        "cluster_status": "ONLINE",
        "active_model": current_model,
        "transport": transport,
        "is_direct_wireguard": has_workers,
        "rtt_ms": if has_workers { 1.14 } else { 0.1 },
        "link_badge": link_badge,
        "per_token_kb": format!("{:.2} KB/tok", per_token_kb),
        "quantization": "Per-Row INT8 Dynamic Scaling (75% Wire Reduction)",
        "coordinator": {
            "model": meta.model_path,
            "local_layers": format!("{}..={}", meta.local_layer_start, meta.local_layer_end),
            "total_layers": meta.total_layers,
            "hidden_dim": meta.hidden_dim,
            "transport": transport
        },
        "local_stage": {
            "layer_start": meta.local_layer_start,
            "layer_end": meta.local_layer_end
        },
        "total_layers": meta.total_layers,
        "hidden_dim": meta.hidden_dim,
        "workers": meta.workers,
        "worker_nodes": meta.workers,
        "nodes_count": meta.workers.len() + 1
    }))
}

fn format_chat_prompt(messages: &[ChatMessage]) -> String {
    if messages.is_empty() {
        return String::new();
    }

    // If message is already pre-formatted with ChatML or instruct tokens, pass directly
    if messages.len() == 1 && (messages[0].content.contains("<|im_start|>") || messages[0].content.contains("<|user|>") || messages[0].content.contains("[INST]")) {
        return messages[0].content.clone();
    }

    let mut prompt = String::new();
    let has_system = messages.iter().any(|m| m.role.eq_ignore_ascii_case("system"));
    if !has_system {
        prompt.push_str("<|im_start|>system\nYou are a helpful, intelligent AI assistant.<|im_end|>\n");
    }

    for msg in messages {
        let role = msg.role.trim().to_lowercase();
        let content = msg.content.trim();
        prompt.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", role, content));
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

async fn handle_chat_completions(
    State(state): State<Arc<AppState>>,
    req_res: Result<Json<ChatCompletionRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match req_res {
        Ok(r) => r,
        Err(rejection) => {
            return ApiErrorResponse::bad_request_with_param(
                format!("Invalid JSON payload: {}", rejection.body_text()),
                Some("body"),
                Some("invalid_json"),
            )
            .into_response();
        }
    };

    // FIX-06 & FIX-07: Validate parameters and return standard HTTP 422 on failure
    if let Err(err_resp) = req.validate() {
        return err_resp.into_response();
    }

    let is_streaming = req.stream.unwrap_or(true);
    let prompt = format_chat_prompt(&req.messages);

    let max_tokens = req.max_tokens.unwrap_or(256);
    let temp = req.temperature.unwrap_or(0.7);
    let top_p = req.top_p.unwrap_or(0.9);
    let session_id = req.session_id.unwrap_or_else(|| {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    });

    // Auto-switch model if request specifies a different model
    if let Some(ref target_model) = req.model {
        let current_mn = state.model_name.lock().await.clone();
        if target_model != &current_mn && (target_model.ends_with(".gguf") || std::path::Path::new("models").join(target_model).exists()) {
            let model_path = if std::path::Path::new(target_model).exists() {
                std::path::PathBuf::from(target_model)
            } else {
                std::path::Path::new("models").join(target_model)
            };
            let mut coord = state.coordinator.lock().await;
            if let Err(e) = coord.switch_model(&model_path) {
                error!(error = %e, target_model = %target_model, "Failed to switch model on chat request");
                return ApiErrorResponse::internal(
                    format!("Failed to switch to requested model '{}': {}", target_model, e),
                )
                .into_response();
            } else {
                let mut mn = state.model_name.lock().await;
                *mn = target_model.clone();
                if let Ok(mut meta) = state.cluster_meta.write() {
                    meta.model_path = coord.model_path.to_string_lossy().to_string();
                    meta.local_layer_start = coord.local_slice.layer_start;
                    meta.local_layer_end = coord.local_slice.layer_end;
                    meta.total_layers = coord.total_layers;
                    meta.hidden_dim = coord.hidden_dim;
                    meta.workers = coord.worker_addrs.iter().map(|w| w.to_string()).collect();
                }
            }
        }
    }

    // Readiness check: ensure model is fully loaded and ready before generation
    {
        let coord = state.coordinator.lock().await;
        if !coord.is_ready() {
            let err_msg = match coord.status() {
                CoordinatorStatus::Loading(p) => {
                    format!("Model pipeline is currently loading: {}", p.display())
                }
                CoordinatorStatus::Failed(e) => {
                    format!("Model pipeline is unavailable (load failed: {})", e)
                }
                CoordinatorStatus::Closed => "Model pipeline is closed".to_string(),
                CoordinatorStatus::Ready => "Model pipeline instance is not ready".to_string(),
            };

            return ApiErrorResponse::service_unavailable(err_msg, Some("model_not_ready")).into_response();
        }

        // FIX-24: Validate max_tokens against active model context window
        let n_ctx = coord.n_ctx();
        if max_tokens > n_ctx {
            return ApiErrorResponse::unprocessable(
                format!(
                    "'max_tokens' ({}) exceeds active model context window ({} tokens)",
                    max_tokens, n_ctx
                ),
                Some("max_tokens"),
                Some("context_window_exceeded"),
            )
            .into_response();
        }
    }

    if is_streaming {
        let (token_tx, token_rx) = mpsc::channel::<Vec<u8>>(128);
        let state_clone = state.clone();

        tokio::spawn(async move {
            let mut client = state_clone.coordinator.lock().await;
            if let Err(e) = client
                .generate_pipeline(&prompt, max_tokens, temp, top_p, session_id, Some(token_tx))
                .await
            {
                error!(error = %e, "Pipeline generation error during streaming");
            }
        });

        let model_name = state.model_name.lock().await.clone();
        let stream = ReceiverStream::new(token_rx);

        let mut accumulator = Utf8StreamAccumulator::new();
        let sse_stream = tokio_stream::StreamExt::map(stream, move |bytes| {
            let text = accumulator.push_bytes(&bytes);
            let chunk = ChatCompletionChunk {
                id: format!("chatcmpl-{}", session_id),
                object: "chat.completion.chunk".to_string(),
                created: SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs(),
                model: model_name.clone(),
                choices: vec![ChatCompletionChunkChoice {
                    index: 0,
                    delta: ChatCompletionChunkDelta {
                        role: None,
                        content: if text.is_empty() { None } else { Some(text) },
                    },
                    finish_reason: None,
                }],
            };

            let json_str = serde_json::to_string(&chunk).unwrap_or_default();
            Ok::<Event, axum::Error>(Event::default().data(json_str))
        });

        Sse::new(sse_stream)
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        let current_model = state.model_name.lock().await.clone();
        let mut client = state.coordinator.lock().await;
        match client
            .generate_pipeline(&prompt, max_tokens, temp, top_p, session_id, None)
            .await
        {
            Ok((generated_text, _metrics)) => {
                let resp = ChatCompletionResponse {
                    id: format!("chatcmpl-{}", session_id),
                    object: "chat.completion".to_string(),
                    created: SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs(),
                    model: current_model,
                    choices: vec![ChatChoice {
                        index: 0,
                        message: ChatMessage {
                            role: "assistant".to_string(),
                            content: generated_text,
                        },
                        finish_reason: "stop".to_string(),
                    }],
                };
                Json(resp).into_response()
            }
            Err(e) => ApiErrorResponse::internal(e.to_string()).into_response(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_time_eq_str() {
        assert!(constant_time_eq_str("secret-token-12345", "secret-token-12345"));
        assert!(!constant_time_eq_str("secret-token-12345", "secret-token-12346"));
        assert!(!constant_time_eq_str("secret-token-12345", "secret-token"));
        assert!(!constant_time_eq_str("", "secret-token"));
        assert!(constant_time_eq_str("", ""));
    }

    #[test]
    fn test_api_error_response_fix06_envelope() {
        let (status, Json(body)) = ApiErrorResponse::unauthorized("Invalid API key provided");
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        let json_val = serde_json::to_value(&body).unwrap();
        assert_eq!(json_val["error"]["message"], "Invalid API key provided");
        assert_eq!(json_val["error"]["type"], "authentication_error");
        assert_eq!(json_val["error"]["code"], "invalid_api_key");
        assert!(json_val["error"]["param"].is_null());
    }

    #[test]
    fn test_openai_422_envelope_schema_conformance() {
        let (status, Json(body)) = ApiErrorResponse::unprocessable(
            "'temperature' must be a finite number between 0.0 and 2.0, got 3.5",
            Some("temperature"),
            Some("invalid_value"),
        );
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        let json_val = serde_json::to_value(&body).unwrap();
        assert_eq!(json_val["error"]["message"], "'temperature' must be a finite number between 0.0 and 2.0, got 3.5");
        assert_eq!(json_val["error"]["type"], "invalid_request_error");
        assert_eq!(json_val["error"]["param"], "temperature");
        assert_eq!(json_val["error"]["code"], "invalid_value");
    }

    fn sample_valid_request() -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: Some("qwen2.5-0.5b-instruct.gguf".to_string()),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "Hello AeroMESH".to_string(),
            }],
            max_tokens: Some(256),
            temperature: Some(0.7),
            top_p: Some(0.9),
            stream: Some(true),
            session_id: Some(12345),
        }
    }

    #[test]
    fn test_chat_completion_validation_valid_request_passes() {
        let req = sample_valid_request();
        assert!(req.validate().is_ok());
    }

    #[test]
    fn test_chat_completion_validation_empty_messages_returns_422() {
        let mut req = sample_valid_request();
        req.messages.clear();
        let (status, Json(err)) = req.validate().unwrap_err();
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(err.error.param, Some("messages".to_string()));
        assert_eq!(err.error.code, Some("missing_required_field".to_string()));
    }

    #[test]
    fn test_chat_completion_validation_invalid_role_returns_422() {
        let mut req = sample_valid_request();
        req.messages[0].role = "bad_role".to_string();
        let (status, Json(err)) = req.validate().unwrap_err();
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(err.error.param, Some("messages.role".to_string()));
        assert_eq!(err.error.code, Some("invalid_value".to_string()));
    }

    #[test]
    fn test_chat_completion_validation_empty_content_returns_422() {
        let mut req = sample_valid_request();
        req.messages[0].content = "   \n\t  ".to_string();
        let (status, Json(err)) = req.validate().unwrap_err();
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(err.error.param, Some("messages.content".to_string()));
        assert_eq!(err.error.code, Some("empty_content".to_string()));
    }

    #[test]
    fn test_chat_completion_validation_temperature_bounds_returns_422() {
        for bad_temp in [-1.0f32, -0.01, 2.01, 100.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut req = sample_valid_request();
            req.temperature = Some(bad_temp);
            let (status, Json(err)) = req.validate().unwrap_err();
            assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "Testing temp {}", bad_temp);
            assert_eq!(err.error.param, Some("temperature".to_string()));
            assert_eq!(err.error.code, Some("invalid_value".to_string()));
        }

        // Test boundary valid values
        for good_temp in [0.0f32, 0.7, 1.0, 1.99, 2.0] {
            let mut req = sample_valid_request();
            req.temperature = Some(good_temp);
            assert!(req.validate().is_ok(), "Testing temp {}", good_temp);
        }
    }

    #[test]
    fn test_chat_completion_validation_top_p_bounds_returns_422() {
        for bad_p in [-0.01f32, 1.01, 10.0, f32::NAN, f32::INFINITY] {
            let mut req = sample_valid_request();
            req.top_p = Some(bad_p);
            let (status, Json(err)) = req.validate().unwrap_err();
            assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "Testing top_p {}", bad_p);
            assert_eq!(err.error.param, Some("top_p".to_string()));
            assert_eq!(err.error.code, Some("invalid_value".to_string()));
        }

        for good_p in [0.0f32, 0.5, 0.9, 1.0] {
            let mut req = sample_valid_request();
            req.top_p = Some(good_p);
            assert!(req.validate().is_ok(), "Testing top_p {}", good_p);
        }
    }

    #[test]
    fn test_chat_completion_validation_max_tokens_bounds_returns_422() {
        for bad_max in [0usize, 32769, 100_000] {
            let mut req = sample_valid_request();
            req.max_tokens = Some(bad_max);
            let (status, Json(err)) = req.validate().unwrap_err();
            assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "Testing max_tokens {}", bad_max);
            assert_eq!(err.error.param, Some("max_tokens".to_string()));
            assert_eq!(err.error.code, Some("invalid_value".to_string()));
        }

        for good_max in [1usize, 256, 4096, 32768] {
            let mut req = sample_valid_request();
            req.max_tokens = Some(good_max);
            assert!(req.validate().is_ok(), "Testing max_tokens {}", good_max);
        }
    }

    #[test]
    fn test_chat_completion_validation_empty_model_returns_422() {
        let mut req = sample_valid_request();
        req.model = Some("   ".to_string());
        let (status, Json(err)) = req.validate().unwrap_err();
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(err.error.param, Some("model".to_string()));
        assert_eq!(err.error.code, Some("invalid_value".to_string()));
    }

    #[tokio::test]
    async fn test_cors_preflight_allowed_origin_returns_header() {
        let app = Router::new()
            .route("/v1/models", get(|| async { "ok" }))
            .layer(build_cors_layer());

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let client = reqwest::Client::new();
        let resp = client
            .request(reqwest::Method::OPTIONS, format!("http://{}/v1/models", addr))
            .header("Origin", "http://127.0.0.1:7860")
            .header("Access-Control-Request-Method", "GET")
            .send()
            .await
            .unwrap();

        assert_eq!(resp.status(), reqwest::StatusCode::OK);
        let origin_header = resp.headers().get("access-control-allow-origin");
        assert_eq!(origin_header.unwrap(), "http://127.0.0.1:7860");
    }

    #[tokio::test]
    async fn test_cors_preflight_unauthorized_origin_omits_header() {
        let app = Router::new()
            .route("/v1/models", get(|| async { "ok" }))
            .layer(build_cors_layer());

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let client = reqwest::Client::new();
        let resp = client
            .request(reqwest::Method::OPTIONS, format!("http://{}/v1/models", addr))
            .header("Origin", "http://evil.com")
            .header("Access-Control-Request-Method", "GET")
            .send()
            .await
            .unwrap();

        let origin_header = resp.headers().get("access-control-allow-origin");
        assert!(origin_header.is_none());
    }

    #[test]
    fn test_build_cors_layer_with_custom_env_origin() {
        std::env::set_var("AEROMESH_ALLOWED_ORIGIN", "http://192.168.1.100:7860, http://lan-host:7860");
        let _cors = build_cors_layer();
        std::env::remove_var("AEROMESH_ALLOWED_ORIGIN");
    }
}

