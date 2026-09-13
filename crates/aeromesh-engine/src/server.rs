use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use anyhow::Result;
use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio::sync::Mutex;
use tokio_stream::wrappers::ReceiverStream;
use tower_http::cors::CorsLayer;
use tracing::{error, info};

use crate::pipeline::{CoordinatorStatus, PipelineCoordinatorClient};

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

#[derive(Clone)]
pub struct AppState {
    pub coordinator: Arc<Mutex<PipelineCoordinatorClient>>,
    pub model_name: Arc<Mutex<String>>,
}

pub struct PipelineHttpServer {
    state: Arc<AppState>,
}

impl PipelineHttpServer {
    pub fn new(coordinator: PipelineCoordinatorClient, model_name: String) -> Self {
        Self {
            state: Arc::new(AppState {
                coordinator: Arc::new(Mutex::new(coordinator)),
                model_name: Arc::new(Mutex::new(model_name)),
            }),
        }
    }

    pub async fn run(&self, host: &str, port: u16) -> Result<()> {
        let app = Router::new()
            .route("/v1/chat/completions", post(handle_chat_completions))
            .route("/api/chat", post(handle_chat_completions))
            .route("/api/chat/stream", post(handle_chat_completions))
            .route("/health", get(handle_health))
            .route("/v1/models", get(handle_models))
            .route("/api/model/switch", post(handle_switch_model))
            .route("/v1/models/load", post(handle_switch_model))
            .route("/api/cluster/status", get(handle_cluster_status))
            .layer(CorsLayer::permissive())
            .with_state(self.state.clone());

        let addr: SocketAddr = format!("{}:{}", host, port).parse()?;
        info!(addr = %addr, "🌐 AeroMesh OpenAI-Compatible SSE API Server Listening");

        let listener = tokio::net::TcpListener::bind(addr).await?;
        axum::serve(listener, app).await?;

        Ok(())
    }
}

async fn handle_health(State(state): State<Arc<AppState>>) -> (StatusCode, &'static str) {
    let coord = state.coordinator.lock().await;
    match coord.status {
        CoordinatorStatus::Ready => (StatusCode::OK, "OK"),
        CoordinatorStatus::Loading(_) => (StatusCode::SERVICE_UNAVAILABLE, "MODEL_LOADING"),
        CoordinatorStatus::Failed(_) => (StatusCode::SERVICE_UNAVAILABLE, "MODEL_FAILED"),
        CoordinatorStatus::Closed => (StatusCode::SERVICE_UNAVAILABLE, "MODEL_CLOSED"),
    }
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
    let client = state.coordinator.lock().await;
    let local_slice = &client.local_slice;
    let workers: Vec<String> = client.worker_addrs.iter().map(|w| w.to_string()).collect();
    let current_model = state.model_name.lock().await.clone();

    Json(serde_json::json!({
        "cluster_status": "ONLINE",
        "active_model": current_model,
        "coordinator": {
            "model": client.model_path.to_string_lossy(),
            "local_layers": format!("{}..={}", local_slice.layer_start, local_slice.layer_end),
            "total_layers": local_slice.total_layers,
            "hidden_dim": client.hidden_dim,
            "transport": "Zero-Weight Activation Streaming"
        },
        "workers": workers,
        "nodes_count": workers.len() + 1
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
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
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
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({
                        "error": {
                            "message": format!("Failed to switch to requested model '{}': {}", target_model, e),
                            "type": "model_load_error",
                            "code": 500
                        }
                    })),
                ).into_response();
            } else {
                let mut mn = state.model_name.lock().await;
                *mn = target_model.clone();
            }
        }
    }

    // Readiness check: ensure model is fully loaded and ready before generation
    {
        let coord = state.coordinator.lock().await;
        if !coord.is_ready() {
            let (status_code, err_msg) = match coord.status() {
                CoordinatorStatus::Loading(p) => (
                    StatusCode::SERVICE_UNAVAILABLE,
                    format!("Model pipeline is currently loading: {}", p.display()),
                ),
                CoordinatorStatus::Failed(e) => (
                    StatusCode::SERVICE_UNAVAILABLE,
                    format!("Model pipeline is unavailable (load failed: {})", e),
                ),
                CoordinatorStatus::Closed => (
                    StatusCode::SERVICE_UNAVAILABLE,
                    "Model pipeline is closed".to_string(),
                ),
                CoordinatorStatus::Ready => (
                    StatusCode::SERVICE_UNAVAILABLE,
                    "Model pipeline instance is not ready".to_string(),
                ),
            };

            return (
                status_code,
                Json(serde_json::json!({
                    "error": {
                        "message": err_msg,
                        "type": "model_unavailable",
                        "code": status_code.as_u16()
                    }
                })),
            ).into_response();
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
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": e.to_string() })),
            )
                .into_response(),
        }
    }
}
