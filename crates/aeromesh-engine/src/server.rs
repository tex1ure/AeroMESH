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

use crate::pipeline::PipelineCoordinatorClient;

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

pub struct AppState {
    pub coordinator: Mutex<PipelineCoordinatorClient>,
    pub model_name: String,
}

pub struct PipelineHttpServer {
    state: Arc<AppState>,
}

impl PipelineHttpServer {
    pub fn new(coordinator: PipelineCoordinatorClient, model_name: String) -> Self {
        Self {
            state: Arc::new(AppState {
                coordinator: Mutex::new(coordinator),
                model_name,
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

async fn handle_health() -> &'static str {
    "OK"
}

async fn handle_models(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "object": "list",
        "data": [
            {
                "id": state.model_name,
                "object": "model",
                "created": SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs(),
                "owned_by": "aeromesh-cluster"
            }
        ]
    }))
}

async fn handle_cluster_status(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let client = state.coordinator.lock().await;
    let local_slice = &client.local_slice;
    let workers: Vec<String> = client.worker_addrs.iter().map(|w| w.to_string()).collect();

    Json(serde_json::json!({
        "cluster_status": "ONLINE",
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

async fn handle_chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    let is_streaming = req.stream.unwrap_or(true);
    let prompt = req
        .messages
        .last()
        .map(|m| m.content.clone())
        .unwrap_or_default();

    let max_tokens = req.max_tokens.unwrap_or(256);
    let temp = req.temperature.unwrap_or(0.7);
    let top_p = req.top_p.unwrap_or(0.9);
    let session_id = req.session_id.unwrap_or_else(|| {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    });

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

        let model_name = state.model_name.clone();
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
                    model: state.model_name.clone(),
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
