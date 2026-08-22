use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use anyhow::{bail, ensure, Context, Result};
use tokio::io::AsyncWriteExt;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, Mutex};
use tracing::{error, info, warn};

use aeromesh_core::{
    ActivationFrame, HandshakeRequest, HandshakeResponse, TokenResponseFrame,
    FLAG_CLEAR_KV, FLAG_IS_PROMPT,
};

use crate::slice_loader::{GgufSliceLoader, LayerSliceConfig, LlamaPipelineInstance};

pub const SOCKET_TIMEOUT: Duration = Duration::from_secs(60);

// ---------------------------------------------------------------------------
// Worker Service (Zero-Weight Pipeline Stage 2 Compute Engine)
// ---------------------------------------------------------------------------

pub struct PipelineWorkerService {
    pub model_path: PathBuf,
    pub slice_config: LayerSliceConfig,
    pub instance: Arc<Mutex<LlamaPipelineInstance>>,
}

impl PipelineWorkerService {
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        slice_config: LayerSliceConfig,
        n_gpu_layers: i32,
    ) -> Result<Self> {
        let path_ref = model_path.as_ref();
        let instance = LlamaPipelineInstance::open(path_ref, slice_config.clone(), n_gpu_layers, 4096)?;

        info!(
            model = %path_ref.display(),
            layers = %format!("{}..={}", slice_config.layer_start, slice_config.layer_end),
            "Initialized Pipeline Worker Service"
        );

        Ok(Self {
            model_path: path_ref.to_path_buf(),
            slice_config,
            instance: Arc::new(Mutex::new(instance)),
        })
    }

    pub async fn run_server(&self, host: &str, port: u16) -> Result<()> {
        let bind_addr = format!("{}:{}", host, port);
        let listener = TcpListener::bind(&bind_addr).await
            .context(format!("Failed to bind worker TCP socket to {}", bind_addr))?;

        info!(
            addr = %bind_addr,
            layers = %format!("{}..={}", self.slice_config.layer_start, self.slice_config.layer_end),
            "⚡ AeroMesh Pipeline Worker listening for coordinator activations"
        );

        loop {
            let (mut socket, peer_addr) = listener.accept().await?;
            socket.set_nodelay(true)?;
            let instance_clone = self.instance.clone();
            let slice_clone = self.slice_config.clone();

            tokio::spawn(async move {
                if let Err(e) = handle_coordinator_connection(&mut socket, peer_addr, instance_clone, slice_clone).await {
                    error!(peer = %peer_addr, error = %e, "Pipeline worker connection closed with error");
                }
            });
        }
    }
}

async fn handle_coordinator_connection(
    socket: &mut TcpStream,
    peer_addr: SocketAddr,
    instance: Arc<Mutex<LlamaPipelineInstance>>,
    slice_config: LayerSliceConfig,
) -> Result<()> {
    // Step 1: Handshake with timeout
    let handshake_fut = HandshakeRequest::decode_async(socket);
    let req = tokio::time::timeout(SOCKET_TIMEOUT, handshake_fut)
        .await
        .context("Handshake timeout")??;

    info!(
        peer = %peer_addr,
        arch = %req.model_architecture,
        hidden_dim = req.hidden_dim,
        total_layers = req.total_layers,
        "Received HandshakeRequest from Coordinator"
    );

    let is_last_stage = slice_config.is_last_stage;
    let resp = HandshakeResponse::ok();
    socket.write_all(&resp.encode()).await?;
    socket.flush().await?;
    info!(peer = %peer_addr, "Handshake accepted. Ready for activation stream.");

    // Step 2: Process incoming activation frames
    let mut active_session_id = 0u64;

    loop {
        let frame_fut = ActivationFrame::decode_async(socket);
        let frame = match tokio::time::timeout(Duration::from_secs(120), frame_fut).await {
            Ok(Ok(f)) => f,
            Ok(Err(_)) => {
                // Clean disconnect or EOF
                break;
            }
            Err(_) => {
                // Connection idle timeout
                break;
            }
        };

        let start_time = Instant::now();
        let mut inst = instance.lock().await;

        // Reset KV-Cache on session change or explicit flag
        if frame.header.session_id != active_session_id || frame.header.has_flag(FLAG_CLEAR_KV) {
            info!(
                session_id = frame.header.session_id,
                prev_session = active_session_id,
                "🧹 Resetting Stage 2 KV-Cache for new session"
            );
            inst.clear_kv_cache();
            active_session_id = frame.header.session_id;
        }

        let seq_len = frame.header.sequence_length.max(1);
        let start_pos = frame.header.token_position;
        let activations = frame.to_f32_vec()?;
        let seq_id = frame.header.sequence_id;

        // Execute Stage 2 GEMM forward pass from incoming activations
        inst.evaluate_activations_to_logits(&activations, seq_len, start_pos)?;

        let eval_duration = start_time.elapsed();
        let eval_time_ms = eval_duration.as_secs_f32() * 1000.0;

        if is_last_stage {
            let resp = TokenResponseFrame {
                session_id: active_session_id,
                sequence_id: seq_id,
                token_id: 0,
                is_eos: false,
                token_text: String::new(),
                eval_time_ms,
            };

            let resp_bytes = resp.encode();
            socket.write_all(&resp_bytes).await?;
            socket.flush().await?;

            println!(
                "  ⚡ [Stage 2 Worker] Step #{:<3} (pos: {}, S={}) | Recv: {:.2} KB payload | Processed in {:.2}ms",
                seq_id + 1,
                start_pos,
                seq_len,
                (frame.payload.len() as f64) / 1024.0,
                eval_time_ms
            );
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Coordinator Pipeline Client (Zero-Weight Pipeline Stage 1 Orchestrator)
// ---------------------------------------------------------------------------

pub struct PipelineCoordinatorClient {
    pub model_path: PathBuf,
    pub local_slice: LayerSliceConfig,
    pub worker_addrs: Vec<SocketAddr>,
    pub instance: LlamaPipelineInstance,
    pub hidden_dim: usize,
    pub total_layers: usize,
    active_stream: Option<TcpStream>,
}

impl PipelineCoordinatorClient {
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        worker_addrs: Vec<SocketAddr>,
        custom_slice: Option<LayerSliceConfig>,
        n_gpu_layers: i32,
    ) -> Result<Self> {
        let path_ref = model_path.as_ref();
        let loader = GgufSliceLoader::open(path_ref)?;
        let total_layers = loader.total_layers;

        let local_slice = if let Some(slice) = custom_slice {
            slice
        } else {
            let num_nodes = worker_addrs.len() + 1;
            let splits = loader.compute_balanced_splits(num_nodes);
            splits.first().cloned().unwrap_or(
                LayerSliceConfig::new(0, total_layers.saturating_sub(1) / 2, total_layers)?,
            )
        };

        let instance = LlamaPipelineInstance::open(path_ref, local_slice.clone(), n_gpu_layers, 4096)?;
        let hidden_dim = instance.hidden_dim;

        info!(
            model = %path_ref.display(),
            workers = worker_addrs.len(),
            local_layers = %format!("{}..={}", local_slice.layer_start, local_slice.layer_end),
            hidden_dim = hidden_dim,
            "Initialized Pipeline Coordinator Client"
        );

        Ok(Self {
            model_path: path_ref.to_path_buf(),
            local_slice,
            worker_addrs,
            instance,
            hidden_dim,
            total_layers,
            active_stream: None,
        })
    }

    pub fn switch_model<P: AsRef<Path>>(&mut self, new_model_path: P) -> Result<()> {
        let path_ref = new_model_path.as_ref();
        if !path_ref.exists() {
            bail!("Model file not found at {:?}", path_ref);
        }

        let loader = GgufSliceLoader::open(path_ref)?;
        let total_layers = loader.total_layers;
        let num_nodes = self.worker_addrs.len() + 1;
        let splits = loader.compute_balanced_splits(num_nodes);
        let local_slice = splits.first().cloned().unwrap_or(
            LayerSliceConfig::new(0, total_layers.saturating_sub(1) / 2, total_layers)?,
        );

        let ngl = local_slice.layer_count() as i32;
        // Explicitly release old model VRAM before loading new model
        self.instance.close();

        let instance = LlamaPipelineInstance::open(path_ref, local_slice.clone(), ngl, 4096)?;
        let hidden_dim = instance.hidden_dim;

        self.model_path = path_ref.to_path_buf();
        self.local_slice = local_slice;
        self.instance = instance;
        self.hidden_dim = hidden_dim;
        self.total_layers = total_layers;
        self.active_stream = None;

        info!(
            model = %self.model_path.display(),
            hidden_dim = self.hidden_dim,
            total_layers = self.total_layers,
            "🔄 Switched active model in coordinator"
        );

        Ok(())
    }

    pub async fn ensure_connected(&mut self) -> Result<()> {
        if self.active_stream.is_some() || self.worker_addrs.is_empty() {
            return Ok(());
        }

        let target_worker = self.worker_addrs[0];
        info!(worker = %target_worker, "Connecting to Stage 2 Pipeline Worker over TCP");

        let connect_fut = TcpStream::connect(target_worker);
        let mut stream = match tokio::time::timeout(SOCKET_TIMEOUT, connect_fut).await {
            Ok(Ok(s)) => s,
            Ok(Err(e)) => {
                warn!(worker = %target_worker, error = %e, "Worker connection failed. Running in coordinator local mode.");
                return Ok(());
            }
            Err(_) => {
                warn!(worker = %target_worker, "Worker connection timed out. Running in coordinator local mode.");
                return Ok(());
            }
        };
        stream.set_nodelay(true)?;

        // Execute Handshake
        let worker_start = (self.local_slice.layer_end + 1) as u32;
        let worker_end = (self.total_layers.saturating_sub(1)) as u32;

        let req = HandshakeRequest {
            version: aeromesh_core::PROTOCOL_VERSION,
            model_architecture: "llama".to_string(),
            hidden_dim: self.hidden_dim as u32,
            total_layers: self.total_layers as u32,
            worker_layer_start: worker_start,
            worker_layer_end: worker_end,
            checksum_prefix: "aeromesh-p2p".to_string(),
        };

        let req_bytes = req.encode();
        if let Err(e) = stream.write_all(&req_bytes).await {
            warn!(error = %e, "Failed to send handshake to worker");
            return Ok(());
        }
        let _ = stream.flush().await;

        let resp_fut = HandshakeResponse::decode_async(&mut stream);
        if let Ok(Ok(resp)) = tokio::time::timeout(SOCKET_TIMEOUT, resp_fut).await {
            if resp.accepted {
                info!(worker = %target_worker, "✅ Handshake verified. Pipeline connected.");
                self.active_stream = Some(stream);
            }
        }

        Ok(())
    }

    pub async fn generate_pipeline(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        _temperature: f32,
        _top_p: f32,
        session_id: u64,
        token_tx: Option<mpsc::Sender<Vec<u8>>>,
    ) -> Result<(String, Vec<String>)> {
        let _ = self.ensure_connected().await;

        println!("\n========================================================");
        println!("   AEROMESH NATIVE ZERO-WEIGHT PIPELINE GENERATION      ");
        println!("========================================================");
        println!("  Stage 1 (Local):    Layers {}..={}", self.local_slice.layer_start, self.local_slice.layer_end);
        println!("  Stage 2 (Remote):   Layers {}..={}", self.local_slice.layer_end + 1, self.total_layers.saturating_sub(1));
        println!("  Hidden Dimension:   {}", self.hidden_dim);
        println!("  Session ID:         {}", session_id);
        println!("  Prompt:             {:?}", prompt);
        println!("--------------------------------------------------------");
        print!("  Output: ");
        let _ = std::io::Write::flush(&mut std::io::stdout());

        let total_start = Instant::now();
        let mut generated_text = String::new();
        let mut total_tokens = 0usize;
        let mut perf_metrics = Vec::new();

        // Step 1: Tokenize prompt
        let prompt_tokens = self.instance.tokenize(prompt, true)?;
        if prompt_tokens.is_empty() {
            bail!("Prompt tokenization produced 0 tokens");
        }

        let prompt_len = prompt_tokens.len() as u32;

        // Reset Stage 1 KV-cache for new session
        self.instance.clear_kv_cache();

        // Step 2: Prefill Stage 1 (Forward pass over all prompt tokens)
        let prefill_start = Instant::now();
        let prefill_activations = self.instance.evaluate_tokens_to_activations(&prompt_tokens, 0)?;
        let prefill_time = prefill_start.elapsed();

        // Sample first token from prefill logits
        let first_token_id = self.instance.sample_next_token(0.7, 0.9, 0)?;
        let is_first_eos = self.instance.is_eog(first_token_id);
        let piece_bytes = self.instance.token_to_piece(first_token_id).unwrap_or_default();
        let first_token_text = String::from_utf8_lossy(&piece_bytes).to_string();

        // Stream Prefill Activation Frame [S * hidden_dim] over TCP if worker is connected
        if let Some(ref mut stream) = self.active_stream {
            let prefill_frame = ActivationFrame::from_f32_matrix(
                session_id,
                0,
                prompt_len,
                0,
                self.local_slice.layer_end as u16,
                self.hidden_dim as u32,
                &prefill_activations,
                FLAG_IS_PROMPT | FLAG_CLEAR_KV,
            );
            let prefill_bytes = prefill_frame.encode();
            if stream.write_all(&prefill_bytes).await.is_ok() {
                let _ = stream.flush().await;
                let resp_fut = TokenResponseFrame::decode_async(stream);
                let _ = tokio::time::timeout(SOCKET_TIMEOUT, resp_fut).await;
            }
        }

        print!("{}", first_token_text);
        let _ = std::io::Write::flush(&mut std::io::stdout());

        generated_text.push_str(&first_token_text);
        if let Some(ref tx) = token_tx {
            if tx.send(piece_bytes).await.is_err() {
                info!("🛑 Client disconnected / aborted during prefill. Halting generation loop.");
                return Ok((generated_text, perf_metrics));
            }
        }

        total_tokens += 1;
        let mut current_token_id = first_token_id;
        let mut current_pos = prompt_len;
        let mut decode_act_buf = vec![0.0f32; self.hidden_dim];

        // Step 3: Autoregressive Decode Loop (S = 1, Zero Heap Churn)
        if !is_first_eos {
            for seq_id in 1..max_tokens {
                let decode_tokens = [current_token_id];
                if let Err(e) = self.instance.evaluate_tokens_to_activations_into(&decode_tokens, current_pos, &mut decode_act_buf) {
                    self.active_stream = None;
                    bail!("Stage 1 decode forward pass error: {}", e);
                }

                // Sample next token ID
                let next_token_id = self.instance.sample_next_token(0.7, 0.9, seq_id as u32)?;
                let token_piece_bytes = self.instance.token_to_piece(next_token_id).unwrap_or_default();
                let token_text = String::from_utf8_lossy(&token_piece_bytes).to_string();

                // Stream single-token activation frame [1 * hidden_dim] over TCP
                if let Some(ref mut stream) = self.active_stream {
                    let decode_frame = ActivationFrame::from_f32_matrix(
                        session_id,
                        seq_id as u64,
                        1,
                        current_pos,
                        self.local_slice.layer_end as u16,
                        self.hidden_dim as u32,
                        &decode_act_buf,
                        0,
                    );
                    let frame_bytes = decode_frame.encode();
                    if stream.write_all(&frame_bytes).await.is_ok() {
                        let _ = stream.flush().await;
                        let resp_fut = TokenResponseFrame::decode_async(stream);
                        let _ = tokio::time::timeout(SOCKET_TIMEOUT, resp_fut).await;
                    }
                }

                let is_eos = self.instance.is_eog(next_token_id)
                    || token_text.contains("<|im_end|>")
                    || token_text.contains("<|endoftext|>")
                    || token_text.contains("<|eot_id|>")
                    || token_text.contains("</s>");

                if is_eos {
                    break;
                }

                print!("{}", token_text);
                let _ = std::io::Write::flush(&mut std::io::stdout());

                generated_text.push_str(&token_text);
                if let Some(ref tx) = token_tx {
                    if tx.send(token_piece_bytes).await.is_err() {
                        info!("🛑 Client disconnected / aborted during decode. Halting generation loop.");
                        break;
                    }
                }

                current_token_id = next_token_id;
                current_pos += 1;
                total_tokens += 1;
            }
        }

        println!("\n========================================================");

        let total_time = total_start.elapsed();
        let tok_per_sec = (total_tokens as f64) / total_time.as_secs_f64();

        perf_metrics.push(format!("Prefill Latency: {:.2} ms ({} tokens)", prefill_time.as_secs_f32() * 1000.0, prompt_len));
        perf_metrics.push(format!("Total Generation Time: {:.2} s", total_time.as_secs_f64()));
        perf_metrics.push(format!("Tokens Generated: {}", total_tokens));
        perf_metrics.push(format!("Inference Speed: {:.2} tok/s", tok_per_sec));
        perf_metrics.push(format!(
            "Network Payload per Step: {:.2} KB (Zero weights on wire)",
            (self.hidden_dim as f64 * 4.0) / 1024.0
        ));

        Ok((generated_text, perf_metrics))
    }
}
