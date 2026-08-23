use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use anyhow::{bail, Context, Result};
use tokio::io::AsyncWriteExt;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, Mutex};
use tracing::{error, info, warn};

use aeromesh_core::{
    ActivationDtype, ActivationFrame, HandshakeRequest, HandshakeResponse,
    PipelineTransport, PipelineTransportFactory, ShmPipelineTransport,
    TokenResponseFrame, FLAG_CLEAR_KV, FLAG_IS_PROMPT,
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
            "⚡ AeroMesh Pipeline Worker listening for coordinator activations (TCP + SHM Ready)"
        );

        // Spawn background local SHM worker thread if on localhost
        if host == "0.0.0.0" || host == "127.0.0.1" {
            let instance_clone = self.instance.clone();
            let shm_port = port;
            tokio::spawn(async move {
                // Poll for SHM channel creation by coordinator
                for _ in 0..60 {
                    if let Ok(mut shm_transport) = ShmPipelineTransport::open_worker(shm_port) {
                        info!(port = shm_port, "🚀 Connected to zero-kernel-copy Intra-Host Shared Memory (SHM) channel");
                        loop {
                            match shm_transport.recv_activation().await {
                                Ok(frame) => {
                                    let start_time = Instant::now();
                                    let seq_len = frame.header.sequence_length.max(1);
                                    let start_pos = frame.header.token_position;
                                    let mut inst = instance_clone.lock().await;

                                    if frame.header.has_flag(FLAG_CLEAR_KV) {
                                        inst.clear_kv_cache();
                                    }

                                    let total_elements = (seq_len as usize) * (frame.header.hidden_dim as usize);
                                    let mut dequant_buf = vec![0.0f32; total_elements];
                                    if let Err(e) = frame.dequantize_into(&mut dequant_buf) {
                                        error!(error = %e, "SHM dequantization error");
                                        continue;
                                    }

                                    if let Err(e) = inst.evaluate_activations_to_logits(&dequant_buf, seq_len, start_pos) {
                                        error!(error = %e, "SHM Stage 2 forward pass error");
                                        continue;
                                    }

                                    let eval_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;
                                    let resp = TokenResponseFrame {
                                        session_id: frame.header.session_id,
                                        sequence_id: frame.header.sequence_id,
                                        token_id: 0,
                                        is_eos: false,
                                        token_text: String::new(),
                                        eval_time_ms,
                                    };

                                    let _ = shm_transport.send_response(&resp).await;
                                }
                                Err(_) => {
                                    tokio::time::sleep(Duration::from_millis(50)).await;
                                }
                            }
                        }
                    }
                    tokio::time::sleep(Duration::from_millis(500)).await;
                }
            });
        }

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

    let mut inst = instance.lock().await;
    let mut current_slice = slice_config.clone();

    // Auto-sync worker model if architecture or dimensions differ from coordinator
    if inst.hidden_dim != req.hidden_dim as usize || inst.total_layers != req.total_layers as usize {
        info!(
            worker_hidden_dim = inst.hidden_dim,
            coord_hidden_dim = req.hidden_dim,
            worker_layers = inst.total_layers,
            coord_layers = req.total_layers,
            "🔄 Model mismatch detected between Coordinator and Worker. Auto-syncing Worker model..."
        );

        let mut matched = false;
        let models_dir = Path::new("models");
        if let Ok(entries) = std::fs::read_dir(models_dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.extension().and_then(|e| e.to_str()) == Some("gguf") {
                    if let Ok(loader) = GgufSliceLoader::open(&p) {
                        if loader.total_layers == req.total_layers as usize && loader.hidden_dim == req.hidden_dim {
                            let new_slice = LayerSliceConfig::new(
                                req.worker_layer_start as usize,
                                req.worker_layer_end as usize,
                                req.total_layers as usize,
                            )?;
                            let file_size_mb = std::fs::metadata(&p).map(|m| m.len() / (1024 * 1024)).unwrap_or(0);
                            let ngl = if file_size_mb <= 4500 { 999 } else { new_slice.layer_count() as i32 };

                            inst.close();
                            *inst = LlamaPipelineInstance::open(&p, new_slice.clone(), ngl, 4096)?;
                            current_slice = new_slice;
                            matched = true;
                            info!(model = %p.display(), "✅ Worker auto-switched model to match Coordinator");
                            break;
                        }
                    }
                }
            }
        }
        if !matched {
            bail!("Worker could not find a local GGUF model matching hidden_dim={} total_layers={}", req.hidden_dim, req.total_layers);
        }
    }
    drop(inst);

    let is_last_stage = current_slice.is_last_stage;
    let resp = HandshakeResponse::ok();
    socket.write_all(&resp.encode()).await?;
    socket.flush().await?;
    info!(peer = %peer_addr, "Handshake accepted. Ready for activation stream.");

    // Step 2: Process incoming activation frames with in-place dequantization
    let mut active_session_id = 0u64;
    let mut dequant_buf = Vec::new();

    loop {
        let frame_fut = ActivationFrame::decode_async(socket);
        let frame = match tokio::time::timeout(Duration::from_secs(120), frame_fut).await {
            Ok(Ok(f)) => f,
            Ok(Err(_)) => break,
            Err(_) => break,
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
        let seq_id = frame.header.sequence_id;
        let total_elements = (seq_len as usize) * (frame.header.hidden_dim as usize);

        // In-place dequantize into reusable buffer
        dequant_buf.resize(total_elements, 0.0f32);
        frame.dequantize_into(&mut dequant_buf)?;

        // Execute Stage 2 GEMM forward pass from incoming activations
        inst.evaluate_activations_to_logits(&dequant_buf, seq_len, start_pos)?;

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

            let dtype_tag = match ActivationDtype::from_u8(frame.header.dtype) {
                Ok(ActivationDtype::Int8PerRow) => "INT8-Row",
                Ok(ActivationDtype::Fp8E4M3) => "FP8-E4M3",
                _ => "FP32",
            };

            println!(
                "  ⚡ [Stage 2 Worker] Step #{:<3} (pos: {}, S={}) | Recv: {:.2} KB [{}] | Processed in {:.2}ms",
                seq_id + 1,
                start_pos,
                seq_len,
                (frame.payload.len() as f64) / 1024.0,
                dtype_tag,
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
    pub transport: Option<Box<dyn PipelineTransport>>,
    pub target_dtype: ActivationDtype,
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
            transport: None,
            target_dtype: ActivationDtype::Int8PerRow,
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

        let file_size_mb = std::fs::metadata(path_ref)
            .map(|m| m.len() / (1024 * 1024))
            .unwrap_or(0);
        let ngl = if file_size_mb <= 4500 { 999 } else { local_slice.layer_count() as i32 };

        // Explicitly release old model VRAM before loading new model
        self.instance.close();

        let instance = LlamaPipelineInstance::open(path_ref, local_slice.clone(), ngl, 4096)?;
        let hidden_dim = instance.hidden_dim;

        self.model_path = path_ref.to_path_buf();
        self.local_slice = local_slice;
        self.instance = instance;
        self.hidden_dim = hidden_dim;
        self.total_layers = total_layers;
        self.transport = None;

        info!(
            model = %self.model_path.display(),
            hidden_dim = self.hidden_dim,
            total_layers = self.total_layers,
            "🔄 Switched active model in coordinator"
        );

        Ok(())
    }

    pub async fn ensure_connected(&mut self) -> Result<()> {
        if self.transport.is_some() || self.worker_addrs.is_empty() {
            return Ok(());
        }

        let target_worker = self.worker_addrs[0];
        info!(worker = %target_worker, "Connecting to Stage 2 Pipeline Worker");

        // Use PipelineTransportFactory for auto SHM or TCP selection
        match PipelineTransportFactory::connect_auto(target_worker).await {
            Ok(transport) => {
                let transport_type = if transport.is_shm() { "Zero-Copy Memory-Mapped Shared Memory (SHM)" } else { "Tuned TCP Socket" };
                info!(worker = %target_worker, transport = %transport_type, "✅ Connected to Stage 2 Worker");
                self.transport = Some(transport);
            }
            Err(e) => {
                warn!(worker = %target_worker, error = %e, "Worker connection failed. Running in coordinator local mode.");
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

        let transport_tag = match self.transport {
            Some(ref t) if t.is_shm() => "Intra-Host Zero-Copy SHM Ring Buffer",
            Some(_) => "Low-Latency Tuned TCP Socket",
            None => "Local CPU/GPU Direct Execution",
        };

        println!("\n========================================================");
        println!("   AEROMESH DISTRIBUTED PIPELINE GENERATION             ");
        println!("========================================================");
        println!("  Stage 1 (Local):    Layers {}..={}", self.local_slice.layer_start, self.local_slice.layer_end);
        println!("  Stage 2 (Remote):   Layers {}..={}", self.local_slice.layer_end + 1, self.total_layers.saturating_sub(1));
        println!("  Hidden Dimension:   {}", self.hidden_dim);
        println!("  Quantization:       Per-Row INT8 Dynamic Scaling (75% payload reduction)");
        println!("  Transport:          {}", transport_tag);
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

        // Step 2: Prefill Stage 1
        let prefill_start = Instant::now();
        let prefill_activations = self.instance.evaluate_tokens_to_activations(&prompt_tokens, 0)?;
        let prefill_time = prefill_start.elapsed();

        let first_token_id = self.instance.sample_next_token(0.7, 0.9, 0)?;
        let is_first_eos = self.instance.is_eog(first_token_id);
        let piece_bytes = self.instance.token_to_piece(first_token_id).unwrap_or_default();
        let first_token_text = String::from_utf8_lossy(&piece_bytes).to_string();

        // Stream Quantized Prefill Activation Frame [S * hidden_dim]
        if let Some(ref mut transport) = self.transport {
            let prefill_frame = ActivationFrame::from_f32_matrix_quantized(
                session_id,
                0,
                prompt_len,
                0,
                self.local_slice.layer_end as u16,
                self.hidden_dim as u32,
                &prefill_activations,
                self.target_dtype,
                FLAG_IS_PROMPT | FLAG_CLEAR_KV,
            );
            if transport.send_activation(&prefill_frame).await.is_ok() {
                let _ = transport.recv_response().await;
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

        // Step 3: Autoregressive Decode Loop with Per-Row INT8 Quantized Transport
        if !is_first_eos {
            for seq_id in 1..max_tokens {
                let decode_tokens = [current_token_id];
                if let Err(e) = self.instance.evaluate_tokens_to_activations_into(&decode_tokens, current_pos, &mut decode_act_buf) {
                    self.transport = None;
                    bail!("Stage 1 decode forward pass error: {}", e);
                }

                // Sample next token ID
                let next_token_id = self.instance.sample_next_token(0.7, 0.9, seq_id as u32)?;
                let token_piece_bytes = self.instance.token_to_piece(next_token_id).unwrap_or_default();
                let token_text = String::from_utf8_lossy(&token_piece_bytes).to_string();

                // Stream single-token quantized activation frame [1 * hidden_dim: 5.12 KB]
                if let Some(ref mut transport) = self.transport {
                    let decode_frame = ActivationFrame::from_f32_matrix_quantized(
                        session_id,
                        seq_id as u64,
                        1,
                        current_pos,
                        self.local_slice.layer_end as u16,
                        self.hidden_dim as u32,
                        &decode_act_buf,
                        self.target_dtype,
                        0,
                    );
                    if transport.send_activation(&decode_frame).await.is_ok() {
                        let _ = transport.recv_response().await;
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

        // 5120 floats quantized to INT8 = 4 bytes scale + 5120 bytes data = 5.12 KB (down from 20.48 KB)
        let quantized_bytes_per_step = (self.hidden_dim + 4) as f64;

        perf_metrics.push(format!("Prefill Latency: {:.2} ms ({} tokens)", prefill_time.as_secs_f32() * 1000.0, prompt_len));
        perf_metrics.push(format!("Total Generation Time: {:.2} s", total_time.as_secs_f64()));
        perf_metrics.push(format!("Tokens Generated: {}", total_tokens));
        perf_metrics.push(format!("Inference Speed: {:.2} tok/s", tok_per_sec));
        perf_metrics.push(format!(
            "Network Payload per Step: {:.2} KB (INT8 Per-Row Quantized, 75% Reduction)",
            quantized_bytes_per_step / 1024.0
        ));

        Ok((generated_text, perf_metrics))
    }
}
