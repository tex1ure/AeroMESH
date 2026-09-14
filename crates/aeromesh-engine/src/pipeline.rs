use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use anyhow::{bail, Context, Result};
use tokio::io::AsyncWriteExt;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, Mutex};
use tracing::{error, info, warn};
use sha2::{Digest, Sha256};

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
        let target_model_path = if path_ref.to_string_lossy().ends_with("_stage2.gguf") {
            path_ref.to_path_buf()
        } else {
            let out_dir = path_ref.parent().unwrap_or_else(|| Path::new("models"));
            let (_, stage2) = crate::gguf_slicer::slice_gguf_for_pipeline(path_ref, out_dir, slice_config.layer_start)?;
            stage2
        };

        let instance = LlamaPipelineInstance::open(&target_model_path, slice_config.clone(), n_gpu_layers, 4096)?;

        info!(
            model = %target_model_path.display(),
            layers = %format!("{}..={}", slice_config.layer_start, slice_config.layer_end),
            "Initialized Pipeline Worker Service"
        );

        Ok(Self {
            model_path: target_model_path,
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
                                    let seq_id = frame.header.sequence_id;
                                    let mut inst = instance_clone.lock().await;

                                    if frame.header.has_flag(FLAG_CLEAR_KV) {
                                        inst.clear_kv_cache();
                                    }

                                    // Dynamic model hot-swap on dimension mismatch
                                    if frame.header.hidden_dim as usize != inst.hidden_dim {
                                        let _ = hot_swap_worker_model(&mut inst, frame.header.hidden_dim, frame.header.layer_index);
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
                                    let next_token_id = inst.sample_next_token(0.7, 0.9, seq_id as u32).unwrap_or(0);
                                    let piece_bytes = inst.token_to_piece(next_token_id).unwrap_or_default();
                                    let token_text = String::from_utf8_lossy(&piece_bytes).to_string();
                                    let is_eos = inst.is_eog(next_token_id)
                                        || token_text.contains("<|im_end|>")
                                        || token_text.contains("<|endoftext|>")
                                        || token_text.contains("<|eot_id|>")
                                        || token_text.contains("</s>");

                                    let resp = TokenResponseFrame {
                                        session_id: frame.header.session_id,
                                        sequence_id: frame.header.sequence_id,
                                        token_id: next_token_id,
                                        is_eos,
                                        token_text,
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

/// Dynamic Model Hot-Swapper: Scans models/ for a .gguf matching target hidden dimension and hot-swaps instance.
fn hot_swap_worker_model(inst: &mut LlamaPipelineInstance, target_hidden_dim: u32, boundary_layer: u16) -> Result<LayerSliceConfig> {
    info!(
            current_dim = inst.hidden_dim,
            target_dim = target_hidden_dim,
            boundary_layer = boundary_layer,
            "🔄 Hot-swapping worker model to match incoming activation dimension..."
        );

        let models_dir = Path::new("models");
        if let Ok(entries) = std::fs::read_dir(models_dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.extension().and_then(|e| e.to_str()) == Some("gguf") && !p.to_string_lossy().contains("_stage1") {
                    if let Ok(loader) = GgufSliceLoader::open(&p) {
                        if loader.hidden_dim == target_hidden_dim {
                            let total_layers = loader.total_layers;
                            let worker_start = (boundary_layer as usize).min(total_layers.saturating_sub(1));
                            let worker_end = total_layers.saturating_sub(1);
                            let new_slice = LayerSliceConfig::new(worker_start, worker_end, total_layers)?;
                            let file_size_mb = std::fs::metadata(&p).map(|m| m.len() / (1024 * 1024)).unwrap_or(0);
                            let ngl = if file_size_mb <= 4500 { 999 } else { new_slice.layer_count() as i32 };

                            let out_dir = p.parent().unwrap_or_else(|| Path::new("models"));
                            let target_path = if p.to_string_lossy().ends_with("_stage2.gguf") {
                                p.clone()
                            } else {
                                let (_, stage2) = crate::gguf_slicer::slice_gguf_for_pipeline(&p, out_dir, worker_start)?;
                                stage2
                            };

                            inst.close();
                            *inst = LlamaPipelineInstance::open(&target_path, new_slice.clone(), ngl, 4096)?;
                            info!(model = %target_path.display(), hidden_dim = loader.hidden_dim, "✅ Worker successfully hot-swapped model");
                            return Ok(new_slice);
                        }
                    }
                }
            }
        }
        bail!("Worker could not find a local GGUF model in 'models/' matching hidden_dim={}", target_hidden_dim)
    }

    async fn handle_coordinator_connection(
        socket: &mut TcpStream,
        peer_addr: SocketAddr,
        instance: Arc<Mutex<LlamaPipelineInstance>>,
        slice_config: LayerSliceConfig,
    ) -> Result<()> {
        // Step 1: Peek magic header to detect whether peer is sending Handshake (AHSK) or direct Activation (AERO)
        let mut peek_buf = [0u8; 4];
        let peek_fut = socket.peek(&mut peek_buf);
        let _ = tokio::time::timeout(SOCKET_TIMEOUT, peek_fut)
            .await
            .context("Connection peek timeout")??;

        let mut current_slice = slice_config.clone();

        if peek_buf == aeromesh_core::HANDSHAKE_REQ_MAGIC {
            let handshake_fut = HandshakeRequest::decode_async(socket);
            let req = match tokio::time::timeout(SOCKET_TIMEOUT, handshake_fut).await {
                Ok(Ok(req)) => req,
                Ok(Err(e)) => {
                    warn!(peer = %peer_addr, error = %e, "🚨 Handshake decoding failed or legacy protocol version detected");
                    let _ = socket.shutdown().await;
                    return Ok(());
                }
                Err(_) => {
                    warn!(peer = %peer_addr, "Handshake timeout");
                    let _ = socket.shutdown().await;
                    return Ok(());
                }
            };

            // Authenticate worker secret
            let expected_secret = std::env::var("AEROMESH_WORKER_SECRET").unwrap_or_default();
            let expected_token: [u8; 32] = Sha256::digest(expected_secret.as_bytes()).into();

            // Constant-time comparison
            let mut diff = 0u8;
            for (a, b) in req.auth_token.iter().zip(expected_token.iter()) {
                diff |= a ^ b;
            }

            if diff != 0 {
                warn!(
                    peer = %peer_addr,
                    "🚨 Handshake rejected: invalid AEROMESH_WORKER_SECRET token from peer. Closing connection."
                );
                let resp = HandshakeResponse::reject("Authentication failed: invalid worker secret");
                let _ = socket.write_all(&resp.encode()).await;
                let _ = socket.flush().await;
                let _ = socket.shutdown().await;
                return Ok(());
            }

            info!(
                peer = %peer_addr,
                arch = %req.model_architecture,
                hidden_dim = req.hidden_dim,
                total_layers = req.total_layers,
                "✅ Received authenticated HandshakeRequest from Coordinator"
            );

            let mut inst = instance.lock().await;

            if inst.hidden_dim != req.hidden_dim as usize || inst.total_layers != req.total_layers as usize {
                current_slice = hot_swap_worker_model(&mut inst, req.hidden_dim, req.worker_layer_start as u16)?;
            }
            drop(inst);

            let resp = HandshakeResponse::ok(expected_token);
            socket.write_all(&resp.encode()).await?;
            socket.flush().await?;
            info!(peer = %peer_addr, "✅ Handshake accepted. Ready for activation stream.");
        }

        // Step 2: Process incoming activation frames with dynamic in-place dequantization & hot-swapping
        let mut active_session_id = 0u64;
        let mut dequant_buf = Vec::new();

        loop {
            let frame_fut = ActivationFrame::decode_async(socket);
            let frame = match tokio::time::timeout(Duration::from_secs(3600), frame_fut).await {
                Ok(Ok(f)) => f,
                Ok(Err(_)) => break,
                Err(_) => break,
            };

            let start_time = Instant::now();
            let mut inst = instance.lock().await;

            // Auto-sync worker model if incoming activation dimension differs from loaded model
            if frame.header.hidden_dim as usize != inst.hidden_dim {
                info!(
                    worker_dim = inst.hidden_dim,
                    incoming_dim = frame.header.hidden_dim,
                    "🔄 Dynamic model dimension mismatch in activation stream. Hot-swapping worker model..."
                );
                if let Ok(new_slice) = hot_swap_worker_model(&mut inst, frame.header.hidden_dim, frame.header.layer_index + 1) {
                    current_slice = new_slice;
                }
            }

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

            // Phase 1 Boundary Watchdog: Validate incoming activations
            if let Err(e) = aeromesh_core::inspect_activations(
                "Worker_Ingress",
                &dequant_buf,
                &[seq_len as usize, frame.header.hidden_dim as usize],
            ) {
                error!("🚨 Worker ingress watchdog assertion failed: {}", e);
            }

            // Execute Stage 2 GEMM forward pass from incoming activations
            inst.evaluate_activations_to_logits(&dequant_buf, seq_len, start_pos)?;

            let eval_duration = start_time.elapsed();
            let eval_time_ms = eval_duration.as_secs_f32() * 1000.0;

            if current_slice.is_last_stage {
                // Sample actual token from Stage 2 LM Head logits
                let next_token_id = inst.sample_next_token(0.7, 0.9, seq_id as u32)?;
                let piece_bytes = inst.token_to_piece(next_token_id).unwrap_or_default();
                let token_text = String::from_utf8_lossy(&piece_bytes).to_string();
                let is_eos = inst.is_eog(next_token_id)
                    || token_text.contains("<|im_end|>")
                    || token_text.contains("<|endoftext|>")
                    || token_text.contains("<|eot_id|>")
                    || token_text.contains("</s>");

                let resp = TokenResponseFrame {
                    session_id: active_session_id,
                    sequence_id: seq_id,
                    token_id: next_token_id,
                    is_eos,
                    token_text,
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
        pub transport: Option<PipelineTransport>,
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

            let target_model_path = if path_ref.to_string_lossy().ends_with("_stage1.gguf") {
                path_ref.to_path_buf()
            } else if !worker_addrs.is_empty() {
                let out_dir = path_ref.parent().unwrap_or_else(|| Path::new("models"));
                let (stage1, _) = crate::gguf_slicer::slice_gguf_for_pipeline(path_ref, out_dir, local_slice.layer_end + 1)?;
                stage1
            } else {
                path_ref.to_path_buf()
            };

            let instance = LlamaPipelineInstance::open(&target_model_path, local_slice.clone(), n_gpu_layers, 4096)?;
            let hidden_dim = instance.hidden_dim;

            info!(
                model = %target_model_path.display(),
                workers = worker_addrs.len(),
                local_layers = %format!("{}..={}", local_slice.layer_start, local_slice.layer_end),
                hidden_dim = hidden_dim,
                "Initialized Pipeline Coordinator Client"
            );

            Ok(Self {
                model_path: target_model_path,
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

            let target_model_path = if path_ref.to_string_lossy().ends_with("_stage1.gguf") {
                path_ref.to_path_buf()
            } else if !self.worker_addrs.is_empty() {
                let out_dir = path_ref.parent().unwrap_or_else(|| Path::new("models"));
                let (stage1, _) = crate::gguf_slicer::slice_gguf_for_pipeline(path_ref, out_dir, local_slice.layer_end + 1)?;
                stage1
            } else {
                path_ref.to_path_buf()
            };

            // Explicitly release old model VRAM before loading new model
            self.instance.close();

            let instance = LlamaPipelineInstance::open(&target_model_path, local_slice.clone(), ngl, 4096)?;
            let hidden_dim = instance.hidden_dim;

            self.model_path = target_model_path;
            self.local_slice = local_slice;
            self.instance = instance;
            self.hidden_dim = hidden_dim;
            self.total_layers = total_layers;
            self.transport = None; // Force fresh transport handshake with new model dimensions

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

        if PipelineTransportFactory::is_local_address(&target_worker) {
            let channel_id = target_worker.port();
            if let Ok(shm_transport) = ShmPipelineTransport::new_coordinator(channel_id) {
                info!(worker = %target_worker, "✅ Connected to Stage 2 Worker via Zero-Copy SHM");
                self.transport = Some(PipelineTransport::Shm(shm_transport));
                return Ok(());
            }
        }

        // Remote TCP Connection & Handshake Execution
        let mut stream = match tokio::time::timeout(SOCKET_TIMEOUT, TcpStream::connect(target_worker)).await {
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

        let worker_start = (self.local_slice.layer_end + 1) as u32;
        let worker_end = (self.total_layers.saturating_sub(1)) as u32;

        let worker_secret = std::env::var("AEROMESH_WORKER_SECRET").unwrap_or_default();
        let auth_token: [u8; 32] = Sha256::digest(worker_secret.as_bytes()).into();

        let req = HandshakeRequest {
            version: aeromesh_core::PROTOCOL_VERSION,
            auth_token,
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
        match tokio::time::timeout(SOCKET_TIMEOUT, resp_fut).await {
            Ok(Ok(resp)) if resp.accepted => {
                let mut diff = 0u8;
                for (a, b) in resp.auth_token.iter().zip(auth_token.iter()) {
                    diff |= a ^ b;
                }
                if diff != 0 {
                    warn!(worker = %target_worker, "🚨 Worker handshake response failed mutual authentication check");
                    return Ok(());
                }
                info!(worker = %target_worker, "✅ Handshake verified with mutual authentication. Connected to remote Stage 2 Worker via Tuned TCP Socket");
                self.transport = Some(PipelineTransport::Tcp(aeromesh_core::TcpPipelineTransport::new(stream)?));
            }
            Ok(Ok(resp)) => {
                warn!(worker = %target_worker, error = %resp.error_message, "🚨 Worker rejected handshake");
            }
            Ok(Err(e)) => {
                warn!(worker = %target_worker, error = %e, "🚨 Worker handshake response error or version mismatch");
            }
            Err(_) => {
                warn!(worker = %target_worker, "Handshake response timed out from worker");
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

        // Phase 1 Boundary Watchdog: Inspect Coordinator prefill egress activations
        if let Err(e) = aeromesh_core::inspect_activations(
            "Coordinator_Egress_Prefill",
            &prefill_activations,
            &[prompt_len as usize, self.hidden_dim],
        ) {
            tracing::error!("🚨 Coordinator prefill egress watchdog assertion failed: {}", e);
        }

        let mut first_token_id = 0;
        let mut first_token_text = String::new();
        let mut is_first_eos = false;

        // Stream Quantized Prefill Activation Frame [S * hidden_dim]
        let mut transport_ok = false;
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
                if let Ok(resp) = transport.recv_response().await {
                    first_token_id = resp.token_id;
                    first_token_text = resp.token_text;
                    is_first_eos = resp.is_eos;
                    transport_ok = true;
                }
            }
        }

        if !transport_ok && !self.worker_addrs.is_empty() {
            tracing::info!("🔄 Re-establishing transport connection to pipeline worker...");
            self.transport = None;
            if self.ensure_connected().await.is_ok() {
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
                        if let Ok(resp) = transport.recv_response().await {
                            first_token_id = resp.token_id;
                            first_token_text = resp.token_text;
                            is_first_eos = resp.is_eos;
                            transport_ok = true;
                        }
                    }
                }
            }
        }

        if !transport_ok && self.transport.is_none() {
            first_token_id = self.instance.sample_next_token(0.7, 0.9, 0)?;
            is_first_eos = self.instance.is_eog(first_token_id);
            let piece_bytes = self.instance.token_to_piece(first_token_id).unwrap_or_default();
            first_token_text = String::from_utf8_lossy(&piece_bytes).to_string();
        }

        print!("{}", first_token_text);
        let _ = std::io::Write::flush(&mut std::io::stdout());

        generated_text.push_str(&first_token_text);
        if let Some(ref tx) = token_tx {
            if tx.send(first_token_text.as_bytes().to_vec()).await.is_err() {
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

                // Phase 1 Boundary Watchdog: Inspect Coordinator decode egress activations
                if let Err(e) = aeromesh_core::inspect_activations(
                    "Coordinator_Egress_Decode",
                    &decode_act_buf,
                    &[1, self.hidden_dim],
                ) {
                    tracing::error!("🚨 Coordinator decode egress watchdog assertion failed: {}", e);
                }

                let mut next_token_id = 0;
                let mut token_text = String::new();
                let mut is_eos = false;
                let mut decode_transport_ok = false;

                // Stream single-token quantized activation frame [1 * hidden_dim: 1.50 KB or 5.12 KB]
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
                        if let Ok(resp) = transport.recv_response().await {
                            next_token_id = resp.token_id;
                            token_text = resp.token_text;
                            is_eos = resp.is_eos;
                            decode_transport_ok = true;
                        }
                    }
                    if !decode_transport_ok {
                        self.transport = None;
                    }
                }
                
                if !decode_transport_ok {
                    next_token_id = self.instance.sample_next_token(0.7, 0.9, seq_id as u32)?;
                    let token_piece_bytes = self.instance.token_to_piece(next_token_id).unwrap_or_default();
                    token_text = String::from_utf8_lossy(&token_piece_bytes).to_string();
                    is_eos = self.instance.is_eog(next_token_id)
                        || token_text.contains("<|im_end|>")
                        || token_text.contains("<|endoftext|>")
                        || token_text.contains("<|eot_id|>")
                        || token_text.contains("</s>");
                }

                if is_eos {
                    break;
                }

                print!("{}", token_text);
                let _ = std::io::Write::flush(&mut std::io::stdout());

                generated_text.push_str(&token_text);
                if let Some(ref tx) = token_tx {
                    if tx.send(token_text.as_bytes().to_vec()).await.is_err() {
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
