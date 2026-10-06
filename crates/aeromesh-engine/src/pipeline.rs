use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use anyhow::{bail, ensure, Context, Result};
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

use crate::slice_loader::{ChatTemplateFormat, GgufSliceLoader, LayerSliceConfig, LlamaPipelineInstance};

pub const STOP_TOKENS: &[&str] = &[
    "<|im_end|>",
    "<|im_start|>",
    "<｜end of sentence｜>",
    "<｜User｜>",
    "<｜Assistant｜>",
    "<｜begin of sentence｜>",
    "<|endoftext|>",
    "<|eot_id|>",
    "<|start_header_id|>",
    "</s>",
];

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
        let target_model_path = if path_ref.to_string_lossy().contains("_stage") {
            path_ref.to_path_buf()
        } else {
            let out_dir = path_ref.parent().unwrap_or_else(|| Path::new("models"));
            let stage_idx = if slice_config.is_first_stage {
                0
            } else {
                1
            };
            crate::gguf_slicer::slice_gguf_stage(path_ref, out_dir, &slice_config, stage_idx, 2)?
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

                                    let total_elements = match (seq_len as usize).checked_mul(frame.header.hidden_dim as usize) {
                                        Some(elems) if elems <= (aeromesh_core::MAX_ACTIVATION_PAYLOAD_BYTES as usize) => elems,
                                        _ => {
                                            error!(
                                                seq_len = seq_len,
                                                dim = frame.header.hidden_dim,
                                                "Oversized or overflowing activation dimensions in SHM; dropping frame"
                                            );
                                            continue;
                                        }
                                    };
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

/// Dynamic Model Hot-Swapper: Scans models/ for a .gguf matching target hidden dimension and hot-swaps instance with stage-aware slicing.
fn hot_swap_worker_model_stage(
    inst: &mut LlamaPipelineInstance,
    target_hidden_dim: u32,
    worker_start: usize,
    worker_end: usize,
    stage_index: usize,
    total_stages: usize,
) -> Result<LayerSliceConfig> {
    info!(
        current_dim = inst.hidden_dim,
        target_dim = target_hidden_dim,
        worker_start = worker_start,
        worker_end = worker_end,
        stage = stage_index + 1,
        total_stages = total_stages,
        "🔄 Hot-swapping worker model to match incoming activation dimension and layer slice..."
    );

    let models_dir = Path::new("models");
    if let Ok(entries) = std::fs::read_dir(models_dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.extension().and_then(|e| e.to_str()) == Some("gguf") && !p.to_string_lossy().contains("_stage") {
                if let Ok(loader) = GgufSliceLoader::open(&p) {
                    if loader.hidden_dim == target_hidden_dim {
                        let total_layers = loader.total_layers;
                        let w_start = worker_start.min(total_layers.saturating_sub(1));
                        let w_end = worker_end.min(total_layers.saturating_sub(1));
                        let new_slice = LayerSliceConfig::new(w_start, w_end, total_layers)?;
                        let file_size_mb = std::fs::metadata(&p).map(|m| m.len() / (1024 * 1024)).unwrap_or(0);
                        let ngl = if file_size_mb <= 4500 { 999 } else { new_slice.layer_count() as i32 };

                        let out_dir = p.parent().unwrap_or_else(|| Path::new("models"));
                        let target_path = crate::gguf_slicer::slice_gguf_stage(&p, out_dir, &new_slice, stage_index, total_stages)?;

                        let new_inst = match LlamaPipelineInstance::open(&target_path, new_slice.clone(), ngl, 4096) {
                            Ok(new_i) => {
                                inst.close();
                                new_i
                            }
                            Err(first_err) => {
                                warn!(error = %first_err, "Worker two-phase hot swap failed (VRAM pressure). Releasing active model to retry...");
                                inst.close();
                                LlamaPipelineInstance::open(&target_path, new_slice.clone(), ngl, 4096)?
                            }
                        };
                        *inst = new_inst;
                        info!(model = %target_path.display(), hidden_dim = loader.hidden_dim, "✅ Worker successfully hot-swapped model");
                        return Ok(new_slice);
                    }
                }
            }
        }
    }
    bail!("Worker could not find a local GGUF model in 'models/' matching hidden_dim={}", target_hidden_dim)
}

fn hot_swap_worker_model(inst: &mut LlamaPipelineInstance, target_hidden_dim: u32, boundary_layer: u16) -> Result<LayerSliceConfig> {
    hot_swap_worker_model_stage(inst, target_hidden_dim, boundary_layer as usize, inst.total_layers.saturating_sub(1), 1, 2)
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
    let mut downstream_socket: Option<TcpStream> = None;
    let mut downstream_peer_str = String::new();
    let mut current_stage_index: u8 = if slice_config.is_last_stage { 2 } else { 1 };
    let mut current_total_stages: u8 = 2;

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

        current_stage_index = req.stage_index;
        current_total_stages = req.total_stages;

        info!(
            peer = %peer_addr,
            stage = req.stage_index,
            total_stages = req.total_stages,
            downstream = %req.downstream_peer,
            arch = %req.model_architecture,
            hidden_dim = req.hidden_dim,
            total_layers = req.total_layers,
            layers = %format!("{}..={}", req.worker_layer_start, req.worker_layer_end),
            "✅ Received authenticated HandshakeRequest"
        );

        let mut inst = instance.lock().await;

        if inst.hidden_dim != req.hidden_dim as usize
            || inst.total_layers != req.total_layers as usize
            || inst.slice_config.layer_start != req.worker_layer_start as usize
            || inst.slice_config.layer_end != req.worker_layer_end as usize
        {
            match hot_swap_worker_model_stage(
                &mut inst,
                req.hidden_dim,
                req.worker_layer_start as usize,
                req.worker_layer_end as usize,
                req.stage_index as usize,
                req.total_stages as usize,
            ) {
                Ok(new_slice) => {
                    current_slice = new_slice;
                }
                Err(e) => {
                    warn!(peer = %peer_addr, error = %e, "🚨 Worker failed to hot-swap model");
                    let resp = HandshakeResponse::reject(format!("Worker failed to hot-swap model: {}", e));
                    let _ = socket.write_all(&resp.encode()).await;
                    let _ = socket.flush().await;
                    let _ = socket.shutdown().await;
                    return Ok(());
                }
            }
        }
        drop(inst);

        // Connect downstream if downstream_peer is specified
        if !req.downstream_peer.is_empty() {
            let peer_parts: Vec<&str> = req.downstream_peer.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
            if !peer_parts.is_empty() {
                let next_peer_addr = peer_parts[0];
                let remaining_peers = peer_parts[1..].join(",");
                info!(
                    current_stage = req.stage_index,
                    next_peer = next_peer_addr,
                    remaining = %remaining_peers,
                    "🔗 Intermediate Stage Worker: Connecting downstream to next pipeline peer..."
                );

                let mut ds_stream = match tokio::time::timeout(SOCKET_TIMEOUT, TcpStream::connect(next_peer_addr)).await {
                    Ok(Ok(s)) => s,
                    Ok(Err(e)) => {
                        warn!(next_peer = next_peer_addr, error = %e, "Failed to connect to downstream pipeline peer");
                        let resp = HandshakeResponse::reject(format!("Downstream connect failed: {}", e));
                        let _ = socket.write_all(&resp.encode()).await;
                        let _ = socket.flush().await;
                        return Ok(());
                    }
                    Err(_) => {
                        warn!(next_peer = next_peer_addr, "Downstream connect timed out");
                        let resp = HandshakeResponse::reject("Downstream connect timed out");
                        let _ = socket.write_all(&resp.encode()).await;
                        let _ = socket.flush().await;
                        return Ok(());
                    }
                };
                ds_stream.set_nodelay(true)?;

                let next_stage = req.stage_index + 1;
                let next_worker_start = req.worker_layer_end + 1;
                let layers_per_stage = (req.total_layers as usize).div_ceil(req.total_stages as usize);
                let next_worker_end = if remaining_peers.is_empty() {
                    req.total_layers.saturating_sub(1)
                } else {
                    ((next_worker_start as usize + layers_per_stage - 1).min(req.total_layers as usize - 1)) as u32
                };

                let downstream_req = HandshakeRequest {
                    version: aeromesh_core::PROTOCOL_VERSION,
                    auth_token: req.auth_token,
                    stage_index: next_stage,
                    total_stages: req.total_stages,
                    downstream_peer: remaining_peers,
                    model_architecture: req.model_architecture.clone(),
                    hidden_dim: req.hidden_dim,
                    total_layers: req.total_layers,
                    worker_layer_start: next_worker_start,
                    worker_layer_end: next_worker_end,
                    checksum_prefix: req.checksum_prefix.clone(),
                };

                let req_bytes = downstream_req.encode();
                if let Err(e) = ds_stream.write_all(&req_bytes).await {
                    warn!(error = %e, "Failed to send HandshakeRequest downstream");
                    let resp = HandshakeResponse::reject("Failed to forward handshake downstream");
                    let _ = socket.write_all(&resp.encode()).await;
                    let _ = socket.flush().await;
                    return Ok(());
                }
                let _ = ds_stream.flush().await;

                let resp_fut = HandshakeResponse::decode_async(&mut ds_stream);
                match tokio::time::timeout(SOCKET_TIMEOUT, resp_fut).await {
                    Ok(Ok(resp)) if resp.accepted => {
                        info!(next_peer = next_peer_addr, "✅ Downstream peer accepted handshake");
                        downstream_socket = Some(ds_stream);
                        downstream_peer_str = next_peer_addr.to_string();
                    }
                    Ok(Ok(resp)) => {
                        warn!(next_peer = next_peer_addr, error = %resp.error_message, "Downstream peer rejected handshake");
                        let r = HandshakeResponse::reject(format!("Downstream peer rejected handshake: {}", resp.error_message));
                        let _ = socket.write_all(&r.encode()).await;
                        let _ = socket.flush().await;
                        return Ok(());
                    }
                    Ok(Err(e)) => {
                        warn!(next_peer = next_peer_addr, error = %e, "Downstream handshake error");
                        let r = HandshakeResponse::reject(format!("Downstream handshake error: {}", e));
                        let _ = socket.write_all(&r.encode()).await;
                        let _ = socket.flush().await;
                        return Ok(());
                    }
                    Err(_) => {
                        warn!(next_peer = next_peer_addr, "Downstream handshake timed out");
                        let r = HandshakeResponse::reject("Downstream handshake timed out");
                        let _ = socket.write_all(&r.encode()).await;
                        let _ = socket.flush().await;
                        return Ok(());
                    }
                }
            }
        }

        let resp = HandshakeResponse::ok(expected_token);
        socket.write_all(&resp.encode()).await?;
        socket.flush().await?;
        info!(peer = %peer_addr, "✅ Handshake accepted. Ready for activation stream.");
    }

    // Step 2: Process incoming activation frames with dynamic in-place dequantization & hot-swapping
    let mut active_session_id = 0u64;
    let mut dequant_buf = Vec::new();
    let mut out_act_buf = Vec::new();

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
            if let Ok(new_slice) = hot_swap_worker_model_stage(
                &mut inst,
                frame.header.hidden_dim,
                current_slice.layer_start,
                current_slice.layer_end,
                current_stage_index as usize,
                current_total_stages as usize,
            ) {
                current_slice = new_slice;
            }
        }

        // Reset KV-Cache on session change or explicit flag
        if frame.header.session_id != active_session_id || frame.header.has_flag(FLAG_CLEAR_KV) {
            info!(
                session_id = frame.header.session_id,
                prev_session = active_session_id,
                "🧹 Resetting Worker KV-Cache for new session"
            );
            inst.clear_kv_cache();
            active_session_id = frame.header.session_id;
        }

        let seq_len = frame.header.sequence_length.max(1);
        let start_pos = frame.header.token_position;
        let seq_id = frame.header.sequence_id;
        let total_elements = (seq_len as usize)
            .checked_mul(frame.header.hidden_dim as usize)
            .ok_or_else(|| anyhow::anyhow!("Integer overflow calculating total elements (seq_len={}, dim={})", seq_len, frame.header.hidden_dim))?;
        ensure!(
            total_elements <= (aeromesh_core::MAX_ACTIVATION_PAYLOAD_BYTES as usize),
            "Total elements ({}) exceeds safety limit ({})",
            total_elements,
            aeromesh_core::MAX_ACTIVATION_PAYLOAD_BYTES
        );

        // In-place dequantize into reusable buffer
        dequant_buf.resize(total_elements, 0.0f32);
        frame.dequantize_into(&mut dequant_buf)?;

        // Boundary Watchdog: Validate incoming activations
        if let Err(e) = aeromesh_core::inspect_activations(
            &format!("Worker_Stage{}_Ingress", current_stage_index),
            &dequant_buf,
            &[seq_len as usize, frame.header.hidden_dim as usize],
        ) {
            error!("🚨 Worker ingress watchdog assertion failed: {}", e);
        }

        if let Some(ref mut ds) = downstream_socket {
            // Intermediate Node Execution: activations -> activations
            inst.evaluate_activations_to_activations(&dequant_buf, seq_len, start_pos, &mut out_act_buf)?;
            let eval_duration = start_time.elapsed();
            let eval_time_ms = eval_duration.as_secs_f32() * 1000.0;
            drop(inst);

            // Package and send downstream
            let next_frame = ActivationFrame::from_f32_matrix_quantized_stage(
                frame.header.session_id,
                frame.header.sequence_id,
                seq_len,
                start_pos,
                current_stage_index,
                current_total_stages,
                current_slice.layer_end as u16,
                frame.header.hidden_dim,
                &out_act_buf,
                ActivationDtype::from_u8(frame.header.dtype).unwrap_or(ActivationDtype::Int8PerRow),
                frame.header.flags,
            );

            let next_bytes = next_frame.encode();
            ds.write_all(&next_bytes).await?;
            ds.flush().await?;

            // Await TokenResponseFrame from downstream
            let token_resp = TokenResponseFrame::decode_async(ds).await?;

            // Forward TokenResponseFrame back upstream
            let resp_bytes = token_resp.encode();
            socket.write_all(&resp_bytes).await?;
            socket.flush().await?;

            let dtype_tag = match ActivationDtype::from_u8(frame.header.dtype) {
                Ok(ActivationDtype::Int8PerRow) => "INT8-Row",
                Ok(ActivationDtype::Fp8E4M3) => "FP8-E4M3",
                _ => "FP32",
            };

            println!(
                "  ⚡ [Stage {} Intermediate] Step #{:<3} (pos: {}, S={}) | In: {:.2} KB [{}] -> Out: {:.2} KB | Forwarded to {} in {:.2}ms",
                current_stage_index,
                seq_id + 1,
                start_pos,
                seq_len,
                (frame.payload.len() as f64) / 1024.0,
                dtype_tag,
                (next_frame.payload.len() as f64) / 1024.0,
                downstream_peer_str,
                eval_time_ms
            );
        } else {
            // Terminal Node Execution: activations -> logits -> sample next token
            inst.evaluate_activations_to_logits(&dequant_buf, seq_len, start_pos)?;
            let eval_duration = start_time.elapsed();
            let eval_time_ms = eval_duration.as_secs_f32() * 1000.0;

            let next_token_id = inst.sample_next_token(0.7, 0.9, seq_id as u32)?;
            let piece_bytes = inst.token_to_piece(next_token_id).unwrap_or_default();
            let token_text = String::from_utf8_lossy(&piece_bytes).to_string();
            let is_eos = inst.is_eog(next_token_id)
                || STOP_TOKENS.iter().any(|&tag| token_text.contains(tag));

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
                "  ⚡ [Stage {} Terminal Worker] Step #{:<3} (pos: {}, S={}) | Recv: {:.2} KB [{}] | Processed in {:.2}ms",
                current_stage_index,
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

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum CoordinatorStatus {
        Ready,
        Loading(PathBuf),
        Failed(String),
        Closed,
    }

    pub struct PipelineCoordinatorClient {
        pub model_path: PathBuf,
        pub local_slice: LayerSliceConfig,
        pub worker_addrs: Vec<SocketAddr>,
        pub instance: Option<LlamaPipelineInstance>,
        pub status: CoordinatorStatus,
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

            let num_stages = worker_addrs.len() + 1;
            let target_model_path = if (num_stages == 2 && path_ref.to_string_lossy().ends_with("_stage1.gguf"))
                || (num_stages > 2 && path_ref.to_string_lossy().ends_with(&format!("_stage1_of_{}.gguf", num_stages))) {
                path_ref.to_path_buf()
            } else if !worker_addrs.is_empty() {
                let out_dir = path_ref.parent().unwrap_or_else(|| Path::new("models"));
                crate::gguf_slicer::slice_gguf_stage(path_ref, out_dir, &local_slice, 0, num_stages)?
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
                instance: Some(instance),
                status: CoordinatorStatus::Ready,
                hidden_dim,
                total_layers,
                transport: None,
                target_dtype: activation_dtype_from_env(),
            })
        }

        pub fn get_instance_mut(&mut self) -> Result<&mut LlamaPipelineInstance> {
            match &mut self.instance {
                Some(inst) if inst.is_ready() => Ok(inst),
                _ => {
                    let reason = match &self.status {
                        CoordinatorStatus::Failed(e) => format!("Model pipeline is unavailable (load failed: {})", e),
                        CoordinatorStatus::Loading(p) => format!("Model pipeline is currently loading ({})", p.display()),
                        CoordinatorStatus::Closed => "Model pipeline is closed".to_string(),
                        CoordinatorStatus::Ready => "Model pipeline instance is not initialized".to_string(),
                    };
                    bail!(reason)
                }
            }
        }

        pub fn get_instance(&self) -> Result<&LlamaPipelineInstance> {
            match &self.instance {
                Some(inst) if inst.is_ready() => Ok(inst),
                _ => {
                    let reason = match &self.status {
                        CoordinatorStatus::Failed(e) => format!("Model pipeline is unavailable (load failed: {})", e),
                        CoordinatorStatus::Loading(p) => format!("Model pipeline is currently loading ({})", p.display()),
                        CoordinatorStatus::Closed => "Model pipeline is closed".to_string(),
                        CoordinatorStatus::Ready => "Model pipeline instance is not initialized".to_string(),
                    };
                    bail!(reason)
                }
            }
        }

        pub fn is_ready(&self) -> bool {
            self.status == CoordinatorStatus::Ready && self.instance.as_ref().map(|i| i.is_ready()).unwrap_or(false)
        }

        pub fn status(&self) -> &CoordinatorStatus {
            &self.status
        }

        pub fn n_ctx(&self) -> usize {
            self.instance.as_ref().map(|i| i.n_ctx).unwrap_or(4096)
        }

        pub fn detect_chat_format(&self) -> ChatTemplateFormat {
            if let Ok(inst) = self.get_instance() {
                inst.detect_chat_format()
            } else {
                ChatTemplateFormat::ChatML
            }
        }

        pub fn switch_model<P: AsRef<Path>>(&mut self, new_model_path: P) -> Result<()> {
            let path_ref = new_model_path.as_ref();
            if !path_ref.exists() {
                bail!("Model file not found at {:?}", path_ref);
            }

            // Step 1: Pre-validate GGUF structure and compute slice before modifying state
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

            let num_stages = self.worker_addrs.len() + 1;
            let target_model_path = if (num_stages == 2 && path_ref.to_string_lossy().ends_with("_stage1.gguf"))
                || (num_stages > 2 && path_ref.to_string_lossy().ends_with(&format!("_stage1_of_{}.gguf", num_stages))) {
                path_ref.to_path_buf()
            } else if !self.worker_addrs.is_empty() {
                let out_dir = path_ref.parent().unwrap_or_else(|| Path::new("models"));
                crate::gguf_slicer::slice_gguf_stage(path_ref, out_dir, &local_slice, 0, num_stages)?
            } else {
                path_ref.to_path_buf()
            };

            // Step 2: Attempt two-phase transactional model loading (zero downtime switch)
            let new_instance = match LlamaPipelineInstance::open(&target_model_path, local_slice.clone(), ngl, 4096) {
                Ok(inst) => {
                    // Two-phase load succeeded! Safely close old instance and take new one
                    if let Some(mut old) = self.instance.take() {
                        old.close();
                    }
                    inst
                }
                Err(first_err) => {
                    // Two-phase load failed (likely CUDA VRAM constraint from having two models concurrent).
                    // Release active model to maximize available VRAM and retry single-model load.
                    warn!(
                        error = %first_err,
                        "Two-phase model load failed (likely VRAM pressure). Freeing active model to retry single-instance load..."
                    );
                    if let Some(mut old) = self.instance.take() {
                        old.close();
                    }
                    self.status = CoordinatorStatus::Loading(target_model_path.clone());

                    match LlamaPipelineInstance::open(&target_model_path, local_slice.clone(), ngl, 4096) {
                        Ok(inst) => inst,
                        Err(second_err) => {
                            let err_msg = format!("Failed to load model {:?}: {}", target_model_path, second_err);
                            error!(error = %err_msg, "Model switch failed. Transitioning coordinator to Failed state.");
                            self.instance = None;
                            self.status = CoordinatorStatus::Failed(err_msg.clone());
                            self.transport = None;
                            bail!(err_msg);
                        }
                    }
                }
            };

            let hidden_dim = new_instance.hidden_dim;
            self.model_path = target_model_path;
            self.local_slice = local_slice;
            self.instance = Some(new_instance);
            self.status = CoordinatorStatus::Ready;
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

        let total_stages = (self.worker_addrs.len() + 1) as u8;
        let downstream_peer = if self.worker_addrs.len() > 1 {
            self.worker_addrs[1..]
                .iter()
                .map(|a| a.to_string())
                .collect::<Vec<_>>()
                .join(",")
        } else {
            String::new()
        };

        let (worker_start, worker_end) = if total_stages == 2 {
            let start = (self.local_slice.layer_end + 1) as u32;
            let end = (self.total_layers.saturating_sub(1)) as u32;
            (start, end)
        } else {
            let layers_per_stage = self.total_layers.div_ceil(total_stages as usize);
            let start = self.local_slice.layer_end + 1;
            let end = (start + layers_per_stage - 1).min(self.total_layers.saturating_sub(1));
            (start as u32, end as u32)
        };

        let worker_secret = std::env::var("AEROMESH_WORKER_SECRET").unwrap_or_default();
        let auth_token: [u8; 32] = Sha256::digest(worker_secret.as_bytes()).into();

        let req = HandshakeRequest {
            version: aeromesh_core::PROTOCOL_VERSION,
            auth_token,
            stage_index: 1,
            total_stages,
            downstream_peer,
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

        let total_stages = (self.worker_addrs.len() + 1) as u8;
        let transport_tag = match self.transport {
            Some(ref t) if t.is_shm() => "Intra-Host Zero-Copy SHM Ring Buffer",
            Some(_) => "Low-Latency Tuned TCP Socket",
            None => "Local CPU/GPU Direct Execution",
        };

        println!("\n========================================================");
        println!("   AEROMESH DISTRIBUTED PIPELINE GENERATION             ");
        println!("========================================================");
        println!("  Pipeline Stages:    {} node(s)", total_stages);
        println!("  Stage 0 (Coord):    Layers {}..={}", self.local_slice.layer_start, self.local_slice.layer_end);
        for (i, w) in self.worker_addrs.iter().enumerate() {
            println!("  Stage {} (Worker):   {}", i + 1, w);
        }
        println!("  Hidden Dimension:   {}", self.hidden_dim);
        println!("  Activation Codec:   {:?} ({:.2} KB per decode step)", self.target_dtype,
            (self.hidden_dim * self.target_dtype.bytes_per_element()) as f64 / 1024.0);
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

        // Verify model readiness before proceeding
        self.get_instance()?;

        // Step 1: Tokenize prompt
        // Bug 2 Fix: If the prompt already begins with a special token sequence (such as
        // <|begin_of_text|> from Llama 3 template or <|im_start|> from ChatML), do NOT pass
        // add_special = true. Otherwise llama.cpp injects a duplicate BOS token, causing the
        // prompt to start with [BOS, BOS], which corrupts attention from token 0.
        let add_special = !prompt.trim_start().starts_with("<|");
        let prompt_tokens = self.get_instance()?.tokenize(prompt, add_special)?;
        if prompt_tokens.is_empty() {
            bail!("Prompt tokenization produced 0 tokens");
        }

        let n_ctx = self.get_instance()?.n_ctx;

        // FIX-24: Validate context window boundaries upfront
        if prompt_tokens.len() >= n_ctx {
            let err_msg = format!(
                "Prompt length ({} tokens) exceeds model maximum context window ({} tokens)",
                prompt_tokens.len(),
                n_ctx
            );
            if let Some(ref tx) = token_tx {
                let _ = tx.send(err_msg.as_bytes().to_vec()).await;
            }
            bail!(err_msg);
        }

        if prompt_tokens.len() + max_tokens > n_ctx {
            let err_msg = format!(
                "Prompt tokens ({}) + requested max_tokens ({}) exceeds model context window of {} tokens (maximum generation capacity: {} tokens)",
                prompt_tokens.len(),
                max_tokens,
                n_ctx,
                n_ctx.saturating_sub(prompt_tokens.len())
            );
            if let Some(ref tx) = token_tx {
                let _ = tx.send(err_msg.as_bytes().to_vec()).await;
            }
            bail!(err_msg);
        }

        let prompt_len = prompt_tokens.len() as u32;

        // Reset Stage 1 KV-cache for new session
        self.get_instance()?.clear_kv_cache();

        // Step 2: Prefill Stage 1
        let prefill_start = Instant::now();
        let prefill_activations = self.get_instance_mut()?.evaluate_tokens_to_activations(&prompt_tokens, 0)?;
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
            let prefill_frame = ActivationFrame::from_f32_matrix_quantized_stage(
                session_id,
                0,
                prompt_len,
                0,
                0,
                total_stages,
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
                    let prefill_frame = ActivationFrame::from_f32_matrix_quantized_stage(
                        session_id,
                        0,
                        prompt_len,
                        0,
                        0,
                        total_stages,
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

        if !transport_ok {
            if !self.worker_addrs.is_empty() {
                let err_msg = format!(
                    "Stage 2 Worker node ({:?}) is disconnected or unreachable. Ensure 'aeromesh worker' is running on the worker machine.",
                    self.worker_addrs
                );
                tracing::error!("{}", err_msg);
                if let Some(ref tx) = token_tx {
                    let _ = tx.send(err_msg.as_bytes().to_vec()).await;
                }
                bail!(err_msg);
            }

            let inst = self.get_instance_mut()?;
            first_token_id = inst.sample_next_token(0.7, 0.9, 0)?;
            is_first_eos = inst.is_eog(first_token_id);
            let piece_bytes = inst.token_to_piece(first_token_id).unwrap_or_default();
            first_token_text = String::from_utf8_lossy(&piece_bytes).to_string();
        }

        let is_first_stop = is_first_eos || STOP_TOKENS.iter().any(|&tag| first_token_text.contains(tag));

        if !is_first_stop && !first_token_text.is_empty() {
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
        }

        let mut current_token_id = first_token_id;
        let mut current_pos = prompt_len;
        let mut decode_act_buf = vec![0.0f32; self.hidden_dim];

        // Step 3: Autoregressive Decode Loop with Per-Row INT8 Quantized Transport
        if !is_first_eos && !is_first_stop {
            for seq_id in 1..max_tokens {
                let decode_tokens = [current_token_id];
                if let Err(e) = self.get_instance_mut()?.evaluate_tokens_to_activations_into(&decode_tokens, current_pos, &mut decode_act_buf) {
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
                    let decode_frame = ActivationFrame::from_f32_matrix_quantized_stage(
                        session_id,
                        seq_id as u64,
                        1,
                        current_pos,
                        0,
                        total_stages,
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
                    if !self.worker_addrs.is_empty() {
                        let err_msg = format!(
                            "Connection to Stage 2 Worker ({:?}) lost during token generation (step {}).",
                            self.worker_addrs, seq_id
                        );
                        tracing::error!("{}", err_msg);
                        if let Some(ref tx) = token_tx {
                            let _ = tx.send(err_msg.as_bytes().to_vec()).await;
                        }
                        bail!(err_msg);
                    }

                    let inst = self.get_instance_mut()?;
                    next_token_id = inst.sample_next_token(0.7, 0.9, seq_id as u32)?;
                    let token_piece_bytes = inst.token_to_piece(next_token_id).unwrap_or_default();
                    token_text = String::from_utf8_lossy(&token_piece_bytes).to_string();
                    is_eos = inst.is_eog(next_token_id)
                        || STOP_TOKENS.iter().any(|&tag| token_text.contains(tag));
                }

                let is_stop_token = is_eos
                    || STOP_TOKENS.iter().any(|&tag| token_text.contains(tag))
                    || token_text == "<|im"
                    || token_text == "<｜";

                if is_stop_token {
                    break;
                }

                print!("{}", token_text);
                let _ = std::io::Write::flush(&mut std::io::stdout());

                generated_text.push_str(&token_text);

                // Check if accumulated text contains any stop sequence across chunk boundaries
                let mut hit_stop = false;
                for &stop_tag in STOP_TOKENS {
                    if let Some(idx) = generated_text.find(stop_tag) {
                        generated_text.truncate(idx);
                        hit_stop = true;
                        break;
                    }
                }
                if hit_stop {
                    break;
                }

                // Phrase-level & character runaway degeneration guard:
                if seq_id % 2 == 0 && is_degenerate_loop(&generated_text) {
                    tracing::warn!("🔁 Repetition loop detected after {} tokens. Halting generation.", total_tokens);
                    if let Some(ref tx) = token_tx {
                        let _ = tx.send(token_text.as_bytes().to_vec()).await;
                    }
                    total_tokens += 1;
                    break;
                }

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
        let bytes_per_step = match self.target_dtype {
            ActivationDtype::Int8PerRow => (self.hidden_dim + 4) as f64,
            dt => (self.hidden_dim * dt.bytes_per_element()) as f64,
        };

        perf_metrics.push(format!("Prefill Latency: {:.2} ms ({} tokens)", prefill_time.as_secs_f32() * 1000.0, prompt_len));
        perf_metrics.push(format!("Total Generation Time: {:.2} s", total_time.as_secs_f64()));
        perf_metrics.push(format!("Tokens Generated: {}", total_tokens));
        perf_metrics.push(format!("Inference Speed: {:.2} tok/s", tok_per_sec));
        perf_metrics.push(format!(
            "Network Payload per Step: {:.2} KB ({:?} activations)",
            bytes_per_step / 1024.0,
            self.target_dtype
        ));

        Ok((generated_text, perf_metrics))
    }
}

/// Activation codec used on the distributed pipeline link.
/// Defaults to lossless BF16: preserves full 8-bit FP32 exponent range so outlier channels
/// survive without distortion while reducing wire bandwidth by 50% (~6 KB/token).
/// Can be overridden with AEROMESH_ACTIVATION_DTYPE (e.g. "bf16", "int8", "fp8", "fp16", "rawf32").
fn activation_dtype_from_env() -> ActivationDtype {
    match std::env::var("AEROMESH_ACTIVATION_DTYPE")
        .unwrap_or_else(|_| "bf16".to_string())
        .to_ascii_lowercase()
        .as_str()
    {
        "int8" => ActivationDtype::Int8PerRow,
        "fp8" | "fp8_e4m3" => ActivationDtype::Fp8E4M3,
        "fp16" | "float16" => ActivationDtype::Fp16,
        "raw" | "rawf32" | "fp32" | "float32" => ActivationDtype::RawF32,
        _ => ActivationDtype::Bf16,
    }
}

/// Returns true when generated text exhibits degenerate loop behavior:
/// 1. Runaway single-character repetitions (e.g. "dddddd", "UUUUUU", "pppppp" >= 5 times).
/// 2. Short n-gram repetitions (2-8 chars repeated >= 4 times continuously).
/// 3. Phrase-level verbatim repetitions across a sliding probe window.
fn is_degenerate_loop(text: &str) -> bool {
    // 1. Immediate single-character runaway check (catches byte fallback tokenizer loops)
    let mut last_char = None;
    let mut repeat_count = 0;
    for c in text.chars() {
        if !c.is_whitespace() {
            if Some(c) == last_char {
                repeat_count += 1;
                if repeat_count >= 5 {
                    return true;
                }
            } else {
                last_char = Some(c);
                repeat_count = 1;
            }
        } else {
            last_char = None;
            repeat_count = 0;
        }
    }

    // 2. Short n-gram loop check in recent 400-byte window (e.g. "mamo" x 4 or "abc" x 4)
    let recent_len = text.len().min(400);
    let mut recent_start = text.len() - recent_len;
    while recent_start < text.len() && !text.is_char_boundary(recent_start) {
        recent_start += 1;
    }
    let recent_slice = &text[recent_start..];
    for n in 2..=8 {
        if recent_slice.len() >= n * 4 {
            let tail = &recent_slice[recent_slice.len() - n..];
            if !tail.trim().is_empty() {
                let tail_repeat = tail.repeat(4);
                if recent_slice.ends_with(&tail_repeat) {
                    return true;
                }
            }
        }
    }

    // 3. Wide-window phrase / sentence loop check
    const WINDOW: usize = 2000;
    const PROBE: usize = 80;
    const MIN_REPEATS: usize = 4;

    let mut start = text.len().saturating_sub(WINDOW);
    while start < text.len() && !text.is_char_boundary(start) {
        start += 1;
    }
    let recent = &text[start..];
    if recent.len() >= PROBE * MIN_REPEATS {
        let mut p = recent.len() - PROBE;
        while p > 0 && !recent.is_char_boundary(p) {
            p -= 1;
        }
        let probe = &recent[p..];
        if probe.trim().chars().count() >= 20 && recent.matches(probe).count() >= MIN_REPEATS {
            return true;
        }
    }

    false
}

#[cfg(test)]
mod tests_degenerate_loop {
    use super::is_degenerate_loop;

    #[test]
    fn detects_single_char_runaway() {
        let text = "Here are the aspects: 1. **In-HetqHxHwshfGhifUHCHHHHkikkkkhhUUUUUU1uuuuuvvrvvrtsttttppppppptttcmddddddddmamo";
        assert!(is_degenerate_loop(text));
    }

    #[test]
    fn detects_short_ngram_runaway() {
        let text = "Analyzing cluster node status now: loop loop loop loop ";
        assert!(is_degenerate_loop(text));
    }

    #[test]
    fn detects_paragraph_loop() {
        let para = "The user is asking for an explanation of photosynthesis. I should explain it clearly. ";
        let text = para.repeat(6);
        assert!(is_degenerate_loop(&text));
    }

    #[test]
    fn detects_short_gibberish_loop() {
        let text = "olai ".repeat(200);
        assert!(is_degenerate_loop(&text));
    }

    #[test]
    fn ignores_normal_text() {
        let text = "Photosynthesis is the process by which green plants convert light energy into chemical energy. \
            It occurs in chloroplasts, using chlorophyll to absorb sunlight. Water is split, releasing oxygen, \
            while carbon dioxide is fixed into glucose through the Calvin cycle in the stroma. The light reactions \
            take place in the thylakoid membranes and produce ATP and NADPH for the dark reactions.";
        assert!(!is_degenerate_loop(text));
    }
}

#[cfg(test)]
mod tests_pipeline_safety {
    use super::*;

    #[test]
    fn test_coordinator_status_rejection_when_failed() {
        let mut client = PipelineCoordinatorClient {
            model_path: PathBuf::from("non_existent_model.gguf"),
            local_slice: LayerSliceConfig::new(0, 0, 1).unwrap(),
            worker_addrs: Vec::new(),
            instance: None,
            status: CoordinatorStatus::Failed("GGUF load failed: Out of Memory".to_string()),
            hidden_dim: 128,
            total_layers: 1,
            transport: None,
            target_dtype: ActivationDtype::RawF32,
        };

        assert!(!client.is_ready());
        let res = client.get_instance_mut();
        assert!(res.is_err());
        let err_msg = res.unwrap_err().to_string();
        assert!(err_msg.contains("Model pipeline is unavailable (load failed:"));
    }

    #[tokio::test]
    async fn test_generate_pipeline_rejects_when_failed() {
        let mut client = PipelineCoordinatorClient {
            model_path: PathBuf::from("non_existent_model.gguf"),
            local_slice: LayerSliceConfig::new(0, 0, 1).unwrap(),
            worker_addrs: Vec::new(),
            instance: None,
            status: CoordinatorStatus::Failed("GGUF load failed: file corrupted".to_string()),
            hidden_dim: 128,
            total_layers: 1,
            transport: None,
            target_dtype: ActivationDtype::RawF32,
        };

        let res = client.generate_pipeline("Hello", 10, 0.7, 0.9, 1234, None).await;
        assert!(res.is_err(), "generate_pipeline must reject when in Failed state without crashing");
        let err_msg = res.unwrap_err().to_string();
        assert!(err_msg.contains("Model pipeline is unavailable (load failed:"));
    }

    #[test]
    fn test_switch_model_invalid_path_preserves_old_model() {
        let mut client = PipelineCoordinatorClient {
            model_path: PathBuf::from("models/mock_stage1.gguf"),
            local_slice: LayerSliceConfig::new(0, 0, 1).unwrap(),
            worker_addrs: Vec::new(),
            instance: Some(LlamaPipelineInstance::dummy_closed()),
            status: CoordinatorStatus::Ready,
            hidden_dim: 128,
            total_layers: 1,
            transport: None,
            target_dtype: ActivationDtype::RawF32,
        };

        let res = client.switch_model("completely_non_existent_model_file.gguf");
        assert!(res.is_err(), "switch_model must fail on non-existent file");
        // Old model instance must NOT have been taken or closed
        assert!(client.instance.is_some(), "Instance must still be present");
        assert_eq!(client.status, CoordinatorStatus::Ready, "Status must remain Ready");
    }

    #[test]
    fn test_coordinator_n_ctx_reporting() {
        let client_with_instance = PipelineCoordinatorClient {
            model_path: PathBuf::from("models/mock_stage1.gguf"),
            local_slice: LayerSliceConfig::new(0, 0, 1).unwrap(),
            worker_addrs: Vec::new(),
            instance: Some(LlamaPipelineInstance::dummy_closed()),
            status: CoordinatorStatus::Ready,
            hidden_dim: 128,
            total_layers: 1,
            transport: None,
            target_dtype: ActivationDtype::RawF32,
        };
        assert_eq!(client_with_instance.n_ctx(), 512);

        let client_without_instance = PipelineCoordinatorClient {
            model_path: PathBuf::from("models/mock_stage1.gguf"),
            local_slice: LayerSliceConfig::new(0, 0, 1).unwrap(),
            worker_addrs: Vec::new(),
            instance: None,
            status: CoordinatorStatus::Closed,
            hidden_dim: 128,
            total_layers: 1,
            transport: None,
            target_dtype: ActivationDtype::RawF32,
        };
        assert_eq!(client_without_instance.n_ctx(), 4096);
    }
}
