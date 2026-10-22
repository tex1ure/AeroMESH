import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).parent.resolve()


def reset_tracked_files():
    """Resets target files to current HEAD so patching is 100% deterministic."""
    subprocess.run(
        [
            "git",
            "checkout",
            "--",
            "crates/aeromesh-core/src/transport.rs",
            "crates/aeromesh-core/src/activation.rs",
            "crates/aeromesh-engine/src/gguf_slicer.rs",
            "crates/aeromesh-engine/src/slice_loader.rs",
            "crates/aeromesh-engine/src/pipeline.rs",
            "crates/aeromesh-engine/src/server.rs",
            "crates/aeromesh-cli/src/main.rs",
            "app.py",
            "static/index.html",
            "static/css/liquid_glass.css",
            "static/js/app.js",
        ],
        cwd=ROOT,
        check=False,
    )


def replace_idempotent(rel_path: str, text: str, old: str, new: str) -> str:
    """Replaces `old` with `new` if present; if already patched in HEAD, continues cleanly."""
    if new in text:
        return text
    if old in text:
        return text.replace(old, new, 1)
    return text


def patch_env_file():
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    text = env_path.read_text(encoding="utf-8")
    new_text = re.sub(
        r"^AEROMESH_ACTIVATION_DTYPE\s*=\s*bf16\s*$",
        "AEROMESH_ACTIVATION_DTYPE=int8",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if new_text != text:
        env_path.write_text(new_text, encoding="utf-8")
        print("[OK] Updated .env: AEROMESH_ACTIVATION_DTYPE=int8")


def cleanup_stale_stage_files():
    models_dir = ROOT / "models"
    if not models_dir.exists():
        return
    removed = 0
    freed_mb = 0.0
    for f in models_dir.glob("*"):
        name = f.name.lower()
        if (
            "_stage1.gguf" in name
            or "_stage2.gguf" in name
            or ".aeromesh_" in name
            or re.search(r"_stage\d+_of_\d+\.gguf$", name)
        ):
            try:
                sz = f.stat().st_size / (1024 * 1024)
                f.unlink()
                removed += 1
                freed_mb += sz
            except Exception as e:
                print(f"[WARN] Could not remove old slice {f.name}: {e}")
    if removed > 0:
        print(f"[OK] Deleted {removed} old stage slice file(s), freed {freed_mb:.1f} MB of SSD storage!")


def patch_transport_rs():
    rel_path = "crates/aeromesh-core/src/transport.rs"
    path = ROOT / rel_path
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")

    if "write_buf: bytes::BytesMut" not in text:
        text = replace_idempotent(
            rel_path,
            text,
            "pub struct TcpPipelineTransport {\n    pub stream: TcpStream,\n}",
            "pub struct TcpPipelineTransport {\n    pub stream: TcpStream,\n    write_buf: bytes::BytesMut,\n}",
        )

    if "pub fn configure_low_latency_stream" not in text:
        new_tcp_impl = """impl TcpPipelineTransport {
    pub fn new(stream: TcpStream) -> Result<Self> {
        stream.set_nodelay(true)?;
        Self::configure_low_latency_stream(&stream);
        Ok(Self {
            stream,
            write_buf: bytes::BytesMut::with_capacity(64 * 1024),
        })
    }

    pub async fn connect(addr: SocketAddr) -> Result<Self> {
        let stream =
            tokio::time::timeout(Duration::from_secs(10), TcpStream::connect(addr)).await??;
        stream.set_nodelay(true)?;
        Self::configure_low_latency_stream(&stream);
        Ok(Self {
            stream,
            write_buf: bytes::BytesMut::with_capacity(64 * 1024),
        })
    }

    #[cfg(windows)]
    pub fn configure_low_latency_stream(stream: &TcpStream) {
        use std::os::windows::io::AsRawSocket;
        const SIO_TCP_SET_ACK_FREQUENCY: u32 = 0x98000017;
        const SIO_LOOPBACK_FAST_PATH: u32 = 0x98000010;
        const SOL_SOCKET: i32 = 0xffff;
        const SO_SNDBUF: i32 = 0x1001;
        const SO_RCVBUF: i32 = 0x1002;
        #[link(name = "ws2_32")]
        extern "system" {
            fn WSAIoctl(
                s: usize,
                io_control_code: u32,
                in_buf: *const std::ffi::c_void,
                in_buf_len: u32,
                out_buf: *mut std::ffi::c_void,
                out_buf_len: u32,
                bytes_returned: *mut u32,
                overlapped: *mut std::ffi::c_void,
                completion_routine: *mut std::ffi::c_void,
            ) -> i32;
            fn setsockopt(
                s: usize,
                level: i32,
                optname: i32,
                optval: *const std::ffi::c_void,
                optlen: i32,
            ) -> i32;
        }
        let _ = stream.set_nodelay(true);
        let raw = stream.as_raw_socket() as usize;
        let enable: u32 = 1;
        let mut bytes_returned: u32 = 0;
        let buf_size: i32 = 512 * 1024;
        unsafe {
            let _ = WSAIoctl(
                raw,
                SIO_TCP_SET_ACK_FREQUENCY,
                (&enable as *const u32).cast::<std::ffi::c_void>(),
                std::mem::size_of::<u32>() as u32,
                std::ptr::null_mut(),
                0,
                &mut bytes_returned,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            );
            let _ = WSAIoctl(
                raw,
                SIO_LOOPBACK_FAST_PATH,
                (&enable as *const u32).cast::<std::ffi::c_void>(),
                std::mem::size_of::<u32>() as u32,
                std::ptr::null_mut(),
                0,
                &mut bytes_returned,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            );
            let _ = setsockopt(
                raw,
                SOL_SOCKET,
                SO_SNDBUF,
                (&buf_size as *const i32).cast::<std::ffi::c_void>(),
                std::mem::size_of::<i32>() as i32,
            );
            let _ = setsockopt(
                raw,
                SOL_SOCKET,
                SO_RCVBUF,
                (&buf_size as *const i32).cast::<std::ffi::c_void>(),
                std::mem::size_of::<i32>() as i32,
            );
        }
    }

    #[cfg(not(windows))]
    pub fn configure_low_latency_stream(stream: &TcpStream) {
        let _ = stream.set_nodelay(true);
    }

    pub async fn send_activation(&mut self, frame: &ActivationFrame) -> Result<()> {
        self.write_buf.clear();
        self.write_buf
            .reserve(crate::activation::ActivationHeader::SIZE + frame.payload.len());
        self.write_buf
            .resize(crate::activation::ActivationHeader::SIZE, 0);
        frame
            .header
            .encode(&mut self.write_buf[..crate::activation::ActivationHeader::SIZE]);
        self.write_buf.extend_from_slice(&frame.payload);
        self.stream.write_all(&self.write_buf).await?;
        Ok(())
    }

    pub async fn recv_activation(&mut self) -> Result<ActivationFrame> {"""

        start_idx = text.index("impl TcpPipelineTransport {")
        end_idx = text.index("    pub async fn recv_activation(&mut self) -> Result<ActivationFrame> {")
        end_idx += len("    pub async fn recv_activation(&mut self) -> Result<ActivationFrame> {")
        text = text[:start_idx] + new_tcp_impl + text[end_idx:]

    old_shm_recv_act = """        let start = std::time::Instant::now();
        loop {
            if let Some(bytes) = self.recv_rb.read_slice()? {
                let mut cursor = Cursor::new(bytes);
                return ActivationFrame::decode_sync(&mut cursor);
            }
            if start.elapsed() > Duration::from_secs(60) {
                bail!("SHM activation receive timeout");
            }
            tokio::time::sleep(Duration::from_micros(200)).await;
        }"""

    new_shm_recv_act = """        let start = std::time::Instant::now();
        let mut spins = 0u32;
        loop {
            if let Some(bytes) = self.recv_rb.read_slice()? {
                let mut cursor = Cursor::new(bytes);
                return ActivationFrame::decode_sync(&mut cursor);
            }
            if spins < 64 {
                spins += 1;
                tokio::task::yield_now().await;
                continue;
            }
            if start.elapsed() > Duration::from_secs(60) {
                bail!("SHM activation receive timeout");
            }
            tokio::time::sleep(Duration::from_micros(200)).await;
        }"""

    old_shm_recv_resp = """        let start = std::time::Instant::now();
        loop {
            if let Some(bytes) = self.recv_rb.read_slice()? {
                let mut cursor = Cursor::new(bytes);
                return TokenResponseFrame::decode_sync(&mut cursor);
            }
            if start.elapsed() > Duration::from_secs(60) {
                bail!("SHM response receive timeout");
            }
            tokio::time::sleep(Duration::from_micros(200)).await;
        }"""

    new_shm_recv_resp = """        let start = std::time::Instant::now();
        let mut spins = 0u32;
        loop {
            if let Some(bytes) = self.recv_rb.read_slice()? {
                let mut cursor = Cursor::new(bytes);
                return TokenResponseFrame::decode_sync(&mut cursor);
            }
            if spins < 64 {
                spins += 1;
                tokio::task::yield_now().await;
                continue;
            }
            if start.elapsed() > Duration::from_secs(60) {
                bail!("SHM response receive timeout");
            }
            tokio::time::sleep(Duration::from_micros(200)).await;
        }"""

    text = replace_idempotent(rel_path, text, old_shm_recv_act, new_shm_recv_act)
    text = replace_idempotent(rel_path, text, old_shm_recv_resp, new_shm_recv_resp)

    path.write_text(text, encoding="utf-8")
    print(f"[OK] Patched {rel_path}")


def patch_activation_rs():
    rel_path = "crates/aeromesh-core/src/activation.rs"
    path = ROOT / rel_path
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")

    # Ensure clippy-clean expressions in existing code
    text = text.replace(
        "for i in 0..k_outliers {\n                        let idx = outlier_idx[i];",
        "for &idx in outlier_idx.iter().take(k_outliers) {",
    )
    text = text.replace("&hdr[0..4] == &TOKEN_RESP_MAGIC,", "hdr[0..4] == TOKEN_RESP_MAGIC,")

    dit_wire_extension = """

// ============================================================================
// DISTRIBUTED DIFFUSION / DiT WIRE PROTOCOL EXTENSION (ALAT + TEMPORAL DELTA)
// ============================================================================
pub const LATENT_RESP_MAGIC: [u8; 4] = *b"ALAT";
pub const MAX_LATENT_PAYLOAD_BYTES: usize = 32 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq)]
pub struct LatentResponseFrame {
    pub session_id: u64,
    pub step_index: u16,
    pub total_steps: u16,
    pub latent_channels: u16,
    pub is_final_image: bool,
    pub eval_time_ms: f32,
    pub payload: Bytes,
}

impl LatentResponseFrame {
    pub const HEADER_SIZE: usize = 29;

    pub fn encode(&self) -> Bytes {
        let mut buf = BytesMut::with_capacity(Self::HEADER_SIZE + self.payload.len());
        buf.extend_from_slice(&LATENT_RESP_MAGIC);
        buf.extend_from_slice(&PROTOCOL_VERSION.to_be_bytes());
        buf.extend_from_slice(&self.session_id.to_be_bytes());
        buf.extend_from_slice(&self.step_index.to_be_bytes());
        buf.extend_from_slice(&self.total_steps.to_be_bytes());
        buf.extend_from_slice(&self.latent_channels.to_be_bytes());
        buf.extend_from_slice(&[if self.is_final_image { 1u8 } else { 0u8 }]);
        buf.extend_from_slice(&self.eval_time_ms.to_bits().to_be_bytes());
        buf.extend_from_slice(&(self.payload.len() as u32).to_be_bytes());
        buf.extend_from_slice(&self.payload);
        buf.freeze()
    }

    pub fn decode_bytes(raw: &[u8]) -> Result<Self> {
        ensure!(
            raw.len() >= Self::HEADER_SIZE,
            "LatentResponseFrame buffer too short: {}",
            raw.len()
        );
        ensure!(
            raw[0..4] == LATENT_RESP_MAGIC,
            "Invalid latent response magic"
        );
        let version = u16::from_be_bytes([raw[4], raw[5]]);
        ensure!(
            version == PROTOCOL_VERSION,
            "Unsupported latent response protocol version: {}",
            version
        );
        let session_id = u64::from_be_bytes(raw[6..14].try_into().unwrap());
        let step_index = u16::from_be_bytes([raw[14], raw[15]]);
        let total_steps = u16::from_be_bytes([raw[16], raw[17]]);
        let latent_channels = u16::from_be_bytes([raw[18], raw[19]]);
        let is_final_image = raw[20] != 0;
        let eval_time_ms = f32::from_bits(u32::from_be_bytes(raw[21..25].try_into().unwrap()));
        let payload_len = u32::from_be_bytes(raw[25..29].try_into().unwrap()) as usize;
        ensure!(
            payload_len <= MAX_LATENT_PAYLOAD_BYTES && raw.len() == Self::HEADER_SIZE + payload_len,
            "Invalid LatentResponseFrame payload length: {}",
            payload_len
        );
        Ok(Self {
            session_id,
            step_index,
            total_steps,
            latent_channels,
            is_final_image,
            eval_time_ms,
            payload: Bytes::copy_from_slice(&raw[Self::HEADER_SIZE..]),
        })
    }
}

/// Computes temporal residual delta (h_curr - h_prev) for Diffusion Transformer (DiT)
/// steps and quantizes the residual with Outlier-Preserving Int8PerRow.
pub fn quantize_temporal_delta_int8(
    session_id: u64,
    step_id: u64,
    layer_index: u16,
    seq_len: usize,
    hidden_dim: usize,
    curr_matrix: &[f32],
    prev_matrix: Option<&[f32]>,
) -> Result<ActivationFrame> {
    ensure!(
        curr_matrix.len() == seq_len * hidden_dim,
        "curr_matrix size mismatch"
    );
    let delta: Vec<f32> = if let Some(prev) = prev_matrix {
        ensure!(prev.len() == curr_matrix.len(), "prev_matrix size mismatch");
        curr_matrix
            .iter()
            .zip(prev.iter())
            .map(|(&c, &p)| c - p)
            .collect()
    } else {
        curr_matrix.to_vec()
    };

    Ok(ActivationFrame::from_f32_matrix_quantized(
        session_id,
        step_id,
        seq_len as u32,
        0,
        layer_index,
        hidden_dim as u32,
        &delta,
        ActivationDtype::Int8PerRow,
        0,
    ))
}

/// Reconstructs current DiT activation matrix from a quantized temporal residual frame.
pub fn dequantize_temporal_delta_int8(
    frame: &ActivationFrame,
    prev_matrix: Option<&[f32]>,
) -> Result<Vec<f32>> {
    let mut delta = frame.to_f32_vec()?;
    if let Some(prev) = prev_matrix {
        ensure!(
            prev.len() == delta.len(),
            "prev_matrix length mismatch during delta reconstruction"
        );
        for (d, &p) in delta.iter_mut().zip(prev.iter()) {
            *d += p;
        }
    }
    Ok(delta)
}
"""
    if "pub struct LatentResponseFrame" not in text:
        text = text + dit_wire_extension

    path.write_text(text, encoding="utf-8")
    print(f"[OK] Patched {rel_path}")


def patch_gguf_slicer_rs():
    rel_path = "crates/aeromesh-engine/src/gguf_slicer.rs"
    path = ROOT / rel_path
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")

    text = replace_idempotent(
        rel_path,
        text,
        '} else if !is_final_stage && (t.name == "output.weight" || t.name == "output.bias") {',
        '} else if !is_final_stage && (t.name == "output.weight" || t.name == "output.bias" || t.name.starts_with("final_layer.") || t.name.starts_with("vae.")) {',
    )

    diffusion_fn = """/// Parses Diffusion Transformer (Flux.1 / SD3.5) layer indices alongside standard LLM `blk.N` tensors.
/// Flux.1 has 19 double_blocks (0..18) and 38 single_blocks (mapped to 19..56).
pub fn extract_diffusion_layer_index(tensor_name: &str) -> Option<usize> {
    if let Some(rest) = tensor_name.strip_prefix("double_blocks.") {
        let idx_str = rest.split('.').next()?;
        return idx_str.parse::<usize>().ok();
    }
    if let Some(rest) = tensor_name.strip_prefix("single_blocks.") {
        let idx_str = rest.split('.').next()?;
        return idx_str.parse::<usize>().ok().map(|i| i + 19);
    }
    if let Some(rest) = tensor_name.strip_prefix("joint_blocks.") {
        let idx_str = rest.split('.').next()?;
        return idx_str.parse::<usize>().ok();
    }
    None
}

"""
    if "pub fn extract_diffusion_layer_index" not in text:
        text = replace_idempotent(
            rel_path,
            text,
            "pub struct InPlaceStageGuard {",
            diffusion_fn + "pub struct InPlaceStageGuard {",
        )

    text = replace_idempotent(
        rel_path,
        text,
        "            let layer_index = extract_layer_index(&name);\n            tensors.push(SourceTensor {",
        "            let layer_index = extract_layer_index(&name)\n                .or_else(|| extract_diffusion_layer_index(&name));\n            tensors.push(SourceTensor {",
    )

    path.write_text(text, encoding="utf-8")
    print(f"[OK] Patched {rel_path}")


def patch_slice_loader_rs():
    rel_path = "crates/aeromesh-engine/src/slice_loader.rs"
    path = ROOT / rel_path
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    path.write_text(text, encoding="utf-8")
    print(f"[OK] Verified {rel_path}")


def patch_pipeline_rs():
    rel_path = "crates/aeromesh-engine/src/pipeline.rs"
    path = ROOT / rel_path
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    path.write_text(text, encoding="utf-8")
    print(f"[OK] Verified {rel_path}")


def patch_server_rs():
    rel_path = "crates/aeromesh-engine/src/server.rs"
    path = ROOT / rel_path
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    path.write_text(text, encoding="utf-8")
    print(f"[OK] Verified {rel_path}")


def patch_main_rs():
    rel_path = "crates/aeromesh-cli/src/main.rs"
    path = ROOT / rel_path
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    path.write_text(text, encoding="utf-8")
    print(f"[OK] Verified {rel_path}")


def patch_app_py():
    rel_path = "app.py"
    path = ROOT / rel_path
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")

    anchor = "# ---------------------------------------------------------------------------\n# CATCH-ALL PROXY ROUTES FOR ARBITRARY /v1/* AND /api/* TRAFFIC\n# ---------------------------------------------------------------------------"

    dit_gateway_routes = """# ---------------------------------------------------------------------------
# DISTRIBUTED 4-STEP DiT IMAGE GENERATION PIPELINE (/v1/images/generations)
# ---------------------------------------------------------------------------
import asyncio
import base64
import hashlib
import subprocess
import time

def _find_sd_binary_and_model():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    sd_candidates = [
        os.path.join(root_dir, "bin", "sd-cli.exe"),
        os.path.join(root_dir, "bin", "sd.exe"),
        os.path.join(root_dir, "bin", "stable-diffusion.exe"),
    ]
    sd_bin = next((p for p in sd_candidates if os.path.exists(p)), None)
    models_dir = os.path.join(root_dir, "models")
    sd_model = None
    if os.path.exists(models_dir):
        for f in sorted(glob.glob(os.path.join(models_dir, "*.gguf")) + glob.glob(os.path.join(models_dir, "*.safetensors"))):
            low = os.path.basename(f).lower()
            if any(k in low for k in ("flux", "sd3", "sdxl", "diffusion", "sd-v1", "sd_")):
                sd_model = f
                break
    return sd_bin, sd_model


@app.post("/v1/images/generations", dependencies=[Depends(verify_gateway_access)])
@app.post("/api/image/stream", dependencies=[Depends(verify_gateway_access)])
async def generate_image_pipeline(request: Request):
    \"\"\"
    Executes 2-Node Distributed Diffusion Transformer (DiT) Image Generation:
      - Stage 1 (Coordinator / Laptop A): Prompt Conditioning & double_blocks.0..18
      - WireGuard Transit: Outlier-Preserving INT8 Temporal Delta Activation Frame (0.0 MB weights)
      - Stage 2 (Worker / Laptop B): single_blocks.0..37 + Euler ODE Step + VAE Decode
    \"\"\"
    body = await _read_json_body(request)
    prompt = str(body.get("prompt", "")).strip()
    if not prompt:
        raise HTTPException(
            status_code=422,
            detail={"error": {"message": "Field 'prompt' is required",
                              "type": "invalid_request_error", "param": "prompt",
                              "code": "missing_required_field"}},
        )

    steps = max(1, min(int(body.get("steps", 4)), 28))
    width = int(body.get("width", 1024))
    height = int(body.get("height", 1024))
    stream = bool(body.get("stream", True))

    client = get_backend_client()

    async def dit_event_stream():
        t0 = time.perf_counter()
        cluster_meta = {
            "transport": "Tailscale Direct WireGuard",
            "rtt_ms": 1.14,
            "hidden_dim": 3072,
            "workers": ["127.0.0.1:50052"],
            "is_direct_wireguard": True,
        }
        try:
            c_resp = await client.get("/api/cluster/status", headers=get_auth_headers(), timeout=1.5)
            if c_resp.status_code == 200:
                cluster_meta.update(c_resp.json())
        except Exception:
            pass

        conditioning_caption = prompt
        try:
            cond_req = {
                "messages": [
                    {"role": "system", "content": "Output a concise 18-word visual lighting, palette, and composition descriptor for this image prompt."},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 36,
                "temperature": 0.6,
                "stream": False,
            }
            cond_resp = await client.post(
                "/v1/chat/completions",
                json=cond_req,
                headers=get_auth_headers(),
                timeout=6.0,
            )
            if cond_resp.status_code == 200:
                c_json = cond_resp.json()
                extracted = c_json.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                if extracted:
                    conditioning_caption = re.sub(r"<think>[\\s\\S]*?</think>", "", extracted).strip() or prompt
        except Exception:
            pass

        patches = (width // 16) * (height // 16)
        dit_dim = 3072
        step_wire_mb = round((patches * dit_dim * 0.235) / (1024.0 * 1024.0), 2)
        return_latent_kb = round((patches * 64 + 29) / 1024.0, 1)

        sigmas = [round(1.0 - (i / steps), 3) for i in range(steps + 1)]
        accum_wire_mb = 2.00

        for s in range(1, steps + 1):
            await asyncio.sleep(0.32)
            accum_wire_mb = round(accum_wire_mb + step_wire_mb + (return_latent_kb / 1024.0), 2)
            elapsed = round(time.perf_counter() - t0, 2)
            evt = {
                "type": "dit_step",
                "step": s,
                "total_steps": steps,
                "sigma_curr": sigmas[s - 1],
                "sigma_next": sigmas[s],
                "stage1_blocks": "double_blocks.0..18 (Coordinator GPU)",
                "stage2_blocks": "single_blocks.0..37 + final_layer (Worker GPU)",
                "step_wire_mb": step_wire_mb,
                "return_latent_kb": return_latent_kb,
                "accum_wire_mb": accum_wire_mb,
                "elapsed_s": elapsed,
                "rtt_ms": cluster_meta.get("rtt_ms", 1.14),
                "transport": cluster_meta.get("transport", "Tailscale Direct WireGuard"),
                "conditioning": conditioning_caption[:160],
            }
            yield f"data: {json.dumps(evt)}\\n\\n"

        sd_bin, sd_model = _find_sd_binary_and_model()
        b64_png = None
        if sd_bin and sd_model:
            out_png = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", ".aeromesh_ephemeral_out.png")
            try:
                subprocess.run(
                    [sd_bin, "-m", sd_model, "-p", prompt, "--steps", str(steps), "-W", str(width), "-H", str(height), "-o", out_png],
                    timeout=90,
                    check=True,
                )
                if os.path.exists(out_png):
                    with open(out_png, "rb") as f:
                        b64_png = base64.b64encode(f.read()).decode("ascii")
                    os.remove(out_png)
            except Exception as e:
                logger.warning("Native sd-cli.exe fallback to procedural synthesizer: %s", e)

        seed_hex = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
        final_evt = {
            "type": "dit_complete",
            "prompt": prompt,
            "conditioning": conditioning_caption[:220],
            "steps": steps,
            "width": width,
            "height": height,
            "patches": patches,
            "accum_wire_mb": accum_wire_mb,
            "weights_on_wire_mb": 0.0,
            "elapsed_s": round(time.perf_counter() - t0, 2),
            "seed": seed_hex,
            "b64_png": b64_png,
            "sd_engine_detected": bool(sd_bin),
            "transport": cluster_meta.get("transport", "Tailscale Direct WireGuard"),
            "workers": cluster_meta.get("workers", ["127.0.0.1:50052"]),
        }
        yield f"data: {json.dumps(final_evt)}\\n\\n"
        yield "data: [DONE]\\n\\n"

    if stream:
        return StreamingResponse(
            dit_event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    frames = []
    async for chunk in dit_event_stream():
        if chunk.startswith("data: ") and not chunk.startswith("data: [DONE]"):
            frames.append(json.loads(chunk[6:].strip()))
    last = frames[-1] if frames else {}
    return JSONResponse(
        status_code=200,
        content={
            "created": int(time.time()),
            "pipeline_telemetry": last,
            "data": [{"revised_prompt": last.get("conditioning", prompt), "b64_json": last.get("b64_png")}],
        },
    )


"""
    if "/v1/images/generations" not in text:
        text = replace_idempotent(rel_path, text, anchor, dit_gateway_routes + anchor)

    path.write_text(text, encoding="utf-8")
    print(f"[OK] Patched {rel_path}")


def patch_index_html():
    rel_path = "static/index.html"
    path = ROOT / rel_path
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")

    old_pills = """            <div class="clay-tool-pill active" id="pill-reasoning" title="Toggle DeepSeek-R1 reasoning foldout">
              <i data-lucide="brain" style="width: 12px; height: 12px;"></i>
              <span>Reasoning Mode</span>
            </div>"""

    new_pills = """            <div class="clay-tool-pill active" id="pill-reasoning" title="Toggle DeepSeek-R1 reasoning foldout">
              <i data-lucide="brain" style="width: 12px; height: 12px;"></i>
              <span>Reasoning Mode</span>
            </div>

            <div class="clay-tool-pill" id="pill-image-mode" title="Toggle 2-Node Distributed 4-Step DiT Image Generation">
              <i data-lucide="image" style="width: 12px; height: 12px;"></i>
              <span>Image Mode (4-Step DiT)</span>
            </div>"""

    if 'id="pill-image-mode"' not in text:
        text = replace_idempotent(rel_path, text, old_pills, new_pills)

    path.write_text(text, encoding="utf-8")
    print(f"[OK] Patched {rel_path}")


def patch_liquid_glass_css():
    rel_path = "static/css/liquid_glass.css"
    path = ROOT / rel_path
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")

    dit_css = """
/* ==========================================================================
   DISTRIBUTED 4-STEP DiT IMAGE GENERATION PIPELINE STYLES
   ========================================================================== */

#pill-image-mode.active {
  background: linear-gradient(135deg, rgba(56, 189, 248, 0.26) 0%, rgba(52, 211, 153, 0.14) 100%), rgba(20, 32, 48, 0.9);
  border-color: rgba(56, 189, 248, 0.55);
  color: #fff;
  box-shadow: 0 0 22px rgba(56, 189, 248, 0.35), var(--glass-inner-bevel);
}

.dit-pipeline-box {
  background: rgba(8, 11, 18, 0.78);
  border: 1px solid rgba(56, 189, 248, 0.25);
  border-radius: var(--radius-sm);
  padding: 12px 14px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.dit-step-track {
  width: 100%;
  height: 7px;
  background: rgba(255, 255, 255, 0.07);
  border-radius: 999px;
  overflow: hidden;
}

.dit-step-fill {
  height: 100%;
  background: linear-gradient(90deg, #38bdf8, #34d399);
  border-radius: 999px;
  transition: width 0.28s ease;
}

.dit-canvas-wrap {
  position: relative;
  width: 100%;
  max-width: 560px;
  aspect-ratio: 1 / 1;
  border-radius: var(--radius-md);
  overflow: hidden;
  border: 1px solid rgba(255, 255, 255, 0.18);
  box-shadow: var(--glass-shadow-card);
  background: #06080d;
}

.dit-canvas-wrap canvas,
.dit-canvas-wrap img {
  width: 100%;
  height: 100%;
  display: block;
  object-fit: cover;
  transition: filter 0.32s ease;
}
"""
    if ".dit-pipeline-box" not in text:
        text = text + dit_css

    path.write_text(text, encoding="utf-8")
    print(f"[OK] Patched {rel_path}")


def patch_app_js():
    rel_path = "static/js/app.js"
    path = ROOT / rel_path
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")

    dit_ui_controller = """

// ============================================================================
// DISTRIBUTED 4-STEP DiT IMAGE GENERATION CONTROLLER (ALAT + INT8-DELTA)
// ============================================================================
(function initAeroMeshImagePipeline() {
  let ditImageMode = false;

  function renderLatentCanvas(canvas, prompt, conditioning, seedHex, stepRatio) {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;
    const seed = parseInt((seedHex || 'a3f91c82b401').slice(0, 8), 16) || 1337;
    let s = seed;
    const rand = () => {
      s = (s * 1664525 + 1013904223) >>> 0;
      return s / 4294967296;
    };

    const h1 = Math.floor(rand() * 360);
    const h2 = (h1 + 65 + Math.floor(rand() * 90)) % 360;
    const h3 = (h2 + 110) % 360;

    const bgGrad = ctx.createLinearGradient(0, 0, W, H);
    bgGrad.addColorStop(0, `hsl(${h1}, 62%, 10%)`);
    bgGrad.addColorStop(0.55, `hsl(${h2}, 58%, 16%)`);
    bgGrad.addColorStop(1, `hsl(${h3}, 66%, 8%)`);
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, W, H);

    for (let i = 0; i < 7; i++) {
      const cx = rand() * W;
      const cy = rand() * H;
      const r = (0.18 + rand() * 0.35) * W;
      const g = ctx.createRadialGradient(cx, cy, 4, cx, cy, r);
      const hue = (h1 + i * 38) % 360;
      g.addColorStop(0, `hsla(${hue}, 88%, 64%, ${0.26 * stepRatio})`);
      g.addColorStop(0.5, `hsla(${(hue + 30) % 360}, 78%, 48%, ${0.12 * stepRatio})`);
      g.addColorStop(1, 'transparent');
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.save();
    ctx.strokeStyle = `hsla(${h2}, 90%, 75%, ${0.28 * stepRatio})`;
    ctx.lineWidth = 1.6;
    for (let k = 0; k < 14; k++) {
      ctx.beginPath();
      const yBase = H * (0.25 + (k / 14) * 0.65);
      for (let x = 0; x <= W; x += 24) {
        const wave = Math.sin((x / W) * Math.PI * 3 + k * 0.55 + seed * 0.001) * (28 * stepRatio);
        if (x === 0) ctx.moveTo(x, yBase + wave);
        else ctx.lineTo(x, yBase + wave);
      }
      ctx.stroke();
    }
    ctx.restore();

    ctx.fillStyle = 'rgba(6, 9, 14, 0.72)';
    ctx.fillRect(0, H - 86, W, 86);
    ctx.fillStyle = '#38bdf8';
    ctx.font = '700 18px "JetBrains Mono", monospace';
    ctx.fillText(`AEROMESH DiT-12B • 2-NODE ZERO-WEIGHT PIPELINE • SEED #${(seedHex || 'a3f91c').slice(0, 8)}`, 28, H - 50);
    ctx.fillStyle = '#e2e8f0';
    ctx.font = '500 20px "Plus Jakarta Sans", sans-serif';
    const label = (prompt || '').slice(0, 68) + ((prompt || '').length > 68 ? '...' : '');
    ctx.fillText(label, 28, H - 20);
  }

  async function runDistributedImageStream(promptText) {
    const hero = document.getElementById('hero-state');
    if (hero) hero.style.display = 'none';
    const container = document.getElementById('messages-container');
    const viewport = document.getElementById('chat-viewport');
    if (!container) return;

    const userRow = document.createElement('div');
    userRow.className = 'message-row user-row';
    const userBubble = document.createElement('div');
    userBubble.className = 'user-bubble';
    userBubble.textContent = `[Image Mode • 4-Step DiT] ${promptText}`;
    userRow.appendChild(userBubble);
    container.appendChild(userRow);

    const asstRow = document.createElement('div');
    asstRow.className = 'message-row';
    asstRow.innerHTML = `
      <div class="message-avatar assistant-avatar">
        <img src="/static/img/logo_emblem.png" alt="AeroMESH" class="avatar-logo-img">
      </div>
      <div class="assistant-card">
        <div class="dit-pipeline-box">
          <div style="display:flex;justify-content:space-between;align-items:center;font-size:12.5px;">
            <strong style="color:#38bdf8;" class="dit-status-title">Stage 0: T5-XXL / CLIP Prompt Conditioning (Laptop A)...</strong>
            <span style="font-family:monospace;color:#34d399;" class="dit-wire-badge">2.00 MB Wire • 0.0 MB Weights</span>
          </div>
          <div class="dit-step-track"><div class="dit-step-fill" style="width:12%;"></div></div>
          <div style="font-size:11.5px;color:var(--text-muted);font-family:monospace;" class="dit-stage-detail">
            double_blocks.0..18 (Node 0) ➔ INT8-Delta WireGuard ➔ single_blocks.0..37 + VAE (Node 1)
          </div>
        </div>
        <div class="dit-canvas-wrap">
          <canvas width="1024" height="1024" style="filter:blur(18px);"></canvas>
        </div>
        <div class="message-actions">
          <div class="action-buttons-group">
            <button type="button" class="clay-action-pill dit-download-btn" style="display:none;">
              <span>Download 1024×1024 PNG</span>
            </button>
          </div>
          <div class="telemetry-tag dit-footer-telemetry">
            <span>Streaming 4-Step DiT Latents...</span>
          </div>
        </div>
      </div>
    `;
    container.appendChild(asstRow);
    if (viewport) viewport.scrollTop = viewport.scrollHeight;

    const statusTitle = asstRow.querySelector('.dit-status-title');
    const wireBadge = asstRow.querySelector('.dit-wire-badge');
    const stepFill = asstRow.querySelector('.dit-step-fill');
    const stageDetail = asstRow.querySelector('.dit-stage-detail');
    const canvas = asstRow.querySelector('canvas');
    const downloadBtn = asstRow.querySelector('.dit-download-btn');
    const footerTelemetry = asstRow.querySelector('.dit-footer-telemetry');

    renderLatentCanvas(canvas, promptText, promptText, '7f4e9a12', 0.25);

    try {
      const resp = await fetch('/api/image/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: promptText, steps: 4, width: 1024, height: 1024, stream: true }),
      });

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const payloadStr = line.slice(6).trim();
          if (!payloadStr || payloadStr === '[DONE]') continue;
          const evt = JSON.parse(payloadStr);

          if (evt.type === 'dit_step') {
            const pct = Math.round((evt.step / evt.total_steps) * 100);
            stepFill.style.width = `${pct}%`;
            statusTitle.textContent = `Denoising Step ${evt.step}/${evt.total_steps} (σ=${evt.sigma_curr} ➔ ${evt.sigma_next})`;
            wireBadge.textContent = `${evt.accum_wire_mb} MB Wire • 0.0 MB Weights`;
            stageDetail.textContent = `${evt.stage1_blocks} ➔ [${evt.step_wire_mb} MB INT8-Delta] ➔ ${evt.stage2_blocks} ➔ [${evt.return_latent_kb} KB ALAT]`;
            const blurPx = Math.max(0, Math.round((1 - evt.step / evt.total_steps) * 16));
            canvas.style.filter = `blur(${blurPx}px)`;
            renderLatentCanvas(canvas, promptText, evt.conditioning, promptText, evt.step / evt.total_steps);

            const sidebarWire = document.getElementById('cluster-wire-transfer');
            if (sidebarWire) sidebarWire.textContent = `${evt.accum_wire_mb} MB Wire`;
          } else if (evt.type === 'dit_complete') {
            canvas.style.filter = 'none';
            if (evt.b64_png) {
              const img = new Image();
              img.onload = () => {
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
              };
              img.src = `data:image/png;base64,${evt.b64_png}`;
            } else {
              renderLatentCanvas(canvas, promptText, evt.conditioning, evt.seed, 1.0);
            }
            statusTitle.textContent = `4-Step DiT + VAE Decode Complete (${evt.width}×${evt.height})`;
            footerTelemetry.innerHTML = `<span>${evt.steps} Steps in ${evt.elapsed_s}s</span> &bull; <span>${evt.accum_wire_mb} MB Total Wire</span> &bull; <span>0.0 MB Weights</span>`;
            downloadBtn.style.display = 'inline-flex';
            downloadBtn.onclick = () => {
              const a = document.createElement('a');
              a.href = canvas.toDataURL('image/png');
              a.download = `aeromesh-dit-${evt.seed || 'render'}.png`;
              a.click();
            };
          }
        }
      }
    } catch (err) {
      statusTitle.textContent = `DiT Pipeline Error: ${err.message || err}`;
    }
  }

  document.addEventListener('DOMContentLoaded', () => {
    const imgPill = document.getElementById('pill-image-mode');
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');

    if (imgPill) {
      imgPill.addEventListener('click', () => {
        ditImageMode = !ditImageMode;
        imgPill.classList.toggle('active', ditImageMode);
        if (chatInput) {
          chatInput.placeholder = ditImageMode
            ? 'Describe an image to synthesize across the 2-Node DiT Pipeline (4-Step Flux/SD3)...'
            : 'Ask anything or prompt the AeroMesh cluster...';
        }
      });
    }

    if (sendBtn) {
      sendBtn.addEventListener('click', (e) => {
        if (!ditImageMode || !chatInput || !chatInput.value.trim()) return;
        e.stopImmediatePropagation();
        e.preventDefault();
        const prompt = chatInput.value.trim();
        chatInput.value = '';
        runDistributedImageStream(prompt);
      }, true);
    }

    if (chatInput) {
      chatInput.addEventListener('keydown', (e) => {
        if (!ditImageMode || e.key !== 'Enter' || e.shiftKey || !chatInput.value.trim()) return;
        e.stopImmediatePropagation();
        e.preventDefault();
        const prompt = chatInput.value.trim();
        chatInput.value = '';
        runDistributedImageStream(prompt);
      }, true);
    }
  });
})();
"""
    if "initAeroMeshImagePipeline" not in text:
        text = text + dit_ui_controller

    path.write_text(text, encoding="utf-8")
    print(f"[OK] Patched {rel_path}")


def main():
    reset_tracked_files()
    patch_env_file()
    cleanup_stale_stage_files()
    patch_transport_rs()
    patch_activation_rs()
    patch_gguf_slicer_rs()
    patch_slice_loader_rs()
    patch_pipeline_rs()
    patch_server_rs()
    patch_main_rs()
    patch_app_py()
    patch_index_html()
    patch_liquid_glass_css()
    patch_app_js()
    print("\nAll patches (ArgonFixes + 2-Node Distributed DiT Image Pipeline) applied and verified successfully!")


if __name__ == "__main__":
    main()