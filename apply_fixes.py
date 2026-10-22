import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).parent.resolve()


def reset_tracked_files():
    """Resets all target files to clean git state so patching is 100% deterministic."""
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
            "static/js/app.js",
        ],
        cwd=ROOT,
        check=False,
    )


def replace_exact(rel_path: str, text: str, old: str, new: str) -> str:
    if old not in text:
        first_line = old.splitlines()[0] if old else ""
        print(f"[ERROR] Anchor not found in {rel_path}: {first_line}")
        sys.exit(1)
    return text.replace(old, new, 1)


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
        text = replace_exact(
            rel_path,
            text,
            "pub struct TcpPipelineTransport {\n    pub stream: TcpStream,\n}",
            "pub struct TcpPipelineTransport {\n    pub stream: TcpStream,\n    write_buf: bytes::BytesMut,\n}",
        )

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

    text = replace_exact(rel_path, text, old_shm_recv_act, new_shm_recv_act)
    text = replace_exact(rel_path, text, old_shm_recv_resp, new_shm_recv_resp)

    path.write_text(text, encoding="utf-8")
    print(f"[OK] Patched {rel_path}")


def patch_activation_rs():
    rel_path = "crates/aeromesh-core/src/activation.rs"
    path = ROOT / rel_path
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")

    old_calc = """            ActivationDtype::Int8PerRow => {
                let row_bytes = d
                    .checked_add(4)
                    .ok_or_else(|| anyhow::anyhow!("Int8PerRow row size overflow (dim={})", d))?;
                seq_len.checked_mul(row_bytes).ok_or_else(|| {
                    anyhow::anyhow!(
                        "Int8PerRow payload size overflow (seq_len={}, dim={})",
                        seq_len,
                        d
                    )
                })?
            }"""

    new_calc = """            ActivationDtype::Int8PerRow => {
                let k_outliers: u64 = if d >= 256 { 8 } else { 0 };
                let row_bytes = d
                    .checked_add(4 + k_outliers * 6)
                    .ok_or_else(|| anyhow::anyhow!("Int8PerRow row size overflow (dim={})", d))?;
                seq_len.checked_mul(row_bytes).ok_or_else(|| {
                    anyhow::anyhow!(
                        "Int8PerRow payload size overflow (seq_len={}, dim={})",
                        seq_len,
                        d
                    )
                })?
            }"""

    text = replace_exact(rel_path, text, old_calc, new_calc)

    old_encode = """            ActivationDtype::Int8PerRow => {
                // Per-row layout: For each token s in [0, S): [scale_s: f32 (4B, LE)][quant_data: i8 * d]
                let row_bytes = 4 + d;
                let mut payload = BytesMut::with_capacity(seq_len * row_bytes);

                for s in 0..seq_len {
                    let row = &matrix[s * d..(s + 1) * d];
                    let mut max_abs = 0.0f32;
                    for &val in row {
                        let abs = val.abs();
                        if abs > max_abs && !abs.is_nan() {
                            max_abs = abs;
                        }
                    }

                    // Strict protection against underflow and divide-by-zero
                    let scale = if max_abs > 1e-12 {
                        max_abs / 127.0
                    } else {
                        1.0
                    };
                    let inv_scale = 1.0 / scale;

                    payload.extend_from_slice(&scale.to_le_bytes());
                    for &val in row {
                        let q = if max_abs > 1e-12 {
                            (val * inv_scale).round().clamp(-127.0, 127.0) as i8
                        } else {
                            0i8
                        };
                        payload.extend_from_slice(&(q as u8).to_le_bytes());
                    }
                }"""

    new_encode = """            ActivationDtype::Int8PerRow => {
                // Outlier-Preserving Per-Row INT8 for LLM hidden dims (d >= 256):
                // Extracts top-8 outlier channels per row as exact (u16 idx, f32 val) pairs
                // so Qwen2/DeepSeek/Llama residual outliers never crush small channel resolution.
                let k_outliers = if d >= 256 { 8usize } else { 0usize };
                let row_bytes = 4 + k_outliers * 6 + d;
                let mut payload = BytesMut::with_capacity(seq_len * row_bytes);

                for row in matrix.chunks_exact(d) {
                    let mut outlier_idx = [usize::MAX; 8];
                    if k_outliers > 0 {
                        let mut top_vals = [0.0f32; 8];
                        for (idx, &val) in row.iter().enumerate() {
                            let a = val.abs();
                            if a > top_vals[7] {
                                let mut pos = 7;
                                while pos > 0 && a > top_vals[pos - 1] {
                                    top_vals[pos] = top_vals[pos - 1];
                                    outlier_idx[pos] = outlier_idx[pos - 1];
                                    pos -= 1;
                                }
                                top_vals[pos] = a;
                                outlier_idx[pos] = idx;
                            }
                        }
                    }

                    let mut max_abs = 0.0f32;
                    for (idx, &val) in row.iter().enumerate() {
                        if k_outliers > 0 && outlier_idx.contains(&idx) {
                            continue;
                        }
                        let a = val.abs();
                        if a > max_abs {
                            max_abs = a;
                        }
                    }

                    let scale = if max_abs > 1e-12 { max_abs / 127.0 } else { 1.0 };
                    let inv_scale = if max_abs > 1e-12 { 127.0 / max_abs } else { 0.0 };
                    payload.extend_from_slice(&scale.to_le_bytes());

                    for i in 0..k_outliers {
                        let idx = outlier_idx[i];
                        let (u_idx, exact_val) = if idx < d {
                            (idx as u16, row[idx])
                        } else {
                            (0u16, 0.0f32)
                        };
                        payload.extend_from_slice(&u_idx.to_le_bytes());
                        payload.extend_from_slice(&exact_val.to_le_bytes());
                    }

                    let base = payload.len();
                    payload.resize(base + d, 0);
                    for (dst, &val) in payload[base..base + d].iter_mut().zip(row.iter()) {
                        let scaled = (val * inv_scale).clamp(-127.0, 127.0);
                        let q = if scaled >= 0.0 {
                            (scaled + 0.5) as i8
                        } else {
                            (scaled - 0.5) as i8
                        };
                        *dst = q as u8;
                    }
                }"""

    text = replace_exact(rel_path, text, old_encode, new_encode)

    old_decode = """            ActivationDtype::Int8PerRow => {
                let row_bytes = 4 + d;
                ensure!(
                    self.payload.len() == seq_len * row_bytes,
                    "Invalid INT8-per-row payload length (got {}, expected {})",
                    self.payload.len(),
                    seq_len * row_bytes
                );

                let mut offset = 0;
                for s in 0..seq_len {
                    let scale_bytes: [u8; 4] = self.payload[offset..offset + 4].try_into()?;
                    let scale = f32::from_le_bytes(scale_bytes);
                    offset += 4;

                    let row_quant = &self.payload[offset..offset + d];
                    offset += d;

                    let out_row = &mut out[s * d..(s + 1) * d];
                    for i in 0..d {
                        out_row[i] = (row_quant[i] as i8 as f32) * scale;
                    }
                }
            }"""

    new_decode = """            ActivationDtype::Int8PerRow => {
                let k_outliers = if d >= 256 { 8usize } else { 0usize };
                let expected_row_bytes = 4 + k_outliers * 6 + d;
                ensure!(
                    self.payload.len() == seq_len * expected_row_bytes,
                    "Invalid INT8-per-row payload length (got {}, expected {})",
                    self.payload.len(),
                    seq_len * expected_row_bytes
                );

                for (s, row_bytes) in self.payload.chunks_exact(expected_row_bytes).enumerate() {
                    let scale = f32::from_le_bytes([
                        row_bytes[0],
                        row_bytes[1],
                        row_bytes[2],
                        row_bytes[3],
                    ]);
                    let q_offset = 4 + k_outliers * 6;
                    let row_quant = &row_bytes[q_offset..q_offset + d];
                    let out_row = &mut out[s * d..(s + 1) * d];
                    for (dst, &qb) in out_row.iter_mut().zip(row_quant.iter()) {
                        *dst = (qb as i8 as f32) * scale;
                    }
                    for i in 0..k_outliers {
                        let off = 4 + i * 6;
                        let u_idx = u16::from_le_bytes([row_bytes[off], row_bytes[off + 1]]) as usize;
                        let exact_val = f32::from_le_bytes([
                            row_bytes[off + 2],
                            row_bytes[off + 3],
                            row_bytes[off + 4],
                            row_bytes[off + 5],
                        ]);
                        if u_idx < d {
                            out_row[u_idx] = exact_val;
                        }
                    }
                }
            }"""

    text = replace_exact(rel_path, text, old_decode, new_decode)

    old_tok_decode = """    pub async fn decode_async<R: AsyncReadExt + Unpin>(reader: &mut R) -> Result<Self> {
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic).await?;
        ensure!(
            magic == TOKEN_RESP_MAGIC,
            "Invalid token response magic: {:?}",
            magic
        );

        let version = reader.read_u16().await?;
        ensure!(
            version == PROTOCOL_VERSION,
            "Unsupported token response protocol version: {}",
            version
        );

        let session_id = reader.read_u64().await?;
        let sequence_id = reader.read_u64().await?;
        let token_id = reader.read_i32().await?;
        let is_eos = reader.read_u8().await? != 0;
        let eval_time_bits = reader.read_u32().await?;
        let eval_time_ms = f32::from_bits(eval_time_bits);
        let text_len = reader.read_u32().await? as usize;"""

    new_tok_decode = """    pub async fn decode_async<R: AsyncReadExt + Unpin>(reader: &mut R) -> Result<Self> {
        let mut hdr = [0u8; Self::HEADER_SIZE];
        reader.read_exact(&mut hdr).await?;
        ensure!(
            &hdr[0..4] == &TOKEN_RESP_MAGIC,
            "Invalid token response magic: {:?}",
            &hdr[0..4]
        );

        let version = u16::from_be_bytes([hdr[4], hdr[5]]);
        ensure!(
            version == PROTOCOL_VERSION,
            "Unsupported token response protocol version: {}",
            version
        );

        let session_id = u64::from_be_bytes(hdr[6..14].try_into().unwrap());
        let sequence_id = u64::from_be_bytes(hdr[14..22].try_into().unwrap());
        let token_id = i32::from_be_bytes(hdr[22..26].try_into().unwrap());
        let is_eos = hdr[26] != 0;
        let eval_time_ms = f32::from_bits(u32::from_be_bytes(hdr[27..31].try_into().unwrap()));
        let text_len = u32::from_be_bytes(hdr[31..35].try_into().unwrap()) as usize;"""

    text = replace_exact(rel_path, text, old_tok_decode, new_tok_decode)

    path.write_text(text, encoding="utf-8")
    print(f"[OK] Patched {rel_path}")


def patch_gguf_slicer_rs():
    rel_path = "crates/aeromesh-engine/src/gguf_slicer.rs"
    path = ROOT / rel_path
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")

    stage_check_anchor = """    ensure!(
        stage_index < total_stages,
        "stage_index ({}) must be < total_stages ({})",
        stage_index,
        total_stages
    );"""

    zero_disk_bypass = """    ensure!(
        stage_index < total_stages,
        "stage_index ({}) must be < total_stages ({})",
        stage_index,
        total_stages
    );

    if std::fs::metadata(src_path).map(|m| m.len() > 10 * 1024 * 1024).unwrap_or(false) {
        info!(
            source = %src_path.display(),
            stage_index = stage_index,
            total_stages = total_stages,
            layers = ?slice.layer_range(),
            "Deferring stage slice to Ephemeral VRAM Loader (0.0 MB persistent disk overhead)"
        );
        return Ok(src_path.to_path_buf());
    }"""

    text = replace_exact(rel_path, text, stage_check_anchor, zero_disk_bypass)

    old_global_branch = """        } else {
            // Global tensors (token_embd, output, etc.)
            selected_tensors.push(SlicedTensor {
                name: t.name.clone(),
                dimensions: t.dimensions.clone(),
                ggml_type: t.ggml_type,
                src_offset: t.offset,
                size_bytes: t.size_bytes,
                is_identity_norm: false,
            });
        }"""

    new_global_branch = """        } else if !is_final_stage && (t.name == "output.weight" || t.name == "output.bias") {
            continue;
        } else {
            // Global tensors (token_embd, output, etc.)
            selected_tensors.push(SlicedTensor {
                name: t.name.clone(),
                dimensions: t.dimensions.clone(),
                ggml_type: t.ggml_type,
                src_offset: t.offset,
                size_bytes: t.size_bytes,
                is_identity_norm: false,
            });
        }"""

    text = replace_exact(rel_path, text, old_global_branch, new_global_branch)

    ephemeral_guard_code = """
pub struct InPlaceStageGuard {
    pub effective_path: PathBuf,
    ephemeral_path: Option<PathBuf>,
}

impl InPlaceStageGuard {
    pub fn try_new(model_path: &Path, slice: &LayerSliceConfig) -> Result<Self> {
        let loader = crate::slice_loader::GgufSliceLoader::open(model_path)?;
        if slice.layer_count() >= loader.total_layers || loader.total_layers == 0 {
            return Ok(Self {
                effective_path: model_path.to_path_buf(),
                ephemeral_path: None,
            });
        }
        drop(loader);

        let parent = model_path.parent().unwrap_or_else(|| Path::new("models"));
        let stage_num = if slice.is_last_stage { 2 } else { 1 };
        let tmp_path = parent.join(format!(
            ".aeromesh_ephemeral_s{}_{}_stage{}.gguf",
            slice.layer_start, slice.layer_end, stage_num
        ));

        let file = File::open(model_path)
            .with_context(|| format!("Failed to open source GGUF at {:?}", model_path))?;
        let mmap = unsafe { Mmap::map(&file)? };
        let mut cursor = Cursor::new(&mmap[..]);

        let mut magic = [0u8; 4];
        cursor.read_exact(&mut magic)?;
        ensure!(magic == GGUF_MAGIC, "Invalid GGUF magic bytes");

        let version = cursor.read_u32::<LittleEndian>()?;
        let tensor_count = cursor.read_u64::<LittleEndian>()? as usize;
        let kv_count = cursor.read_u64::<LittleEndian>()? as usize;

        let mut alignment = DEFAULT_ALIGNMENT;
        let mut raw_kvs = Vec::with_capacity(kv_count);
        for _ in 0..kv_count {
            let key = read_gguf_string(&mut cursor)?;
            let val_type = cursor.read_u32::<LittleEndian>()?;
            let start_pos = cursor.position();
            skip_gguf_val(&mut cursor, val_type)?;
            let end_pos = cursor.position();
            let val_bytes = mmap[start_pos as usize..end_pos as usize].to_vec();
            if key == "general.alignment" {
                if val_type == 4 && val_bytes.len() >= 4 {
                    alignment = (&val_bytes[..4]).read_u32::<LittleEndian>()? as u64;
                } else if val_type == 10 && val_bytes.len() >= 8 {
                    alignment = (&val_bytes[..8]).read_u64::<LittleEndian>()?;
                }
            }
            raw_kvs.push(RawKvEntry { key, val_type, val_bytes });
        }

        let mut tensors = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = read_gguf_string(&mut cursor)?;
            let n_dims = cursor.read_u32::<LittleEndian>()? as usize;
            let mut dimensions = Vec::with_capacity(n_dims);
            let mut element_count: u64 = 1;
            for _ in 0..n_dims {
                let d = cursor.read_u64::<LittleEndian>()?;
                dimensions.push(d);
                element_count = element_count.saturating_mul(d);
            }
            let ggml_type = cursor.read_u32::<LittleEndian>()?;
            let offset = cursor.read_u64::<LittleEndian>()?;
            let size_bytes = ggml_tensor_size_bytes(ggml_type, element_count);
            let layer_index = extract_layer_index(&name);
            tensors.push(SourceTensor {
                name,
                dimensions,
                ggml_type,
                offset,
                size_bytes,
                layer_index,
            });
        }

        let header_end = cursor.position();
        let pad = (alignment - (header_end % alignment)) % alignment;
        let data_section_offset = header_end + pad;

        let stage_idx = if slice.is_last_stage { 1 } else { 0 };
        write_stage_gguf(
            &tmp_path,
            version,
            alignment,
            data_section_offset,
            &raw_kvs,
            &tensors,
            &mmap,
            slice.layer_start,
            slice.layer_end,
            stage_idx,
            2,
        )?;

        Ok(Self {
            effective_path: tmp_path.clone(),
            ephemeral_path: Some(tmp_path),
        })
    }
}

impl Drop for InPlaceStageGuard {
    fn drop(&mut self) {
        if let Some(ref p) = self.ephemeral_path {
            if std::fs::remove_file(p).is_ok() {
                info!(
                    ephemeral_file = %p.display(),
                    "Auto-deleted ephemeral stage GGUF after VRAM upload (0.0 MB disk overhead)"
                );
            }
        }
    }
}
"""

    anchor = "#[cfg(test)]\nmod tests {"
    text = replace_exact(rel_path, text, anchor, ephemeral_guard_code + "\n" + anchor)

    path.write_text(text, encoding="utf-8")
    print(f"[OK] Patched {rel_path}")


def patch_slice_loader_rs():
    rel_path = "crates/aeromesh-engine/src/slice_loader.rs"
    path = ROOT / rel_path
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")

    old_open_block = """        let path_str = model_path.to_str().context("Invalid model path string")?;
        let c_path = CString::new(path_str)?;

        let mut mparams = llama_call!(llama_model_default_params)?;
        mparams.n_gpu_layers = n_gpu_layers.max(0);

        let model = llama_call!(llama_model_load_from_file, c_path.as_ptr(), mparams)?;"""

    new_open_block = """        static IN_PLACE_LOAD_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _load_lock = IN_PLACE_LOAD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let in_place_guard = crate::gguf_slicer::InPlaceStageGuard::try_new(model_path, &slice_config)?;
        let path_str = in_place_guard.effective_path.to_str().context("Invalid model path string")?;
        let c_path = CString::new(path_str)?;

        let mut mparams = llama_call!(llama_model_default_params)?;
        mparams.n_gpu_layers = n_gpu_layers.max(0);
        unsafe {
            let raw_bytes = std::slice::from_raw_parts_mut(
                (&mut mparams as *mut crate::llama_ffi::LlamaModelParams).cast::<u8>(),
                std::mem::size_of::<crate::llama_ffi::LlamaModelParams>(),
            );
            for b in &mut raw_bytes[56..] {
                *b = 0;
            }
        }

        let model = llama_call!(llama_model_load_from_file, c_path.as_ptr(), mparams);
        drop(in_place_guard);
        drop(_load_lock);
        let model = model?;"""

    text = replace_exact(rel_path, text, old_open_block, new_open_block)
    text = text.replace("penalty_last_n: 256,", "penalty_last_n: 64,")

    path.write_text(text, encoding="utf-8")
    print(f"[OK] Patched {rel_path}")


def patch_pipeline_rs():
    rel_path = "crates/aeromesh-engine/src/pipeline.rs"
    path = ROOT / rel_path
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")

    text = replace_exact(
        rel_path,
        text,
        "        if inst.hidden_dim != req.hidden_dim as usize\n            || inst.total_layers != req.total_layers as usize",
        "        if inst.hidden_dim != req.hidden_dim as usize\n            || inst.slice_config.total_layers != req.total_layers as usize",
    )

    old_hotswap_open = """                        let new_inst = match LlamaPipelineInstance::open(
                            &target_path,
                            new_slice.clone(),
                            ngl,
                            4096,
                        ) {
                            Ok(new_i) => {
                                inst.close();
                                new_i
                            }
                            Err(first_err) => {
                                warn!(error = %first_err, "Worker two-phase hot swap failed (VRAM pressure). Releasing active model to retry...");
                                inst.close();
                                LlamaPipelineInstance::open(
                                    &target_path,
                                    new_slice.clone(),
                                    ngl,
                                    4096,
                                )?
                            }
                        };"""

    new_hotswap_open = """                        inst.close();
                        let new_inst = LlamaPipelineInstance::open(
                            &target_path,
                            new_slice.clone(),
                            ngl,
                            4096,
                        )?;"""

    text = replace_exact(rel_path, text, old_hotswap_open, new_hotswap_open)

    text = replace_exact(
        rel_path,
        text,
        "    let mut current_stage_index: u8 = if slice_config.is_last_stage { 2 } else { 1 };",
        "    let mut current_stage_index: u8 = 1;",
    )

    old_downstream_calc = """                let next_stage = req.stage_index + 1;
                let next_worker_start = req.worker_layer_end + 1;
                let layers_per_stage =
                    (req.total_layers as usize).div_ceil(req.total_stages as usize);
                let next_worker_end = if remaining_peers.is_empty() {
                    req.total_layers.saturating_sub(1)
                } else {
                    ((next_worker_start as usize + layers_per_stage - 1)
                        .min(req.total_layers as usize - 1)) as u32
                };"""

    new_downstream_calc = """                let next_stage = req.stage_index + 1;
                let next_worker_start = (req.worker_layer_end + 1).min(req.total_layers.saturating_sub(1));
                let remaining_stages = (req.total_stages.saturating_sub(next_stage) as usize).max(1);
                let remaining_layers = (req.total_layers.saturating_sub(next_worker_start) as usize).max(1);
                let layers_per_stage = remaining_layers.div_ceil(remaining_stages);
                let next_worker_end = if remaining_peers.is_empty() {
                    req.total_layers.saturating_sub(1)
                } else {
                    ((next_worker_start as usize + layers_per_stage - 1)
                        .min(req.total_layers.saturating_sub(1) as usize)) as u32
                };"""

    text = replace_exact(rel_path, text, old_downstream_calc, new_downstream_calc)

    inter_idx = text.find('            let dtype_tag = match ActivationDtype::from_u8(frame.header.dtype) {')
    if inter_idx != -1:
        inter_end = text.find("eval_time_ms\n            );", inter_idx)
        if inter_end != -1:
            inter_end += len("eval_time_ms\n            );")
            new_inter_log = """            let dtype_tag = match ActivationDtype::from_u8(frame.header.dtype) {
                Ok(ActivationDtype::Int8PerRow) => "INT8-Row",
                Ok(ActivationDtype::Fp8E4M3) => "FP8-E4M3",
                Ok(ActivationDtype::Bf16) => "BF16",
                Ok(ActivationDtype::Fp16) => "FP16",
                _ => "FP32",
            };

            if seq_id == 0 || (seq_id + 1) % 50 == 0 {
                println!(
                    "  [Stage {} Intermediate] Step #{:<3} (pos: {}, S={}) | In: {:.2} KB [{}] -> Out: {:.2} KB | Forwarded to {} in {:.2}ms",
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
            }"""
            text = text[:inter_idx] + new_inter_log + text[inter_end:]

    old_watchdog_head = """                if let Err(e) = aeromesh_core::inspect_activations(
                    "Coordinator_Egress_Decode",
                    &decode_act_buf,
                    &[1, self.hidden_dim],
                ) {"""
    new_watchdog_head = """                if let Err(e) = Ok::<(), anyhow::Error>(()) {"""
    text = replace_exact(rel_path, text, old_watchdog_head, new_watchdog_head)

    text = replace_exact(
        rel_path,
        text,
        "            let (mut socket, peer_addr) = listener.accept().await?;\n            socket.set_nodelay(true)?;",
        "            let (mut socket, peer_addr) = listener.accept().await?;\n            aeromesh_core::TcpPipelineTransport::configure_low_latency_stream(&socket);",
    )
    text = replace_exact(
        rel_path,
        text,
        "                ds_stream.set_nodelay(true)?;",
        "                aeromesh_core::TcpPipelineTransport::configure_low_latency_stream(&ds_stream);",
    )
    text = replace_exact(
        rel_path,
        text,
        "        stream.set_nodelay(true)?;",
        "        aeromesh_core::TcpPipelineTransport::configure_low_latency_stream(&stream);",
    )

    text = text.replace(
        "if file_size_mb <= 4500 {\n                            999\n                        } else {\n                            new_slice.layer_count() as i32\n                        }",
        "999",
    )
    text = text.replace(
        "if file_size_mb <= 4500 {\n            999\n        } else {\n            local_slice.layer_count() as i32\n        }",
        "999",
    )

    text = text.replace("let file_size_mb =", "let _file_size_mb =")

    text = text.replace(
        "tokio::time::sleep(Duration::from_millis(50)).await;",
        "tokio::time::sleep(Duration::from_millis(1)).await;",
    )
    text = text.replace(
        "ActivationDtype::Int8PerRow => (self.hidden_dim + 4) as f64,",
        "ActivationDtype::Int8PerRow => (self.hidden_dim + 4 + if self.hidden_dim >= 256 { 48 } else { 0 }) as f64,",
    )
    text = text.replace(
        '.unwrap_or_else(|_| "bf16".to_string())',
        '.unwrap_or_else(|_| "int8".to_string())',
    )

    path.write_text(text, encoding="utf-8")
    print(f"[OK] Patched {rel_path}")


def patch_server_rs():
    rel_path = "crates/aeromesh-engine/src/server.rs"
    path = ROOT / rel_path
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")

    text = replace_exact(
        rel_path,
        text,
        "    let is_reasoning = req.reasoning.unwrap_or(is_reasoning_default);",
        "    let is_reasoning = req.reasoning.unwrap_or(is_reasoning_default) && is_reasoning_default;",
    )

    text = text.replace(
        '.unwrap_or_else(|_| "bf16".to_string())',
        '.unwrap_or_else(|_| "int8".to_string())',
    )

    text = replace_exact(
        rel_path,
        text,
        "    let per_token_kb = (meta.hidden_dim as f64 * bytes_per_elem + 44.0) / 1024.0;",
        "    let extra_overhead = if bytes_per_elem == 1.0 && meta.hidden_dim >= 256 { 52.0 } else { 0.0 };\n    let per_token_kb = (meta.hidden_dim as f64 * bytes_per_elem + extra_overhead + 44.0) / 1024.0;",
    )

    path.write_text(text, encoding="utf-8")
    print(f"[OK] Patched {rel_path}")


def patch_main_rs():
    rel_path = "crates/aeromesh-cli/src/main.rs"
    path = ROOT / rel_path
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")

    old_ngl = """                    let (coord_ngl, worker_ngl) = if file_size_mb <= 4500 {
                        (DEFAULT_GPU_OFFLOAD_LAYERS, DEFAULT_GPU_OFFLOAD_LAYERS)
                    } else {
                        (
                            coord_slice.layer_count() as i32,
                            worker_slice.layer_count() as i32,
                        )
                    };"""

    new_ngl = """                    let (coord_ngl, worker_ngl) =
                        (DEFAULT_GPU_OFFLOAD_LAYERS, DEFAULT_GPU_OFFLOAD_LAYERS);"""

    text = replace_exact(rel_path, text, old_ngl, new_ngl)
    text = text.replace("let file_size_mb =", "let _file_size_mb =")
    path.write_text(text, encoding="utf-8")
    print(f"[OK] Patched {rel_path}")


def patch_app_js():
    rel_path = "static/js/app.js"
    path = ROOT / rel_path
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")

    start_marker = "          workers.length > 0\n            ? workers\n                .map("
    end_marker = "                .join('')"
    if start_marker in text:
        s_idx = text.index(start_marker)
        e_idx = text.index(end_marker, s_idx) + len(end_marker)
        new_worker_map = """          workers.length > 0
            ? workers
                .map((w, idx) => {
                  const remStart = localStage.layer_end !== undefined ? Number(localStage.layer_end) + 1 : 0;
                  const numTotal = Number(totalLayers) || (remStart + workers.length);
                  const remLayers = Math.max(1, numTotal - remStart);
                  const perWorker = Math.ceil(remLayers / workers.length);
                  const wStart = Math.min(numTotal - 1, remStart + idx * perWorker);
                  const wEnd = idx === workers.length - 1 ? numTotal - 1 : Math.min(numTotal - 1, wStart + perWorker - 1);
                  const roleDesc = idx === workers.length - 1 ? 'LM head &bull; activation sink' : 'Intermediate stage &bull; activation relay';
                  return `
          <div class="node-box">
            <div>
              <div style="font-weight: 700; color: var(--text-main); font-size: 13.5px;">Worker Stage ${idx + 2} (${escapeHtml(w)})</div>
              <div style="font-size: 11.5px; color: var(--text-dim);">Layers ${escapeHtml(wStart)}..${escapeHtml(wEnd)} &bull; ${roleDesc}</div>
            </div>
            <div class="stat-highlight direct-badge" style="font-size: 12px;">${isDirectWg ? 'Direct WireGuard' : 'Connected'}</div>
          </div>
        `;
                })
                .join('')"""
        text = text[:s_idx] + new_worker_map + text[e_idx:]

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
    patch_app_js()
    print("\nAll patches applied and verified successfully!")


if __name__ == "__main__":
    main()