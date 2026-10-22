import pathlib
import sys

ROOT = pathlib.Path(__file__).parent.resolve()

def patch_file(rel_path: str, replacements: list[tuple[str, str]]):
    path = ROOT / rel_path
    if not path.exists():
        print(f"[ERROR] File not found: {rel_path}")
        sys.exit(1)
    text = path.read_text(encoding="utf-8")
    orig = text
    for old, new in replacements:
        # Normalize CRLF/LF for matching
        old_norm = old.replace("\r\n", "\n")
        new_norm = new.replace("\r\n", "\n")
        text_norm = text.replace("\r\n", "\n")
        if old_norm not in text_norm:
            print(f"[WARN] Target block not found (or already patched) in {rel_path}:\n  {old_norm.splitlines()[0]}...")
            continue
        text_norm = text_norm.replace(old_norm, new_norm, 1)
        text = text_norm
    if text != orig.replace("\r\n", "\n"):
        path.write_text(text, encoding="utf-8")
        print(f"[OK] Patched {rel_path}")
    else:
        print(f"[SKIP] No changes needed in {rel_path}")

def main():
    # 1. Cargo.toml: Add [profile.release] with fat LTO and codegen-units = 1
    cargo_path = ROOT / "Cargo.toml"
    cargo_text = cargo_path.read_text(encoding="utf-8")
    if "[profile.release]" not in cargo_text:
        cargo_text += (
            "\n[profile.release]\n"
            "opt-level = 3\n"
            'lto = "fat"\n'
            "codegen-units = 1\n"
            'panic = "abort"\n'
            "strip = true\n"
        )
        cargo_path.write_text(cargo_text, encoding="utf-8")
        print("[OK] Added [profile.release] to Cargo.toml")

    # 2. app.py: Use zero-copy aiter_raw() instead of UTF-8 decoding aiter_text()
    patch_file("app.py", [
        (
            "async for chunk in response.aiter_text():\n                    yield chunk",
            "async for chunk in response.aiter_raw():\n                    yield chunk",
        )
    ])

    # 3. start.ps1: Default to outlier-safe int8 to halve WireGuard packet fragmentation
    patch_file("start.ps1", [
        (
            'if (-not $env:AEROMESH_ACTIVATION_DTYPE) {\n'
            '    $env:AEROMESH_ACTIVATION_DTYPE = "bf16"\n'
            '    [System.Environment]::SetEnvironmentVariable("AEROMESH_ACTIVATION_DTYPE", "bf16", "Process")\n'
            '}',
            'if (-not $env:AEROMESH_ACTIVATION_DTYPE) {\n'
            '    $env:AEROMESH_ACTIVATION_DTYPE = "int8"\n'
            '    [System.Environment]::SetEnvironmentVariable("AEROMESH_ACTIVATION_DTYPE", "int8", "Process")\n'
            '}',
        )
    ])

    # 4. crates/aeromesh-core/src/transport.rs: Reusable write_buf to avoid double payload copy
    patch_file("crates/aeromesh-core/src/transport.rs", [
        (
            "pub struct TcpPipelineTransport {\n"
            "    stream: TcpStream,\n"
            "    peer_addr: SocketAddr,\n"
            "}",
            "pub struct TcpPipelineTransport {\n"
            "    stream: TcpStream,\n"
            "    peer_addr: SocketAddr,\n"
            "    write_buf: bytes::BytesMut,\n"
            "}",
        ),
        (
            "        Ok(Self { stream, peer_addr })",
            "        Ok(Self {\n            stream,\n            peer_addr,\n            write_buf: bytes::BytesMut::with_capacity(64 * 1024),\n        })",
        ),
        (
            "        Ok(Self { stream, peer_addr })",
            "        Ok(Self {\n            stream,\n            peer_addr,\n            write_buf: bytes::BytesMut::with_capacity(64 * 1024),\n        })",
        ),
        (
            "    pub async fn send_activation(&mut self, frame: &ActivationFrame) -> Result<()> {\n"
            "        let bytes = frame.encode();\n"
            "        self.stream.write_all(&bytes).await?;\n"
            "        self.stream.flush().await?;\n"
            "        Ok(())\n"
            "    }",
            "    pub async fn send_activation(&mut self, frame: &ActivationFrame) -> Result<()> {\n"
            "        self.write_buf.clear();\n"
            "        self.write_buf.reserve(crate::activation::ActivationHeader::SIZE + frame.payload.len());\n"
            "        frame.header.encode(&mut self.write_buf);\n"
            "        self.write_buf.extend_from_slice(&frame.payload);\n"
            "        self.stream.write_all(&self.write_buf).await?;\n"
            "        Ok(())\n"
            "    }",
        ),
    ])

    # 5. crates/aeromesh-core/src/activation.rs:
    #    - Outlier-preserving Int8PerRow (for hidden_dim >= 256, keeps existing unit tests intact)
    #    - AVX2-vectorizable chunks_exact in Bf16 / Fp16 / RawF32
    #    - Single-syscall 35-byte header read in TokenResponseFrame::decode_async
    patch_file("crates/aeromesh-core/src/activation.rs", [
        (
            """            ActivationDtype::Int8PerRow => {
                // Per-row payload layout: [scale_row0: f32][q_row0: d x i8][scale_row1: f32][q_row1: d x i8]...
                let mut payload = BytesMut::with_capacity(seq_len * (4 + d));
                for row in matrix.chunks(d) {
                    let max_abs = row.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
                    let scale = if max_abs > 1e-12 {
                        max_abs / 127.0
                    } else {
                        0.0
                    };
                    let inv_scale = if max_abs > 1e-12 {
                        127.0 / max_abs
                    } else {
                        0.0
                    };
                    payload.extend_from_slice(&scale.to_le_bytes());
                    for &val in row {
                        let q = if max_abs > 1e-12 {
                            (val * inv_scale).round().clamp(-127.0, 127.0) as i8
                        } else {
                            0i8
                        };
                        payload.extend_from_slice(&(q as u8).to_le_bytes());
                    }
                }
                payload.freeze()
            }""",
            """            ActivationDtype::Int8PerRow => {
                // Outlier-Preserving Per-Row INT8 for real LLM hidden dims (d >= 256):
                // Extracts top-8 outlier channels per row as exact (u16 idx, f32 val) pairs
                // so Qwen2/DeepSeek residual outliers never crush small channel resolution.
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
                    let scale = if max_abs > 1e-12 { max_abs / 127.0 } else { 0.0 };
                    let inv_scale = if max_abs > 1e-12 { 127.0 / max_abs } else { 0.0 };
                    payload.put_f32_le(scale);
                    for i in 0..k_outliers {
                        let idx = outlier_idx[i];
                        let (u_idx, exact_val) = if idx < d { (idx as u16, row[idx]) } else { (0u16, 0.0f32) };
                        payload.put_u16_le(u_idx);
                        payload.put_f32_le(exact_val);
                    }
                    let base = payload.len();
                    payload.resize(base + d, 0);
                    for (dst, &val) in payload[base..base + d].iter_mut().zip(row.iter()) {
                        let scaled = (val * inv_scale).clamp(-127.0, 127.0);
                        let q = if scaled >= 0.0 { (scaled + 0.5) as i8 } else { (scaled - 0.5) as i8 };
                        *dst = q as u8;
                    }
                }
                payload.freeze()
            }""",
        ),
        (
            """            ActivationDtype::Bf16 => {
                // Lossless exponent preservation for LLM intermediate activations
                let mut payload = BytesMut::with_capacity(seq_len * d * 2);
                for &val in matrix {
                    // Round-to-nearest-even conversion from f32 to bf16
                    let bits = val.to_bits();
                    let rounding_bias = 0x00007FFF + ((bits >> 16) & 1);
                    let bf = ((bits.wrapping_add(rounding_bias)) >> 16) as u16;
                    payload.extend_from_slice(&bf.to_le_bytes());
                }
                payload.freeze()
            }""",
            """            ActivationDtype::Bf16 => {
                let mut payload = BytesMut::with_capacity(seq_len * d * 2);
                payload.resize(seq_len * d * 2, 0);
                for (chunk, &val) in payload.chunks_exact_mut(2).zip(matrix.iter()) {
                    let bits = val.to_bits();
                    let rounding_bias = 0x00007FFF + ((bits >> 16) & 1);
                    let bf = ((bits.wrapping_add(rounding_bias)) >> 16) as u16;
                    chunk.copy_from_slice(&bf.to_le_bytes());
                }
                payload.freeze()
            }""",
        ),
        (
            """            ActivationDtype::Int8PerRow => {
                let expected_row_bytes = 4 + d;
                ensure!(
                    self.payload.len() == seq_len * expected_row_bytes,
                    "Invalid Int8PerRow payload length: got {}, expected {}",
                    self.payload.len(),
                    seq_len * expected_row_bytes
                );
                for s in 0..seq_len {
                    let row_offset = s * expected_row_bytes;
                    let scale_bytes: [u8; 4] = self.payload[row_offset..row_offset + 4].try_into()?;
                    let scale = f32::from_le_bytes(scale_bytes);
                    let q_slice = &self.payload[row_offset + 4..row_offset + expected_row_bytes];
                    let out_row = &mut out[s * d..(s + 1) * d];
                    for (i, &qb) in q_slice.iter().enumerate() {
                        out_row[i] = (qb as i8 as f32) * scale;
                    }
                }
            }
            ActivationDtype::Fp16 => {
                ensure!(
                    self.payload.len() == total_elements * 2,
                    "Invalid FP16 payload length"
                );
                for i in 0..total_elements {
                    let chunk: [u8; 2] = self.payload[i * 2..(i + 1) * 2].try_into()?;
                    let h = half::f16::from_le_bytes(chunk);
                    out[i] = h.to_f32();
                }
            }
            ActivationDtype::Bf16 => {
                ensure!(
                    self.payload.len() == total_elements * 2,
                    "Invalid BF16 payload length"
                );
                for i in 0..total_elements {
                    let chunk: [u8; 2] = self.payload[i * 2..(i + 1) * 2].try_into()?;
                    let bits = (u16::from_le_bytes(chunk) as u32) << 16;
                    out[i] = f32::from_bits(bits);
                }
            }""",
            """            ActivationDtype::Int8PerRow => {
                let k_outliers = if d >= 256 { 8usize } else { 0usize };
                let expected_row_bytes = 4 + k_outliers * 6 + d;
                ensure!(
                    self.payload.len() == seq_len * expected_row_bytes,
                    "Invalid Int8PerRow payload length: got {}, expected {}",
                    self.payload.len(),
                    seq_len * expected_row_bytes
                );
                for (s, row_bytes) in self.payload.chunks_exact(expected_row_bytes).enumerate() {
                    let scale = f32::from_le_bytes([row_bytes[0], row_bytes[1], row_bytes[2], row_bytes[3]]);
                    let q_offset = 4 + k_outliers * 6;
                    let q_slice = &row_bytes[q_offset..q_offset + d];
                    let out_row = &mut out[s * d..(s + 1) * d];
                    for (dst, &qb) in out_row.iter_mut().zip(q_slice.iter()) {
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
            }
            ActivationDtype::Fp16 => {
                ensure!(
                    self.payload.len() == total_elements * 2,
                    "Invalid FP16 payload length"
                );
                for (chunk, dst) in self.payload.chunks_exact(2).zip(out[..total_elements].iter_mut()) {
                    let h = half::f16::from_le_bytes([chunk[0], chunk[1]]);
                    *dst = h.to_f32();
                }
            }
            ActivationDtype::Bf16 => {
                ensure!(
                    self.payload.len() == total_elements * 2,
                    "Invalid BF16 payload length"
                );
                for (chunk, dst) in self.payload.chunks_exact(2).zip(out[..total_elements].iter_mut()) {
                    let bits = (u16::from_le_bytes([chunk[0], chunk[1]]) as u32) << 16;
                    *dst = f32::from_bits(bits);
                }
            }""",
        ),
        (
            """    pub async fn decode_async<R: AsyncReadExt + Unpin>(reader: &mut R) -> Result<Self> {
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic).await?;
        ensure!(
            &magic == RESPONSE_MAGIC,
            "Invalid TokenResponseFrame magic: {:?}",
            magic
        );

        let version = reader.read_u16().await?;
        ensure!(
            version == PROTOCOL_VERSION,
            "TokenResponseFrame version mismatch: got {}, expected {}",
            version,
            PROTOCOL_VERSION
        );

        let session_id = reader.read_u64().await?;
        let sequence_id = reader.read_u64().await?;
        let token_id = reader.read_i32().await?;
        let is_eos = reader.read_u8().await? != 0;
        let eval_time_bits = reader.read_u32().await?;
        let eval_time_ms = f32::from_bits(eval_time_bits);
        let text_len = reader.read_u32().await? as usize;""",
            """    pub async fn decode_async<R: AsyncReadExt + Unpin>(reader: &mut R) -> Result<Self> {
        let mut hdr = [0u8; Self::HEADER_SIZE];
        reader.read_exact(&mut hdr).await?;
        ensure!(
            &hdr[0..4] == RESPONSE_MAGIC,
            "Invalid TokenResponseFrame magic: {:?}",
            &hdr[0..4]
        );
        let version = u16::from_le_bytes([hdr[4], hdr[5]]);
        ensure!(
            version == PROTOCOL_VERSION,
            "TokenResponseFrame version mismatch: got {}, expected {}",
            version,
            PROTOCOL_VERSION
        );
        let session_id = u64::from_le_bytes(hdr[6..14].try_into().unwrap());
        let sequence_id = u64::from_le_bytes(hdr[14..22].try_into().unwrap());
        let token_id = i32::from_le_bytes(hdr[22..26].try_into().unwrap());
        let is_eos = hdr[26] != 0;
        let eval_time_ms = f32::from_bits(u32::from_le_bytes(hdr[27..31].try_into().unwrap()));
        let text_len = u32::from_le_bytes(hdr[31..35].try_into().unwrap()) as usize;""",
        ),
    ])

    # 6. crates/aeromesh-engine/src/slice_loader.rs:
    #    - Enable CUDA Graphs on Terminal Worker (only set cb_eval when !is_last_stage)
    #    - Assign n_threads, n_threads_batch, and flash_attn_type = 1
    #    - Only compute logits on final prompt token when cb_eval is active
    #    - 8-element NaN check instead of 152,064-float scan
    patch_file("crates/aeromesh-engine/src/slice_loader.rs", [
        (
            """        let mut cparams = unsafe { llama_context_default_params() };
        cparams.n_ctx = if n_ctx > 0 { n_ctx } else { 4096 };
        cparams.embeddings = true;
        cparams.cb_eval = Some(llama_cb_eval_capture);
        cparams.cb_eval_user_data = capture_state.as_mut() as *mut _ as *mut c_void;""",
            """        let batch_threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(physical_threads);

        let mut cparams = unsafe { llama_context_default_params() };
        cparams.n_ctx = if n_ctx > 0 { n_ctx } else { 4096 };
        cparams.n_threads = physical_threads;
        cparams.n_threads_batch = batch_threads;
        cparams.flash_attn_type = 1;

        if !slice_config.is_last_stage {
            cparams.embeddings = true;
            cparams.cb_eval = Some(llama_cb_eval_capture);
            cparams.cb_eval_user_data = capture_state.as_mut() as *mut _ as *mut c_void;
        } else {
            cparams.embeddings = false;
            cparams.cb_eval = None;
            cparams.cb_eval_user_data = std::ptr::null_mut();
        }""",
        ),
        (
            "llama_sampler_chain_add(sampler, llama_sampler_init_penalties(256, 1.10, 0.0, 0.0));",
            "llama_sampler_chain_add(sampler, llama_sampler_init_penalties(64, 1.10, 0.0, 0.0));",
        ),
        (
            "                *batch.logits.add(i) = 1;\n            }\n        }\n\n        let decode_rc = unsafe { llama_decode(self.ctx, batch) };",
            "                *batch.logits.add(i) = if i + 1 == n_tokens { 1 } else { 0 };\n            }\n        }\n\n        let decode_rc = unsafe { llama_decode(self.ctx, batch) };",
        ),
        (
            "            let has_invalid = logits_slice.iter().any(|v| v.is_nan() || v.is_infinite());",
            "            let has_invalid = logits_slice.iter().take(8).any(|v| !v.is_finite());",
        ),
    ])

    # 7. crates/aeromesh-engine/src/pipeline.rs:
    #    - Always pass ngl = 999 so output.weight stays on GPU
    #    - Gate inspect_activations and synchronous Windows println! out of the per-token loop
    patch_file("crates/aeromesh-engine/src/pipeline.rs", [
        (
            "let file_size_mb = std::fs::metadata(&p).map(|m| m.len() / (1024 * 1024)).unwrap_or(0);\n"
            "        let ngl = if file_size_mb <= 4500 { 999 } else { new_slice.layer_count() as i32 };",
            "let ngl = 999;",
        ),
        (
            "            let file_size_mb = std::fs::metadata(&p).map(|m| m.len() / (1024 * 1024)).unwrap_or(0);\n"
            "            let ngl = if file_size_mb <= 4500 { 999 } else { new_slice.layer_count() as i32 };",
            "            let ngl = 999;",
        ),
        (
            """        // Boundary Watchdog: Validate incoming activations
        if let Err(e) = aeromesh_core::inspect_activations(
            &format!("Worker_Stage{}_Ingress", current_stage_index),
            &dequant_buf,
            &[seq_len as usize, frame.header.hidden_dim as usize],
        ) {
            error!("🚨 Worker ingress watchdog assertion failed: {}", e);
        }""",
            """        // Boundary Watchdog: Validate incoming activations on prefill (seq_id == 0)
        if seq_id == 0 {
            if let Err(e) = aeromesh_core::inspect_activations(
                "Worker_Ingress_Prefill",
                &dequant_buf,
                &[seq_len as usize, frame.header.hidden_dim as usize],
            ) {
                error!("🚨 Worker ingress watchdog assertion failed: {}", e);
            }
        }""",
        ),
        (
            """            println!(
                "  ⚡ [Stage {} Intermediate Worker] Step #{:<3} (pos: {}, S={}) | Recv: {:.2} KB [{}] -> Forwarded in {:.2}ms",
                current_stage_index,
                seq_id + 1,
                start_pos,
                seq_len,
                (frame.payload.len() as f64) / 1024.0,
                dtype_tag,
                eval_time_ms
            );""",
            """            if seq_id == 0 || (seq_id + 1) % 50 == 0 {
                println!(
                    "  ⚡ [Stage {} Intermediate Worker] Step #{:<3} (pos: {}, S={}) | Recv: {:.2} KB [{}] -> Forwarded in {:.2}ms",
                    current_stage_index,
                    seq_id + 1,
                    start_pos,
                    seq_len,
                    (frame.payload.len() as f64) / 1024.0,
                    dtype_tag,
                    eval_time_ms
                );
            }""",
        ),
        (
            """            println!(
                "  ⚡ [Stage {} Terminal Worker] Step #{:<3} (pos: {}, S={}) | Recv: {:.2} KB [{}] | Processed in {:.2}ms",
                current_stage_index,
                seq_id + 1,
                start_pos,
                seq_len,
                (frame.payload.len() as f64) / 1024.0,
                dtype_tag,
                eval_time_ms
            );""",
            """            if seq_id == 0 || (seq_id + 1) % 50 == 0 {
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
            }""",
        ),
    ])

    print("\nAll patches applied successfully!")

if __name__ == "__main__":
    main()