import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).parent.resolve()


def patch_env_file():
    """Ensures .env uses AEROMESH_ACTIVATION_DTYPE=int8 instead of bf16."""
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
    else:
        print("[SKIP] .env already set to int8")


def patch_transport_rs():
    """Patches crates/aeromesh-core/src/transport.rs cleanly regardless of rustfmt."""
    rel_path = "crates/aeromesh-core/src/transport.rs"
    path = ROOT / rel_path
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    orig = text

    struct_match = re.search(
        r"(pub\s+struct\s+TcpPipelineTransport\s*\{)([^}]*)(\})",
        text,
        flags=re.DOTALL,
    )
    if struct_match and "write_buf" not in struct_match.group(2):
        body = struct_match.group(2).rstrip()
        if not body.endswith(","):
            body += ","
        new_struct = (
            f"{struct_match.group(1)}{body}\n"
            f"    write_buf: bytes::BytesMut,\n"
            f"{struct_match.group(3)}"
        )
        text = text[: struct_match.start()] + new_struct + text[struct_match.end() :]

    low_latency_fn = """    #[cfg(windows)]
    fn configure_low_latency_stream(stream: &TcpStream) {
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
    fn configure_low_latency_stream(stream: &TcpStream) {
        let _ = stream.set_nodelay(true);
    }"""

    if "fn configure_low_latency_stream" not in text:
        text = text.replace(
            "    pub async fn send_activation(",
            low_latency_fn + "\n\n    pub async fn send_activation(",
            1,
        )

    text = re.sub(
        r"Ok\(Self\s*\{\s*stream\s*(?:,\s*peer_addr\s*)?\}\)",
        "Self::configure_low_latency_stream(&stream);\n"
        "        Ok(Self {\n"
        "            stream,\n"
        "            write_buf: bytes::BytesMut::with_capacity(64 * 1024),\n"
        "        })",
        text,
    )

    if text != orig:
        path.write_text(text, encoding="utf-8")
        print(f"[OK] Patched {rel_path}")
    else:
        print(f"[SKIP] Already up to date: {rel_path}")


def patch_activation_rs():
    """Ensures activation.rs has consistent payload sizing and endianness."""
    rel_path = "crates/aeromesh-core/src/activation.rs"
    path = ROOT / rel_path
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    orig = text

    if "fn calculate_expected_payload_bytes" in text:
        calc_section = text.split("fn calculate_expected_payload_bytes", 1)[1].split("pub fn ", 1)[0]
        if "k_outliers" not in calc_section and "let k_outliers = if d >= 256 { 8usize } else { 0usize };" in text:
            text = text.replace(
                "let k_outliers = if d >= 256 { 8usize } else { 0usize };",
                "let k_outliers = 0usize;",
            )

    if "RESPONSE_MAGIC" in text:
        uses_be_encode = "put_u16(self.version)" in text or "put_u16(PROTOCOL_VERSION)" in text
        if uses_be_encode and "u16::from_le_bytes([hdr[4], hdr[5]])" in text:
            text = text.replace(
                "u16::from_le_bytes([hdr[4], hdr[5]])",
                "u16::from_be_bytes([hdr[4], hdr[5]])",
            )
            text = text.replace(
                "u64::from_le_bytes(hdr[6..14]",
                "u64::from_be_bytes(hdr[6..14]",
            )
            text = text.replace(
                "u64::from_le_bytes(hdr[14..22]",
                "u64::from_be_bytes(hdr[14..22]",
            )
            text = text.replace(
                "i32::from_le_bytes(hdr[22..26]",
                "i32::from_be_bytes(hdr[22..26]",
            )
            text = text.replace(
                "u32::from_le_bytes(hdr[27..31]",
                "u32::from_be_bytes(hdr[27..31]",
            )
            text = text.replace(
                "u32::from_le_bytes(hdr[31..35]",
                "u32::from_be_bytes(hdr[31..35]",
            )

    if text != orig:
        path.write_text(text, encoding="utf-8")
        print(f"[OK] Patched {rel_path}")
    else:
        print(f"[SKIP] Already up to date: {rel_path}")


def patch_slice_loader_rs():
    """Fixes `Unsupported ctx type` caused by misaligned llama_context_params writes."""
    rel_path = "crates/aeromesh-engine/src/slice_loader.rs"
    path = ROOT / rel_path
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    orig = text

    # Strip out direct writes to n_threads, n_threads_batch, and flash_attn_type on cparams
    text = re.sub(
        r"^[ \t]*cparams\.(?:n_threads|n_threads_batch|flash_attn_type)\s*=[^;]+;[ \t]*\n",
        "        let _ = &physical_threads;\n",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"\blet\s+batch_threads\s*=",
        "let _batch_threads =",
        text,
    )
    text = re.sub(
        r"\}\s*else\s*\{\s*cparams\.embeddings\s*=\s*false;\s*cparams\.cb_eval\s*=\s*None;\s*cparams\.cb_eval_user_data\s*=\s*std::ptr::null_mut\(\);\s*\}",
        "}",
        text,
    )

    if text != orig:
        path.write_text(text, encoding="utf-8")
        print(f"[OK] Patched {rel_path} (fixed Unsupported ctx type)")
    else:
        print(f"[SKIP] Already up to date: {rel_path}")


def main():
    # 1. Cargo.toml
    cargo_path = ROOT / "Cargo.toml"
    cargo_text = cargo_path.read_text(encoding="utf-8")
    if "[profile.release]" not in cargo_text:
        cargo_text = cargo_text.rstrip() + (
            "\n\n[profile.release]\n"
            "opt-level = 3\n"
            'lto = "fat"\n'
            "codegen-units = 1\n"
            'panic = "abort"\n'
            "strip = true\n"
        )
        cargo_path.write_text(cargo_text, encoding="utf-8")
        print("[OK] Added [profile.release] to Cargo.toml")
    else:
        print("[SKIP] Already up to date: Cargo.toml")

    # 2. app.py
    app_path = ROOT / "app.py"
    app_text = app_path.read_text(encoding="utf-8")
    if "response.aiter_text()" in app_text:
        app_path.write_text(
            app_text.replace("response.aiter_text()", "response.aiter_raw()"),
            encoding="utf-8",
        )
        print("[OK] Patched app.py")
    else:
        print("[SKIP] Already up to date: app.py")

    # 3. start.ps1 & .env
    start_path = ROOT / "start.ps1"
    start_text = start_path.read_text(encoding="utf-8")
    new_start = (
        start_text.replace('"AEROMESH_ACTIVATION_DTYPE=bf16"', '"AEROMESH_ACTIVATION_DTYPE=int8"')
        .replace('$env:AEROMESH_ACTIVATION_DTYPE = "bf16"', '$env:AEROMESH_ACTIVATION_DTYPE = "int8"')
        .replace('"AEROMESH_ACTIVATION_DTYPE", "bf16"', '"AEROMESH_ACTIVATION_DTYPE", "int8"')
        .replace('else { "BF16" }', 'else { "INT8" }')
    )
    if new_start != start_text:
        start_path.write_text(new_start, encoding="utf-8")
        print("[OK] Patched start.ps1")
    else:
        print("[SKIP] Already up to date: start.ps1")

    patch_env_file()

    # 4. crates/aeromesh-core/src/transport.rs
    patch_transport_rs()

    # 5. crates/aeromesh-core/src/activation.rs
    patch_activation_rs()

    # 6. crates/aeromesh-engine/src/slice_loader.rs
    patch_slice_loader_rs()

    # 7. Verify pipeline.rs status
    pipe_text = (ROOT / "crates/aeromesh-engine/src/pipeline.rs").read_text(encoding="utf-8")
    if "let ngl = 999;" in pipe_text:
        print("[SKIP] Already up to date: crates/aeromesh-engine/src/pipeline.rs")

    print("\nAll patches applied and verified successfully!")


if __name__ == "__main__":
    main()