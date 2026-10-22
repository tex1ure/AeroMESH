# AeroMESH Changelog

All notable changes to the AeroMESH project are documented in this file.

## [v0.2.0-ArgonFixes] - 2026-10-22

### Performance & Networking
- **Windows TCP Delayed-ACK Bypass (`transport.rs`, `pipeline.rs`)**: Configured `SIO_TCP_SET_ACK_FREQUENCY = 1`, `SIO_LOOPBACK_FAST_PATH = 1`, `TCP_NODELAY`, and 512 KB `SO_SNDBUF`/`SO_RCVBUF` socket buffers via Win32 `WSAIoctl` on all Coordinator and Worker sockets. Eliminates the 200ms delayed ACK stall over Tailscale's 1280-byte MTU, boosting throughput from 2–5 TPS to 15–30 TPS.
- **Zero-Allocation Frame Write Buffer (`transport.rs`)**: Added reusable `BytesMut` write buffer in `TcpPipelineTransport::send_activation` to coalesce the 44-byte `ActivationHeader` and payload into a single syscall.
- **SHM Spin-Yield Polling (`transport.rs`)**: Added 64-iteration cooperative `yield_now()` spin loop before microsecond sleep in `ShmPipelineTransport`, reducing local shared-memory handoff latency.

### Compression & Numerical Stability
- **Outlier-Preserving Per-Row INT8 Codec (`activation.rs`, `server.rs`, `pipeline.rs`)**: Upgraded `Int8PerRow` for hidden dimensions $d \ge 256$ to extract the Top-8 outlier channels per row as exact `(u16 idx, f32 val)` pairs before quantizing normal channels to INT8. Prevents quantization collapse and repetition loops on Qwen, DeepSeek, and Llama models.
- **Big-Endian Token Frame Decoding Fix (`activation.rs`)**: Fixed `TokenResponseFrame::decode_async` to read header fields using `from_be_bytes`, matching the Big-Endian wire encoder.

### Storage & Engine Architecture
- **Ephemeral VRAM Stage Loading (`gguf_slicer.rs`, `slice_loader.rs`)**: Introduced `InPlaceStageGuard` with `use_mmap = false` to load sliced stage tensors directly into GPU VRAM and immediately delete the temporary `.aeromesh_ephemeral_*` file on drop, achieving **0.0 MB persistent disk overhead**.
- **Non-Terminal LM Head Stripping (`gguf_slicer.rs`)**: Omitted `output.weight` and `output.bias` from non-terminal stage slices, saving hundreds of megabytes of VRAM on Stage 1 nodes.
- **Worker Hot-Swap False-Positive Fix (`pipeline.rs`)**: Fixed handshake layer count check (`inst.slice_config.total_layers != req.total_layers`) so workers no longer reload weights on every incoming connection.
- **Standalone & Multi-Stage Pipeline Compatibility (`slice_loader.rs`, `pipeline.rs`, `app.js`)**: Preserved `cb_eval` and `embeddings` across standalone 1-node and multi-node configurations, fixed N-worker downstream layer range calculation, and updated the Cluster Topology HUD for 3+ node chains.

## [v0.1.0] - 2026-10-10

### Security & Authentication
- **Gateway Loopback & Key Guard (SEC-01 / Tier 0)**: Secured `app.py` auth verification with fail-closed loopback detection and authorization checks.
- **Upstream Credential Injection**: Stripped incoming client `Authorization` header prior to injecting gateway credentials.
- **Bounded HTTP Payloads**: Enforced request payload ceilings on gateway routes.
- **XSS & Content Sanitization**: Pinned CDN assets with Subresource Integrity (SRI) hashes and configured DOMPurify markdown sanitization.

### Architecture & Engine
- **Runtime FFI Dynamic Resolution**: Switched native `llama.cpp` symbols to runtime dynamic loading (`GetProcAddress`).
- **Protocol v4 Wire Format**: Formalized 44-byte binary `ActivationHeader` supporting multi-node topologies, token positions, and KV flags.
- **Identity-RMSNorm Injection**: Extended GGUF slicer identity normalization support to F32, F16, BF16, and Q8_0.
- **Real-Time Generation Abort**: Added atomic cancellation flag for `/api/chat/abort`.