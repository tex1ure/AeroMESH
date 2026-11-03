# AeroMESH Changelog

All notable changes, architectural updates, and security hardening milestones for the AeroMESH project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [v0.2.0-ArgonFixes] - 2026-10-22

### Added
- **Sub-Millisecond Windows TCP Socket Tuning (`transport.rs`, `pipeline.rs`)**:
  - Implemented Win32 `WSAIoctl` controls on all coordinator and worker network sockets:
    - `SIO_TCP_SET_ACK_FREQUENCY = 1` (`0x98000017`): Eliminates the 200 ms Windows TCP Delayed-ACK stall over Tailscale WireGuard's 1280-byte MTU, lifting generation throughput from 2–5 tok/s to 15–25 tok/s.
    - `SIO_LOOPBACK_FAST_PATH = 1` (`0x98000010`): Enables fast-path routing on loopback connections.
    - `TCP_NODELAY = true`: Disables Nagle's buffering algorithm.
    - 512 KB `SO_SNDBUF` and `SO_RCVBUF` socket buffers.
- **Outlier-Preserving Per-Row INT8 Codec (`activation.rs`, `server.rs`, `pipeline.rs`)**:
  - Upgraded `Int8PerRow` for hidden dimensions $d \ge 256$ to isolate and extract the Top-8 outlier channels per row as exact `(u16 index, f32 value)` pairs (48 bytes) before quantizing remaining channels to INT8.
  - Eliminates numerical scale collapse and repetition loops on DeepSeek-R1 and Qwen-2.5 models while achieving a 74.8% reduction in wire bandwidth (3.05 KB/tok for 3B, 5.05 KB/tok for 14B).
- **Ephemeral VRAM Stage Loading (`InPlaceStageGuard` in `gguf_slicer.rs`, `slice_loader.rs`)**:
  - Direct upload of sliced stage tensors into GPU VRAM with `use_mmap = false`, followed by immediate unlinking and deletion of temporary `.aeromesh_ephemeral_*` slice files upon initialization.
  - Reduces persistent disk overhead across the entire cluster to **0.0 MB**.
- **Coalesced Frame Write Buffer (`transport.rs`)**:
  - Added reusable `BytesMut` buffer in `TcpPipelineTransport::send_activation` to merge the 44-byte `ActivationHeader` and activation payload into a single network system call.
- **Lock-Free SHM Spin-Yielding (`shm.rs`, `transport.rs`)**:
  - Integrated a 64-iteration cooperative `tokio::task::yield_now()` spin loop before microsecond sleep in shared-memory transport, reducing local handoff latency to < 100 microseconds.
- **Procedural DiT / Stable Diffusion Pipeline (`app.py`, `/v1/images/generations`)**:
  - Added support for procedural image generation conditioned via distributed mesh textual prompt expansion.
- **Native Desktop Studio Application (`aeromesh_desktop.rs`, `AeroMesh-Desktop.exe`)**:
  - Compiled Windows frameless native desktop studio embedding coordinator, worker, and Web UI.

### Fixed
- **Big-Endian Token Frame Decoding (`activation.rs`)**:
  - Corrected `TokenResponseFrame::decode_async` to unpack header fields using `from_be_bytes`, matching the Big-Endian wire encoder specification.
- **Non-Terminal LM Head Stripping (`gguf_slicer.rs`)**:
  - Stripped `output.weight` and `output.bias` tensors from non-terminal Stage 1 slices, saving hundreds of megabytes of VRAM and avoiding unused tensor initialization overhead.
- **Worker Handshake False-Positive Reloading (`pipeline.rs`)**:
  - Fixed layer count check (`inst.slice_config.total_layers != req.total_layers`) in the worker service loop to prevent unnecessary weight reloads on incoming peer reconnections.
- **Multi-Stage Chain Compatibility (`slice_loader.rs`, `pipeline.rs`, `app.js`)**:
  - Fixed downstream layer range computation for clusters with 3+ nodes and updated the Liquid Glass Cluster HUD to visualize multi-stage topologies.

---

## [v0.1.5] - 2026-10-15

### Security & Hardening
- **FastAPI Gateway Hardening (FIX-16)**:
  - Added persistent connection-pooled `httpx.AsyncClient` with unbuffered SSE token streaming (`X-Accel-Buffering: no`).
  - Added RFC 7230 hop-by-hop header stripping and upstream credential injection.
  - Implemented graceful offline degradation: dynamic local `models/*.gguf` scanning when the coordinator backend is offline.
  - Added `/health` (gateway liveness) and `/ready` (end-to-end backend readiness) probes.
- **Win32 Job Object Lifecycle Containment (`job_object.rs`)**:
  - Bound native child processes to a Win32 Job Object configured with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`, ensuring automatic kernel reclamation of GPU VRAM on exit.
- **OpenAI-Compatible Error Envelopes (FIX-06, FIX-07)**:
  - Standardized JSON error response schema across all 400, 401, 404, 422, 429, 500, and 503 status codes.
- **Context-Window Guard (FIX-24)**:
  - Early rejection of requests where `prompt_tokens + max_tokens > n_ctx` before allocating GPU computation buffers.
- **Real-Time Generation Abort (`/api/chat/abort`, FIX-19)**:
  - Added atomic cancellation token halting the autoregressive generation loop upon client abort.

---

## [v0.1.0] - 2026-10-01

### Added
- **Core Zero-Weight Distributed Pipeline**:
  - Monolithic base GGUF pre-positioning with deterministic layer slicing along transformer block boundaries.
  - Identity RMSNorm injection ($\mathbf{w} = [1.0, 1.0, \dots, 1.0]$) resolving the Double Normalization Problem.
- **Binary Wire Protocol v4 (`activation.rs`)**:
  - Formalized 44-byte `ActivationHeader` and 35-byte `TokenResponseFrame` with Big-Endian byte encoding.
  - Cryptographic mutual handshake with SHA-256 worker token authentication.
- **Claymorphic Liquid Glass Web UI (`static/`)**:
  - Dependency-free dark theme Single Page Application with live KaTeX math rendering, syntax highlighting, and cluster telemetry HUD.
- **Dynamic C++ FFI Loader (`llama_ffi.rs`)**:
  - Runtime dynamic symbol resolution (`GetProcAddress`) for CUDA-compiled `llama.cpp` runtime libraries.
