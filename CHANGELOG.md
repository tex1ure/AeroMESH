# Changelog

Notable changes to AeroMESH are documented here following [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [v0.3.0] - 2026-11-21

### Added
- **Windows Setup Wizard (`installer/AeroMesh.iss`, `installer/build-installer.ps1`)**:
  - Full Inno Setup x64 installer (`AeroMesh-Setup-x64.exe`) with VS Code-style component checkboxes: NVIDIA CUDA / CPU `llama.cpp` runtime DLLs, Tailscale VPN check, Desktop / Start Menu shortcuts, PATH registration, and Windows Firewall rules for `:50052`, `:50053`, and `:50054`.
  - Bundled standalone Python embeddable runtime so end users do not need to install Python, Rust, or Visual Studio.
  - Portable `setup.ps1` improvements including `-CloneLlamaCpp` support and dynamic FFI loading without requiring MinGW `dlltool.exe`.
- **Model Hub Tab (`static/js/models_hub.js`, `app.py` `/api/models/*`)**:
  - LM Studio / Ollama-style model manager built into the Studio sidebar.
  - Curated multi-GPU catalog + live Hugging Face Hub GGUF repository search and quantization file picker.
  - Resumable background downloads with live progress bar, MB/s speed, ETA, and cancellation.
  - Custom watched model folders (link existing LM Studio or Ollama directories with zero duplication), native Windows file/folder picker (`/api/models/browse`), and drag-and-drop streaming `.gguf`/`.safetensors` uploads.
- **One-Click Cluster Pairing & Secret Sync (`static/js/cluster_pair.js`, `app.py` `/api/cluster/pair/*`)**:
  - Ephemeral encrypted pairing listener on TCP `:50053` protected by a 6-digit one-time PIN (PBKDF2-HMAC-SHA256 + Encrypt-then-MAC HMAC-SHA256, 5-minute TTL, 5-attempt lockout).
  - Copyable one-click invite tokens (`AERO-<ip>:50053-<pin>`) for both Tailscale and Wi-Fi LAN, plus automatic nearby host discovery.
  - Saved peer registry showing live GPU info, key fingerprint match (`KEY-XXXX-XXXX`), matching GGUF models, one-click **Use as Worker**, and one-click **Rotate Cluster Key**.
- **Automatic Multi-Node Model Check & Peer Model Sync (`peer_model_sync.py`, `app.py` `/api/cluster/models/*`)**:
  - Background TCP `:50054` sync daemon (`aeromesh-sync-v1`) using mutual challenge-response HMAC-SHA256 derived from `AEROMESH_WORKER_SECRET` with fast deterministic GGUF header + content fingerprinting (`size` + head/mid/tail SHA-256).
  - Cluster tab **Multi-Node Model Check & Peer Sync** matrix detecting `Verified Match`, `Missing on Peer`, `GGUF Hash Mismatch`, and `Available on Peer` with one-click resumable LAN/Tailscale **Push**, **Push to All Missing**, **Overwrite & Sync**, and **Pull** transfers (live progress, MB/s, ETA, and Cancel).
  - Automatic **Pre-Flight Model Check** inside the **Node Control Center** modal with inline one-click **Sync Model to Worker** before launching distributed inference.
- **Hardware-Aware VRAM Fit Meter & Auto Layer-Split Planner (`vram_planner.py`, `app.py` `/api/cluster/hardware` & `/api/cluster/fit/plan`, `static/js/model_hub.js`)**:
  - Fast binary GGUF v2/v3 KV metadata header reader + pre-download model architecture profiler and heterogeneous multi-node GPU VRAM + System RAM layer allocator.
  - Interactive **VRAM Fit Meter & Auto Layer-Split Planner** in the **Model Hub** with live `Cluster` / `This PC` toggle, context length selector (`2K`–`32K`), `+ Simulate Worker` GPU presets, proportional multi-node layer split bar, per-node GPU VRAM & System RAM spill meters with exact `--layers` CLI flags, quantization/context recommendations, and live fit badges (`Fits Local GPU`, `2-Node VRAM Fit`, `Tight VRAM Fit`, `GPU + RAM Spill`, `Exceeds Memory`) across Catalog cards, Hugging Face quantizations, and Local Library rows.
  - Inline **VRAM Fit & Auto-Split Bar** inside the **Node Control Center** modal with optional one-click `Auto-Split (--layers ...)` launch support wired through `POST /api/node/start`.
- **Inference Settings, Context Length & System Prompt Drawer (`static/index.html`, `static/js/app.js`, `crates/aeromesh-cli/src/main.rs`, `crates/aeromesh-engine/src/slice_loader.rs`)**:
  - Right slide-over drawer accessible from the top nav (`Inference`) and composer quick-tune pill (`T 0.7 · 4K`), offering per-chat **Temperature** (`0.0–2.0`), **Top-P** (`0.05–1.0`), **Max Output Tokens** (`256–8192`), **Context Length (`n_ctx`: 4K / 8K / 16K / 32K)**, **GPU Offload Layers (`-ngl`: All GPU `-1` or custom `0..999`)**, and **Custom System Prompts / Personas** saved per chat and as reusable user presets.
  - Added `--ctx` and `--ngl` CLI flags to `aeromesh serve` and `aeromesh start` plus `AEROMESH_N_CTX` support in `LlamaPipelineInstance::open`.
- **OpenAI-Compatible Local API Server Panel, Tool Calling & JSON Mode (`openai_compat.py`, `static/js/dev_server.js`, `app.py`, `crates/aeromesh-engine/src/server.rs`)**:
  - New **API Server** workspace tab displaying the live base URL (`http://127.0.0.1:8080/v1`), Gateway/LAN/Tailscale URLs, masked `AEROMESH_API_KEY` with Reveal/Copy/Rotate controls, toggleable LAN exposure (`127.0.0.1` vs `0.0.0.0` with automatic Bearer auth enforcement), copy-paste snippets for **cURL**, **OpenAI Python SDK**, **Tool Calling & JSON**, **Continue.dev**, **Cline / Roo Code**, and **Cursor / Open WebUI**, plus a live protocol tester.
  - Full OpenAI **Tool Calling** (`tools`, `tool_choice`, `role: "tool"`, `tool_calls` unary & SSE streaming) and **JSON Mode** (`response_format`: `json_object` / `json_schema`) support across both the FastAPI gateway and the Rust Axum coordinator.
- **Offline Vendored UI Assets (`static/vendor/`)**:
  - Bundled Lucide, Marked, DOMPurify, Highlight.js, and KaTeX locally with automatic SRI-pinned CDN fallback so the Studio works reliably offline.

### Fixed
- **Cluster Pairing Layout & Responsive Grid (`static/css/liquid_glass.css`)**:
  - Fixed two-column grid overflow and edge overlap on the Cluster Pairing page across desktop and narrower viewports (`minmax(0, 1fr)`, proper flex wrapping, and clean card padding).
- **Documentation & Visual Assets**:
  - Rewrote `README.md` for conciseness, moved deep engineering write-ups into `docs/architecture/engineering-and-protocol.md`, replaced wide ASCII diagrams with mobile-safe GitHub Mermaid charts, refreshed all UI screenshots, and added a walkthrough GIF + HD MP4 demo.

---

## [v0.2.0-ArgonFixes] - 2026-10-22

### Added
- **Windows TCP Socket Tuning (`transport.rs`, `pipeline.rs`)**:
  - Win32 `WSAIoctl` controls (`SIO_TCP_SET_ACK_FREQUENCY = 1`, `SIO_LOOPBACK_FAST_PATH = 1`), `TCP_NODELAY = true`, and 512 KB `SO_SNDBUF`/`SO_RCVBUF` buffers, eliminating the 200 ms Windows Delayed-ACK stall over Tailscale WireGuard's 1280-byte MTU and lifting throughput from 2–5 tok/s to 15–25 tok/s.
- **Outlier-Preserving Per-Row INT8 Codec (`activation.rs`, `server.rs`, `pipeline.rs`)**:
  - Extracts the Top-8 outlier channels per row as exact `(u16 index, f32 value)` pairs (48 bytes) before quantizing the remaining channels to INT8. Cuts wire bandwidth by 74.8% (3.05 KB/tok for 3B, 5.05 KB/tok for 14B) without repetition loops on DeepSeek-R1 or Qwen 2.5.
- **Ephemeral VRAM Stage Loading (`InPlaceStageGuard` in `gguf_slicer.rs`, `slice_loader.rs`)**:
  - Loads sliced tensors directly into VRAM (`use_mmap = false`) and immediately unlinks temporary `.aeromesh_ephemeral_*` slice files, leaving **0.0 MB** persistent disk overhead.
- **Coalesced Frame Writes (`transport.rs`)**:
  - Reusable `BytesMut` buffer sends the 44-byte `ActivationHeader` and payload in a single write syscall.
- **Lock-Free SHM Spin-Yielding (`shm.rs`, `transport.rs`)**:
  - 64-iteration cooperative `tokio::task::yield_now()` spin loop cuts intra-host shared-memory handoff latency below 100 microseconds.
- **Distributed Diffusion Pipeline (`app.py`, `/v1/images/generations`)**:
  - Procedural DiT / Stable Diffusion image generation conditioned by distributed LLM prompt expansion.
- **Desktop Studio Binary (`aeromesh_desktop.rs`, `AeroMesh-Desktop.exe`)**:
  - Frameless native Windows desktop wrapper embedding the Studio UI.

### Fixed
- **Big-Endian Token Frame Decoding (`activation.rs`)**: Aligned `TokenResponseFrame::decode_async` with the Big-Endian encoder.
- **Non-Terminal LM Head Stripping (`gguf_slicer.rs`)**: Stripped `output.weight` and `output.bias` from non-terminal slices to save VRAM.
- **Worker Handshake False Reloads (`pipeline.rs`)**: Fixed layer-count comparison so peer reconnects do not trigger redundant weight reloads.
- **Multi-Stage Chain Indexing (`slice_loader.rs`, `pipeline.rs`, `app.js`)**: Fixed layer range calculation for 3+ node clusters and updated the Cluster HUD.

---

## [v0.1.5] - 2026-10-15

### Security & Hardening
- **FastAPI Gateway Hardening (FIX-16)**: Connection-pooled `httpx.AsyncClient`, unbuffered SSE streaming (`X-Accel-Buffering: no`), RFC 7230 hop-by-hop stripping, upstream Bearer injection, offline local model scanning, and `/health` + `/ready` probes.
- **Win32 Job Object Containment (`job_object.rs`)**: `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` ensures child processes terminate and release GPU VRAM on exit or crash.
- **OpenAI-Compatible Errors (FIX-06, FIX-07)**: Standardized JSON error responses across 400, 401, 404, 422, 429, 500, and 503.
- **Context-Window Guard (FIX-24)**: Rejects requests where `prompt_tokens + max_tokens > n_ctx` with HTTP 422 before GPU allocation.
- **Generation Abort (`/api/chat/abort`, FIX-19)**: Atomic cancellation halts active decoding immediately.

---

## [v0.1.0] - 2026-10-01

### Added
- **Zero-Weight Distributed Pipeline**: Symmetric local `.gguf` slicing along transformer block boundaries with Identity RMSNorm ($\mathbf{w} = [1.0, \dots, 1.0]$) on non-terminal stages.
- **Wire Protocol v4 (`activation.rs`)**: 44-byte `ActivationHeader`, 35-byte `TokenResponseFrame`, Big-Endian framing, and SHA-256 mutual handshake.
- **Web UI (`static/`)**: Dark glass single-page interface with Markdown, KaTeX math, Highlight.js code blocks, and live cluster telemetry.
- **Dynamic C++ FFI Loader (`llama_ffi.rs`)**: Runtime symbol resolution (`GetProcAddress`) for CUDA `llama.cpp` DLLs.
