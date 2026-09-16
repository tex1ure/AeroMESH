# AeroMESH: Project Status & Master Fixes Tracker

**Last Updated**: September 15, 2026  
**Current Git Branch**: `main` (`commit 6dfc3b7`)  
**Workspace Test Suite**: **43 / 43 Unit Tests Passing** (`cargo test --workspace`)  
**Release Build Status**: **Clean / Zero Errors** (`cargo check --workspace --release`)

---

## 1. Executive Summary

Over the recent audit cycles, the codebase underwent critical hardening against remote crashes, memory exhaustion attacks, native access violations, and arithmetic underflows. 

Both feature branches (`P0-Fixes` and `upX`) have been **successfully unified and merged into `main`**. The system now features:
- **Zero-Trust Mutual Authentication** on all control surfaces and binary TCP mesh sockets (`FIX-03`).
- **Strict Local Origin CORS Restriction** preventing CSRF and drive-by web attacks (`FIX-11`).
- **Transactional Model Hot-Swapping** immune to native segmentation faults (`CRASH-01`).
- **Zero-Allocation GGUF Model Slicing** immune to underflow panics (`CRASH-02`).
- **Bounded Frame Deserialization** immune to wire memory DoS attacks (`FIX-05`).
- **Request Parameter Validation & Structured 422 Envelopes** adhering strictly to OpenAI schema (`FIX-06` & `FIX-07`).
- **Context Window & VRAM Boundary Guard** preventing out-of-bounds KV-cache memory allocation (`FIX-24`).
- **Real-Time Telemetry Synchronization** with live inferring HUD indicators.
- **Comprehensive Benchmark & Security Documentation** (`BENCHMARKS.md`, `SECURITY.md`, `README.md`).

Below is the complete, plain-English breakdown of what has been fixed, what is still remaining, and how remaining tasks are prioritized.

---

## 2. Completed Fixes (Now Live in `main`)

### A. Engine & Crash Immunity
| Issue ID | What Was Broken | How It Was Fixed | Affected Files |
|---|---|---|---|
| **CRASH-01** | **Dangling Null Pointer Segfault on `switch_model`**<br>Calling `switch_model` freed C++ pointers and left `vocab` dangling. The next tokenization request crashed the coordinator with native `EXCEPTION_ACCESS_VIOLATION` (`0xC0000005`). | Added pointer nullification on `close()`, guarded all C++ FFI entrypoints with `ensure_ready()`, implemented transactional two-phase model loading with VRAM fallback, and added typed HTTP 500/503 errors. | `slice_loader.rs`<br>`pipeline.rs`<br>`server.rs` |
| **CRASH-02** | **Arithmetic Underflow Panic in Slicer (`0 - 24`)**<br>In `gguf_slicer.rs`, `writer.get_ref().metadata()?.len() - 24` queried an unflushed buffer (0 bytes), executing `0u64 - 24u64` and panicking on every new slice. | Removed dead underflow calculation and temporary cursor pass; implemented zero-allocation padding with static `[u8; 4096]` buffers; hardened string bounds; added end-to-end unit test. | `gguf_slicer.rs` |
| **FIX-05** | **Unbounded Wire Memory Allocation DoS**<br>Frames read raw length prefixes and immediately executed `vec![0u8; len]`, allowing remote attackers to crash nodes via Out-Of-Memory (OOM). | Added hard resource ceilings (`MAX_ACTIVATION_PAYLOAD_BYTES = 128MB`), mathematical shape invariant checks ($S \times d$), and checked multiplication (`checked_mul`) before allocation. | `activation.rs`<br>`shm.rs` |
| **FIX-05 / SEC** | **Exact GGUF Quantization Tensor Sizing**<br>Approximating tensor sizes caused inaccurate offset calculations for advanced quantization types. | Implemented `ggml_tensor_size_bytes` with exact block-size arithmetic for all GGML formats (Q4_0, Q4_K, Q5_K, Q6_K, IQ types). | `gguf_slicer.rs` |
| **RESILIENCE** | **Worker Prefill Transport Auto-Reconnection**<br>If a worker dropped during prefill, the coordinator would fail permanently. | Added automatic transport re-establishment loop to reconnect to worker peers on dropped connections. | `pipeline.rs` |

### B. Security & Authentication
| Issue ID | What Was Broken | How It Was Fixed | Affected Files |
|---|---|---|---|
| **FIX-03** | **No Authentication on HTTP API or TCP Worker Port**<br>Any network peer could invoke `/v1/chat/completions` or send raw frames to TCP port 50052. | (1) Added Axum `auth_middleware` checking `Authorization: Bearer <AEROMESH_API_KEY>` with constant-time equality.<br>(2) FastAPI gateway forwards the bearer token.<br>(3) Upgraded binary protocol to `PROTOCOL_VERSION = 3` with 32-byte mutual auth token (`AEROMESH_WORKER_SECRET`) on port 50052. | `server.rs`<br>`app.py`<br>`activation.rs`<br>`pipeline.rs` |
| **FIX-11** | **Overly Permissive CORS Allowed Arbitrary External Origins**<br>In `server.rs`, Axum used `.layer(CorsLayer::permissive())`, allowing any external website in the user's browser to send cross-origin requests to `localhost:8080`. | Replaced permissive CORS with strict `build_cors_layer()` allowing only local Web UI origins (`http://127.0.0.1:7860`, `http://localhost:7860`) with optional `AEROMESH_ALLOWED_ORIGIN` env var override. Exempted preflight `OPTIONS` requests from authentication. Added 3 automated tests. | `server.rs` |
| **FIX-08** | **No Rate Limiting on API Endpoints**<br>Unauthenticated rapid health probes or spamming completions requests could overwhelm the async runtime and exhaust inference resources. | Added Tower `RateLimitLayer` and `BufferLayer` across two isolated sub-routers: a 60 req/min pool for `/health` (1024 buffer) and a shared 15 req/min pool for `/v1/chat/completions`, `/api/chat`, and `/api/chat/stream` (128 buffer). Created `map_rate_limit_error` mapping Tower errors to standard OpenAI 429 JSON envelope (`rate_limit_error`). Added 2 unit/integration tests. | `server.rs`<br>`Cargo.toml` |
| **SEC-DOC** | **Missing Threat Model & Security Disclosure**<br>No documentation of trust boundaries or ports. | Created [`SECURITY.md`](SECURITY.md) covering threat model, timing attack mitigations, control surfaces, and loopback rules. | `SECURITY.md` |

### C. Performance, Telemetry & Documentation
| Issue ID | What Was Broken | How It Was Fixed | Affected Files |
|---|---|---|---|
| **FIX-01** | **No Empirical Benchmark Data**<br>Performance claims (tok/s, payload size) were unmeasured. | Created [`BENCHMARKS.md`](BENCHMARKS.md) with empirical hardware tables, TTFT, tokens/sec, and payload compression (5.04 KB/tok vs 20.48 KB/tok, 75% reduction). | `BENCHMARKS.md` |
| **FIX-15** | **Inadequate Technical Documentation**<br>Project lacked deep architecture and protocol documentation. | Rewrote root [`README.md`](README.md) (32 KB) with detailed pipeline diagrams, memory topologies, CLI reference, and tensor slicing specs. | `README.md` |
| **FIX-14 / FIX-02** | **Missing Live Demo Battlecard & Presenter Playbook**<br>No step-by-step procedure existed for orchestrating a zero-failure live presentation across two laptops or recovering from venue Wi-Fi drops. | Created [`DEMO_SCRIPT.md`](DEMO_SCRIPT.md) with pre-flight checklist, exact PowerShell launch commands for Laptop A and B, expected console logs, screen-by-screen architectural talking points, and an instant single-machine emergency fallback command (`.\start.ps1 all`) using shared memory IPC. | `DEMO_SCRIPT.md` |
| **FIX-17** | **Standalone API Reference Documentation**<br>Missing comprehensive schemas, parameter constraints, and SSE wire formats. | Created [`API_REFERENCE.md`](API_REFERENCE.md) covering request/response schemas, validation rules, cURL examples, rate limits, and OpenAI-compatible error envelopes. | `API_REFERENCE.md` |
| **FIX-18** | **Architectural Limitations & Boundaries (`LIMITATIONS.md`)**<br>Missing proactive documentation of operational boundaries and hardware requirements. | Published [`LIMITATIONS.md`](LIMITATIONS.md) documenting the 2-node topology limit, Win32 Job Object lifecycle controls, zero-weight storage prerequisites, and stateless session constraints. | `LIMITATIONS.md`<br>`README.md` |
| **FIX-25 & FIX-27** | **Missing License & Contributing Guide**<br>Repository lacked explicit legal licensing declarations and community contribution guidelines. | Added dual MIT OR Apache-2.0 [`LICENSE`](LICENSE), [`LICENSE-MIT`](LICENSE-MIT), [`LICENSE-APACHE`](LICENSE-APACHE), and comprehensive [`CONTRIBUTING.md`](CONTRIBUTING.md) with environment setup, build, test, and code style standards. | `LICENSE`<br>`LICENSE-MIT`<br>`LICENSE-APACHE`<br>`CONTRIBUTING.md`<br>`README.md` |
| **FIX-22** | **Automated CI/CD Pipeline (`.github/workflows/ci.yml`)**<br>Repository lacked automated continuous integration to validate test suite and release builds on pushes and PRs. | Added GitHub Actions workflow ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) configured for `windows-latest` running `cargo test --workspace` and `cargo check --workspace --release` with Cargo artifact caching (`Swatinem/rust-cache@v2`). | `.github/workflows/ci.yml`<br>`README.md` |
| **FIX-04** | **No UI Screenshots or Visual Proof**<br>The README lacked visual evidence of the working claymorphic web dashboard, model controls, or live inference streaming. | Captured 4 high-resolution (1920x1080) PNG screenshots (`01-chat-workspace.png`, `02-model-loading.png`, `03-streaming-response.png`, `04-cluster-hud.png`) in `docs/screenshots/` and embedded them in `README.md` as an HTML gallery table. | `docs/screenshots/*`<br>`README.md` |
| **TELEMETRY** | **UI Showed "Coordinator Offline" During Generation**<br>Polling `/api/cluster/status` contended on the generation lock. | Added thread-safe `cluster_meta: RwLock<ClusterMeta>` to `AppState`, live inferring indicators (`⚡ Inferring (P2P)`), generating pulse CSS animations, and dynamic token counters. | `server.rs`<br>`app.js`<br>`claymorphic.css` |

### D. Input Validation & Bounds Safety
| Issue ID | What Was Broken | How It Was Fixed | Affected Files |
|---|---|---|---|
| **FIX-06** | **Unvalidated Chat Completion Requests**<br>`/v1/chat/completions` accepted empty message arrays, invalid roles, blank content, extreme/negative temperatures, or `NaN`/`Inf` that caused floating-point math domain crashes in native softmax sampling. | Implemented `ChatCompletionRequest::validate()` enforcing non-empty `messages`, valid roles (`system`, `user`, `assistant`, `tool`, `function`), non-blank content, `temperature` $\in [0.0, 2.0]$, `top_p` $\in [0.0, 1.0]$, and `max_tokens` $\in [1, 32768]$. | `server.rs` |
| **FIX-07** | **Non-Standard Error Envelopes on Validation Failure**<br>Validation rejections returned Axum plain text or non-standard JSON, causing third-party OpenAI client SDKs to crash with deserialization errors. | Enhanced `ApiErrorResponse` with structured constructors (`unprocessable`, `bad_request_with_param`, `service_unavailable`, `not_found`) conforming to OpenAI schema (`{"error": {"message": "...", "type": "invalid_request_error", "param": "...", "code": 422}}`). Added `JsonRejection` interception for HTTP 400 on malformed JSON syntax. | `server.rs` |
| **FIX-24** | **No Context Window / VRAM Boundary Enforcement**<br>Prompts exceeding `n_ctx = 4096` or requests where `prompt_tokens + max_tokens > n_ctx` were not checked upfront, causing out-of-bounds KV-cache allocation or silent segmentation faults in native CUDA kernels. | Added pre-flight token count validation (`prompt_tokens.len() + max_tokens <= n_ctx`). Rejects overflowing requests upfront with typed HTTP 422 error: `"Total tokens (X prompt + Y max_tokens = Z) exceed model context window of 4096"`. Added context validation guards across `LlamaPipelineInstance` and `PipelineCoordinatorClient`. | `server.rs`<br>`pipeline.rs` |

### E. Frontend UX & Stream Lifecycle (FIX-19)
| Issue ID | What Was Broken | How It Was Fixed | Affected Files |
|---|---|---|---|
| **FIX-19** | **No "Stop Generating" Abort Capability**<br>Streaming responses could not be cancelled by the user mid-generation, causing unwanted token generation, wasted GPU cycles, and UI lockup until max_tokens elapsed. | Made `send-btn` context-aware (dual-purpose Send/Stop toggle), wired up `AbortController` to the fetch SSE stream, handled `AbortError` gracefully in console (`Generation aborted by user.`), added `Escape` key abort trigger, and integrated `/api/chat/abort` backup endpoints across Axum and FastAPI with native token loop cancellation. | `static/js/app.js`<br>`app.py`<br>`server.rs`<br>`claymorphic.css` |

---

## 3. Remaining Fixes & Action Items

### Tier 1: High-Impact Code & Security Fixes (Completed)

All critical code and security fixes (`FIX-08` Rate Limiting, `FIX-11` Strict CORS, `FIX-19` Abort Button) have been implemented and validated with 43 automated workspace unit and integration tests.

---

### Tier 2: Essential Documentation & Presentation Assets (Completed)

All documentation, demo battlecards, API specifications, licensing files, and UI visual proof assets have been fully completed and integrated into `main`:

- ✅ **FIX-04 (P0, Documentation): UI Screenshots & Visual Proof**: Captured 4 clean, high-resolution (1920x1080) PNG screenshots (`01-chat-workspace.png`, `02-model-loading.png`, `03-streaming-response.png`, `04-cluster-hud.png`) in `docs/screenshots/` and embedded them directly into `README.md` in a responsive two-column preview gallery.

---

### Tier 3: Accepted Architectural Constraints (Deliberately Kept As-Is)

These items were evaluated during the ruthless review and determined to be **correct architectural decisions for the scope of this project** (do not rewrite):

- **FIX-16 (FastAPI Proxy Gateway)**: Adds a localhost hop between the browser and Axum, but serves static assets, provides loopback proxying, and enables rapid Python prototyping. Documented as intended architecture.
- **FIX-20 (Conversation Persistence)**: Local LLM engines (like llama.cpp server and Ollama) operate statelessly in memory. Disk persistence is not needed for the core distributed inference engine.
- **FIX-21 (`pipeline.rs` Refactoring)**: Contains both coordinator client and worker service (~900 lines). Fully tested and stable; refactoring into multiple files is deferred to post-demo maintenance.
- **FIX-23 (HTTP TLS Termination)**: Inter-node traffic is encrypted at Layer 3 via Tailscale WireGuard. Localhost traffic (7860/8080) does not require TLS certificates.
- **FIX-26 (Mobile Responsiveness)**: The target use case is multi-laptop distributed edge inference; desktop UI layout is the primary presentation target.
- **FIX-28 (3+ Node Clusters)**: The project specifically demonstrates heterogeneous 2-node pipeline parallelism across consumer laptops. Multi-node N-stage pipelines are marked on the future roadmap.
- **FIX-29 (Windows-Only / Win32 Job Objects)**: Windows is specifically targeted because consumer gaming laptops with NVIDIA GPUs run Windows 11; Win32 Job Objects guarantee zero orphaned GPU worker processes.
- **FIX-30 (Docker Containerization)**: Consumer Windows GPU passthrough inside Docker is notoriously brittle; native PowerShell execution directly accesses the NVIDIA CUDA drivers with zero virtualization overhead.

---

## 4. Master Status Summary Matrix

| Category | Total Issues | Resolved | Remaining Actionable | Accepted Constraints |
|---|:---:|:---:|:---:|:---:|
| **P0 (Critical / Blockers)** | 5 | **5** | **0** | 0 |
| **P1 (High Priority)** | 10 | **10** | **0** | 0 |
| **P2 (Medium Priority)** | 9 | **5** | **0** | 4 |
| **P3 (Low Priority)** | 6 | **2** | **0** | 4 |
| **Engine Crash Fixes** | 2 | **2** | **0** | 0 |
| **Total** | **32** | **24** | **0** (100% Resolved) | **8** |

---

## 5. Recommended Execution Roadmap

All actionable fixes across the repository are now **100% Complete**:

1. **Step 1 (Code & Security - 100% Complete)**:
   - ✅ `FIX-19` ("Stop Generating" abort button) completed across frontend & backend.
   - ✅ `FIX-08` (Rate limiting middleware) completed with isolated health & completions pools and OpenAI error envelope.
2. **Step 2 (Documentation Suite - 100% Complete)**:
   - ✅ `DEMO_SCRIPT.md` (`FIX-14` / `FIX-02`) completed presenter battlecard and failover guide.
   - ✅ `API_REFERENCE.md` (`FIX-17`) completed standalone HTTP API specification and wire schemas.
   - ✅ `LIMITATIONS.md` (`FIX-18`) completed architectural boundaries and hardware trade-offs disclosure.
   - ✅ `LICENSE` (`FIX-25`), `LICENSE-MIT`, `LICENSE-APACHE`, and `CONTRIBUTING.md` (`FIX-27`) completed dual-license legal and contributor guidelines.
3. **Step 3 (DevOps & Visual Proof - 100% Complete)**:
   - ✅ `.github/workflows/ci.yml` (`FIX-22`) completed automated GitHub Actions CI workflow.
   - ✅ `FIX-04` UI screenshots captured and embedded into `README.md`.
