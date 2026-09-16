# AeroMESH — Complete Fixes & Remaining Tasks List

> **Current Status**: All P0 blockers, critical engine crashes, input validation bounds, context window guards, rate limiting, and CORS restrictions are **FIXED and tested** (43/43 workspace tests passing).  
> **Git Branch**: `main` and `upX` (synchronized with `origin/main` at commit `3b8d685`).  
> **Purpose of this file**: A simple, human-readable checklist of everything that was fixed, everything that remains to be done, and architectural decisions kept as-is.

---

## Quick Summary Matrix

| Category | Total Items | Completed | Remaining To Do | Accepted by Design |
|---|:---:|:---:|:---:|:---:|
| **Critical Engine & Crash Fixes (P0)** | 5 | **5** (100%) | 0 | 0 |
| **Security & Authentication (P0 / P1)** | 4 | **4** (100%) | 0 | 0 |
| **Input Validation & Safety (P1 / P2)** | 3 | **3** (100%) | 0 | 0 |
| **Frontend & Usability (P2)** | 2 | **2** (100%) | 0 | 0 |
| **Documentation & Presentation (P1 / P2)** | 7 | **7** (100%) | 0 | 0 |
| **DevOps & CI (P2)** | 1 | **1** (100%) | 0 | 0 |
| **Legal & Open Source Polish (P3)** | 2 | **2** (100%) | 0 | 0 |
| **Intentional Design Decisions (No Action)** | 8 | 0 | 0 | **8** |
| **Total** | **32** | **24** | **0** | **8** |

---

## 1. What We Already Fixed & Tested ([X] DONE)

These are critical bugs, crashes, vulnerabilities, and validation defects that were identified and completely patched. All 43 automated unit tests currently pass.

### [X] 1. Dangling Pointer Segfault on Model Switch (`CRASH-01`)
- **What was broken**: When switching models (`/v1/chat/completions`), the engine called `close()` which freed underlying C++ memory and left pointers null. If loading the new model failed, the coordinator kept running with null pointers. The very next chat request attempted to tokenize using a null vocabulary pointer, causing an instant native segmentation fault (`0xC0000005`) that crashed the entire program.
- **How it was fixed**: 
  1. Pointer nullification on `close()` so dangling pointers can never be accessed.
  2. Guarded all C++ entrypoints (`tokenize`, `generate`, `embed`) with `ensure_ready()` checks.
  3. Implemented a **transactional 2-phase load**: if the new model fails to load, the old model stays active or an explicit HTTP 503/500 JSON error is returned without crashing.
- **Code Locations**: `crates/aeromesh-engine/src/slice_loader.rs`, `pipeline.rs`, `server.rs`.

### [X] 2. Arithmetic Underflow Panic in GGUF Slicer (`CRASH-02`)
- **What was broken**: In `gguf_slicer.rs`, the code calculated padding using `writer.get_ref().metadata()?.len() - 24`. Because `writer` was a buffered writer (`BufWriter`) that had not flushed to disk yet, `metadata()?.len()` returned `0`. Calculating `0u64 - 24u64` caused an instant integer underflow panic in debug mode and wrapped to 18 quintillion in release mode, corrupting every sliced model file.
- **How it was fixed**: Removed the dead underflow subtraction, eliminated unnecessary cursor passes, implemented zero-allocation padding with static `[u8; 4096]` buffers, and added an end-to-end synthetic GGUF slicing test.
- **Code Location**: `crates/aeromesh-engine/src/gguf_slicer.rs`.

### [X] 3. Unbounded Network Memory Allocation DoS (`FIX-05`)
- **What was broken**: Binary network frames read payload lengths directly from untrusted incoming network bytes and immediately ran `vec![0u8; len]`. An attacker could send a 4 GB length prefix and force an Out-Of-Memory (OOM) crash on any node in the cluster.
- **How it was fixed**: Added hard memory limits (`MAX_ACTIVATION_PAYLOAD_BYTES = 128MB`), verified tensor shape invariants ($S \times d == \text{elements}$), and used checked arithmetic (`checked_mul`) before allocating any memory.
- **Code Locations**: `crates/aeromesh-core/src/activation.rs`, `crates/aeromesh-core/src/shm.rs`.

### [X] 4. Missing Authentication on API & TCP Mesh Port (`FIX-03`)
- **What was broken**: The HTTP API on port 8080 and binary TCP mesh port 50052 were completely open. Anyone on the local WiFi could send prompts, execute inference, or inject malicious activation tensors.
- **How it was fixed**:
  1. Added Axum HTTP `auth_middleware` checking `Authorization: Bearer <AEROMESH_API_KEY>` with constant-time equality check (prevents timing attacks).
  2. Upgraded binary mesh wire protocol to `PROTOCOL_VERSION = 3` with 32-byte mutual auth token (`AEROMESH_WORKER_SECRET`).
  3. Configured FastAPI frontend gateway (`app.py`) to automatically forward the bearer token.
- **Code Locations**: `crates/aeromesh-engine/src/server.rs`, `pipeline.rs`, `crates/aeromesh-core/src/activation.rs`, `app.py`.

### [X] 5. Exact Quantization Tensor Size Calculation (`FIX-05 / Sizer`)
- **What was broken**: The GGUF slicer previously estimated tensor sizes, leading to miscalculated byte offsets when splitting models with advanced block quantization (Q4_K, Q5_K, Q6_K, IQ types).
- **How it was fixed**: Implemented `ggml_tensor_size_bytes` with exact block-size arithmetic for all GGML quantization formats.
- **Code Location**: `crates/aeromesh-engine/src/gguf_slicer.rs`.

### [X] 6. Worker Prefill Auto-Reconnection (`RESILIENCE`)
- **What was broken**: If the TCP connection between coordinator and worker dropped temporarily during prefill, the coordinator would fail permanently and crash future requests.
- **How it was fixed**: Added an automatic transport re-establishment loop to reconnect to worker peers on dropped connections.
- **Code Location**: `crates/aeromesh-engine/src/pipeline.rs`.

### [X] 7. UI Telemetry Desynchronization (`TELEMETRY`)
- **What was broken**: During active token generation, polling `/api/cluster/status` blocked on the coordinator's generation lock. This caused the web UI to erroneously flash "Coordinator Offline" while tokens were actively streaming.
- **How it was fixed**: Decoupled telemetry from generation lock using `cluster_meta: RwLock<ClusterMeta>`. Added live inferring status badges (`⚡ Inferring (P2P)`), generating pulse animations, and real-time tokens/sec calculations.
- **Code Locations**: `crates/aeromesh-engine/src/server.rs`, `static/js/app.js`, `static/css/claymorphic.css`.

### [X] 8. Missing Empirical Benchmarks (`FIX-01`)
- **What was broken**: Claims of 75% network payload reduction and distributed tokens/sec had no published empirical measurements.
- **How it was fixed**: Published [`BENCHMARKS.md`](BENCHMARKS.md) with measured hardware tables, Time-To-First-Token (TTFT), tokens/sec, and payload compression data (5.04 KB/tok vs 20.48 KB/tok).
- **File**: `BENCHMARKS.md`.

### [X] 9. Security & Threat Model Disclosure (`SEC-DOC`)
- **What was broken**: No documentation of network trust boundaries, ports, or security expectations.
- **How it was fixed**: Created [`SECURITY.md`](SECURITY.md) documenting authentication, timing attack protection, network trust boundaries, and vulnerability reporting.
- **File**: `SECURITY.md`.

### [X] 10. Master Technical Documentation Overhaul (`FIX-15`)
- **What was broken**: The root README lacked pipeline diagrams, memory layout specs, and comprehensive CLI usage.
- **How it was fixed**: Rewrote [`README.md`](README.md) (32 KB) with detailed pipeline diagrams, memory topologies, CLI reference, and tensor slicing specs.
- **File**: `README.md`.

### [X] 11. Git Branch Unification & Merge
- **What was broken**: Changes were split across two branches (`P0-Fixes` and `upX`), with potential merge conflicts in `activation.rs`, `pipeline.rs`, and `server.rs`.
- **How it was fixed**: Merged `origin/P0-Fixes` into `upX`, resolved all conflicts cleanly, verified with 26 passing unit tests, and fast-forward merged into `main`.
- **Branches**: `main` and `upX` are now unified at `fa3837a`.

### [X] 12. Chat Completions Parameter Bounds Checking (`FIX-06`)
- **What was broken**: `/v1/chat/completions` accepted completely unvalidated requests. Empty message arrays produced empty prompts that wasted GPU cycles; messages with invalid roles or empty content corrupted ChatML prompt formatting; negative or extreme temperatures (`< 0.0` or `> 2.0`, `NaN`, `Inf`) caused floating-point math domain crashes in native softmax sampling; invalid `top_p` or zero/excessive `max_tokens` threatened engine stability.
- **How it was fixed**: Implemented `ChatCompletionRequest::validate()` verifying `messages` is non-empty, roles are valid (`system`, `user`, `assistant`, `tool`, `function`), content is non-blank, `temperature` $\in [0.0, 2.0]$, `top_p` $\in [0.0, 1.0]$, and `max_tokens` $\in [1, 32768]$.
- **File**: `crates/aeromesh-engine/src/server.rs`.

### [X] 13. OpenAI 422 JSON Error Envelopes (`FIX-07`)
- **What was broken**: When requests failed validation, the server either returned Axum's default plain-text rejections, non-standard `{ "error": "string" }` JSON, or HTTP 500 with integer codes. Third-party OpenAI SDKs (Python `openai`, LangChain) failed with deserialization errors.
- **How it was fixed**: Enhanced `ApiErrorResponse` with `unprocessable`, `bad_request_with_param`, `service_unavailable`, and `not_found` constructors conforming strictly to the OpenAI JSON schema (`{"error": {"message": "...", "type": "invalid_request_error", "param": "...", "code": "..."}}`). Added `JsonRejection` interception returning HTTP 400 on malformed JSON syntax.
- **File**: `crates/aeromesh-engine/src/server.rs`.

### [X] 14. Context Window Boundary Guard (`FIX-24`)
- **What was broken**: Prompts and generation parameters were never verified against the model context size (`n_ctx = 4096`). Prompts exceeding 4096 tokens or requests with `prompt_tokens + max_tokens > n_ctx` attempted out-of-bounds KV cache allocation, causing silent truncation or segmentation faults inside native CUDA kernels.
- **How it was fixed**: Added `PipelineCoordinatorClient::n_ctx()` exposing the active instance context window. In `handle_chat_completions`, requests with `max_tokens > n_ctx` are rejected upfront with `HTTP 422 Unprocessable Entity` (`code: "context_window_exceeded"`). In `generate_pipeline`, prompt token counts are checked after tokenization: if `prompt_tokens.len() >= n_ctx` or `prompt_tokens.len() + max_tokens > n_ctx`, generation immediately returns an explicit descriptive error and forwards it to the streaming channel without crashing.
- **Files**: `crates/aeromesh-engine/src/pipeline.rs`, `crates/aeromesh-engine/src/server.rs`.

---

## 2. What Is Remaining To Do ([ ] TO-DO)

These are the remaining actionable tasks, ranked by order of priority and ease of implementation.

### High Priority — Code & Security (Do First)

#### [x] 1. Restrict CORS from `*` to Localhost (`FIX-11`)
- **What is needed**: In `crates/aeromesh-engine/src/server.rs`, replace `.layer(CorsLayer::permissive())` with an explicit origin whitelist:
  - Allow `http://127.0.0.1:7860` and `http://localhost:7860` (the FastAPI frontend).
- **Why it matters**: Permissive CORS allows any website you visit in your web browser to quietly send requests to your local LLM engine.
- **Effort**: ~10 minutes | **Difficulty**: Very Easy

#### [x] 2. "Stop Generating" Abort Button (`FIX-19`)
- **What is needed**: In `static/js/app.js`, attach a JavaScript `AbortController` to the Server-Sent Events (SSE) fetch stream. When the user clicks the "Stop" button in the UI, call `controller.abort()` to halt token streaming immediately.
- **Why it matters**: Essential user experience feature; lets the user cancel long generations without having to refresh the browser.
- **Effort**: ~15 minutes | **Difficulty**: Easy

#### [x] 3. Rate Limiting Middleware (`FIX-08`)
- **What is needed**: In `crates/aeromesh-engine/src/server.rs`, attach `tower::limit::RateLimitLayer` to routes (e.g., 60 req/min for `/health`, 15 req/min for `/v1/chat/completions`).
- **Why it matters**: Protects the engine from accidental request floods or loops.
- **Effort**: ~25 minutes | **Difficulty**: Easy

---

### Medium Priority — Documentation & Demo Assets (Quick Wins)

#### [x] 6. Presenter Demo Battlecard (`DEMO_SCRIPT.md` / `FIX-14`)
- **What is needed**: Create a simple, step-by-step markdown cheat sheet for running a flawless live demonstration:
  - Exact PowerShell commands to run on Laptop A (Coordinator) and Laptop B (Worker).
  - Expected console outputs so the presenter knows it succeeded.
  - 3 key talking points to explain to judges/viewers (P2P mesh, 75% activation compression, zero-weight network transfer).
  - Emergency fallback command (`.\start.ps1 all` on single machine) if the venue WiFi fails.
- **Why it matters**: Prevents demo day panic and guarantees a polished live presentation.
- **Effort**: ~15 minutes | **Difficulty**: Very Easy

#### [x] 7. Standalone API Reference (`API_REFERENCE.md` / `FIX-17`) - **Status**: Completed. Fully documented endpoints, curl commands, JSON schemas, SSE formats, and error matrices.
- **What is needed**: Create a clean developer document listing:
  - `/v1/chat/completions` (JSON format, parameters, curl example, SSE streaming example).
  - `/v1/models` (list available models).
  - `/health` (health check response codes).
  - `/api/cluster/status` (telemetry JSON format).
- **Why it matters**: Anyone who wants to build on AeroMESH or write scripts can see all API endpoints at a glance.
- **Effort**: ~15 minutes | **Difficulty**: Very Easy

#### [X] 8. Architectural Limitations & Boundaries (`LIMITATIONS.md` / `FIX-18`)
- **Status**: Completed. Fully documented the 2-node topology boundary, Win32 Job Object containment rationale, zero-weight storage requirements, and ephemeral context lifecycles.
- **What is needed**: A proactive document explaining known design boundaries:
  - 2-node maximum in current pipeline design.
  - Windows-only Win32 Job Object requirement.
  - Both nodes require local GGUF model files.
  - Ephemeral conversation history (in-memory, no database).
- **Why it matters**: Preempts reviewer critiques by showing these were conscious engineering decisions, not oversights.
- **Effort**: ~10 minutes | **Difficulty**: Very Easy

#### [X] 9. Automated GitHub Actions CI (`.github/workflows/ci.yml` / `FIX-22`)
- **Status**: Completed. Created `.github/workflows/ci.yml` configured on `windows-latest` runner running `cargo test --workspace` and `cargo check --workspace --release` with Cargo caching.
- **What is needed**: Create `.github/workflows/ci.yml` that automatically runs `cargo test --workspace` and `cargo check --workspace --release` on every push and pull request.
- **Why it matters**: Proves to viewers and contributors that code quality is continuously tested.
- **Effort**: ~15 minutes | **Difficulty**: Easy

#### [X] 10. UI Screenshots in README (`FIX-04`)
- **Status**: Completed. Captured 4 clean, high-resolution (1920x1080) PNG screenshots (`01-chat-workspace.png`, `02-model-loading.png`, `03-streaming-response.png`, `04-cluster-hud.png`) under `docs/screenshots/` and embedded them directly into `README.md` in a responsive two-column preview gallery.
- **What was added**: Visual proof of the working claymorphic chat workspace, model selection & slice panel, mid-stream token generation with DeepSeek-R1 `<think>` foldout, syntax highlighting & KaTeX math rendering, and cluster topology HUD modal.
- **Why it matters**: Visual proof establishes credibility, showcases the claymorphic design, and proves the system is a functional, polished edge cluster.
- **Effort**: ~20 minutes | **Difficulty**: Easy

---

### Low Priority — Open-Source Legal & Community Polish

#### [X] 11. Open Source License (`LICENSE` / `FIX-25`)
- **Status**: Completed. Dual-licensed under MIT OR Apache-2.0. Published root `LICENSE`, `LICENSE-MIT`, and `LICENSE-APACHE`.
- **What is needed**: Add standard Apache-2.0 or MIT `LICENSE` file in the root directory.
- **Effort**: ~5 minutes | **Difficulty**: Trivial

#### [X] 12. Contribution Guidelines (`CONTRIBUTING.md` / `FIX-27`)
- **Status**: Completed. Published comprehensive `CONTRIBUTING.md` detailing setup, build, test, and code style standards.
- **What is needed**: Add standard `CONTRIBUTING.md` outlining repository structure, build commands, and pull request guidelines.
- **Effort**: ~10 minutes | **Difficulty**: Very Easy

---

## 3. Intentional Architectural Decisions ([-] NO ACTION NEEDED)

During the ruthless review, the following 8 items were carefully evaluated. They are **deliberate engineering choices** appropriate for this project. **Do not rewrite or change them**:

| Item ID | Topic | Why It Is Left As-Is |
|---|---|---|
| **FIX-16** | **FastAPI Proxy Gateway** | Documented and hardened intentional architecture ([`docs/architecture/fastapi-proxy-gateway.md`](docs/architecture/fastapi-proxy-gateway.md)). FastAPI on port 7860 serves the static claymorphic frontend, provides browser single-origin isolation, injects Bearer credentials, handles unbuffered SSE token streaming, and provides graceful offline fallback. Validated by 9 automated tests in [`tests/test_fastapi_gateway.py`](tests/test_fastapi_gateway.py). |
| **FIX-20** | **Conversation Persistence** | Documented and hardened as intentional architecture / non-goal ([`docs/architecture/stateless-conversation-policy.md`](docs/architecture/stateless-conversation-policy.md)). The core inference engine is stateless with respect to conversation history; model weights are file-based, while activation and KV cache buffers live exclusively in volatile RAM/VRAM. Application-level conversation persistence is handled client-side via browser `localStorage` (`aeromesh_conversations_v2`) and Markdown export. |
| **FIX-21** | **`pipeline.rs` Modularity** | Intentionally deferred to post-demo maintenance to preserve stability on the demo-critical inference path ([`docs/architecture/pipeline-refactoring-plan.md`](docs/architecture/pipeline-refactoring-plan.md)). Module is 100% stable, fully covered by unit tests, and annotated with a deferral safety note. Detailed post-demo extraction blueprint is documented. |
| **FIX-23** | **HTTP TLS Termination** | Documented and hardened as intentional architecture / non-goal ([`docs/architecture/tls-transport-security-policy.md`](docs/architecture/tls-transport-security-policy.md) and [`SECURITY.md`](SECURITY.md)). Localhost HTTP traffic (7860/8080) relies on OS loopback process isolation and does not require TLS certificates. Cross-machine mesh traffic is encrypted at Layer 3 via Tailscale WireGuard (ChaCha20-Poly1305). Runtime warnings alert on non-loopback binds. Edge TLS termination is delegated to Tailscale Serve, Caddy, or Nginx with unbuffered SSE streaming. |
| **FIX-26** | **Mobile Responsive UI** | Documented and hardened as intentional product scope / desktop-first architecture ([`docs/architecture/device-targeting-responsive-policy.md`](docs/architecture/device-targeting-responsive-policy.md) and [`README.md`](README.md)). The primary target is multi-laptop distributed edge inference on desktop and laptop displays (`1920x1080`, `1600x900`, `1366x768`). Added non-destructive additive responsive guardrails via `static/css/responsive.css` and `static/js/app.js` (viewport meta, mobile drawer sidebar, compact top navbar, and vertically scrollable unclipped modals) while keeping desktop layout 100% untouched. |
| **FIX-28** | **3+ Node Clusters** | Documented and hardened as intentional product scope / 2-node architecture with formal N-stage roadmap ([`docs/architecture/n-stage-pipeline-roadmap.md`](docs/architecture/n-stage-pipeline-roadmap.md), [`ROADMAP.md`](ROADMAP.md), and [`LIMITATIONS.md`](LIMITATIONS.md)). The current release demonstrates heterogeneous 2-node pipeline parallelism: Stage 1 on coordinator and Stage 2 with true output norm and LM head on worker. Linear $N$-stage pipelines ($N \ge 3$ nodes) with multi-split GGUF slicing, intermediate token-less GEMM evaluation, and direct tail-to-coordinator return socket channels are scheduled for Milestone 3. |
| **FIX-29** | **Windows-Only / Win32 Job Objects** | Target hardware is Windows 11 gaming laptops with NVIDIA GPUs. Win32 Job Objects cleanly terminate all worker processes with zero orphaned GPU memory leaks. |
| **FIX-30** | **Docker Containerization** | Consumer Windows GPU passthrough inside Docker containers is notoriously brittle; running natively via PowerShell directly leverages host CUDA drivers with zero virtualization overhead. |

---

## 4. Suggested Order of Attack

To finish all remaining items efficiently, tackle them in this order:

1. **Step 1 (Code & Security — ~35 mins)**:
   - Implement `FIX-06 & FIX-07` (Input validation) + `FIX-11` (CORS restriction) in `crates/aeromesh-engine/src/server.rs`.
   - Implement `FIX-19` ("Stop Generating" abort button) in `static/js/app.js`.
   - Run `cargo test --workspace` to ensure all tests pass.

2. **Step 2 (Documentation & Demo Assets — ~30 mins)**:
   - Create `DEMO_SCRIPT.md` (`FIX-14`).
   - Create `API_REFERENCE.md` (`FIX-17`).
   - Create `LIMITATIONS.md` (`FIX-18`).

3. **Step 3 (Polish & DevOps — 100% Complete)**:
   - ✅ Create `LICENSE` (`FIX-25`), `LICENSE-MIT`, `LICENSE-APACHE`, and `CONTRIBUTING.md` (`FIX-27`).
   - ✅ Create `.github/workflows/ci.yml` (`FIX-22`).
   - ✅ Take and embed UI screenshots (`FIX-04`).
