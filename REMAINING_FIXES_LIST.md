# AeroMESH — Complete Fixes & Remaining Tasks List

> **Current Status**: All P0 blockers, critical engine crashes, and input validation bounds are **FIXED and tested** (35/35 workspace tests passing).  
> **Git Branch**: `main` (synchronized with `origin/main` at commit `fa3837a`).  
> **Purpose of this file**: A simple, human-readable checklist of everything that was fixed, everything that remains to be done, and architectural decisions kept as-is.

---

## Quick Summary Matrix

| Category | Total Items | Completed | Remaining To Do | Accepted by Design |
|---|:---:|:---:|:---:|:---:|
| **Critical Engine & Crash Fixes (P0)** | 5 | **5** (100%) | 0 | 0 |
| **Security & Authentication (P0 / P1)** | 4 | **2** (50%) | 2 | 0 |
| **Input Validation & Safety (P1 / P2)** | 3 | **2** (67%) | 1 | 0 |
| **Frontend & Usability (P2)** | 2 | **1** (50%) | 1 | 0 |
| **Documentation & Presentation (P1 / P2)** | 6 | **3** (50%) | 3 | 0 |
| **Legal & Open Source Polish (P3)** | 3 | 0 | 3 | 0 |
| **Intentional Design Decisions (No Action)** | 8 | 0 | 0 | **8** |
| **Total** | **31** | **13** | **10** | **8** |

---

## 1. What We Already Fixed & Tested ([X] DONE)

These are critical bugs, crashes, vulnerabilities, and validation defects that were identified and completely patched. All 35 automated unit tests currently pass.

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

---

## 2. What Is Remaining To Do ([ ] TO-DO)

These are the remaining actionable tasks, ranked by order of priority and ease of implementation.

### High Priority — Code & Security (Do First)

#### [ ] 1. Restrict CORS from `*` to Localhost (`FIX-11`)
- **What is needed**: In `crates/aeromesh-engine/src/server.rs`, replace `.layer(CorsLayer::permissive())` with an explicit origin whitelist:
  - Allow `http://127.0.0.1:7860` and `http://localhost:7860` (the FastAPI frontend).
- **Why it matters**: Permissive CORS allows any website you visit in your web browser to quietly send requests to your local LLM engine.
- **Effort**: ~10 minutes | **Difficulty**: Very Easy

#### [ ] 3. "Stop Generating" Abort Button (`FIX-19`)
- **What is needed**: In `static/js/app.js`, attach a JavaScript `AbortController` to the Server-Sent Events (SSE) fetch stream. When the user clicks the "Stop" button in the UI, call `controller.abort()` to halt token streaming immediately.
- **Why it matters**: Essential user experience feature; lets the user cancel long generations without having to refresh the browser.
- **Effort**: ~15 minutes | **Difficulty**: Easy

#### [ ] 4. Context Window Boundary Guard (`FIX-24`)
- **What is needed**: In `crates/aeromesh-engine/src/pipeline.rs`, check that `prompt_tokens.len() + max_tokens <= n_ctx` (e.g. 4096). If the prompt is too long, return an explicit error instead of letting llama.cpp truncate or fail silently.
- **Why it matters**: Prevents out-of-bounds KV cache allocation.
- **Effort**: ~15 minutes | **Difficulty**: Easy

#### [ ] 5. Rate Limiting Middleware (`FIX-08`)
- **What is needed**: In `crates/aeromesh-engine/src/server.rs`, attach `tower::limit::RateLimitLayer` to routes (e.g., 60 req/min for `/health`, 15 req/min for `/v1/chat/completions`).
- **Why it matters**: Protects the engine from accidental request floods or loops.
- **Effort**: ~25 minutes | **Difficulty**: Easy

---

### Medium Priority — Documentation & Demo Assets (Quick Wins)

#### [ ] 6. Presenter Demo Battlecard (`DEMO_SCRIPT.md` / `FIX-14`)
- **What is needed**: Create a simple, step-by-step markdown cheat sheet for running a flawless live demonstration:
  - Exact PowerShell commands to run on Laptop A (Coordinator) and Laptop B (Worker).
  - Expected console outputs so the presenter knows it succeeded.
  - 3 key talking points to explain to judges/viewers (P2P mesh, 75% activation compression, zero-weight network transfer).
  - Emergency fallback command (`.\start.ps1 all` on single machine) if the venue WiFi fails.
- **Why it matters**: Prevents demo day panic and guarantees a polished live presentation.
- **Effort**: ~15 minutes | **Difficulty**: Very Easy

#### [ ] 7. Standalone API Reference (`API_REFERENCE.md` / `FIX-17`)
- **What is needed**: Create a clean developer document listing:
  - `/v1/chat/completions` (JSON format, parameters, curl example, SSE streaming example).
  - `/v1/models` (list available models).
  - `/health` (health check response codes).
  - `/api/cluster/status` (telemetry JSON format).
- **Why it matters**: Anyone who wants to build on AeroMESH or write scripts can see all API endpoints at a glance.
- **Effort**: ~15 minutes | **Difficulty**: Very Easy

#### [ ] 8. Architectural Limitations & Boundaries (`LIMITATIONS.md` / `FIX-18`)
- **What is needed**: A proactive document explaining known design boundaries:
  - 2-node maximum in current pipeline design.
  - Windows-only Win32 Job Object requirement.
  - Both nodes require local GGUF model files.
  - Ephemeral conversation history (in-memory, no database).
- **Why it matters**: Preempts reviewer critiques by showing these were conscious engineering decisions, not oversights.
- **Effort**: ~10 minutes | **Difficulty**: Very Easy

#### [ ] 9. Automated GitHub Actions CI (`.github/workflows/ci.yml` / `FIX-22`)
- **What is needed**: Create `.github/workflows/ci.yml` that automatically runs `cargo test --workspace` and `cargo check --workspace --release` on every push and pull request.
- **Why it matters**: Proves to viewers and contributors that code quality is continuously tested.
- **Effort**: ~15 minutes | **Difficulty**: Easy

#### [ ] 10. UI Screenshots in README (`FIX-04`)
- **What is needed**: Take 3–4 clean screenshots of the claymorphic UI (chat screen, model loading, telemetry HUD) and add them to `README.md`.
- **Why it matters**: Visual proof makes the project look 10x more impressive instantly.
- **Effort**: ~20 minutes | **Difficulty**: Easy

---

### Low Priority — Open-Source Legal & Community Polish

#### [ ] 11. Open Source License (`LICENSE` / `FIX-25`)
- **What is needed**: Add standard Apache-2.0 or MIT `LICENSE` file in the root directory.
- **Effort**: ~5 minutes | **Difficulty**: Trivial

#### [ ] 12. Contribution Guidelines (`CONTRIBUTING.md` / `FIX-27`)
- **What is needed**: Add standard `CONTRIBUTING.md` outlining repository structure, build commands, and pull request guidelines.
- **Effort**: ~10 minutes | **Difficulty**: Very Easy

---

## 3. Intentional Architectural Decisions ([-] NO ACTION NEEDED)

During the ruthless review, the following 8 items were carefully evaluated. They are **deliberate engineering choices** appropriate for this project. **Do not rewrite or change them**:

| Item ID | Topic | Why It Is Left As-Is |
|---|---|---|
| **FIX-16** | **FastAPI Proxy Layer** | FastAPI on port 7860 serves the static HTML/CSS/JS frontend and proxies API requests to Axum on port 8080. This cleanly decouples the frontend from the Rust engine and allows quick Python scripting without recompiling Rust. |
| **FIX-20** | **Conversation Persistence** | Chat history is kept in memory during the browser session, not saved to disk or SQLite. Local inference engines (like llama.cpp server and Ollama) are intentionally stateless. |
| **FIX-21** | **`pipeline.rs` Modularity** | `pipeline.rs` contains both coordinator logic and worker service (~900 lines). It works reliably and is covered by unit tests; refactoring into multiple files before a demo introduces unnecessary regression risk. |
| **FIX-23** | **HTTP TLS Termination** | Traffic between laptops is already encrypted at the network layer via Tailscale WireGuard. Localhost traffic (7860/8080) does not require HTTPS certificates. |
| **FIX-26** | **Mobile Responsive UI** | The target demo is on laptops and desktop screens. Mobile responsive layout is a future enhancement. |
| **FIX-28** | **3+ Node Clusters** | AeroMESH specifically demonstrates heterogeneous 2-node pipeline parallelism across consumer laptops. N-stage clusters are part of the future roadmap. |
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

3. **Step 3 (Polish & DevOps — ~25 mins)**:
   - Create `LICENSE` (`FIX-25`) and `CONTRIBUTING.md` (`FIX-27`).
   - Create `.github/workflows/ci.yml` (`FIX-22`).
   - Take and embed UI screenshots (`FIX-04`).
