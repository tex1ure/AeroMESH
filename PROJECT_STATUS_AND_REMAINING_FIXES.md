# AeroMESH: Project Status & Master Fixes Tracker

**Last Updated**: September 15, 2026  
**Current Git Branch**: `main` (`commit fa3837a`)  
**Workspace Test Suite**: **26 / 26 Unit Tests Passing** (`cargo test --workspace`)  
**Release Build Status**: **Clean / Zero Errors** (`cargo check --workspace --release`)

---

## 1. Executive Summary

Over the recent audit cycles, the codebase underwent critical hardening against remote crashes, memory exhaustion attacks, native access violations, and arithmetic underflows. 

Both feature branches (`P0-Fixes` and `upX`) have been **successfully unified and merged into `main`**. The system now features:
- **Zero-Trust Mutual Authentication** on all control surfaces and binary TCP mesh sockets.
- **Transactional Model Hot-Swapping** immune to native segmentation faults.
- **Zero-Allocation GGUF Model Slicing** immune to underflow panics.
- **Bounded Frame Deserialization** immune to wire memory DoS attacks.
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
| **SEC-DOC** | **Missing Threat Model & Security Disclosure**<br>No documentation of trust boundaries or ports. | Created [`SECURITY.md`](SECURITY.md) covering threat model, timing attack mitigations, control surfaces, and loopback rules. | `SECURITY.md` |

### C. Performance, Telemetry & Documentation
| Issue ID | What Was Broken | How It Was Fixed | Affected Files |
|---|---|---|---|
| **FIX-01** | **No Empirical Benchmark Data**<br>Performance claims (tok/s, payload size) were unmeasured. | Created [`BENCHMARKS.md`](BENCHMARKS.md) with empirical hardware tables, TTFT, tokens/sec, and payload compression (5.04 KB/tok vs 20.48 KB/tok, 75% reduction). | `BENCHMARKS.md` |
| **FIX-15** | **Inadequate Technical Documentation**<br>Project lacked deep architecture and protocol documentation. | Rewrote root [`README.md`](README.md) (32 KB) with detailed pipeline diagrams, memory topologies, CLI reference, and tensor slicing specs. | `README.md` |
| **TELEMETRY** | **UI Showed "Coordinator Offline" During Generation**<br>Polling `/api/cluster/status` contended on the generation lock. | Added thread-safe `cluster_meta: RwLock<ClusterMeta>` to `AppState`, live inferring indicators (`⚡ Inferring (P2P)`), generating pulse CSS animations, and dynamic token counters. | `server.rs`<br>`app.js`<br>`claymorphic.css` |

---

## 3. Remaining Fixes & Action Items

### Tier 1: High-Impact Code & Security Fixes (Immediate Next Steps)

These are actionable code improvements that directly increase reliability, security, and demo usability:

1. **FIX-06 & FIX-07 (P1, Backend): Request Input Validation & Structured 422 Errors**
   - **Current State**: `/v1/chat/completions` accepts requests without checking parameters.
   - **Required Action**: Validate that `messages` is non-empty, each message has a non-empty `role` and `content`, `temperature` $\in [0.0, 2.0]$, `top_p` $\in [0.0, 1.0]$, and `max_tokens` $\in [1, 32768]$. Return standard OpenAI-compatible HTTP 422 JSON error if invalid:
     ```json
     {"error": {"message": "temperature must be between 0.0 and 2.0", "type": "invalid_request_error", "code": 422}}
     ```
   - **Effort**: Easy (~30 mins) • **Impact**: High

2. **FIX-11 (P1, Security): Restrict CORS from `*` to Local Origins**
   - **Current State**: In `server.rs`, Axum uses `.layer(CorsLayer::permissive())`, allowing any malicious site in the user's browser to send requests to `localhost:8080`.
   - **Required Action**: Restrict CORS headers to `http://127.0.0.1:7860` and `http://localhost:7860`.
   - **Effort**: Very Easy (~10 mins) • **Impact**: High

3. **FIX-19 (P2, Frontend): "Stop Generating" Abort Button**
   - **Current State**: Once a response starts streaming, the user cannot cancel it without refreshing the page.
   - **Required Action**: Attach an `AbortController` to the fetch/SSE stream in `static/js/app.js` and wire up the UI Stop button to trigger `abort()`.
   - **Effort**: Easy (~20 mins) • **Impact**: Medium

4. **FIX-24 (P2, Performance): Context Window / VRAM Boundary Guard**
   - **Current State**: Prompts exceeding model context (`n_ctx = 4096`) are not rejected upfront.
   - **Required Action**: Validate `prompt_tokens.len() + max_tokens <= n_ctx`. If exceeded, return an explicit error: `"Context window of 4096 tokens exceeded"`.
   - **Effort**: Easy (~15 mins) • **Impact**: Medium

5. **FIX-08 (P1, Security): Rate Limiting Middleware**
   - **Current State**: No rate limiter on Axum routes.
   - **Required Action**: Attach `tower::limit::RateLimitLayer` (e.g. 60 req/min for health, 15 req/min for completions).
   - **Effort**: Easy (~30 mins) • **Impact**: Medium

---

### Tier 2: Essential Documentation & Presentation Assets (Quick Wins)

These items require no complex code changes, take minimal time, and significantly boost the professional polish of the repository:

6. **FIX-14 / FIX-02 (P1, Demo): `DEMO_SCRIPT.md`**
   - **Description**: A comprehensive presenter's battlecard containing:
     - Exact PowerShell commands to run on Laptop A (Coordinator) and Laptop B (Worker).
     - Expected console outputs.
     - 3 key talking points per screen (zero-weight transfer, 75% compression, P2P Tailscale mesh).
     - Instant fallback command (`.\start.ps1 all` for single-machine loopback) if network fails.
   - **Effort**: Easy (~20 mins) • **Impact**: High

7. **FIX-17 (P2, Documentation): `API_REFERENCE.md`**
   - **Description**: Standalone API documentation containing curl examples, request/response JSON schemas, SSE event formats, and error codes for `/v1/chat/completions`, `/v1/models`, `/health`, and `/api/cluster/status`.
   - **Effort**: Easy (~20 mins) • **Impact**: Medium

8. **FIX-18 (P2, Documentation): `LIMITATIONS.md`**
   - **Description**: Proactive disclosure of architectural boundaries (preempts reviewer critique):
     - 2-node maximum in current pipeline design.
     - Windows-only Win32 Job Object requirement.
     - Both nodes require local GGUF model files.
     - Ephemeral conversation history (no database persistence).
   - **Effort**: Very Easy (~15 mins) • **Impact**: Medium

9. **FIX-25 & FIX-27 (P3, Legal/Community): `LICENSE` & `CONTRIBUTING.md`**
   - **Description**: Add standard Apache-2.0 / MIT `LICENSE` file in root and concise `CONTRIBUTING.md` with build steps and code style guidelines.
   - **Effort**: Very Easy (~10 mins) • **Impact**: Medium

10. **FIX-22 (P2, DevOps): CI/CD Pipeline (`.github/workflows/ci.yml`)**
    - **Description**: Automated GitHub Actions workflow running `cargo test --workspace` and `cargo check --workspace --release` on every pull request and push.
    - **Effort**: Easy (~15 mins) • **Impact**: Medium

11. **FIX-04 (P0, Documentation): UI Screenshots & Visual Proof**
    - **Description**: Capture 3–5 clean screenshots of the claymorphic web UI (chat state, model loading, telemetry HUD modal, streaming response) and embed them into `README.md`.
    - **Effort**: Easy (~20 mins) • **Impact**: High

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
| **P0 (Critical / Blockers)** | 5 | 3 | 2 (Demo script, Screenshots) | 0 |
| **P1 (High Priority)** | 10 | 2 | 8 (Validation, CORS, Rate limit, etc.) | 0 |
| **P2 (Medium Priority)** | 9 | 1 | 4 (API doc, Limitations, Stop btn, CI) | 4 |
| **P3 (Low Priority)** | 6 | 0 | 2 (License, Contributing) | 4 |
| **Engine Crash Fixes** | 2 | 2 | 0 | 0 |
| **Total** | **32** | **10** | **14** | **8** |

---

## 5. Recommended Execution Roadmap

To get the project submission-ready in the shortest time:

1. **Step 1 (Code & Security)**:
   - Implement `FIX-06` (Request validation on `/v1/chat/completions`) + `FIX-11` (CORS restriction) in `server.rs`.
   - Implement `FIX-19` ("Stop Generating" abort button) in `app.js`.
2. **Step 2 (Documentation Suite)**:
   - Generate `DEMO_SCRIPT.md` (`FIX-14` / `FIX-02`).
   - Generate `API_REFERENCE.md` (`FIX-17`) and `LIMITATIONS.md` (`FIX-18`).
   - Add `LICENSE` (`FIX-25`) and `CONTRIBUTING.md` (`FIX-27`).
3. **Step 3 (DevOps & CI)**:
   - Add `.github/workflows/ci.yml` (`FIX-22`).
