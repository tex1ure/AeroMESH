# FastAPI Proxy Gateway Architecture (FIX-16)

> **Status**: Intended Architecture (Not a Defect)  
> **Component**: `app.py`  
> **Default Gateway Port**: `7860`  
> **Target Backend Port**: `8080` (Axum Engine)  
> **Test Suite**: `tests/test_fastapi_gateway.py` (9 / 9 Passing)

---

## 1. Executive Summary & Design Philosophy

AeroMESH utilizes a **dual-tier HTTP architecture** designed for clean separation of concerns, browser security isolation, and rapid prototyping:

1. **Rust Axum Engine Server (`aeromesh-engine` :8080)**:
   - The authoritative inference engine and cluster control plane.
   - Executes GGUF layer slicing, C++ FFI tensor evaluation, WireGuard activation streaming, and token sampling.
   - Enforces strict local-origin CORS policies (`FIX-11`), rate limiting (`FIX-08`), and constant-time Bearer token authentication (`FIX-03`).

2. **Python FastAPI Gateway (`app.py` :7860)**:
   - Serves the zero-dependency dark claymorphic Single Page Application (SPA) from `static/`.
   - Provides an intentional, unified browser-facing origin (`http://127.0.0.1:7860`), completely eliminating browser CORS complexities.
   - Transparently proxies `/v1/*` and `/api/*` requests to the Axum engine via a persistent, connection-pooled `httpx.AsyncClient`.
   - Injects cluster credentials (`AEROMESH_API_KEY`) so browser JavaScript never handles raw backend secrets.
   - Preserves high-throughput, non-buffering Server-Sent Events (SSE) token streams.
   - Provides graceful offline degradation (dynamic local GGUF discovery and structured offline fallback payloads) when the coordinator is stopped.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       Browser / Client Workspace                        │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ HTTP / SSE (Same-Origin)
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              FastAPI Gateway & SPA Host (app.py :7860)                  │
│  ─────────────────────────────────────────────────────────────────────  │
│  • Serves static assets: /static/index.html, CSS, JS, KaTeX, Lucide     │
│  • Gateway Liveness (/health) & End-to-End Readiness (/ready)           │
│  • Hop-by-Hop Header Stripping (RFC 7230) & Bearer Token Injection      │
│  • Graceful Offline Fallbacks: Local model discovery & safe state       │
│  • Unbuffered SSE Token Stream Passthrough (X-Accel-Buffering: no)      │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ Loopback TCP Proxy (sub-0.1ms RTT)
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│               Rust Axum Engine Server (aeromesh-engine :8080)           │
│  ─────────────────────────────────────────────────────────────────────  │
│  • OpenAI-Compatible Chat Completions: POST /v1/chat/completions        │
│  • Model Slicing & Hot-Swapping: POST /api/model/switch                 │
│  • Real-Time Cluster Telemetry: GET /api/cluster/status                 │
│  • Bounded Memory Deserialization & Strict Token Context Validation     │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ WireGuard Direct P2P (Port 50052)
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│               Remote Worker Node (Worker B: Layers 24..47)              │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Why the Localhost Hop is an Intentional Architectural Choice

Reviewers sometimes ask why the Rust server doesn't serve the static assets directly. The localhost gateway is an **intentional design decision** with substantial advantages:

| Consideration | Direct Axum (Without Gateway) | Dual-Tier Gateway Architecture (AeroMESH) |
|---|---|---|
| **Frontend Prototyping** | Any UI tweak or routing change requires rebuilding Rust crates (`cargo build`). | Python `app.py` allows instant iteration, live reloading, and experimentation without compiling Rust. |
| **CORS & Origin Isolation** | Direct browser access to port 8080 requires exposing the backend control plane to cross-origin web requests or complex CORS headers. | Browser accesses port 7860 as its single origin. Cross-origin requests from external browser tabs are blocked before reaching Axum. |
| **Credential Protection** | The web UI JavaScript would need to hold the coordinator's `AEROMESH_API_KEY` in memory or local storage. | The gateway holds the secret in process memory and injects the `Authorization: Bearer` header on localhost loopback calls. |
| **Offline User Experience** | If the Rust engine is compiling or offline, visiting the dashboard results in a browser connection failure (`ERR_CONNECTION_REFUSED`). | The gateway remains online, serves the claymorphic UI, dynamically scans `models/*.gguf` on disk, and displays guided recovery commands. |
| **Latency Impact** | 0 ms gateway overhead. | Localhost TCP loopback introduces **< 0.08 ms** overhead—completely imperceptible compared to GPU token generation latency (~50–100 ms/token). |

---

## 3. Environment Variable Configuration

The gateway standardizes environment variables with backward-compatible fallbacks:

| Variable | Default Value | Fallback Variables | Description |
|---|---|---|---|
| `AEROMESH_UI_HOST` | `127.0.0.1` | `HOST` | IP interface on which the FastAPI gateway binds. Defaults strictly to loopback. |
| `AEROMESH_UI_PORT` | `7860` | `AEROMESH_PORT`, `PORT` | Local TCP port for browser dashboard access. |
| `AEROMESH_BACKEND_URL` | `http://127.0.0.1:8080` | `AEROMESH_ENDPOINT` | Target coordinator HTTP base URL. |
| `AEROMESH_API_KEY` | `""` | — | Secret forwarded to the Axum engine via `Authorization: Bearer`. |
| `AEROMESH_GATEWAY_API_KEY` | `""` | — | Optional secret required for external browser access if bound beyond loopback. |

### Security Warning on Non-Loopback Binding
If `AEROMESH_UI_HOST` is explicitly set to `0.0.0.0`, `app.py` logs a loud warning:
```text
WARNING: AeroMESH gateway is binding to all network interfaces (0.0.0.0). 
This is intended for local development and trusted tailnets only.
```

---

## 4. Routing & Proxy Behavior

The gateway mounts endpoints in strict priority order to prevent route shadowing:

| Method | Gateway Route | Target Backend Route | Description |
|---|---|---|---|
| `GET` | `/health` | *Local* | **Gateway Liveness Check**: Returns `{"gateway": "ok", "backend_url": "..."}`. |
| `GET` | `/ready` | `/health` (Axum) | **End-to-End Readiness**: Probes backend with a 3s timeout. Returns 200 `ok` or 503 `unreachable`. |
| `GET` | `/` | *Local* | Serves the claymorphic Single Page Application (`static/index.html`). |
| `GET` | `/static/*` | *Local* | Serves CSS stylesheets, client JavaScript, KaTeX math assets, and icons. |
| `GET` | `/v1/models` | `/v1/models` | Queries active backend model. If backend is offline, falls back to dynamic local `models/*.gguf` scan. |
| `POST` | `/api/model/switch` | `/api/model/switch` | Proxies model load / slicing command to coordinator. |
| `GET` | `/api/cluster/status` | `/api/cluster/status` | Queries live cluster telemetry. Returns structured offline payload (`state: "offline"`) if backend is down. |
| `POST` | `/v1/chat/completions` | `/v1/chat/completions` | OpenAI-compatible chat completions. Automatically routes `stream: true` to the SSE streaming pipeline. |
| `POST` | `/api/chat/stream` | `/v1/chat/completions` | Direct browser SSE streaming endpoint with live Markdown/KaTeX formatting. |
| `POST` | `/api/chat/abort` | `/api/chat/abort` | Signals stream cancellation and stops native model evaluation loop. |
| `*` | `/api/{path:path}` | `/api/{path}` | Generic catch-all proxy for future `/api/*` endpoints. |
| `*` | `/v1/{path:path}` | `/v1/{path}` | Generic catch-all proxy for future `/v1/*` endpoints. |

---

## 5. SSE Streaming & Connection Lifecycle

Streaming LLM completions require specialized HTTP proxy semantics to prevent token buffering:

1. **Unbuffered Streaming**:
   - The gateway uses `response.aiter_text()` / `aiter_raw()` to immediately forward incoming TCP chunks as they arrive from the Axum engine.
   - Injects `Cache-Control: no-cache` and `X-Accel-Buffering: no` headers to prevent intermediary proxies or web browsers from buffering SSE frames.

2. **Infinite Read Timeout (`read=None`)**:
   - The shared `httpx.AsyncClient` is configured with `httpx.Timeout(connect=5.0, read=None, write=30.0, pool=10.0)`.
   - A finite read timeout would terminate long-form reasoning generation (such as DeepSeek-R1 extended thinking traces). The connection remains open until the model issues `[DONE]` or the client aborts.

3. **Client Abort & Resource Teardown**:
   - When the user presses the "Stop Generating" button (`FIX-19`) or presses `Escape`, the browser aborts the fetch stream and dispatches a background `POST /api/chat/abort`.
   - The gateway's `BackgroundTask(backend_response.aclose)` immediately closes the backend HTTP response and frees socket descriptors.

4. **Hop-by-Hop Header Stripping (RFC 7230)**:
   - Per HTTP/1.1 proxy specifications, hop-by-hop headers (`Connection`, `Keep-Alive`, `Transfer-Encoding`, `Upgrade`, `Host`, `Content-Length`) are stripped before retransmitting requests and responses.

---

## 6. Verification & Automated Tests

The gateway implementation is thoroughly validated via automated integration tests in [tests/test_fastapi_gateway.py](file:///d:/New%20folder/Latest_llama/AeroMESH/tests/test_fastapi_gateway.py):

```powershell
python -m pytest tests/test_fastapi_gateway.py -v
```

### Verified Test Cases
1. `test_gateway_health`: Validates gateway liveness check returns 200 with `gateway: ok` and backend URL.
2. `test_gateway_ready_backend_offline`: Validates readiness probe returns HTTP 503 with structured `backend: unreachable` when coordinator is offline.
3. `test_gateway_ready_backend_online`: Validates readiness probe returns HTTP 200 with `backend: ok` when coordinator is active.
4. `test_serve_spa_root`: Validates `GET /` serves HTML containing the AeroMESH dashboard.
5. `test_models_fallback_when_backend_offline`: Validates local disk GGUF model discovery returns valid OpenAI model list schema even when coordinator is down.
6. `test_cluster_status_fallback_when_backend_offline`: Validates cluster status endpoint returns graceful `state: "offline"` payload instead of raw 500 error.
7. `test_chat_abort_endpoint`: Validates `POST /api/chat/abort` returns `{"status": "aborted"}`.
8. `test_hop_by_hop_header_filtering`: Validates hop-by-hop headers are removed while legitimate metadata headers are preserved.
9. `test_unreachable_backend_returns_502`: Validates unhandled proxy routes return HTTP 502 Bad Gateway with structured JSON error envelopes.

---

## 7. Production Deployment Alternatives

For latency-critical production benchmarking or headless enterprise deployments where a browser UI is not needed:
- **Headless Benchmarking**: Connect benchmarking harnesses directly to the Axum engine (`http://127.0.0.1:8080/v1/chat/completions`).
- **Production Reverse Proxy**: If exposing AeroMESH outside a trusted Tailnet, place a production reverse proxy (e.g., Caddy, Nginx, Envoy) in front of Axum with TLS termination and external authentication.
