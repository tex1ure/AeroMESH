# FastAPI Proxy Gateway Architecture (FIX-16)

> **Status**: Production Architecture  
> **Component**: `app.py`  
> **Default Gateway Port**: `7860`  
> **Target Backend Port**: `8080` (Axum Engine)  
> **Test Suite**: `tests/test_fastapi_gateway.py` (27 / 27 Passing)

---

## Table of Contents

- [1. Executive Summary & Design Philosophy](#1-executive-summary--design-philosophy)
- [2. Why the Localhost Hop is an Intentional Architectural Choice](#2-why-the-localhost-hop-is-an-intentional-architectural-choice)
- [3. Environment Variable Configuration](#3-environment-variable-configuration)
- [4. Complete Routing & Proxy Behavior](#4-complete-routing--proxy-behavior)
- [5. SSE Streaming & Connection Lifecycle](#5-sse-streaming--connection-lifecycle)
- [6. Graceful Offline Degradation & Local Model Discovery](#6-graceful-offline-degradation--local-model-discovery)
- [7. Procedural DiT / Stable Diffusion Pipeline](#7-procedural-dit--stable-diffusion-pipeline)
- [8. Automated Test Suite (27 Test Cases)](#8-automated-test-suite-27-test-cases)
- [9. Production Deployment Alternatives](#9-production-deployment-alternatives)

---

## 1. Executive Summary & Design Philosophy

AeroMESH utilizes an intentional **dual-tier HTTP architecture** designed for clean separation of concerns, browser security isolation, and rapid UI development:

1. **Rust Axum Engine Server (`aeromesh-engine` :8080)**:
   - The authoritative inference engine and cluster control plane.
   - Executes GGUF layer slicing, native C++ FFI tensor evaluation, WireGuard activation streaming, and token sampling.
   - Enforces strict local-origin CORS policies, rate limiting, and constant-time Bearer token authentication.

2. **Python FastAPI Gateway (`app.py` :7860)**:
   - Serves the zero-dependency dark claymorphic Single Page Application (SPA) from `static/`.
   - Provides an intentional, unified browser-facing origin (`http://127.0.0.1:7860`), eliminating browser cross-origin complexities.
   - Transparently proxies `/v1/*` and `/api/*` requests to the Axum engine via a persistent, connection-pooled `httpx.AsyncClient`.
   - Injects cluster credentials (`AEROMESH_API_KEY`) so browser JavaScript never handles raw backend secrets.
   - Preserves high-throughput, non-buffering Server-Sent Events (SSE) token streams (`X-Accel-Buffering: no`).
   - Delivers graceful offline degradation: dynamically scans local `models/*.gguf` on disk and provides structured fallback states when the coordinator is offline.

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
│  • Request Size Ceiling: MAX_REQUEST_BYTES = 8 MB                       │
│  • Graceful Offline Fallbacks: Dynamic local model scan & safe HUD      │
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
                                     │ Tailscale Direct WireGuard (:50052)
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│               Remote Worker Node (Worker B: Layers 23..47)              │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Why the Localhost Hop is an Intentional Architectural Choice

Developers occasionally ask why the Rust server does not serve the static assets directly. The localhost gateway is an **intentional design decision** providing concrete engineering advantages:

| Consideration | Direct Axum (Without Gateway) | Dual-Tier Gateway Architecture (AeroMESH) |
|---|---|---|
| **Frontend Prototyping** | Any UI tweak or routing change requires rebuilding Rust crates (`cargo build`). | Python `app.py` allows instant iteration, live reloading, and styling adjustments without compiling Rust. |
| **CORS & Origin Isolation** | Direct browser access to port 8080 requires exposing the backend control plane to cross-origin web requests or complex CORS headers. | Browser accesses port 7860 as its single origin. Cross-origin requests from external browser tabs are blocked before reaching Axum. |
| **Credential Protection** | The web UI JavaScript would need to hold the coordinator's `AEROMESH_API_KEY` in memory or local storage. | The gateway holds the secret in process memory and injects the `Authorization: Bearer` header on localhost loopback calls. |
| **Offline User Experience** | If the Rust engine is compiling or offline, visiting the dashboard results in a browser connection failure (`ERR_CONNECTION_REFUSED`). | The gateway remains online, serves the claymorphic UI, dynamically scans `models/*.gguf` on disk, and displays guided recovery commands. |
| **Latency Impact** | 0 ms gateway overhead. | Localhost TCP loopback introduces **< 0.08 ms** overhead—completely imperceptible compared to GPU token generation latency (~40–100 ms/token). |

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
| `AEROMESH_MAX_REQUEST_BYTES` | `8388608` (8 MB) | — | Maximum request body size accepted before returning 413 / 422. |

### Security Warning on Non-Loopback Binding
If `AEROMESH_UI_HOST` is explicitly set beyond loopback, `app.py` logs a loud warning and requires `AEROMESH_GATEWAY_API_KEY` for all external callers:
```text
WARNING: AeroMESH gateway is binding to '0.0.0.0', which is not loopback.
Requests from non-loopback clients require AEROMESH_GATEWAY_API_KEY.
This is intended for trusted tailnets only.
```

---

## 4. Complete Routing & Proxy Behavior

The gateway mounts endpoints in strict priority order to prevent route shadowing:

| Method | Gateway Route | Target Backend Route | Description |
|---|---|---|---|
| `GET` | `/health` | *Local* | **Gateway Liveness Check**: Returns `{"gateway": "ok", "backend_url": "..."}`. |
| `GET` | `/ready` | `/health` (Axum) | **End-to-End Readiness**: Probes backend with a 3s timeout. Returns 200 `ok` or 503 `unreachable`. |
| `GET` | `/` | *Local* | Serves the claymorphic Single Page Application (`static/index.html`). |
| `GET` | `/static/*` | *Local* | Serves CSS stylesheets, client JavaScript, KaTeX math assets, and vector icons. |
| `GET` | `/v1/models` | `/v1/models` | Queries active backend model. If backend is offline, falls back to dynamic local `models/*.gguf` scan. |
| `POST` | `/api/model/switch` | `/api/model/switch` | Proxies model load / slicing command to coordinator. |
| `POST` | `/v1/models/load` | `/api/model/switch` | Alias route for model hot-swapping. |
| `GET` | `/api/cluster/status` | `/api/cluster/status` | Queries live cluster telemetry. Returns structured offline payload (`state: "offline"`) if backend is down. |
| `GET` | `/api/cluster/discover` | *Local* | Discovers reachable mesh nodes on local Tailnet. |
| `GET` | `/api/node/state` | *Local* | Returns local node background process state. |
| `POST` | `/api/node/start` | *Local* | Starts a local worker or coordinator daemon. |
| `POST` | `/api/node/stop` | *Local* | Halts local child node processes. |
| `POST` | `/v1/chat/completions` | `/v1/chat/completions` | OpenAI-compatible chat completions. Automatically routes `stream: true` to SSE streaming pipeline. |
| `POST` | `/api/chat` | `/v1/chat/completions` | Alias endpoint for chat completions. |
| `POST` | `/api/chat/stream` | `/v1/chat/completions` | Direct browser SSE streaming endpoint with live Markdown/KaTeX formatting. |
| `POST` | `/api/chat/abort` | `/api/chat/abort` | Signals stream cancellation and stops native model evaluation loop. |
| `POST` | `/v1/images/generations` | *Local* | Procedural DiT / Stable Diffusion image generation pipeline. |
| `POST` | `/api/image/stream` | *Local* | SSE streaming diffusion generation with mesh prompt conditioning. |
| `*` | `/api/{path:path}` | `/api/{path}` | Generic catch-all proxy for auxiliary `/api/*` endpoints. |
| `*` | `/v1/{path:path}` | `/v1/{path}` | Generic catch-all proxy for auxiliary `/v1/*` endpoints. |

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
   - When the user presses the "Stop Generating" button or presses `Escape`, the browser aborts the fetch stream and dispatches a background `POST /api/chat/abort`.
   - The gateway's `BackgroundTask(backend_response.aclose)` immediately closes the backend HTTP response and frees socket descriptors.

4. **Hop-by-Hop Header Stripping (RFC 7230)**:
   - Per HTTP/1.1 proxy specifications, hop-by-hop headers (`Connection`, `Keep-Alive`, `Transfer-Encoding`, `Upgrade`, `Host`, `Content-Length`) are stripped before retransmitting requests and responses.

---

## 6. Graceful Offline Degradation & Local Model Discovery

When the coordinator backend is offline (e.g., during recompilation or node restart), the gateway remains responsive:
- **`GET /v1/models`**: Instead of throwing `502 Bad Gateway`, the gateway scans `models/*.gguf` on disk, reads file metadata, and returns a valid OpenAI model list indicating `is_active: false`.
- **`GET /api/cluster/status`**: Emits a structured telemetry object (`state: "offline"`, `connected: false`) enabling the UI HUD to display helpful troubleshooting tips rather than a blank screen.

---

## 7. Procedural DiT / Stable Diffusion Pipeline

AeroMESH features an integrated diffusion generation pipeline (`POST /v1/images/generations` and `POST /api/image/stream`):
- Bridges distributed prompt expansion across the LLM mesh with local procedural image synthesis.
- Streams multi-stage progress events (`dit_mesh_conditioning`, `dit_generation_progress`, `dit_output`) directly to the Single Page Application.

---

## 8. Automated Test Suite (27 Test Cases)

The gateway implementation is thoroughly validated via 27 automated integration tests in [tests/test_fastapi_gateway.py](../../tests/test_fastapi_gateway.py):

```powershell
# Run the complete test suite
.\.venv\Scripts\pytest.exe tests/test_fastapi_gateway.py -v
```

### Key Test Categories Verified:
- **Gateway Health & Readiness**: Liveness (`/health`) and end-to-end backend probing (`/ready`) when online and offline.
- **Header Sanitization**: Removal of hop-by-hop headers and secure injection of upstream Bearer credentials.
- **Loopback Enforcement & Auth**: Verification that loopback clients succeed and external unauthenticated requests fail closed with 401.
- **Offline Model & Cluster Fallbacks**: Preservation of UI integrity when the backend engine is stopped.
- **Chat & Abort Endpoints**: Validation of request proxies, streaming options, and cancellation triggers.
- **Diffusion Pipeline Routes**: Validation of image generation schemas, prompt checking, and streaming events.

---

## 9. Production Deployment Alternatives

For latency-critical production benchmarking or headless enterprise deployments:
- **Direct Backend Benchmarking**: Connect high-throughput benchmarking harnesses directly to the Axum engine (`http://127.0.0.1:8080/v1/chat/completions`).
- **Production Reverse Proxy**: When exposing AeroMESH beyond a trusted Tailnet, place a production reverse proxy (e.g., Caddy, Nginx, Cloudflare Tunnel) in front of the gateway with TLS termination.
