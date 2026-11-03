# AeroMESH HTTP API & WebSocket/SSE Reference

> **Base URLs**
> - **Direct Coordinator Engine (Axum)**: `http://127.0.0.1:8080`
> - **Web Gateway Proxy (FastAPI)**: `http://127.0.0.1:7860` (proxies to `:8080`, injects bearer credentials securely, hosts SPA)
> - **Native Desktop Studio**: `http://127.0.0.1:7865` (`AeroMesh-Desktop.exe`)
>
> **Authentication**: All endpoints except `/health` and static assets require `Authorization: Bearer <AEROMESH_API_KEY>`.
> Token comparison uses SHA-256 constant-time evaluation (`constant_time_eq_str`) to eliminate execution-time side-channel attacks.
> If `AEROMESH_API_KEY` is not configured, protected routes fail closed with HTTP `401 Unauthorized`.

---

## Table of Contents

- [1. Unified Error Envelope (OpenAI-Compatible)](#1-unified-error-envelope-openai-compatible)
- [2. POST /v1/chat/completions & /api/chat/stream](#2-post-v1chatcompletions--apichatstream)
- [3. POST /api/chat/abort](#3-post-apichatabort)
- [4. GET /v1/models](#4-get-v1models)
- [5. POST /api/model/switch (alias /v1/models/load)](#5-post-apimodelswitch-alias-v1modelsload)
- [6. GET /health & GET /ready (Health Probes)](#6-get-health--get-ready-health-probes)
- [7. GET /api/cluster/status & /api/cluster/discover](#7-get-apiclusterstatus--apiclusterdiscover)
- [8. Node Lifecycle Management (/api/node/*)](#8-node-lifecycle-management-apinodestart-stop-state)
- [9. POST /v1/images/generations & /api/image/stream (Procedural DiT / SD)](#9-post-v1imagesgenerations--apiimagestream)
- [10. Gateway Reverse Proxy & Header Stripping Architecture](#10-gateway-reverse-proxy--header-stripping-architecture)

---

## 1. Unified Error Envelope (OpenAI-Compatible)

Pre-flight validation errors and runtime failures on all `/v1/*` and `/api/*` endpoints strictly follow OpenAI's error envelope:

```json
{
  "error": {
    "message": "Human-readable description of error condition",
    "type": "invalid_request_error",
    "param": "temperature",
    "code": "invalid_value"
  }
}
```

| HTTP Status | `error.type` | `error.code` | Condition |
|:---:|:---|:---|:---|
| **400** | `invalid_request_error` | `invalid_json` | Malformed JSON request body or payload framing error |
| **401** | `authentication_error` | `invalid_api_key` | Missing, malformed, or invalid Bearer token |
| **404** | `invalid_request_error` | `not_found` | Requested `.gguf` file not found in local `models/` directory |
| **422** | `invalid_request_error` | `missing_required_field` | `messages` array is empty or missing required prompt |
| **422** | `invalid_request_error` | `invalid_value` | Unsupported role, out-of-bounds parameters, or whitespace-only model |
| **422** | `invalid_request_error` | `empty_content` | Message contains empty or whitespace-only content |
| **422** | `invalid_request_error` | `context_window_exceeded` | `prompt_tokens + max_tokens` exceeds model context window `n_ctx` |
| **429** | `rate_limit_error` | `rate_limit_exceeded` | Rate limit breached on `/health` (60 req/min) or completions (15 req/min) |
| **500** | `api_error` | `null` | Engine runtime or native C++ FFI evaluation failure |
| **502** | `api_error` | `backend_unreachable` | FastAPI gateway unable to reach upstream Axum coordinator |
| **503** | `model_unavailable` | `model_not_ready` | Cluster state is currently Loading, Failed, or Closed |

---

## 2. POST /v1/chat/completions & /api/chat/stream

Primary chat generation endpoint. Routes `/api/chat` and `/api/chat/stream` point to this identical handler.

**Headers**:
- `Content-Type: application/json`
- `Authorization: Bearer <key>`

### Request Schema
```json
{
  "model": "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
  "messages": [{ "role": "system", "content": "You are a helpful assistant." },
    { "role": "user", "content": "Explain zero-weight pipelined inference." }
  ],
  "max_tokens": 512,
  "temperature": 0.6,
  "top_p": 0.9,
  "stream": true,
  "session_id": 1726400000000
}
```

### Parameter Validation Rules
- **`messages`**: Array of `ChatMessage` objects. Must contain at least one item.
  - `role`: Must be one of `"system"`, `"user"`, `"assistant"`, `"tool"`, `"function"`.
  - `content`: Non-empty string (whitespace-only rejected with 422).
- **`temperature`**: Finite float where `0.0 <= temperature <= 2.0` (Default: `0.7`).
- **`top_p`**: Finite float where `0.0 <= top_p <= 1.0` (Default: `0.9`).
- **`max_tokens`**: Integer where `1 <= max_tokens <= 32768` and `prompt_tokens + max_tokens <= n_ctx` (Default: `256`).
- **`model`**: Optional string. If specified and different from the active model, triggers a transactional model hot-swap if the target `.gguf` exists in `models/`.
- **`stream`**: Boolean flag indicating unary JSON response or SSE token streaming (Default: `true`).
- **`session_id`**: Optional 64-bit integer monotonically identifying the conversation turn.

### Non-Streaming Response (`"stream": false`)
```json
{
  "id": "chatcmpl-1726400000000",
  "object": "chat.completion",
  "created": 1726400000,
  "model": "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
  "choices": [{
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Zero-weight pipelined inference aggregates distributed GPUs by keeping weights strictly local..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 18,
    "completion_tokens": 84,
    "total_tokens": 102
  }
}
```

### Streaming Response (`"stream": true`, `text/event-stream`)
Intermediate tokens are emitted via an internal `Utf8StreamAccumulator` ensuring multi-byte UTF-8 codepoints (such as emojis or complex symbols) are never split across SSE chunk boundaries.

```text
data: {"id":"chatcmpl-1726400000000","object":"chat.completion.chunk","created":1726400000,"model":"DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf","choices":[{"index":0,"delta":{"content":"Zero"}}]}

data: {"id":"chatcmpl-1726400000000","object":"chat.completion.chunk","created":1726400000,"model":"DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf","choices":[{"index":0,"delta":{"content":"-weight"}}]}

data: {"id":"chatcmpl-1726400000000","object":"chat.completion.chunk","created":1726400000,"model":"DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf","choices":[{"index":0,"delta":{"content":" pipelined"}}]}

data: [DONE]
```

---

## 3. POST /api/chat/abort

Atomically halts active token generation across all cluster nodes and flushes transient context KV states.

**Headers**:
- `Authorization: Bearer <key>`

### Request
```powershell
curl.exe -s -X POST http://127.0.0.1:8080/api/chat/abort `
  -H "Authorization: Bearer $env:AEROMESH_API_KEY"
```

### Response (200 OK)
```json
{
  "status": "aborted",
  "message": "Active generation halted successfully"
}
```

---

## 4. GET /v1/models

Lists all monolithic base `.gguf` model files located in the `models/` directory. Internal ephemeral stage slice files (such as `.aeromesh_ephemeral_*`) are automatically filtered out.

```powershell
curl.exe -s http://127.0.0.1:8080/v1/models -H "Authorization: Bearer $env:AEROMESH_API_KEY"
```

### Response Format
```json
{
  "object": "list",
  "data": [{
      "id": "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
      "object": "model",
      "created": 1726400000,
      "owned_by": "local_disk",
      "is_active": true
    },
    {
      "id": "test.gguf",
      "object": "model",
      "created": 1726400000,
      "owned_by": "local_disk",
      "is_active": false
    }
  ]
}
```

---

## 5. POST /api/model/switch (alias /v1/models/load)

Executes transactional model switching across the cluster. If loading the target model fails, the previous model context remains active without memory deallocation or dangling native C++ FFI pointers.

### Request Body
```json
{
  "model": "test.gguf"
}
```

### Success Response (200 OK)
```json
{
  "status": "ok",
  "active_model": "test.gguf",
  "total_layers": 28,
  "hidden_dim": 3072
}
```

---

## 6. GET /health & GET /ready (Health Probes)

High-frequency readiness and liveness probes for monitoring and load balancers. Rate-limited to 60 requests per minute.

```powershell
# Probe gateway liveness
curl.exe -si http://127.0.0.1:7860/health

# Probe end-to-end readiness
curl.exe -si http://127.0.0.1:7860/ready
```

| Route | Upstream Condition | Status Code | Returned Body |
|:---|:---|:---:|:---|
| `/health` | Gateway process running | **200** | `{"status": "ok", "state": "online_idle"}` |
| `/ready` | Axum coordinator online | **200** | `{"status": "ready", "backend": "ok"}` |
| `/ready` | Axum coordinator stopped | **503** | `{"status": "degraded", "backend": "unreachable"}` |

---

## 7. GET /api/cluster/status & /api/cluster/discover

Returns active cluster topology, Direct WireGuard tunnel quality, inter-node round-trip latency, and wire payload compression metrics.

```powershell
curl.exe -s http://127.0.0.1:8080/api/cluster/status -H "Authorization: Bearer $env:AEROMESH_API_KEY"
```

### Response Format (Distributed P2P WireGuard Mode)
```json
{
  "state": "online_idle",
  "status": "ok",
  "connected": true,
  "cluster_status": "ONLINE",
  "active_model": "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
  "transport": "Tailscale Direct WireGuard",
  "is_direct_wireguard": true,
  "rtt_ms": 1.14,
  "link_badge": "Direct WireGuard (P2P Mesh)",
  "per_token_kb": "5.05 KB/tok",
  "quantization": "Outlier-Preserving INT8 (Top-8 FP32 + INT8 Row)",
  "coordinator": {
    "model": "models/DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
    "local_layers": "0..=22",
    "total_layers": 48,
    "hidden_dim": 5120,
    "transport": "Tailscale Direct WireGuard"
  },
  "local_stage": {
    "layer_start": 0,
    "layer_end": 22
  },
  "total_layers": 48,
  "hidden_dim": 5120,
  "workers": ["100.101.147.24:50052"],
  "worker_nodes": ["100.101.147.24:50052"],
  "nodes_count": 2
}
```

---

## 8. Node Lifecycle Management (/api/node/start, /stop, /state)

Enables remote orchestrators to inspect and command background node processes.

### GET /api/node/state
Returns local node daemon state, active model, process ID, and port allocations.

### POST /api/node/start
Starts a local background coordinator or worker node process.
```json
{
  "role": "worker",
  "model": "models/test.gguf",
  "port": 50052
}
```

### POST /api/node/stop
Gracefully halts local child node processes.

---

## 9. POST /v1/images/generations & /api/image/stream

Procedural Diffusion Transformer (DiT) / Stable Diffusion image generation pipeline.

**Headers**:
- `Content-Type: application/json`
- `Authorization: Bearer <key>`

### Request Body
```json
{
  "prompt": "Cyberpunk terminal displaying glowing distributed neural network nodes, 8k resolution, cinematic lighting",
  "steps": 4,
  "width": 512,
  "height": 512,
  "stream": true
}
```

### SSE Streaming Events
When `stream: true`, the endpoint streams status events as the distributed mesh conditions the prompt and generates intermediate latents:
1. `event: dit_mesh_conditioning` (Coordinator layer split and prompt enhancement)
2. `event: dit_generation_progress` (Denoising step counter)
3. `event: dit_output` (Final base64 encoded PNG image)

---

## 10. Gateway Reverse Proxy & Header Stripping Architecture

The Python FastAPI gateway (`app.py` :7860) enforces strict HTTP proxy isolation:
- **`MAX_REQUEST_BYTES = 8388608` (8 MB)**: Prevents denial-of-service memory exhaustion on large prompts or images.
- **RFC 7230 Hop-by-Hop Filter**: Strips `Connection`, `Keep-Alive`, `Proxy-Authenticate`, `Proxy-Authorization`, `TE`, `Trailers`, `Transfer-Encoding`, and `Upgrade` headers.
- **Upstream Credential Injection**: Strips any caller-supplied `Authorization` header, injecting the canonical `Authorization: Bearer <AEROMESH_API_KEY>` secret before forwarding to the Axum backend on `:8080`.
- **Unbuffered SSE Passthrough**: Injects `Cache-Control: no-cache` and `X-Accel-Buffering: no` with an infinite read timeout (`httpx.Timeout(read=None)`).
