# AeroMESH HTTP API Reference (FIX-17)

> **Base URLs**
> - **Direct Coordinator (Axum)**: `http://127.0.0.1:8080`
> - **Web Gateway Proxy (FastAPI)**: `http://127.0.0.1:7860` (proxies requests to `:8080` and attaches credentials securely)
>
> **Authentication**: All endpoints except `/health` require `Authorization: Bearer <AEROMESH_API_KEY>`.
> Token comparison uses SHA-256 constant-time evaluation (`constant_time_eq_str`) to eliminate timing side-channels. 
> If `AEROMESH_API_KEY` is not set in the environment, all protected routes return `401 Unauthorized`.

---

## 1. Unified Error Envelope (OpenAI-Compatible)

Pre-flight validation errors and initialization failures on `/v1/chat/completions` use the standardized envelope:

```json
{
  "error": {
    "message": "Human-readable description",
    "type": "invalid_request_error",
    "param": "temperature",
    "code": "invalid_value"
  }
}
```

| HTTP Status | `error.type` | `error.code` | Condition |
|---|---|---|---|
| **400** | `invalid_request_error` | `invalid_json` | Malformed JSON request payload |
| **401** | `authentication_error` | `invalid_api_key` | Missing, malformed, or mismatching Bearer token |
| **404** | `invalid_request_error` | `not_found` | Requested .gguf file not found in `models/` |
| **422** | `invalid_request_error` | `missing_required_field` | `messages` array is empty |
| **422** | `invalid_request_error` | `invalid_value` | Unsupported role, out-of-bounds parameters, or blank model |
| **422** | `invalid_request_error` | `empty_content` | Message contains empty or whitespace-only content |
| **422** | `invalid_request_error` | `context_window_exceeded` | `max_tokens` exceeds model context window `n_ctx` (FIX-24) |
| **429** | `rate_limit_error` | `rate_limit_exceeded` | Request rate limit exceeded on `/health` (60 req/min) or completions pool (15 req/min) (FIX-08) |
| **500** | `api_error` | `null` | Engine runtime or C++ FFI evaluation failure |
| **503** | `model_unavailable` | `model_not_ready` | Pipeline state is Loading, Failed, or Closed |

---

## 2. POST /v1/chat/completions

Primary chat generation endpoint. Routes `/api/chat` and `/api/chat/stream` point to this identical handler.

**Headers**:
- `Content-Type: application/json`
- `Authorization: Bearer <key>`

### Request Schema
```json
{
  "model": "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
  "messages": [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user", "content": "Explain zero-weight pipelined inference." }
  ],
  "max_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.9,
  "stream": true,
  "session_id": 1726400000000
}
```

### Field Constraints & Validation Rules (Returns 422 on breach)
- **`messages`**: Must contain at least one item. Every message requires a valid role (`system`, `user`, `assistant`, `tool`, `function`) and nonblank content.
- **`temperature`**: Finite float where $0.0 \le \text{temperature} \le 2.0$ (Default: 0.7).
- **`top_p`**: Finite float where $0.0 \le \text{top\_p} \le 1.0$ (Default: 0.9).
- **`max_tokens`**: Integer where $1 \le \text{max\_tokens} \le 32768$ and $\text{max\_tokens} \le n\_ctx$ of the active model (Default: 256).
- **`model`**: Optional string. If specified and different from the active model, triggers an automatic transactional hot-swap if present in `models/`.
- **`stream`**: Boolean flag indicating unary response or SSE token streaming (Default: `true`).
- Single-message inputs containing existing ChatML (`<|im_start|>`) or `[INST]` markers are passed directly without re-templating.

### Non-Streaming Response (`"stream": false`)
```json
{
  "id": "chatcmpl-1726400000000",
  "object": "chat.completion",
  "created": 1726400000,
  "model": "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Zero-weight pipelined inference distributes model layers..."
      },
      "finish_reason": "stop"
    }
  ]
}
```

### Streaming Response (`"stream": true`, `text/event-stream`)
Chunks are processed via an internal `Utf8StreamAccumulator` to avoid splitting multi-byte UTF-8 codepoints across boundaries. Fields with `None` values (such as `role` and `finish_reason`) are omitted from intermediate chunk serialization via `#[serde(skip_serializing_if = "Option::is_none")]`. The stream terminates upon channel close (EOF).

```text
data: {"id":"chatcmpl-1726400000000","object":"chat.completion.chunk","created":1726400000,"model":"DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf","choices":[{"index":0,"delta":{"content":"Zero"}}]}

data: {"id":"chatcmpl-1726400000000","object":"chat.completion.chunk","created":1726400000,"model":"DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf","choices":[{"index":0,"delta":{"content":"-weight"}}]}
```

### cURL Examples

#### Unary Non-Streaming Call
```powershell
curl.exe -s http://127.0.0.1:8080/v1/chat/completions `
  -H "Content-Type: application/json" `
  -H "Authorization: Bearer $env:AEROMESH_API_KEY" `
  -d "{\"model\":\"DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf\",\"messages\":[{\"role\":\"user\",\"content\":\"Ping\"}],\"stream\":false}"
```

#### Streaming Call (-N disables output buffering)
```powershell
curl.exe -sN http://127.0.0.1:8080/v1/chat/completions `
  -H "Content-Type: application/json" `
  -H "Authorization: Bearer $env:AEROMESH_API_KEY" `
  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Count 1 to 5\"}],\"stream\":true}"
```

---

## 3. GET /v1/models

Enumerates all `.gguf` files present in the `models/` directory, identifying the active model via `is_active`.

```powershell
curl.exe -s http://127.0.0.1:8080/v1/models -H "Authorization: Bearer $env:AEROMESH_API_KEY"
```

### Response Format
```json
{
  "object": "list",
  "data": [
    {
      "id": "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
      "object": "model",
      "created": 1726400000,
      "owned_by": "local_disk",
      "is_active": true
    },
    {
      "id": "qwen2.5-0.5b-instruct.gguf",
      "object": "model",
      "created": 1726400000,
      "owned_by": "local_disk",
      "is_active": false
    }
  ]
}
```

---

## 4. POST /api/model/switch (alias /v1/models/load)

Executes transactional model switching. If loading the target model fails, the previous model remains bound without pointer deallocation or dangling memory references (CRASH-01).

### Request
```powershell
curl.exe -s -X POST http://127.0.0.1:8080/api/model/switch `
  -H "Content-Type: application/json" `
  -H "Authorization: Bearer $env:AEROMESH_API_KEY" `
  -d "{\"model\":\"qwen2.5-0.5b-instruct.gguf\"}"
```

### Success Response (200 OK)
```json
{
  "status": "ok",
  "active_model": "qwen2.5-0.5b-instruct.gguf",
  "total_layers": 24,
  "hidden_dim": 896
}
```

> **Note**: On failure, this endpoint returns a plain text status message accompanied by HTTP 404 (file missing) or HTTP 500 (load failure).

---

## 5. GET /health (Authentication Exempt)

High-frequency readiness and liveness probe for cluster monitors and load balancers. Rate-limited to 60 requests per minute (FIX-08).

```powershell
curl.exe -si http://127.0.0.1:8080/health
```

| Coordinator Status | HTTP Code | Returned `state` |
|---|---|---|
| `CoordinatorStatus::Ready` | **200** | `online_idle` |
| Active token generation lock held | **200** | `online_generating` |
| `CoordinatorStatus::Loading(_)` | **503** | `model_loading` |
| `CoordinatorStatus::Failed(_)` | **503** | `model_failed` |
| `CoordinatorStatus::Closed` | **503** | `model_closed` |

### Response Format
```json
{
  "status": "ok",
  "state": "online_idle",
  "service": "aeromesh-coordinator"
}
```

---

## 6. GET /api/cluster/status

Returns telemetry, inter-node latency, and tensor shape metadata consumed by the Claymorphic HUD.

```powershell
curl.exe -s http://127.0.0.1:8080/api/cluster/status -H "Authorization: Bearer $env:AEROMESH_API_KEY"
```

### Response Format (Distributed P2P Mode)
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
  "per_token_kb": "5.04 KB/tok",
  "quantization": "Per-Row INT8 Dynamic Scaling (75% Wire Reduction)",
  "coordinator": {
    "model": "models/DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
    "local_layers": "0..=24",
    "total_layers": 49,
    "hidden_dim": 5120,
    "transport": "Tailscale Direct WireGuard"
  },
  "local_stage": {
    "layer_start": 0,
    "layer_end": 24
  },
  "total_layers": 49,
  "hidden_dim": 5120,
  "workers": ["100.101.147.24:50052"],
  "worker_nodes": ["100.101.147.24:50052"],
  "nodes_count": 2
}
```

> **Note**: When operating in standalone single-machine loopback mode, the payload shifts to `"transport": "Intra-Host Shared Memory (SHM)"`, `"link_badge": "SHM Ring Buffer (Loopback)"`, `"rtt_ms": 0.1`, and `"nodes_count": 1`.

---

## 7. Gateway Proxy Architecture (`app.py` :7860)

- **Security Boundary**: The FastAPI application on port 7860 handles static asset hosting and acts as a gateway proxy for all `/v1/*` and `/api/*` endpoints.
- **Credential Injection**: Local browsers accessing the dashboard do not need direct access to `AEROMESH_API_KEY`. The gateway injects the `Authorization: Bearer` header on proxy requests to `:8080`.
- **Cancellation**: The gateway provides an additional endpoint `POST /api/chat/abort` to signal stream termination and forward the abort to the Axum engine.
