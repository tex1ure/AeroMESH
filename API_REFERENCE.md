# AeroMESH HTTP & SSE API Reference

Base URLs:
- **Coordinator Engine (Axum):** `http://127.0.0.1:8080`
- **Web Gateway Proxy (FastAPI):** `http://127.0.0.1:7860` (proxies to `:8080`, injects Bearer credentials, serves the UI, and hosts the local Model Hub and Cluster Pairing APIs)
- **Desktop Studio:** `http://127.0.0.1:7865` (`AeroMesh-Desktop.exe`)

**Authentication:** All coordinator routes except `/health` require `Authorization: Bearer <AEROMESH_API_KEY>`, verified with constant-time SHA-256 comparison (`constant_time_eq_str`). If `AEROMESH_API_KEY` is unset, protected routes fail closed with `401 Unauthorized`. Gateway routes on `:7860` allow loopback callers (`127.0.0.1` / `::1`) directly and require `AEROMESH_GATEWAY_API_KEY` for non-loopback requests.

---

## Contents

1. [Error Format (OpenAI-Compatible)](#1-error-format-openai-compatible)
2. [Chat Completions & SSE Streaming](#2-chat-completions--sse-streaming)
3. [Abort Active Generation](#3-abort-active-generation)
4. [Models List & Hot-Swap](#4-models-list--hot-swap)
5. [Health & Readiness Probes](#5-health--readiness-probes)
6. [Cluster Status & Worker Discovery](#6-cluster-status--worker-discovery)
7. [One-Click Cluster Pairing & Secret Sync](#7-one-click-cluster-pairing--secret-sync)
8. [Model Hub (Discovery, Downloads & Custom Folders)](#8-model-hub-discovery-downloads--custom-folders)
9. [Node Lifecycle Management](#9-node-lifecycle-management)
10. [Distributed Diffusion / Image Generation](#10-distributed-diffusion--image-generation)
11. [Gateway Proxy Behavior](#11-gateway-proxy-behavior)
12. [Developer API Server & Runtime Configuration](#12-developer-api-server--runtime-configuration)

---

## 1. Error Format (OpenAI-Compatible)

Validation and runtime errors across `/v1/*` and `/api/*` use the standard OpenAI error envelope:

```json
{
  "error": {
    "message": "Human-readable error description",
    "type": "invalid_request_error",
    "param": "temperature",
    "code": "invalid_value"
  }
}
```

| Status | `error.type` | `error.code` | Cause |
|:---:|:---|:---|:---|
| **400** | `invalid_request_error` | `invalid_json` | Malformed JSON body or framing error |
| **401** | `authentication_error` | `invalid_api_key` | Missing or invalid Bearer token |
| **404** | `invalid_request_error` | `not_found` | Requested `.gguf` model not found in local library |
| **422** | `invalid_request_error` | `missing_required_field` | Empty `messages` array or missing prompt |
| **422** | `invalid_request_error` | `invalid_value` | Unsupported role, out-of-range parameter, or blank model |
| **422** | `invalid_request_error` | `empty_content` | Message content is empty or whitespace-only |
| **422** | `invalid_request_error` | `context_window_exceeded` | `prompt_tokens + max_tokens` exceeds `n_ctx` |
| **429** | `rate_limit_error` | `rate_limit_exceeded` | Exceeded rate limit (`/health`: 60/min, completions: 15/min) |
| **500** | `api_error` | `null` | Runtime or native `llama.cpp` FFI failure |
| **502** | `api_error` | `backend_unreachable` | Gateway cannot reach the Axum coordinator on `:8080` |
| **503** | `model_unavailable` | `model_not_ready` | Cluster is loading, failed, or shut down |

---

## 2. Chat Completions & SSE Streaming

`POST /v1/chat/completions` (aliases: `POST /api/chat`, `POST /api/chat/stream`)

```json
{
  "model": "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
  "messages": [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user", "content": "Explain zero-weight pipelined inference." }
  ],
  "max_tokens": 512,
  "temperature": 0.6,
  "top_p": 0.9,
  "stream": true,
  "session_id": 1726400000000
}
```

### Validation Rules
- `messages`: non-empty array; `role` must be `system`, `user`, `assistant`, `tool`, or `function`; `content` may be a string, OpenAI content-part array, or `null` when an `assistant` message carries `tool_calls`.
- `temperature`: finite float in `[0.0, 2.0]` (default `0.7`).
- `top_p`: finite float in `[0.0, 1.0]` (default `0.9`).
- `max_tokens`: integer in `[1, 32768]`, and `prompt_tokens + max_tokens <= n_ctx` (default `256`).
- `tools` & `tool_choice`: standard OpenAI function tool definitions (`[{"type":"function","function":{"name":...,"parameters":...}}]`) with `tool_choice` (`"auto"`, `"none"`, `"required"`, or `{"type":"function","function":{"name":...}}`). Extracted function invocations return `finish_reason: "tool_calls"` and `message.tool_calls` (both unary and SSE streaming).
- `response_format`: supports `{"type":"json_object"}` and `{"type":"json_schema","json_schema":{...}}`, unwrapping markdown JSON fences automatically.
- `model`: optional; if different from the active model, triggers a transactional hot-swap.
- `stream`: boolean (default `true`).
- `session_id`: optional 64-bit integer identifying the conversation turn.

### Unary Response (`"stream": false`)
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
        "content": "Zero-weight pipelined inference keeps model weights local on each GPU..."
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

### SSE Stream (`"stream": true`, `text/event-stream`)
Multi-byte UTF-8 sequences are buffered through `Utf8StreamAccumulator` so codepoints (e.g. emojis or CJK characters) are never split across chunks:

```text
data: {"id":"chatcmpl-1726400000000","object":"chat.completion.chunk","created":1726400000,"model":"DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf","choices":[{"index":0,"delta":{"content":"Zero"}}]}

data: {"id":"chatcmpl-1726400000000","object":"chat.completion.chunk","created":1726400000,"model":"DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf","choices":[{"index":0,"delta":{"content":"-weight"}}]}

data: [DONE]
```

---

## 3. Abort Active Generation

`POST /api/chat/abort`

Halts in-flight token generation across all pipeline stages and clears transient KV state.

```powershell
curl.exe -s -X POST http://127.0.0.1:8080/api/chat/abort `
  -H "Authorization: Bearer $env:AEROMESH_API_KEY"
```

```json
{
  "status": "aborted",
  "message": "Active generation halted successfully"
}
```

---

## 4. Models List & Hot-Swap

### `GET /v1/models`
Lists available base `.gguf` files (excluding internal `.aeromesh_ephemeral_*` temporary slices). When accessed through `:7860` while the coordinator is offline, falls back to scanning local and linked model folders.

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
    }
  ]
}
```

### `POST /api/model/switch` (alias `POST /v1/models/load`)
Transactionally switches the active model across the cluster. If loading fails, the previous model stays loaded.

```json
// Request
{ "model": "Qwen2.5-7B-Instruct-Q4_K_M.gguf" }

// Response (200 OK)
{
  "status": "ok",
  "active_model": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
  "total_layers": 28,
  "hidden_dim": 3584
}
```

---

## 5. Health & Readiness Probes

Rate-limited to 60 requests/min.

| Route | Condition | Status | Body |
|:---|:---|:---:|:---|
| `GET /health` | Process alive | `200` | `{"status": "ok", "state": "online_idle"}` (coordinator) or `{"gateway": "ok", ...}` (gateway) |
| `GET /ready` | Coordinator reachable | `200` | `{"status": "ready", "backend": "ok"}` |
| `GET /ready` | Coordinator stopped | `503` | `{"status": "degraded", "backend": "unreachable"}` |

---

## 6. Cluster Status & Worker Discovery

### `GET /api/cluster/status`
Returns active topology, WireGuard link quality, inter-node RTT, layer split, and wire payload size.

```json
{
  "state": "online_idle",
  "status": "ok",
  "connected": true,
  "active_model": "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
  "transport": "Tailscale Direct WireGuard",
  "is_direct_wireguard": true,
  "rtt_ms": 1.14,
  "per_token_kb": "5.05 KB/tok",
  "quantization": "Outlier-Preserving INT8 (Top-8 FP32 + INT8 Row)",
  "local_stage": { "layer_start": 0, "layer_end": 22 },
  "total_layers": 48,
  "hidden_dim": 5120,
  "worker_nodes": ["100.101.147.24:50052"],
  "nodes_count": 2
}
```

### `GET /api/cluster/discover`
Scans Tailscale peers and local subnet hosts for active worker daemons listening on `:50052`.

---

## 7. One-Click Cluster Pairing & Secret Sync

Hosted by the FastAPI gateway (`:7860`) to securely pair two or more machines over Wi-Fi or Tailscale without manually editing `.env` files. Pairing uses an ephemeral TCP listener on port `50053` encrypted with PBKDF2-HMAC-SHA256 + Encrypt-then-MAC HMAC-SHA256 derived from a one-time 6-digit PIN.

| Method | Route | Description |
|:---|:---|:---|
| `GET` | `/api/cluster/pair/status` | Returns local cluster key fingerprint (`KEY-XXXX-XXXX`), Wi-Fi & Tailscale IPs, active host session info, and saved paired peers with live model compatibility. |
| `POST` | `/api/cluster/pair/start` | Opens a 5-minute encrypted pairing listener on TCP `:50053`. Returns a 6-digit PIN and copyable `AERO-<ip>:50053-<pin>` invite tokens for both Tailscale and Wi-Fi LAN. |
| `POST` | `/api/cluster/pair/stop` | Immediately closes the active `:50053` pairing listener. |
| `GET` | `/api/cluster/pair/discover` | Probes Wi-Fi LAN and Tailscale peers for active `:50053` pairing hosts. |
| `POST` | `/api/cluster/pair/join` | Connects to a host using `invite_code` (or `host` + `pin`), authenticates and decrypts `AEROMESH_WORKER_SECRET`, persists it to `.env`, and exchanges GPU/model inventories. |
| `POST` | `/api/cluster/pair/rotate` | Generates a fresh 256-bit `AEROMESH_WORKER_SECRET` and marks older peers as needing re-pairing. |
| `POST` | `/api/cluster/pair/remove` | Removes a saved peer (`{"id": "..."}`) from the local paired-node registry. |

### Example: Join a Pairing Session
```json
// POST /api/cluster/pair/join
{
  "invite_code": "AERO-100.64.0.12:50053-482915"
}
```
Or with explicit host and PIN:
```json
{
  "host": "192.168.1.42",
  "port": 50053,
  "pin": "482915"
}
```

### 7.1 Automatic Multi-Node Model Check & Peer Model Sync (`:50054`)

Every AeroMESH gateway runs an authenticated TCP sync daemon (`aeromesh-sync-v1`, default port `50054`) using mutual challenge-response HMAC-SHA256 derived from `AEROMESH_WORKER_SECRET` to verify exact `.gguf` header + content fingerprints across nodes and stream missing models over LAN or Tailscale (resumable `.aeromesh-sync.part` chunks). Runtime inference remains strictly `0.0 MB` weight transfer.

| Method | Route | Description |
|:---|:---|:---|
| `POST` | `/api/cluster/models/check` | Queries paired peers (or explicit `peers` CSV/list and optional `model`) on TCP `:50054` and returns a per-model compatibility matrix (`ready`, `needs_sync`, `mismatch`, `missing_locally`, `peers_offline`). |
| `POST` | `/api/cluster/models/sync` | Starts a background resumable `push` (local -> peer) or `pull` (peer -> local) GGUF transfer (`direction`, `model`, `peer_id` or `peer_host`, optional `sync_port`). |
| `GET` | `/api/cluster/models/sync/jobs` | Lists active and recent peer model sync jobs with `percent`, `transferred_label`, `speed_mbps`, `eta_seconds`, and `resumed_from_bytes`. |
| `POST` | `/api/cluster/models/sync/cancel` | Cancels an active sync transfer (`{"id": "sync-..."}`), preserving partial `.aeromesh-sync.part` progress for later resume. |

### 7.2 Hardware-Aware VRAM Fit Meter & Auto Layer-Split Planner

Computes per-node VRAM and System RAM budgets across heterogeneous GPUs (e.g., an 8 GB RTX 4060 coordinator paired with a 6 GB RTX 3060 worker and 16 GB system RAM) using exact binary GGUF v2/v3 KV headers for local files and architecture profiles for pre-download catalog/Hugging Face models.

| Method | Route | Description |
|:---|:---|:---|
| `GET` | `/api/cluster/hardware` | Returns detected local GPU VRAM + System RAM alongside paired worker VRAM/RAM totals (`nodes`, `total_vram_mb`, `total_ram_mb`). |
| `POST` | `/api/cluster/fit/plan` | Computes a proportional layer-split plan and fit verdict (`fits_vram`, `tight_vram`, `ram_offload`, `wont_fit`), per-stage VRAM/RAM usage, exact `--layers <start>..<end>` flags, and quantization/context recommendations. Accepts `model`, optional `context_length` (512..131072), `size_bytes` / `size_hint`, `mode` (`cluster` \| `local`), `peers`, or `custom_nodes` (up to 8 simulated nodes). |

---

## 8. Model Hub (Discovery, Downloads & Custom Folders)

Hosted by the FastAPI gateway (`:7860`) so users can search, download, upload, or link GGUF and diffusion models from any local drive (including existing LM Studio or Ollama directories).

| Method | Route | Description |
|:---|:---|:---|
| `GET` | `/api/models/library` | Returns managed directory path, free disk space, watched custom folders, installed LLM & diffusion models, and the curated model catalog. |
| `POST` | `/api/models/paths` | Add or remove a custom watched model folder (`{"action": "add" \| "remove", "path": "D:\\LMStudio\\models"}`). |
| `POST` | `/api/models/import-path` | Link (zero-copy) or copy a specific `.gguf` / `.safetensors` file or folder (`{"path": "...", "mode": "link" \| "copy"}`). |
| `POST` | `/api/models/browse` | Opens a native Windows file or folder picker dialog (`{"mode": "file" \| "folder"}`) and returns the selected path. |
| `POST` | `/api/models/upload` | Streaming binary upload (`Content-Type: application/octet-stream`, `X-Filename` header) for drag-and-drop model installation. |
| `POST` | `/api/models/delete` | Deletes a managed model file from the AeroMESH managed folder (`{"filename": "..."}`). |
| `GET` | `/api/models/hf/search?q=...&limit=12` | Searches Hugging Face Hub for GGUF repositories. |
| `GET` | `/api/models/hf/files?repo=...` | Lists `.gguf` quantization variants and file sizes in a Hugging Face repository. |
| `POST` | `/api/models/download` | Starts a background resumable download (`{"repo": "...", "filename": "..."}` or `{"url": "..."}`, optional `hf_token`). |
| `GET` | `/api/models/downloads` | Returns active and recent download jobs with progress percentage, speed (MB/s), and ETA. |
| `POST` | `/api/models/download/cancel` | Cancels an active download job (`{"id": "..."}`). |

---

## 9. Node Lifecycle Management

Hosted by the FastAPI gateway (`:7860`) for starting, stopping, and inspecting local coordinator or worker processes from the UI or scripts.

- **`GET /api/node/state`** — Returns local role, detected GPU, structured `hardware` VRAM/RAM metrics, Wi-Fi/Tailscale IPs, cluster fingerprint, installed models, and recent node logs.
- **`POST /api/node/start`** — Spawns a local coordinator or worker process (optionally passing a strictly validated `--layers <start>..<end>` range from the Auto Layer-Split Planner):
  ```json
  {
    "role": "coordinator",
    "model": "Qwen2.5-14B-Instruct-Q4_K_M.gguf",
    "peers": "100.64.0.29:50052",
    "layers": "0..27"
  }
  ```
- **`POST /api/node/stop`** — Gracefully stops the managed child process.

---

## 10. Distributed Diffusion / Image Generation

`POST /v1/images/generations` and `POST /api/image/stream`

```json
{
  "prompt": "Minimalist architectural concrete courtyard at golden hour, 35mm photography",
  "steps": 6,
  "width": 768,
  "height": 768,
  "stream": true
}
```

When `stream: true`, SSE emits:
1. `event: dit_mesh_conditioning` — distributed LLM prompt conditioning metadata
2. `event: dit_generation_progress` — denoising step counter
3. `event: dit_output` — base64 PNG output and transfer stats

---

## 11. Gateway Proxy Behavior

The FastAPI gateway (`app.py`, port `7860`) sits in front of the Axum engine (`:8080`) with:
- **`MAX_REQUEST_BYTES = 8388608` (8 MB)** on JSON proxy routes (streaming model uploads use chunked disk streaming separately).
- **RFC 7230 hop-by-hop stripping** for `Connection`, `Keep-Alive`, `Proxy-Authenticate`, `Proxy-Authorization`, `TE`, `Trailers`, `Transfer-Encoding`, and `Upgrade`.
- **Credential isolation**: caller `Authorization` headers are stripped and replaced with `AEROMESH_API_KEY` on loopback calls to `:8080`.
- **Unbuffered SSE forwarding**: `Cache-Control: no-cache`, `X-Accel-Buffering: no`, and `httpx.Timeout(read=None)`.

---

## 12. Developer API Server & Runtime Configuration

Hosted by the FastAPI gateway (`:7860`) and surfaced via the **API Server** tab and **Inference Settings Drawer**:

- **`GET /api/server/config`** — Returns live OpenAI endpoint URLs (`base_url`, `gateway_base_url`, `lan_url`, `tailscale_url`), masked and live `AEROMESH_API_KEY`, LAN exposure state (`lan_exposed`, `bind_host`), active runtime defaults (`n_ctx`, `ngl`), capability flags (`tool_calling`, `json_mode`, `json_schema`), and copy-ready IDE/SDK snippets (`curl`, `python`, `tools_json`, `continue`, `cline`, `cursor`).
- **`POST /api/server/config`** — Validates and persists `n_ctx` (`4096` / `8192` / `16384` / `32768`), `ngl` (`-1` for all GPU layers or `0..999`), and `lan_exposed` (`127.0.0.1` vs `0.0.0.0`). Enabling LAN exposure automatically ensures `AEROMESH_API_KEY` is generated in `.env`. Pass `"apply_restart": true` to restart the active managed node immediately with `--ctx`, `--ngl`, and `--bind`.
- **`POST /api/server/api-key/rotate`** — Generates a new `sk-aeromesh-...` API key, persists it atomically to `.env`, and returns the updated configuration payload.

