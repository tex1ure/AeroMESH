# AeroMESH Security Architecture & Threat Model (SEC-01)

## 1. Threat Model Overview

AeroMESH implements a zero-trust, defense-in-depth security model across all three primary control and data surfaces:

```
[ Web Browser ]
      │ (Unauthenticated, bound to 127.0.0.1)
      ▼
[ FastAPI Gateway :7860 ]
      │ (Bearer Token: AEROMESH_API_KEY, bound to 127.0.0.1)
      ▼
[ Axum Coordinator :8080 ]
      │ (Mutual Auth Handshake: AEROMESH_WORKER_SECRET, Protocol v3)
      ▼ (Over Tailscale WireGuard Tunnel)
[ TCP Worker Node :50052 ]
```

### Control Surfaces & Trust Boundaries

| Interface | Default Binding | Protocol / Transport | Authentication Mechanism | Public Exposure |
|---|---|---|---|---|
| **Web Gateway (UI / Proxy)** | `127.0.0.1:7860` | HTTP / SSE | Localhost loopback; optional `verify_gateway_access` bearer | **NEVER** expose to public internet without reverse proxy + auth |
| **Coordinator API (Axum)** | `127.0.0.1:8080` | HTTP / JSON | Constant-time Bearer token check (`AEROMESH_API_KEY`) | **NEVER** expose to public internet; strictly loopback or Tailscale |
| **P2P Worker Mesh** | `0.0.0.0:50052` (Tailscale) | Binary TCP (Magic `0xAE20`) | Protocol v3 mutual auth token (`AEROMESH_WORKER_SECRET`) | **RESTRICT** via Tailscale WireGuard ACLs only |

---

## 2. Authentication Implementations

### A. Axum Coordinator API (Port 8080)
- **Middleware**: `auth_middleware` attached to all routes via `axum::middleware::from_fn`.
- **Exemptions**: `/health` endpoint is unauthenticated to allow low-overhead liveness probes and container health checks.
- **Timing Attack Mitigation**: Secret comparison uses `constant_time_eq_str`, which SHA-256 hashes both operands before XOR accumulation, preventing execution-time side-channel attacks.
- **Error Standard (FIX-06)**: Failed authentication returns HTTP 401 with the standardized OpenAI-compatible error envelope:
  ```json
  {
    "error": {
      "message": "Invalid or missing Bearer token. Supply valid AEROMESH_API_KEY in Authorization header.",
      "type": "authentication_error",
      "param": null,
      "code": "invalid_api_key"
    }
  }
  ```

### B. FastAPI Web Gateway (Port 7860)
- **Secret Ingestion**: Automatically reads `AEROMESH_API_KEY` from `.env` or system environment.
- **Proxy Injection**: Injects `Authorization: Bearer <AEROMESH_API_KEY>` on all upstream `httpx` calls to coordinator port 8080 (`/v1/models`, `/v1/chat/completions`, `/api/cluster/status`, `/health`).
- **Loopback Protection**: `verify_gateway_access` dependency ensures requests originate from loopback (`127.0.0.1` / `::1`) or provide valid credentials, preventing unauthorized LAN access if port 7860 is accidentally bound broadly.

### C. Worker TCP Handshake (Port 50052)
- **Protocol Version**: Bumped to `PROTOCOL_VERSION = 3`.
- **Legacy Packet Rejection**: Nodes running legacy protocol version 2 (lacking authentication fields) are immediately rejected with an explicit warning log:
  ```
  Handshake rejected: received legacy protocol version 2 (missing auth_token). AeroMesh requires PROTOCOL_VERSION 3 with mutual authentication.
  ```
- **Mutual Handshake Token**:
  - Derived from `AEROMESH_WORKER_SECRET` using SHA-256 into a 32-byte token (`[u8; 32]`).
  - Transmitted in `HandshakeRequest::auth_token`.
  - Verified by the worker in constant time (`constant_time_eq_32`).
  - Mismatched tokens result in immediate connection teardown and a security warning log:
    ```
    🚨 Handshake rejected: invalid AEROMESH_WORKER_SECRET token from peer. Closing connection.
    ```
  - Upon verification, the worker echoes back the token in `HandshakeResponse::ok` so the coordinator can verify the remote worker also possesses the shared secret.

---

## 3. Secret Management & Tooling

### Automated Generation via `start.ps1`
On first run, `start.ps1` checks for `.env`. If absent or missing keys:
1. Generates 32-byte cryptographically secure random hex tokens via `System.Security.Cryptography.RandomNumberGenerator`.
2. Persists them into `.env`.
3. Sets `$env:AEROMESH_API_KEY`, `$env:AEROMESH_WORKER_SECRET`, and `$env:LLAMA_API_KEY` in the execution environment.
4. Children processes (Axum, FastAPI, Worker) automatically inherit these tokens.

### `.gitignore` Enforcement
`.env` and `.env.*` are strictly excluded from version control to prevent secret leakage in commits or PRs.

---

## 4. Tailscale & Remote Exposure Requirements

> [!WARNING]
> Do NOT expose port 8080 or port 50052 to the public internet. AeroMESH distributed execution is designed specifically for peer-to-peer Tailscale WireGuard networks.

When deploying across multiple physical machines (e.g., Laptop A as Coordinator, Laptop B as Worker):

1. **Join Private Tailnet**: Both machines must authenticate to the same private Tailscale tailnet.
2. **Tailscale ACL Policy**: Configure your Tailscale ACLs (`tailscale.com/admin/acls`) so only designated coordinator nodes can reach port 50052 on worker nodes:
   ```json
   {
     "acls": [
       {
         "action": "accept",
         "src": ["tag:coordinator"],
         "dst": ["tag:worker:50052"]
       }
     ]
   }
   ```
3. **P2P WireGuard Reachability**: Verify direct WireGuard connectivity using:
   ```powershell
   tailscale ping <worker-tailscale-ip>
   cargo run --bin aeromesh -- probe <worker-tailscale-ip>:50052
   ```

---

## 5. Transport Security & TLS Termination Policy (FIX-23)

AeroMESH explicitly delegates cryptographic transport security to the network and operating system layers:

### A. Localhost HTTP Traffic (Ports 7860 & 8080)
- **Zero TLS Requirement**: Communication between the local web browser, the FastAPI gateway (`:7860`), and the Axum engine (`:8080`) is plain HTTP.
- **Kernel Process Boundary**: Packets on `127.0.0.1` / `::1` never leave the local host network stack and cannot be intercepted by remote network peers.
- **Local Origin Isolation**: Axum restricts CORS strictly to `127.0.0.1:7860` (`FIX-11`), preventing cross-origin browser script attacks.

### B. Inter-Node Cluster Traffic (Port 50052)
- **Layer 3 WireGuard Encryption**: All cross-machine traffic (activation vectors, token streams, handshake frames) must be routed over Tailscale.
- **Noise Protocol Primitives**: Tailscale provides ChaCha20-Poly1305 authenticated encryption with ephemeral key exchanges, eliminating the need for application-layer TLS certificate management.

### C. Edge TLS Termination for Public Access
AeroMESH core binaries do not manage X.509 certificates, Let's Encrypt ACME renewal, or HTTPS listeners. If public or tailnet-wide HTTPS is required, terminate TLS externally:
- **Tailscale Serve**: `tailscale serve https / http://127.0.0.1:7860` (provides automated Let's Encrypt certificates for your tailnet).
- **Reverse Proxy**: Deploy Caddy or Nginx in front of port 7860 with `proxy_buffering off` to preserve unbuffered SSE token streaming.

For full architectural specifications, see [`docs/architecture/tls-transport-security-policy.md`](docs/architecture/tls-transport-security-policy.md).
