# AeroMESH Security Architecture & Threat Model (SEC-01)

**Document ID**: SEC-2026-ARCH-01  
**Target System**: AeroMESH Distributed LLM Cluster Engine (Protocol v4)  
**Security Standard**: Zero-Trust Multi-Tier Defense-in-Depth

---

## Table of Contents

- [1. Executive Threat Model](#1-executive-threat-model)
- [2. Control Surfaces & Trust Boundaries](#2-control-surfaces--trust-boundaries)
- [3. Authentication & Authorization Implementations](#3-authentication--authorization-implementations)
  - [A. Web Gateway Proxy (Port 7860)](#a-web-gateway-proxy-port-7860)
  - [B. Coordinator API Engine (Port 8080)](#b-coordinator-api-engine-port-8080)
  - [C. Worker P2P TCP Mesh (Port 50052)](#c-worker-p2p-tcp-mesh-port-50052)
- [4. Cryptographic Primitives & Side-Channel Resistance](#4-cryptographic-primitives--side-channel-resistance)
- [5. Wire Protocol Bounded Deserialization (Anti-DoS)](#5-wire-protocol-bounded-deserialization-anti-dos)
- [6. Secret Management & Automated Provisioning](#6-secret-management--automated-provisioning)
- [7. Client-Side Security & XSS Mitigation](#7-client-side-security--xss-mitigation)
- [8. Tailscale WireGuard Network Policy & ACLs](#8-tailscale-wireguard-network-policy--acls)
- [9. Vulnerability Disclosure & Reporting Policy](#9-vulnerability-disclosure--reporting-policy)

---

## 1. Executive Threat Model

AeroMESH is designed to run distributed tensor pipelines across consumer devices connected over untrusted home Wi-Fi networks or public Tailscale tailnets. The architecture implements a **multi-tier defense-in-depth model** where each layer assumes the adjacent layer could be compromised:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        CLIENT WORKSPACE (Tier 0)                       │
│  • Web Browser Single Page Application (Liquid Glass UI)               │
│  • DOMPurify Sanitization · Subresource Integrity (SRI) Pinned Assets  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP / SSE (Loopback: 127.0.0.1)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     FASTAPI GATEWAY PROXY (Tier 1)                     │
│  • Bounded Payloads: MAX_REQUEST_BYTES = 8 MB                          │
│  • Loopback Enforcement: verify_gateway_access Guard                   │
│  • RFC 7230 Hop-by-Hop Filter · Upstream Credential Injection          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Authenticated HTTP Loopback (:8080)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                    AXUM COORDINATOR ENGINE (Tier 2)                    │
│  • Constant-Time Bearer Verification: AEROMESH_API_KEY                 │
│  • Rate Limiting: 60 req/min (health), 15 req/min (chat completions)   │
│  • Strict Input Validation (OpenAI-compatible 422 envelopes)           │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Protocol v4 Binary Mesh (:50052)
                                    │ [Tailscale WireGuard UDP Tunnel]
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      REMOTE WORKER NODE (Tier 3)                       │
│  • Constant-Time Handshake Auth: AEROMESH_WORKER_SECRET (Protocol v4)  │
│  • Strict Frame Length Caps (128 MB max, S*d invariant verification)   │
│  • Win32 Job Object Containment (SafeProcessJob - Auto VRAM reclaim)   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Control Surfaces & Trust Boundaries

| Interface / Surface | Default Network Binding | Transport Protocol | Authentication Method | Exposure Recommendation |
|:---|:---|:---|:---|:---|
| **Web Gateway Proxy** | `127.0.0.1:7860` | HTTP / SSE | Loopback Origin or `AEROMESH_GATEWAY_API_KEY` | **Strictly Loopback** (Do not expose to public internet) |
| **Coordinator API** | `127.0.0.1:8080` | HTTP / JSON | Constant-time Bearer token (`AEROMESH_API_KEY`) | **Strictly Loopback** or Tailscale IP |
| **P2P Worker Mesh** | `0.0.0.0:50052` | Binary TCP (`AERO`) | SHA-256 Handshake Token (`AEROMESH_WORKER_SECRET`) | **Restrict via Tailscale WireGuard ACLs only** |

---

## 3. Authentication & Authorization Implementations

### A. Web Gateway Proxy (Port 7860)
The gateway proxy (`app.py`) controls browser ingress:
- **Loopback Protection (`verify_gateway_access`)**: Permitted loopback origins (`127.0.0.1`, `localhost`, `::1`) are allowed direct access. Calls originating from non-loopback IPs must present a matching `Authorization: Bearer <AEROMESH_GATEWAY_API_KEY>`; if unconfigured, non-loopback access is rejected immediately (`401 Unauthorized`).
- **Credential Ownership**: The gateway owns the upstream credentials. Any client-provided `Authorization` header is stripped before the gateway injects its internal `AEROMESH_API_KEY`, preventing client impersonation or header duplication.
- **Request Size Ceilings**: Enforces `MAX_REQUEST_BYTES = 8_388_608` (8 MB). Payloads exceeding this limit are aborted prior to buffering in memory.

### B. Coordinator API Engine (Port 8080)
The Axum backend (`crates/aeromesh-engine/src/server.rs`) protects inference execution:
- **Constant-Time Verification**: Uses `constant_time_eq_str` to validate incoming Bearer tokens against `AEROMESH_API_KEY`.
- **Public Probes**: Only the low-overhead liveness route (`GET /health`) is exempt from authentication to support health checks.
- **Rate Limiting**: Axum tower middleware enforces isolated bucket pools: 60 req/min for health probes and 15 req/min for chat completions, emitting standard OpenAI 429 errors upon overrun.

### C. Worker P2P TCP Mesh (Port 50052)
The worker daemon (`crates/aeromesh-core/src/activation.rs` & `aeromesh-engine/src/pipeline.rs`) accepts activation streams:
- **Protocol Version Pinning**: Workers reject frames with protocol version `< 4`, preventing backward-compatibility downgrade attacks.
- **Mutual Handshake Authentication**: Nodes derive a 32-byte cryptographic token from `AEROMESH_WORKER_SECRET` using SHA-256. Mismatched handshake tokens trigger immediate socket closure and security alerts.

---

## 4. Cryptographic Primitives & Side-Channel Resistance

To prevent execution-time side-channel attacks (timing attacks) where an adversary infers token characters by measuring microsecond response differentials, all secret comparisons employ constant-time evaluation:

```rust
// Constant-time string equality via SHA-256 XOR accumulation
pub fn constant_time_eq_str(a: &str, b: &str) -> bool {
    let hash_a = sha2::Sha256::digest(a.as_bytes());
    let hash_b = sha2::Sha256::digest(b.as_bytes());
    constant_time_eq_32(&hash_a.into(), &hash_b.into())
}

// 32-byte constant-time XOR accumulation
pub fn constant_time_eq_32(a: &[u8; 32], b: &[u8; 32]) -> bool {
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}
```

---

## 5. Wire Protocol Bounded Deserialization (Anti-DoS)

Untrusted network input can trigger remote Denial-of-Service (DoS) via memory exhaustion if buffer lengths are derived directly from wire headers. AeroMESH implements strict mathematical validation on all incoming binary frames:

1. **Absolute Payload Ceiling**: Frame payloads cannot exceed `MAX_FRAME_PAYLOAD_BYTES = 134_217_728` (128 MB).
2. **Dimension Upper Bounds**:
   - Sequence length: $S \le 131,072$ tokens.
   - Hidden dimension: $d_{\text{model}} \le 65,536$.
3. **Mathematical Size Invariant**:
   The header `payload_bytes` field must strictly match the byte count implied by $S \times d \times \text{dtype}$. Any frame exhibiting length discrepancies or integer overflow during multiplication is discarded without memory allocation.

---

## 6. Secret Management & Automated Provisioning

### Cryptographic Generation (`start.ps1`)
On first startup, the cluster launcher verifies the existence of `.env`. If absent or unpopulated, it generates cryptographically secure 256-bit random tokens using `System.Security.Cryptography.RandomNumberGenerator`:

```powershell
$bytes = New-Object byte[] 32
[Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
$secret = -join ($bytes | ForEach-Object { "{0:x2}" -f $_ })
```

### Version Control Exclusion
`.env`, `.env.*`, and temporary model slices (`.aeromesh_ephemeral_*`) are explicitly ignored in `.gitignore`, preventing accidental commits of production keys.

---

## 7. Client-Side Security & XSS Mitigation

1. **DOMPurify Sanitization**: All markdown and assistant reasoning outputs rendered by the Single Page Application (`static/js/app.js`) are sanitized via DOMPurify before DOM injection.
2. **Subresource Integrity (SRI)**: All CDN assets (KaTeX, Highlight.js, Lucide Icons) are pinned with cryptographic SHA-384 / SHA-512 SRI integrity hashes.
3. **CORS Isolation**: The coordinator Axum server rejects cross-origin browser requests, restricting CORS exclusively to the local web gateway origin (`http://127.0.0.1:7860`).

---

## 8. Tailscale WireGuard Network Policy & ACLs

When distributing inference across multiple physical machines, deploy nodes strictly within a private Tailscale tailnet. Configure access control lists (ACLs) to isolate worker nodes:

```json
{
  "acls": [{
      "action": "accept",
      "src": ["tag:coordinator"],
      "dst": ["tag:worker:50052"]
    }
  ],
  "tagOwners": {
    "tag:coordinator": ["autogroup:admin"],
    "tag:worker": ["autogroup:admin"]
  }
}
```

---

## 9. Vulnerability Disclosure & Reporting Policy

If you discover a security vulnerability or authentication bypass in AeroMESH:
- **Do not open a public GitHub issue**.
- Email security reports directly to the core engineering maintainers.
- Include detailed reproduction steps, network capture traces, and environment details.
- We will acknowledge receipt within 48 hours and coordinate a coordinated security patch.
