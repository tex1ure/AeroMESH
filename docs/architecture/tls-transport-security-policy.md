# TLS Termination & Transport Security Policy (FIX-23)

> **Status**: Intended Architecture (Deliberate Design Non-Goal)  
> **Component**: `aeromesh-engine`, `app.py`, `aeromesh-cli`, `aeromesh-core`  
> **Classification**: Architectural Decision Record (ADR)  
> **Related Documents**: `SECURITY.md`, `BENCHMARKS.md`, `docs/architecture/fastapi-proxy-gateway.md`

---

## 1. Executive Summary & Policy Statement

AeroMESH delegates cryptographic transport confidentiality directly to the operating system and network layers:

> **AeroMESH does not implement application-level TLS certificate generation, X.509 trust store validation, or HTTPS termination inside the core engine or web gateway.**
> Localhost HTTP communication (`:7860` and `:8080`) relies on kernel-enforced loopback process isolation. Inter-node cluster communication across physical machines relies on **Tailscale Layer 3 WireGuard encryption** (ChaCha20-Poly1305).

TLS termination, if required for compliance or external exposure, is classified as an **edge deployment concern** and must be handled externally via **Tailscale Serve**, a VPN, or a dedicated reverse proxy (e.g., Caddy, Nginx, Envoy).

---

## 2. Traffic Classification & Security Boundaries

AeroMESH partitions cluster communication into four distinct planes:

| Traffic Plane | Ports | Protocols | Encryption & Protection | Scope |
|---|---|---|---|---|
| **Web Browser to SPA Gateway** | `7860` | HTTP / SSE | Loopback process isolation (`127.0.0.1`); TLS not required | Same machine only |
| **Gateway to Axum Engine API** | `8080` | HTTP / JSON / SSE | Loopback process isolation + `AEROMESH_API_KEY` Bearer token | Same machine only |
| **Coordinator to Worker Mesh** | `50052` | Binary TCP (Magic `0xAE20`) | **Tailscale WireGuard (Layer 3)** + Protocol v3 SHA-256 mutual handshake | Inter-node across network |
| **Peer Telemetry & Diagnostics** | `50052` / Daemon | WireGuard / JSON | Tailscale ACLs + direct P2P WireGuard quality inspection | Inter-node across network |

```
[ Browser ]
    │ Plain HTTP (127.0.0.1:7860) — Protected by OS process isolation
    ▼
[ FastAPI Gateway ]
    │ Plain HTTP + Bearer Token (127.0.0.1:8080) — Loopback only
    ▼
[ Axum Coordinator ]
    │
    │ ══════════════════════════════════════════════════════════════
    │ TAILSCALE WIREGUARD TUNNEL (Layer 3 ChaCha20-Poly1305)
    │ Mutual Handshake Secret: AEROMESH_WORKER_SECRET (Protocol v3)
    │ ══════════════════════════════════════════════════════════════
    ▼
[ Remote Worker Node ]
```

---

## 3. Technical Rationale: Why TLS in Core is an Anti-Pattern

### 3.1 Localhost Traffic is Protected by Kernel User Boundaries
On modern operating systems (Windows and Linux), loopback network traffic (`127.0.0.1` / `::1`) never leaves the host network stack. Packets are transferred in-memory via virtual software interfaces. Non-privileged network peers on the local LAN cannot sniff loopback traffic. Encrypting loopback packets with TLS introduces CPU overhead and certificate management without adding meaningful security against unprivileged local users.

### 3.2 Tailscale Provides Robust Layer 3 WireGuard Encryption
AeroMESH is designed to cluster consumer hardware across distributed edge environments. Rather than reinventing transport encryption, AeroMESH pairs natively with Tailscale:
- **WireGuard Cryptography**: Tailscale authenticates and encrypts all inter-node packets using state-of-the-art Noise protocol primitives (Curve25519, ChaCha20-Poly1305, BLAKE2s).
- **Zero-Configuration Mesh**: WireGuard operates below the application layer, automatically protecting TCP port `50052` without requiring per-node TLS certificate authorities, SAN configurations, or renewal jobs.
- **Direct P2P NAT Traversal**: DERP-assisted hole punching creates direct UDP WireGuard tunnels with 1–2ms latencies, as verified by `aeromesh probe`.

### 3.3 Avoidance of Certificate Management Complexity
Embedding TLS inside the Rust and Python binaries would require:
- Generating self-signed certificates, which trigger browser trust warnings (`SEC_ERROR_UNKNOWN_ISSUER`).
- Integrating ACME clients (Let's Encrypt), which fail on local/private networks lacking public DNS records.
- Storing and rotating private keys on local disks.
- Complex client-side CA trust bundle configuration.

By delegating transport security to Tailscale, the cluster remains completely operational offline and requires zero manual certificate maintenance.

---

## 4. Non-Loopback Binding Warnings

To prevent inadvertent exposure, both `app.py` and `server.rs` log prominent warnings whenever they are configured to bind to a non-loopback interface:

### 4.1 Axum Engine (`server.rs`)
```rust
if !addr.ip().is_loopback() {
    warn!(
        addr = %addr,
        "⚠️ AeroMesh engine API is binding to a non-loopback interface. Ensure this interface is protected by Tailscale WireGuard or a trusted firewall."
    );
}
```

### 4.2 FastAPI Gateway (`app.py`)
```python
if AEROMESH_UI_HOST not in {"127.0.0.1", "::1", "localhost"}:
    logger.warning(
        "AeroMESH web gateway is binding to non-loopback interface (%s). "
        "Ensure this interface is protected by Tailscale or trusted VPN.",
        AEROMESH_UI_HOST,
    )
```

---

## 5. Production Edge Deployment & Reverse Proxying

When organizations require public HTTPS URLs or centralized TLS termination (e.g., exposing an internal company LLM endpoint), TLS must terminate at the reverse proxy layer in front of AeroMESH.

### Option A: Tailscale Serve (Zero-Configuration HTTPS)
If the cluster operates within a corporate tailnet, enable Tailscale Serve to obtain an official signed HTTPS certificate for the dashboard:
```powershell
tailscale serve https / http://127.0.0.1:7860
```
Tailscale automatically provisions a Let's Encrypt certificate for `<machine-name>.<tailnet>.ts.net` with zero application changes.

### Option B: Caddy Reverse Proxy
Caddy provides automated Let's Encrypt certificates and passes SSE token streams without buffering:
```caddy
llm.example.internal {
    reverse_proxy 127.0.0.1:7860
}
```

### Option C: Nginx Reverse Proxy
When deploying behind Nginx, **buffering must be explicitly disabled** so Server-Sent Events (SSE) token generation streams immediately to clients:
```nginx
server {
    listen 443 ssl http2;
    server_name llm.example.internal;

    ssl_certificate     /etc/ssl/certs/aeromesh.crt;
    ssl_certificate_key /etc/ssl/private/aeromesh.key;

    location / {
        proxy_pass http://127.0.0.1:7860;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;

        # CRITICAL FOR SSE STREAMING: Disable proxy buffering
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 600s;
        chunked_transfer_encoding on;
    }
}
```

---

## 6. Security Guarantees & Summary

| Scenario | Transport Protection | Operational Action Required |
|---|---|---|
| **Local Single-Machine Test** | OS Loopback Isolation | None (Plain HTTP on `127.0.0.1`) |
| **Multi-Laptop P2P Cluster** | Tailscale WireGuard Layer 3 | Join shared tailnet; verify via `aeromesh probe` |
| **Tailnet-Wide Web Access** | Tailscale Serve (HTTPS) | Run `tailscale serve https / http://127.0.0.1:7860` |
| **Enterprise Public Ingress** | Edge Reverse Proxy (Caddy/Nginx) | Terminate TLS at proxy with `proxy_buffering off` |
| **Core Application Binaries** | **Zero TLS Complexity** | **None (Decoupled by design)** |
