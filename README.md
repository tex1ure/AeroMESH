# AeroMesh: Distributed Fault-Tolerant LLM Cluster Engine

AeroMesh aggregates heterogeneous consumer laptops (Windows + NVIDIA GPUs) into a unified, high-throughput LLM inference cluster interconnected via **Tailscale** and local Wi-Fi with **0.0 MB model weights transferred across the wire**.

---

## ⚡ Zero-Weight P2P Pipeline Architecture

In traditional distributed inference (like default GGML RPC), model weights are streamed from the host to remote worker GPUs over the network on every startup. Over Wi-Fi or VPN tunnels, this causes severe latency bottlenecks.

**AeroMesh completely eliminates network weight transfers**:
1. **Local Layer Slicing (`mmap`)**: Each laptop stores the model `.gguf` file on its local SSD. Using memory-mapped I/O, Node A maps Layers `0..24` and Node B maps Layers `25..48`.
2. **Autoregressive Loopback**:
   - **Node A (Coordinator)**: Embeds token with `token_embd.weight`, evaluates Layers `0..24`, and streams only the activation vector ($B \times 1 \times d_{\text{model}}$, ~8–16 KB) to Node B.
   - **Node B (Worker / Final Stage)**: Receives activations, evaluates Layers `25..48`, applies `output_norm` + LM Head (`output.weight`), and samples the next token ID $T_{i+1}$.
   - **Loopback**: Node B streams `TokenResponseFrame` (containing token text and token ID) back to Node A to seed the next forward step.
3. **Context & KV-Cache Synchronization**: `ActivationHeader` carries `session_id`, `token_position`, and `FLAG_CLEAR_KV` flags to ensure multi-turn chat sessions remain in sync without context drift.
4. **Shape & Handshake Guards**: Nodes negotiate `HandshakeRequest`/`HandshakeResponse` before inference, validating architecture, hidden dimensions, layer boundaries, and GGUF checksums.
5. **Wi-Fi Jitter & Socket Guards**: Socket reads are guarded by a 10-second timeout to handle transient packet drops without hanging the user interface.

```
                                 [TAILSCALE MESH NETWORK / DIRECT WIREGUARD]
                                                     │
               ┌─────────────────────────────────────┴─────────────────────────────────────┐
               ▼                                                                           ▼
     ┌──────────────────┐                                                        ┌──────────────────┐
     │  Coordinator     │   Activation Vectors (~8–16 KB/token)                  │  Worker Laptop   │
     │  (e.g. Laptop A) │ ─────────────────────────────────────────────────────> │  (e.g. Laptop B) │
     │  Layers 0..24    │ <───────────────────────────────────────────────────── │  Layers 25..48   │
     │  (Local GGUF)    │          TokenResponseFrame (Token Text, Token ID)     │  (Local GGUF)    │
     └──────────────────┘                                                        └──────────────────┘
               ▲
               │ OpenAI-Compatible API (http://127.0.0.1:8080/v1/chat/completions)
               ▼
     ┌──────────────────┐
     │  Gradio Web UI   │
     │     (app.py)     │
     └──────────────────┘
```

---

## Quickstart Guide for Teammates

### 1. Prerequisites
- **OS**: Windows 10/11 (64-bit)
- **GPU**: NVIDIA GPU with updated drivers
- **Git**: [Git for Windows](https://git-scm.com/) installed
- **Network**: [Tailscale](https://tailscale.com) installed and signed into your team account.

---

### 2. Setup Your Node in 1 Step

In **PowerShell** inside the `AeroMESH` project folder, run:
```powershell
.\setup.ps1
```
*This checks your GPU, verifies Tailscale, installs Rust if needed, synchronizes native CUDA binaries, and compiles the `aeromesh` binary.*

---

### 3. How to Run the Cluster (Zero-Weight Transfer)

#### Step 1: Start the Worker (Laptop B - e.g. Tailscale IP `100.101.147.24`)
Tell the worker to slice and load only its assigned upper layers from its local SSD:
```powershell
cargo run --bin aeromesh -- worker --model "models/DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf" --layers 25..48 --port 50052
```

#### Step 2: Start the Coordinator & API Server (Laptop A)
Start the coordinator with an OpenAI-compatible HTTP API server for Gradio and web clients:
```powershell
cargo run --bin aeromesh -- serve `
    --model "models/DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf" `
    --layers 0..24 `
    --peers "100.101.147.24:50052" `
    --port 8080
```
*Alternatively, for a single CLI prompt test:*
```powershell
cargo run --bin aeromesh -- coordinator `
    --model "models/DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf" `
    --layers 0..24 `
    --peers "100.101.147.24:50052" `
    --mode pipeline `
    --prompt "Explain distributed GPU clustering in one sentence."
```

#### Step 3: Launch the Web UI
In a separate terminal on the Coordinator laptop:
```powershell
python app.py
```
Open `http://127.0.0.1:7860` in your browser. Real-time streaming chat connects directly to the Zero-Weight Pipeline API.

---

## CLI Command Reference

| Command | Description |
|---|---|
| `aeromesh serve --model <path> --layers <range> --peers <ips> [--port 8080]` | Launches persistent OpenAI-compatible HTTP API server for Gradio / Web UI. |
| `aeromesh worker --model <path> --layers <range>` | Starts native Zero-Weight Pipeline worker with local mmap layer slicing. |
| `aeromesh coordinator --model <path> --layers <range> --peers <ips> --mode pipeline` | Runs native Zero-Weight P2P Pipeline parallel token generation. |
| `aeromesh slice-info --model <path> --layers <range>` | Inspects GGUF layer ranges, tensor counts, and VRAM memory-mapping footprint. |
| `aeromesh model-check <file.gguf>` | Inspects GGUF metadata, tensor counts, and verifies block checksum. |
| `aeromesh probe <ip:port>` | Probes TCP RTT latency and detects Tailscale Direct WireGuard vs DERP Relay. |
| `aeromesh status` | Discovers active cluster nodes across Tailscale mesh network. |

---

## Project Architecture

```
AeroMESH/
├── setup.ps1                 # Automated 1-click bootstrap script
├── Cargo.toml                # Rust workspace configuration
├── app.py                    # Gradio Web UI (streams from AeroMesh Pipeline API)
├── pyproject.toml            # Python dependencies (gradio, requests, httpx)
├── crates/
│   ├── aeromesh-core/        # Binary activation wire protocol, handshake, Tailscale prober, domain types
│   ├── aeromesh-engine/      # Zero-copy GGUF slice loader (mmap), P2P pipeline supervisor, Axum HTTP API server
│   └── aeromesh-cli/         # Unified 'aeromesh' CLI binary (worker, coordinator, serve, probe, status)
├── llama.cpp/                # Native llama.cpp submodule/repository
├── bin/                      # Native CUDA llama.cpp backend executables and DLLs
└── models/                   # Local .gguf model repository
```
