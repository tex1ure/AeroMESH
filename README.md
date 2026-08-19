# AeroMesh: Distributed Fault-Tolerant LLM Cluster Engine

AeroMesh aggregates heterogeneous consumer laptops (Windows + NVIDIA GPUs) into a unified, high-throughput LLM inference cluster interconnected via **Tailscale** and local networks with **0.0 MB model weights transferred across the wire**.

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
*This checks your GPU, verifies Tailscale, ensures `llama.cpp` and required folders are present, installs Rust if needed, synchronizes native CUDA binaries, and compiles the `aeromesh` binary.*

---

### 3. How to Run the Cluster

#### Option A: Zero-Weight P2P Pipeline Mode (Recommended for Wi-Fi & Tailscale)
*Each laptop possesses the `.gguf` model file on local disk and loads only its assigned layer subset.*

1. **On Worker Laptops (e.g. Laptop B)**:
   ```powershell
   cargo run --bin aeromesh -- worker --model "models/DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf" --layers 25..48 --port 50052
   ```
2. **On Coordinator Laptop (e.g. Laptop A)**:
   ```powershell
   cargo run --bin aeromesh -- coordinator `
       --model "models/DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf" `
       --peers "100.101.147.24:50052" `
       --mode pipeline `
       --prompt "Explain distributed GPU clustering in one sentence."
   ```

#### Option B: Supervised CUDA RPC Backend Mode
1. **On Worker Laptops**:
   ```powershell
   cargo run --bin aeromesh -- worker --port 50052
   ```
2. **On Coordinator Laptop**:
   ```powershell
   cargo run --bin aeromesh -- coordinator `
       --model "models/DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf" `
       --peers "100.101.147.24:50052" `
       --mode rpc `
       --ngl -1 `
       --prompt "Explain distributed GPU clustering in one sentence."
   ```

---

## CLI Command Reference

| Command | Description |
|---|---|
| `aeromesh worker --model <path> --layers <range>` | Starts native Zero-Weight Pipeline worker with local mmap layer slicing. |
| `aeromesh worker --port 50052` | Starts supervised CUDA RPC backend worker in leak-proof Windows Job Object. |
| `aeromesh coordinator --model <path> --peers <ips> --mode pipeline` | Runs native Zero-Weight P2P Pipeline parallel token generation. |
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
├── crates/
│   ├── aeromesh-core/        # Binary activation wire protocol, Tailscale prober, domain types
│   ├── aeromesh-engine/      # Zero-copy GGUF slice loader (mmap), P2P pipeline supervisor, Windows Job Object
│   └── aeromesh-cli/         # Unified 'aeromesh' CLI binary
├── llama.cpp/                # Native llama.cpp submodule/repository
├── bin/                      # Native CUDA llama.cpp backend executables and DLLs
└── models/                   # Local .gguf model repository
```
