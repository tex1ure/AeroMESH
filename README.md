# AeroMesh: Distributed Fault-Tolerant LLM Cluster Engine

AeroMesh aggregates heterogeneous consumer laptops (Windows + NVIDIA GPUs) into a unified, high-throughput LLM inference cluster interconnected via **Tailscale** and local networks.

---

## Quickstart Guide for Teammates

### 1. Prerequisites
- **OS**: Windows 10/11 (64-bit)
- **GPU**: NVIDIA GPU with updated drivers
- **Git**: [Git for Windows](https://git-scm.com/) installed
- **Network**: [Tailscale](https://tailscale.com) installed and signed into your team account.

---

### 2. Clone the Repository & llama.cpp

Open **PowerShell** and clone the AeroMesh repository along with the `llama.cpp` backend:

```powershell
# 1. Clone AeroMesh repository
git clone https://github.com/tex1ure/AeroMESH.git
cd AeroMESH

# 2. Clone llama.cpp repository
git clone https://github.com/ggerganov/llama.cpp.git
```

> **Note:** If you haven't cloned `llama.cpp` manually, `setup.ps1` will automatically detect and clone it during the setup step.

---

### 3. Setup Your Node in 1 Step

In **PowerShell** inside the `AeroMESH` project folder, run:
```powershell
.\setup.ps1
```
*This checks your GPU, verifies Tailscale, ensures `llama.cpp` and required folders are present, installs Rust if needed, and builds the `aeromesh` binary.*

---

### 4. How to Run the Cluster

#### On Worker Laptops (e.g., Laptop B & C):
1. Place the `.gguf` model file inside the `models/` folder.
2. Start the worker daemon:
   ```powershell
   cargo run --bin aeromesh -- worker --port 50052
   ```
3. Get your Tailscale IP address:
   ```powershell
   tailscale ip -4
   ```
   *(Share this IP address, e.g. `100.122.125.95`, with the Coordinator operator).*

---

#### On the Coordinator Laptop (e.g., Laptop A):
1. Verify the model file hash across nodes:
   ```powershell
   cargo run --bin aeromesh -- model-check "models/test.gguf"
   ```
2. Test connection speed to a worker:
   ```powershell
   cargo run --bin aeromesh -- probe "100.122.125.95:50052"
   ```
3. Run distributed multi-node inference:
   ```powershell
   cargo run --bin aeromesh -- coordinator `
       --model "models/test.gguf" `
       --peers "100.122.125.95:50052" `
       --ngl -1 `
       --prompt "Explain distributed GPU clustering in one short sentence."
   ```

---

## CLI Command Reference

| Command | Description |
|---|---|
| `aeromesh worker --port 50052` | Starts a supervised CUDA RPC backend worker in a leak-proof Windows Job Object. |
| `aeromesh coordinator --model <path> --peers <ips>` | Orchestrates distributed inference across active Tailscale nodes. |
| `aeromesh model-check <file.gguf>` | Inspects GGUF metadata, tensor counts, and verifies block checksum. |
| `aeromesh probe <ip:port>` | Probes TCP RTT latency and detects Tailscale Direct WireGuard vs DERP Relay. |

---

## Project Architecture

```
AeroMESH/
├── setup.ps1                 # Automated 1-click bootstrap script
├── Cargo.toml                # Rust workspace configuration
├── crates/
│   ├── aeromesh-core/        # Core domain types, Tailscale prober, error definitions
│   ├── aeromesh-engine/      # Windows Job Object supervisor, GGUF parser, llama process manager
│   └── aeromesh-cli/         # Unified 'aeromesh' CLI binary
├── llama.cpp/                # Native llama.cpp submodule/repository
├── bin/                      # Native CUDA llama.cpp backend executables and DLLs
└── models/                   # Local .gguf model repository
```
