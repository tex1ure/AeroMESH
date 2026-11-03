# Contributing to AeroMESH

Thank you for your interest in contributing to AeroMESH!

AeroMESH is an open-source distributed LLM cluster engine written in Rust and Python that aggregates consumer GPUs across Tailscale mesh networks or local area networks.

Contributions are welcome across distributed inference math, wire protocol hardening, Windows/Linux compatibility, networking optimizations, automated testing, and UI enhancements.

---

## Table of Contents

- [Code of Conduct & License](#code-of-conduct--license)
- [Architecture & Protocol Invariants (Protocol v4)](#architecture--protocol-invariants-protocol-v4)
- [Development Environment Setup](#development-environment-setup)
- [Building the Project from Source](#building-the-project-from-source)
- [Running the Automated Test Suites (92 Tests)](#running-the-automated-test-suites-92-tests)
- [Local Mesh Testing & Cluster Simulation](#local-mesh-testing--cluster-simulation)
- [Code Style & Best Practices](#code-style--best-practices)
- [Pull Request Guidelines](#pull-request-guidelines)
- [Security Vulnerability Reporting](#security-vulnerability-reporting)

---

## Code of Conduct & License

By contributing to AeroMESH, you agree that your contributions will be licensed under the **Apache License, Version 2.0**. See the [LICENSE](LICENSE) file for complete details.

---

## Architecture & Protocol Invariants (Protocol v4)

AeroMESH implements a **Zero-Weight P2P Pipeline Architecture** operating under **Protocol Version 4**:

```
[Control Plane: Star Topology]
Coordinator (Node 0)  <-- Handshake & Health (TCP) -->  Worker 1, Worker 2, ... Worker N-1

[Data Plane: Sequential Chain + 1-Hop Feedback]
Node 0 (Layers 0..K0)  == ActivationFrame ==>  Node 1 (Layers K0+1..K1)  == ActivationFrame ==>  Node N-1 (Terminal)
        ^                                                                                              │
        └──────────────────────── TokenResponseFrame (~48 bytes, 1-Hop Direct) ────────────────────────┘
```

### Core Mathematical & Wire Protocol Invariants
When modifying `aeromesh-core` or `aeromesh-engine`, you must strictly maintain the following invariants:

1. **0.0 MB Network Weight Transfers**:
   - Model weights must never be transmitted over the network. Each machine retains the base `.gguf` file locally in `models/` and performs ephemeral slicing directly into GPU VRAM (`InPlaceStageGuard`).
   - Only intermediate activation vectors (~3.05 KB to 5.05 KB per token step) cross the wire.
2. **Identity RMSNorm in Non-Terminal Stages**:
   - Intermediate stages ($0 \dots N-2$) replace `output_norm.weight` with an **Identity RMSNorm** tensor ($\mathbf{w} = [1.0, 1.0, \dots, 1.0]$) to prevent premature scale distortion and preserve residual stream variance across hops.
   - Only the terminal stage ($N-1$) applies the genuine source `output_norm.weight` and LM head projection.
3. **RoPE Phase Coherence**:
   - Rotary positional embeddings depend strictly on global sequence position $p$.
   - All nodes must propagate `ActivationHeader.token_position`, assigning `batch.pos[i] = token_position + i` into the native `llama_batch`.
4. **1-Hop Direct Token Feedback**:
   - When the terminal stage finishes sampling the next token, it streams a `TokenResponseFrame` directly back to the Coordinator (Node 0). Intermediate stages are bypassed on token returns, eliminating $(N-1)$ hops of round-trip bubble latency.
5. **Double BOS Prevention**:
   - When tokenizing prompts, do not inject duplicate `<|begin_of_text|>` or BOS tokens if the prompt already begins with special token markers.
6. **KV-Cache Synchronization**:
   - The coordinator signals session resets and aborts by setting `FLAG_CLEAR_KV (0x01)` in `ActivationHeader.flags`. Workers must intercept this flag and clear their local context KV cache immediately.
7. **32-Byte Tensor Alignment**:
   - All sliced GGUF partitions written by `gguf_slicer` must enforce 32-byte boundary alignment for every tensor to maintain zero-copy memory mapping performance on AVX2/AVX-512 and CUDA.

---

## Development Environment Setup

AeroMESH targets Windows 10/11 x64 with NVIDIA CUDA acceleration.

### Required Prerequisites
- **Rust Toolchain**: Rust 1.78+ (`stable-x86_64-pc-windows-gnu` or MSVC).
- **MinGW / w64devkit**: Contains `dlltool.exe` and GNU utilities (add `C:\w64devkit\bin` to `PATH` if using GNU).
- **Native llama.cpp Runtime**: Kept in `bin/` (`llama.dll`, `ggml.dll`, `ggml-cuda.dll`, `ggml-base.dll`, `ggml-cpu.dll`).
- **Python**: Python 3.11 or newer (Python 3.12 recommended).
- **Tailscale**: For cross-machine WireGuard mesh networking.
- **Git**

### Automated Bootstrap Script
```powershell
.\setup.ps1
```

### Python Virtual Environment Setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```
Or using `uv`:
```powershell
uv sync
```

---

## Building the Project from Source

Compile the complete Rust workspace:
```powershell
# Ensure w64devkit and cargo are on your environment PATH
$env:PATH = "C:\w64devkit\bin;C:\Users\$env:USERNAME\.cargo\bin;$env:PATH"

# Build all binaries in release mode with Thin LTO
cargo build --release --workspace

# Copy native runtime DLLs to target/release
Copy-Item -Path "bin\*.dll" -Destination "target\release" -Force
```

---

## Running the Automated Test Suites (92 Tests)

AeroMESH maintains an automated test suite across Rust and Python:

```powershell
# 1. Run all 65 Rust workspace unit & integration tests
# (19 in aeromesh-core + 46 in aeromesh-engine)
cargo test --workspace

# 2. Run all 27 Python FastAPI Gateway integration tests
# (Ensure background coordinator is stopped first to allow offline fallback tests to pass)
Stop-Process -Name "aeromesh" -Force -ErrorAction SilentlyContinue
.\.venv\Scripts\pytest.exe tests/test_fastapi_gateway.py
```

### Formatting & Linting
```powershell
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
```

---

## Local Mesh Testing & Cluster Simulation

### Method 1: Unified Interactive Launcher (`start.ps1`)
```powershell
# Run full local mesh on a single machine via zero-copy Shared Memory (SHM)
.\start.ps1 all -Model "models/test.gguf"

# Start Coordinator Node (Laptop A)
.\start.ps1 coordinator -Peers "100.101.147.24:50052" -Model "models/test.gguf"

# Start Worker Node (Laptop B)
.\start.ps1 worker -Port 50052 -Model "models/test.gguf"
```

### Method 2: Manual CLI Workflow (`aeromesh.exe`)
```powershell
# Start worker listener
.\target\release\aeromesh.exe worker --port 50052 --model "models/test.gguf"

# Start coordinator server
.\target\release\aeromesh.exe start --role coordinator --model "models/test.gguf" --port 8080 --peers "127.0.0.1:50052"
```

Web UI is served at: `http://127.0.0.1:7860`

---

## Code Style & Best Practices

### Rust Guidelines
- Use Rust 2021 edition conventions.
- Adhere to standard naming conventions (`snake_case` functions, `CamelCase` types, `SCREAMING_SNAKE_CASE` constants).
- Prefer explicit typed errors (`thiserror` in libraries, `anyhow::Result` in application binaries).
- Never allow unhandled panics in network loops, wire protocol decoders, or streaming generators.
- When changing wire frames, verify `PROTOCOL_VERSION = 4` in `crates/aeromesh-core/src/activation.rs` and update unit tests.
- Always include automated unit tests for serialization, bounds checking, and integer overflow validation.

### Python Guidelines
- Target Python 3.12+ standards.
- Provide explicit type annotations on endpoints, helper functions, and background tasks.
- Keep FastAPI gateway proxy routes clean and filter hop-by-hop headers (`Connection`, `Upgrade`, `Keep-Alive`).

### Web UI Guidelines
- Keep frontend assets lightweight and dependency-free.
- Use vanilla JavaScript, semantic HTML5, and modern CSS (`liquid_glass.css`).
- Maintain SSE stream parsing and real-time telemetry HUD updates without blocking page rendering.

---

## Pull Request Guidelines

1. **Keep Changes Focused**: Separate wire protocol changes from engine optimizations or UI improvements.
2. **Preserve Mathematical Invariants**: Ensure non-terminal stages maintain Identity RMSNorm and sequence offsets stay aligned.
3. **Verify Tests**: All 92 workspace tests must pass cleanly (`cargo test --workspace` and `pytest`).
4. **Code Quality**: Verify that `cargo fmt` and `cargo clippy` pass without warnings.
5. **Documentation**: Update `README.md`, `API_REFERENCE.md`, or `BENCHMARKS.md` if your change introduces new flags, wire headers, or CLI parameters.

---

## Security Vulnerability Reporting

For security-sensitive vulnerabilities or authentication bypass issues, do not open a public issue. Follow the reporting guidelines in [SECURITY.md](SECURITY.md) or contact the core maintainers directly.
