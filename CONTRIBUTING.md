# Contributing to AeroMESH

Thanks for your interest in contributing to AeroMESH!

AeroMESH is a distributed, zero-weight pipeline-parallel inference engine written in Rust and Python that aggregates consumer GPUs across Tailscale mesh networks or local area networks.

Contributions are welcome for distributed inference math, protocol hardening, Windows/Linux compatibility, networking improvements, testing, and UI work.

---

## License

By contributing to AeroMESH, you agree that your contributions will be licensed under the **Apache License, Version 2.0**.

See [LICENSE](LICENSE) for details.

---

## Architecture & Protocol Overview (v4)

AeroMESH implements a **Zero-Weight P2P Pipeline Architecture** operating under **Protocol Version 4**:

```
[Control Plane: Star Topology]
Coordinator (Node 0)  <-- Handshake & Health (TCP) -->  Worker 1, Worker 2, ... Worker N-1

[Data Plane: Chain + 1-Hop Feedback]
Node 0 (Layers 0..K0)  == ActivationFrame ==>  Node 1 (Layers K0+1..K1)  == ActivationFrame ==>  Node N-1 (Terminal)
        ^                                                                                              │
        └──────────────────────── TokenResponseFrame (~48 bytes, 1-Hop) ──────────────────────────────┘
```

### Core Mathematical & Wire Protocol Invariants

When contributing changes to `aeromesh-core` or `aeromesh-engine`, you must preserve these mathematical and protocol invariants:

1. **0.0 MB Network Weight Transfers**:
   - Model weights are never streamed over the network. Each machine retains the base `.gguf` file locally in `models/` and performs local zero-copy slicing for its assigned stage partition (`model_stage_X_of_Y.gguf`).
   - Only intermediate activation vectors (~1.5 KB to 16 KB per step) are transferred over the wire.

2. **Identity RMSNorm in Non-Terminal Stages**:
   - Intermediate stages ($0 \dots N-2$) replace `output_norm.weight` with an **Identity RMSNorm** tensor ($\mathbf{w} = [1.0, 1.0, \dots, 1.0]$) to prevent premature scale distortion and preserve channel variance across hops.
   - Only the terminal stage ($N-1$) applies the genuine source `output_norm.weight` and LM head projection.
   - Evaluators extract unnormalized activations using the `cb_eval` callback on tensor `l_out-<layer>` via `safe_ggml_backend_tensor_get`.

3. **RoPE Phase Coherence**:
   - Rotary positional embeddings depend strictly on global sequence position $p$.
   - The coordinator and all workers must propagate and forward `ActivationHeader.token_position`, assigning `batch.pos[i] = token_position + i` into the `llama_batch`.

4. **1-Hop Direct Token Feedback**:
   - When the terminal stage finishes sampling the next token, it streams a `TokenResponseFrame` directly back to the Coordinator (Node 0) via `coordinator_addr`. Intermediate stages are not involved in token return relaying, eliminating $(N-1)$ hops of pipeline bubble latency.

5. **Double BOS Prevention**:
   - When tokenizing prompts, do not inject duplicate `<|begin_of_text|>` or BOS tokens if the prompt or chat template already begins with special token markers (`<|`).

6. **KV-Cache Synchronization**:
   - The coordinator signals session resets and aborts by setting `FLAG_CLEAR_KV (0x01)` in `ActivationHeader.flags`. Workers must intercept this flag and clear their local context KV cache immediately.

7. **32-Byte Zero-Copy Tensor Alignment**:
   - All sliced GGUF partitions written by `gguf_slicer` must ensure 32-byte boundary alignment for every tensor to maintain zero-copy memory mapping (`mmap`) performance on AVX-512 and CUDA.

---

## Development Environment

AeroMESH currently targets Windows 10/11 with NVIDIA CUDA acceleration.

### Required Tools:
- **Rust Toolchain**: 1.78+ (recommended: `stable-x86_64-pc-windows-gnu` or MSVC).
- **MinGW / w64devkit**: Contains `dlltool.exe` and GNU binutils (add `C:\w64devkit\bin` to `PATH` if using the GNU toolchain).
- **Native llama.cpp Binaries & DLLs**: Kept in `bin/` (`llama.dll`, `ggml.dll`, `ggml-cuda.dll`, `ggml-base.dll`, `ggml-cpu.dll`, `ggml-rpc-server.exe`).
- **Python**: 3.11 or newer (preferably 3.12).
- **Tailscale**: Optional for secure cross-machine WireGuard mesh networking.
- **Git**

### Automated Setup:
```powershell
.\setup.ps1
```

### Python Virtual Environment:
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

## Building AeroMESH

Compile the complete Rust workspace:
```powershell
# Ensure w64devkit and cargo are in PATH
$env:PATH = "C:\w64devkit\bin;C:\Users\$env:USERNAME\.cargo\bin;$env:PATH"
cargo build --release --bin aeromesh
```

After building, copy the CUDA runtime DLLs from `bin/` to the release directory so the binary can load them:
```powershell
Copy-Item -Path "bin\*.dll" -Destination "target\release" -Force
```

Test the release binary:
```powershell
.\target\release\aeromesh.exe --help
```

---

## Running Automated Tests

`aeromesh-engine` resolves native `llama.cpp` symbols dynamically at runtime via `GetProcAddress` rather than static linking. As a result, all Rust unit tests run and pass cleanly even on clean runner environments without CUDA or native DLLs installed.

When running on a machine with CUDA hardware, add `bin/` to `PATH` to exercise native compute pathways:

```powershell
$binDir = (Resolve-Path "bin").Path
$env:PATH = "C:\w64devkit\bin;C:\Users\$env:USERNAME\.cargo\bin;$binDir;$env:PATH"

# Run all unit tests across the workspace (65 tests: 19 core + 46 engine)
cargo test --workspace

# Run lib tests specifically
cargo test --lib
```

Run formatting and linting:
```powershell
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
```

Run FastAPI Gateway integration tests (27 tests):
```powershell
pytest tests/test_fastapi_gateway.py
```

---

## Testing Cluster Meshes Locally

### 1. Unified 1-Click Launcher (`start.ps1`)
The interactive PowerShell script handles port conflict resolution, secret generation, and background orchestration:

- **Full Local Test Mesh** (Worker + Coordinator + Web UI on one machine):
  ```powershell
  .\start.ps1 all
  ```
- **Coordinator Node** (Laptop A):
  ```powershell
  .\start.ps1 coordinator -Peers "100.66.49.50:50052"
  ```
  *(For 3+ nodes, pass comma-separated peer endpoints: `-Peers "100.66.49.50:50052,100.66.49.51:50052"`)*
- **Worker Node** (Laptop B / Laptop C):
  ```powershell
  .\start.ps1 worker -Port 50052
  ```

### 2. Manual CLI Workflow (`aeromesh.exe`)
```powershell
# Worker:
.\target\release\aeromesh.exe worker --port 50052 --model "models/test.gguf"

# Coordinator:
.\target\release\aeromesh.exe start --role coordinator --model "models/test.gguf" --port 8080 --peers "127.0.0.1:50052"
```

Web UI is served at: `http://127.0.0.1:7860`

---

## Code Style & Conventions

### Rust
- Use Rust 2021 edition idioms.
- Adhere to standard naming conventions (`snake_case` functions, `CamelCase` types, `SCREAMING_SNAKE_CASE` constants).
- Prefer explicit typed errors (`thiserror` in libraries, `anyhow::Result` in application layers).
- Never allow unhandled panics in network loops, protocol decoders, or streaming generators.
- When adding or changing wire frames, increment/verify `PROTOCOL_VERSION` in `crates/aeromesh-core/src/activation.rs` and update unit tests.
- Always include automated unit tests for serialization, bounds checking, and integer overflow validation.

### Python & Gateway
- Target Python 3.12+ conventions.
- Provide explicit type annotations on endpoints, helper functions, and background tasks.
- Keep FastAPI gateway proxy routes clean and filter hop-by-hop headers (`Connection`, `Upgrade`, `Keep-Alive`).

### Web UI
- Keep frontend assets lightweight and dependency-free.
- Use vanilla JavaScript, HTML5 semantic elements, and modern CSS (`liquid_glass.css`).
- Maintain SSE stream parsing and real-time telemetry HUD updates without blocking page rendering.

---

## Pull Request Guidelines

1. **Keep Changes Focused**: Separate wire protocol changes from engine optimizations or UI improvements.
2. **Preserve Mathematical Invariants**: Ensure non-terminal stages maintain Identity RMSNorm and sequence offsets stay aligned.
3. **Verify Tests**: All workspace tests must pass (`cargo test --workspace`).
4. **Formatting & Lints**: Run `cargo fmt --check` and `cargo clippy`.
5. **Clear Documentation**: Document any new flags, wire headers, or CLI parameters in `README.md` and `CONTRIBUTING.md`.
6. **Reference Issues**: Link corresponding bug reports or pull requests (e.g., `Fixes #12`).

---

## Security

For security-sensitive vulnerabilities or authentication bypass issues, do not open a public issue. Follow the reporting guidelines in `SECURITY.md` or contact the core maintainers directly.
