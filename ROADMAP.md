# AeroMESH Development Roadmap

This document outlines the engineering trajectory, current production boundaries, and planned milestones for the AeroMESH distributed edge inference system.

---

## Milestone 1: Heterogeneous 2-Node Pipeline Parallelism (v0.1.x) — Current / 100% Complete

The primary objective of AeroMESH v0.1.x is verified, low-latency distributed pipeline parallelism across two heterogeneous consumer laptops running Windows 11 with NVIDIA GPUs.

- [x] **Zero-Weight Distributed Pipeline**: Moving 4–16 KB activation tensors over WireGuard instead of transferring 8–30 GB model weights across consumer Wi-Fi (`LIMITATIONS.md`).
- [x] **Heterogeneous 2-Node Topology**: Coordinator evaluates Stage 1 (Embeddings + Layers $0 \dots K$), Worker evaluates Stage 2 (Layers $K+1 \dots N-1$ + True Output Norm + LM Head).
- [x] **Deterministic Local GGUF Slicing**: Dynamic header and tensor slicing with Identity RMSNorm substitution for seamless non-tail layer execution (`crates/aeromesh-engine/src/gguf_slicer.rs`).
- [x] **Protocol v3 Mutual Authentication**: Cryptographic SHA-256 HMAC mutual handshake verifying shared secret (`AEROMESH_WORKER_SECRET`) before admitting worker nodes.
- [x] **Direct Tailscale WireGuard Transport**: Automatic peer discovery, direct UDP hole punching (1–2 ms RTT), and latency admission quality checks ($\le 150\text{ ms}$).
- [x] **Win32 Job Object Process Isolation**: Automatic kernel cleanup of worker and backend daemons (`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`), guaranteeing zero orphaned GPU memory leaks.
- [x] **Claymorphic Web Dashboard**: Dependency-free luxury warm dark UI featuring unbuffered SSE token streaming, DeepSeek-R1 reasoning foldouts, KaTeX math, syntax highlighting, and live cluster HUD.
- [x] **Desktop-First Responsive Guardrails (`FIX-26`)**: Non-destructive tablet/mobile fallbacks (`static/css/responsive.css`), overlay drawer sidebar, and compact navigation preserving 100% desktop fidelity.
- [x] **Stateless Privacy & Transport Security (`FIX-20`, `FIX-23`)**: Zero-trace loopback HTTP isolation with edge TLS termination delegation (Tailscale Serve, Caddy, Nginx).

---

## Milestone 2: Near-Term Maintenance & Polish (v0.2.x) — Post-Demo

Targeted enhancements to improve developer velocity and automated quality checks following the live presentation:

- [ ] **`pipeline.rs` Modularization (`FIX-21`)**:
  - Extract `PipelineCoordinatorClient` into `crates/aeromesh-engine/src/pipeline/coordinator_client.rs`.
  - Extract `PipelineWorkerService` into `crates/aeromesh-engine/src/pipeline/worker_service.rs`.
  - Retain `pipeline.rs` as a thin module root with public re-exports (full blueprint: `docs/architecture/pipeline-refactoring-plan.md`).
- [ ] **Automated Multi-Runner CI/CD (`FIX-22`)**:
  - GitHub Actions workflow running cargo tests and release-mode checks on Windows runners (`.github/workflows/ci.yml`).
- [ ] **Dynamic Model Cache Management**:
  - Automatic purging of stale temporary GGUF slices in `%TEMP%` when storage thresholds are exceeded.

---

## Milestone 3: 3+ Node N-Stage Pipelines (v0.3.x) — Future Roadmap (`FIX-28`)

Extending pipeline parallelism from 2-node point-to-point to arbitrary $N$-stage linear pipeline rings across 3 or more physical machines:

- [ ] **Multi-Split GGUF Slicer**:
  - Extend `gguf_slicer.rs` to support arbitrary layer split arrays (e.g. `--splits 16,32,48`).
  - Generate intermediate slice files with Identity RMSNorm and zero LM head overhead.
- [ ] **Intermediate Stage Relay Service**:
  - Ingest `ActivationFrame` from upstream peer, evaluate middle layer GEMM, and forward fresh `ActivationFrame` downstream without token sampling.
- [ ] **Direct Tail-to-Coordinator Token Return**:
  - Tail node connects directly back to Node 0 over a dedicated return socket to deliver `TokenResponseFrame` packets, avoiding multi-hop reverse transit.
- [ ] **Cumulative WireGuard Latency Budgeting**:
  - Multi-hop RTT verification ensuring total pipeline round-trip latency remains within interactive streaming thresholds ($\le 200\text{ ms}$).
- [ ] **Multi-Node Cluster HUD**:
  - Visualize 3+ stage chains, per-hop latencies, and intermediate stage status directly in the web UI.
  - Full architectural specification: `docs/architecture/n-stage-pipeline-roadmap.md`.

---

## Milestone 4: Cross-Platform & Infrastructure Scaling (v0.4.x) — Long-Term

- [ ] **POSIX Kernel Process Containment (`FIX-29`)**:
  - Abstract `job_object.rs` with `prctl(PR_SET_PDEATHSIG, SIGKILL)` on Linux and systemd cgroups v2 containment for headless multi-GPU edge nodes.
- [ ] **Apple Silicon Acceleration**:
  - Metal backend integration in llama.cpp runner for heterogeneous Mac + Windows edge clustering.
- [ ] **P2P Model Weight Pre-Seeding**:
  - Integrated local BitTorrent or Tailscale peer-to-peer hash-verified GGUF file distribution prior to cluster launch.
