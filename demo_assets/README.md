# AeroMESH Demonstration Video Assets

This directory contains empirical video recordings demonstrating the execution modes of AeroMESH.

---

## 1. Primary Video: Distributed 2-Node Cluster over Tailscale Direct WireGuard (`DIST-01`)
- **File**: [`demo_distributed_2node_tailscale_direct.mp4`](file:///d:/New%20folder/Latest_llama/AeroMESH/demo_assets/demo_distributed_2node_tailscale_direct.mp4)
- **Resolution**: 1080p (1920x1080) @ 30 fps
- **Duration**: ~46 seconds
- **Thesis Proof**: True pipeline parallelism across consumer hardware without model weight transfer over the wire.
- **Key Evidence Captured**:
  1. **Title & Architecture Card**: Summary of 2-node cluster running DeepSeek-R1-Distill-14B / Qwen-2.5-14B (8.37 GB, 48 layers).
  2. **Cluster Topology HUD Inspection**: Real-time HUD modal querying `GET /api/cluster/status` showing:
     - **Coordinator Node (Laptop A • Local)**: Layers `0..22` on NVIDIA RTX 4060 GPU (`100.66.49.50`).
     - **Link Transport**: `Tailscale Direct WireGuard` badge with **1.14 ms RTT ping** (non-DERP direct UDP 41641 peer connection).
     - **Worker Node (Mesh Peer Stage 2)**: Layers `23..47` + LM Head on port `50052`.
     - **Wire Metrics**: Per-token activation size `5.04 KB/tok`, Zero Model Weights on Wire (`0.0 MB`).
  3. **Live Streaming Generation**: Browser session streaming prompt tokens:
     - Assistant card showing live reasoning and streaming response text.
     - Live telemetry displaying token generation speed (`1.4 tok/s`), wire payload accumulating to `1.33 MB Wire`, per-token readout (`5.16 KB/tok`), and `🛡️ Direct WireGuard` badge.
  4. **Final Completed Generation & Telemetry**: Completed response with finalized telemetry tag:
     - `⚡ 1.4 tok/s • 🌐 1.33 MB Wire • 📦 5.16 KB/tok • 🛡️ Direct WireGuard`
  5. **Empirical Benchmarks Summary**: Table of recorded latency, bandwidth, and quantization metrics.

---

## 2. Fallback Video: Single-Machine Zero-Copy Shared Memory (SHM) Baseline
- **File**: [`demo_shm_single_node_fallback.mp4`](file:///d:/New%20folder/Latest_llama/AeroMESH/demo_assets/demo_shm_single_node_fallback.mp4)
- **Resolution**: 1080p @ 30 fps
- **Thesis Proof**: Ultra-low-latency intra-host ring buffer communication between two local processes without socket overhead.
- **Key Evidence Captured**:
  - Top bar and sidebar displaying single-node SHM mode.
  - Telemetry showing `0.0 MB Wire (SHM)` and zero kernel copy latency.
  - Generation speed of 38.5 tok/s on local test model.

---

## Reproduction Instructions
To reproduce the distributed demonstration shown in `demo_distributed_2node_tailscale_direct.mp4`, follow [SETUP_INSTRUCTIONS.md](file:///d:/New%20folder/Latest_llama/AeroMESH/SETUP_INSTRUCTIONS.md) Scenario A:
```powershell
# Node 2 (Worker - Stage 2):
.\target\release\aeromesh.exe worker --model "models/DS.gguf" --layers 24..47 --port 50052

# Node 1 (Coordinator - Stage 1 + API):
.\target\release\aeromesh.exe serve --model "models/DS.gguf" --layers 0..23 --peers "<WORKER_TAILSCALE_IP>:50052" --port 8080

# Web Interface:
.venv\Scripts\python.exe app.py
```
Open `http://127.0.0.1:7860/` in browser to observe HUD and stream responses.
