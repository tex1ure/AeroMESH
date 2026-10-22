# AeroMESH Empirical Benchmark Report

**Document ID**: BM-2026-DIST-02  
**Target System**: AeroMESH Distributed Heterogeneous LLM Pipeline (Protocol v4)  
**Models Under Test**:
- `Llama-3.2-3B-Instruct-Q8_0` (`test.gguf`, 3.19 GB, 28 Layers, $d_{\text{model}} = 3072$)
- `DeepSeek-R1-Distill-Qwen-14B-Q4_K_M` (`DS.gguf`, 8.37 GB, 48 Layers, $d_{\text{model}} = 5120$)

---

## 1. Executive Summary

AeroMESH resolves the three primary bottlenecks of consumer distributed LLM inference:
1. **Network Weight Transfer Overhead**: Solved via **0.0 MB model weights on the wire** (pre-positioned base GGUF).
2. **SSD Storage Duplication**: Solved via **Ephemeral VRAM Stage Loading (`InPlaceStageGuard`)**, which uploads only the active layer range into GPU VRAM and immediately deletes temporary slice files (**0.0 MB persistent disk overhead**).
3. **Tailscale / Wi-Fi Packet Latency**: Solved via **Outlier-Preserving Per-Row INT8 quantization** (~75% wire reduction) paired with **Win32 `SIO_TCP_SET_ACK_FREQUENCY = 1`** to eliminate the 200ms Windows TCP Delayed ACK stall on odd-segment WireGuard frames.

---

## 2. Measured Performance: Llama 3.2 3B Instruct (`test.gguf`, Q8_0)

| Metric | Measured Value | Notes |
| :--- | :--- | :--- |
| **Model Architecture** | `llama` (28 layers, $d = 3072$, 255 tensors) | `test.gguf` (3.19 GB Q8_0) |
| **Pipeline Partition** | Stage 1: Layers `0..13` • Stage 2: Layers `14..27` + LM Head | Full CUDA GPU Offload (`ngl = 999`) |
| **Sustained Generation Speed** | **24.9 tok/s** (~40.1 ms/tok) | Verified in Web UI & Coordinator telemetry |
| **Activation Codec** | `Int8PerRow` (Outlier-Preserving INT8) | $3072\text{ B} + 4\text{ B scale} + 48\text{ B outliers}$ |
| **Wire Payload per Token** | **3.05 KB/tok** (3,124 B payload + 44 B header) | 74.6% reduction vs 12.0 KB FP32 |
| **Persistent Disk Overhead** | **0.0 MB** | Ephemeral stage files auto-deleted after VRAM load |
| **Model Weights on Wire** | **0.0 MB** | Zero-Weight P2P Architecture |

---

## 3. Measured Performance: DeepSeek-R1-Distill-Qwen-14B (`DS.gguf`, Q4_K_M)

| Property | Coordinator Node (Stage 1) | Worker Node (Stage 2) |
| :--- | :--- | :--- |
| **Hardware** | Laptop A (RTX 4060 Laptop 8GB VRAM) | Laptop B (Peer Node `:50052`) |
| **Model Partition** | Layers `0..22` (23 layers) | Layers `23..47` (25 layers) + LM Head |
| **VRAM Footprint** | **4.12 GB** (fits comfortably in 8GB VRAM) | **4.85 GB** (split-allocated) |
| **Persistent Disk Overhead** | **0.0 MB** (Ephemeral VRAM upload) | **0.0 MB** (Ephemeral VRAM upload) |
| **Wire Payload per Token** | **5.05 KB/tok** ($5120 + 52\text{ B}$ payload) | **35 B + UTF-8** (`TokenResponseFrame`) |
| **Network Transport** | Tailscale Direct WireGuard (UDP 41641) | `SIO_TCP_SET_ACK_FREQUENCY = 1` |
| **Round-Trip Ping** | **1.14 ms** (Direct P2P Mesh) | **1.14 ms** (Direct P2P Mesh) |

### Why `SIO_TCP_SET_ACK_FREQUENCY = 1` Was Critical Over Tailscale
- Tailscale's WireGuard interface uses an MTU of **1280 bytes** (TCP MSS = **1240 bytes**).
- For $d = 5120$, an `ActivationFrame` is $5,172 + 44 = 5,216\text{ bytes}$, which fragments into **5 TCP segments** ($\lceil 5216 / 1240 \rceil = 5$).
- Because 5 is an **odd number**, Windows' default Delayed ACK algorithm (`TcpAckFrequency = 2`) acknowledged segments 1–2 and 3–4, then waited **200 ms** on the 5th segment before sending an ACK—capping generation at **2–5 tok/s**.
- Applying `SIO_TCP_SET_ACK_FREQUENCY = 1` via `WSAIoctl` forces immediate per-segment ACKs, restoring **15–30 tok/s** over Tailscale.

---

## 4. Architectural Comparison: AeroMESH vs Alternatives

| Metric / Feature | AeroMESH (`ArgonFixes`) | Standard GGML RPC (`llama.cpp`) | Weight-Streaming (Exo / Petals) |
| :--- | :--- | :--- | :--- |
| **Model Weights on Wire** | **0.0 MB** | 0.0 MB (if cached) / GBs uncached | 8.37 GB – 14.0 GB cold-start |
| **Duplicate Slice Files on Disk** | **0.0 MB** (Ephemeral VRAM Loader) | 0.0 MB | High cache churn |
| **Wire Payload / Token ($d=5120$)** | **5.05 KB** (Outlier-Preserving INT8) | 20.0 KB (Uncompressed FP32) | Variable |
| **Windows Tailscale Delayed ACK** | **Bypassed (<1 ms ACK)** | Suffers 200ms stalls on odd frames | High latency |
| **VRAM Requirement (14B Q4_K_M)** | **4.12 GB** (Enables 8GB GPUs) | ~4.5 GB | High buffer churn |