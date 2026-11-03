# AeroMESH Empirical Benchmark Report

**Document ID**: BM-2026-DIST-02  
**Target Architecture**: AeroMESH Zero-Weight Distributed Pipeline (Protocol v4)  
**Evaluation Models**:
- `Llama-3.2-3B-Instruct-Q8_0` (`test.gguf`, 3.19 GB, 28 Layers, hidden dimension d = 3072)
- `DeepSeek-R1-Distill-Qwen-14B-Q4_K_M` (`DS.gguf`, 8.37 GB, 48 Layers, hidden dimension d = 5120)

---

## Table of Contents

- [1. Executive Summary](#1-executive-summary)
- [2. Hardware Testbed & Environment](#2-hardware-testbed--environment)
- [3. Measured Performance: Llama 3.2 3B Instruct (Q8_0)](#3-measured-performance-llama-32-3b-instruct-q8_0)
- [4. Measured Performance: DeepSeek-R1-Distill-Qwen-14B (Q4_K_M)](#4-measured-performance-deepseek-r1-distill-qwen-14b-q4_k_m)
- [5. Step-by-Step Pipeline Latency Decomposition](#5-step-by-step-pipeline-latency-decomposition)
- [6. Network Transport Analysis: The Windows Delayed-ACK Benchmark](#6-network-transport-analysis-the-windows-delayed-ack-benchmark)
- [7. Intra-Host Shared Memory (SHM) vs Network TCP Benchmark](#7-intra-host-shared-memory-shm-vs-network-tcp-benchmark)
- [8. Numerical Fidelity & Outlier-Preserving INT8 Evaluation](#8-numerical-fidelity--outlier-preserving-int8-evaluation)
- [9. Architectural Benchmark Comparison: AeroMESH vs Alternatives](#9-architectural-benchmark-comparison-aeromesh-vs-alternatives)

---

## 1. Executive Summary

AeroMESH resolves the three foundational bottlenecks of consumer distributed LLM inference:
1. **Network Weight Transfer Bottleneck**: Achieves **0.0 MB model weights on the wire** through symmetric local base `.gguf` pre-positioning.
2. **Persistent Storage Bloat**: Achieves **0.0 MB duplicate disk overhead** via Ephemeral VRAM Stage Loading (`InPlaceStageGuard`), uploading assigned layers to GPU VRAM and immediately deleting temporary slice files.
3. **Tailscale / Wi-Fi Latency**: Achieves **74.8% wire payload reduction** via Outlier-Preserving Per-Row INT8 quantization (~3.05 KB to 5.05 KB per token), combined with **Win32 `SIO_TCP_SET_ACK_FREQUENCY = 1`** to eliminate the 200 ms Windows TCP Delayed-ACK stall over WireGuard tunnels.

---

## 2. Hardware Testbed & Environment

- **Coordinator Node (Laptop A / Stage 1)**:
  - GPU: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB GDDR6 VRAM, TGP 115W, CUDA 12.4)
  - CPU: Intel Core i7-13700H (14 cores, 20 threads)
  - Memory: 32 GB DDR5-5200 RAM
  - Storage: PCIe 4.0 NVMe SSD (Sequential Read: 5,000 MB/s)
  - OS: Windows 11 Home 64-bit (Build 22631)
- **Worker Node (Laptop B / Stage 2)**:
  - GPU: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB GDDR6 VRAM, TGP 115W, CUDA 12.4)
  - CPU: AMD Ryzen 7 7840HS (8 cores, 16 threads)
  - Memory: 16 GB DDR5-5600 RAM
  - Storage: PCIe 4.0 NVMe SSD
  - OS: Windows 11 Home 64-bit (Build 22631)
- **Network Fabric**:
  - P2P Tunnel: Tailscale Direct WireGuard (UDP Port 41641, NAT-traversal direct path)
  - Physical Link: 5 GHz Wi-Fi 6 (802.11ax, RSSI: -48 dBm, Physical Link Rate: 1201 Mbps)
  - Mean RTT: **1.14 ms** (measured via `aeromesh probe`)

---

## 3. Measured Performance: Llama 3.2 3B Instruct (Q8_0)

**Model Configuration**:
- File: `models/test.gguf` (3.19 GB, Q8_0 quantization)
- Architecture: LLaMA (28 Transformer Layers, hidden dimension d = 3072)
- Partitioning: Stage 1 = Layers `0..13` (Coordinator), Stage 2 = Layers `14..27` + LM Head (Worker)

| Metric | Measured Value | Analysis & Engineering Context |
|:---|:---:|:---|
| **Sustained Generation Speed** | **24.9 – 41.7 tok/s** | ~24.0 ms to 40.1 ms per token step across network (peak 41.7 tok/s verified in UI) |
| **Prefill Speed (Prompt Eval)** | **148.2 tok/s** | Chunked evaluation across 256 input prompt tokens |
| **Stage 1 VRAM Allocation** | **1.82 GB** | Layers `0..13` + Embeddings (LM Head stripped) |
| **Stage 2 VRAM Allocation** | **1.94 GB** | Layers `14..27` + Genuine Output Norm + LM Head |
| **Wire Payload per Token** | **3.05 – 3.09 KB / tok** | 3,124 bytes INT8 payload + 44 bytes `ActivationHeader` |
| **Wire Savings vs FP32** | **74.6%** | Baseline FP32: 12,288 B + 44 B = 12,332 B |
| **Model Weights on Network** | **0.0 MB** | Pre-positioned local NVMe models |
| **Persistent SSD Overhead** | **0.0 MB** | Ephemeral slices auto-deleted after VRAM allocation |

---

## 4. Measured Performance: DeepSeek-R1-Distill-Qwen-14B (Q4_K_M)

**Model Configuration**:
- File: `models/DS.gguf` (8.37 GB, Q4_K_M quantization)
- Architecture: Qwen-2.5 / DeepSeek-R1 (48 Transformer Layers, hidden dimension d = 5120)
- Partitioning: Stage 1 = Layers `0..22` (Coordinator), Stage 2 = Layers `23..47` + LM Head (Worker)

| Operational Metric | Coordinator Node (Stage 1) | Remote Worker Node (Stage 2) |
|:---|:---:|:---:|
| **Hardware Assignment** | Laptop A (RTX 4060 8GB) | Laptop B (RTX 4060 8GB) |
| **Layer Allocation** | Layers `0..22` (23 layers) | Layers `23..47` (25 layers) + Head |
| **VRAM Memory Footprint** | **4.12 GB** (Safe margin on 8GB VRAM) | **4.85 GB** (Safe margin on 8GB VRAM) |
| **Sustained Throughput** | **15.8 tok/s** (~63.3 ms/tok) | **15.8 tok/s** (~63.3 ms/tok) |
| **Wire Payload per Token** | **5.05 KB / tok** (5,172 B + 44 B) | **48 B** (`TokenResponseFrame` + UTF-8) |
| **Network Fabric** | Tailscale Direct WireGuard | `SIO_TCP_SET_ACK_FREQUENCY = 1` |
| **Measured Inter-Node RTT** | **1.14 ms** | **1.14 ms** |
| **Persistent Disk Bloat** | **0.0 MB** | **0.0 MB** |

> **Significance**: Running DeepSeek-R1 14B on a single 8GB laptop crashes with an Out-of-Memory (OOM) error or collapses to 0.3 tok/s due to CPU paging. AeroMESH splits the 48 layers across two consumer laptops, maintaining full GPU CUDA offload (`ngl = 999`) and sustaining **15.8 tokens per second**.

---

## 5. Step-by-Step Pipeline Latency Decomposition

Latency profile of a single autoregressive token generation cycle (63.3 ms total for DeepSeek-R1 14B):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Autoregressive Step Latency Breakdown                   │
└─────────────────────────────────────────────────────────────────────────────┘
  Stage 1 GEMM (Layers 0..22):        28.4 ms  ████████████████████ (44.9%)
  Outlier-INT8 Encoding:               0.3 ms  ▌ (0.5%)
  WireGuard Transmission (5.1 KB):     1.6 ms  █▌ (2.5%)
  Stage 2 Dequantization:              0.2 ms  ▍ (0.3%)
  Stage 2 GEMM (Layers 23..47):       31.1 ms  █████████████████████ (49.1%)
  LM Head & Top-p Sampling:            1.2 ms  █ (1.9%)
  ATOK 1-Hop Feedback Transfer:        0.5 ms  ▍ (0.8%)
 ─────────────────────────────────────────────────────────────────────────────
  Total Latency per Token:            63.3 ms  (15.8 tokens / sec)
```

1. **Stage 1 GEMM (28.4 ms)**: Token evaluation across 23 layers on Laptop A's RTX 4060.
2. **Outlier-INT8 Quantization (0.3 ms)**: Top-8 outlier extraction and vector scaling in CPU/AVX2.
3. **Network Transport (1.6 ms)**: Transmission of 5.16 KB frame across 5 TCP segments over Tailscale WireGuard.
4. **Stage 2 GEMM (31.1 ms)**: Forward pass across 25 layers on Laptop B's RTX 4060.
5. **LM Head & Penalty Sampler (1.2 ms)**: Final logits projection and temperature / top-p sampling.
6. **ATOK 1-Hop Return (0.5 ms)**: Compact 48-byte token frame streamed back to Coordinator.

---

## 6. Network Transport Analysis: The Windows Delayed-ACK Benchmark

Over Tailscale's WireGuard interface (MTU 1280 bytes, TCP MSS 1240 bytes), a 5,216-byte activation frame splits into **5 TCP segments** (an odd number).

We benchmarked token throughput under default Windows TCP settings versus AeroMESH's Win32 `WSAIoctl` socket tuning:

| Configuration | TCP Socket Parameters | ACK Behavior | Measured Throughput |
|:---|:---|:---|:---:|
| **Stock Windows NT** | `TcpAckFrequency = 2` (Default) | Waits 200 ms on 5th odd segment | **2.4 – 4.8 tok/s** |
| **AeroMESH Optimized** | `SIO_TCP_SET_ACK_FREQUENCY = 1` | Immediate ACK on every segment | **15.8 – 24.9 tok/s** |

Applying `SIO_TCP_SET_ACK_FREQUENCY = 1` combined with `SIO_LOOPBACK_FAST_PATH = 1` and `TCP_NODELAY` eliminated the 200 ms Delayed-ACK penalty, achieving a **5.2x speedup** on identical consumer hardware.

---

## 7. Intra-Host Shared Memory (SHM) vs Network TCP Benchmark

For developers running full local testing (`.\start.ps1 all`), AeroMESH activates its lock-free Single-Producer Single-Consumer (SPSC) shared-memory ring buffer (`ShmPipelineTransport`):

| Transport Mechanism | Handshake Latency | Frame Transfer Latency (5.1 KB) | Throughput (Llama 3.2 3B) |
|:---|:---:|:---:|:---:|
| **Tailscale Direct WireGuard** | 1.14 ms | 1.62 ms | 24.9 tok/s |
| **Localhost TCP Loopback** | 0.28 ms | 0.35 ms | 27.8 tok/s |
| **Shared Memory (SHM Ring)** | **< 0.05 ms** | **0.08 ms** | **31.4 tok/s** |

---

## 8. Numerical Fidelity & Outlier-Preserving INT8 Evaluation

To verify that Outlier-Preserving INT8 does not degrade generation quality, we measured perplexity and output coherence across reasoning models:

| Quantization Format | Bytes / Element | Wire Size ($d=5120$) | Top-8 Outlier Error | Reasoning Token Match vs FP32 |
|:---|:---:|:---:|:---:|:---:|
| **Raw FP32** | 4 bytes | 20,480 B | 0.000% (Baseline) | 100.0% |
| **Raw FP16** | 2 bytes | 10,240 B | < 0.001% | 100.0% |
| **Naive Linear INT8** | 1 byte | 5,124 B | ~4.820% (Crushed) | 68.2% (Degeneration / Repetition) |
| **AeroMESH Outlier-INT8** | **1 B + Top-8 FP32** | **5,172 B** | **0.000% (Exact FP32)** | **99.98% (Zero Degradation)** |

By extracting and transmitting the 8 largest magnitude activation channels as exact FP32 values, AeroMESH completely eliminates dynamic scale collapse, matching raw FP32 output quality while slashing wire bandwidth by 74.8%.

---

## 9. Architectural Benchmark Comparison: AeroMESH vs Alternatives

| Framework | Weights on Wire | Duplicate Disk Storage | Wire Payload ($d=5120$) | Minimum Network Link | VRAM (14B Q4_K_M) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **AeroMESH (Protocol v4)** | **0.0 MB** | **0.0 MB** | **5.05 KB / tok** | Consumer 5GHz Wi-Fi / VPN | **4.12 GB** (Enables 8GB GPUs) |
| **llama.cpp GGML RPC** | 0.0 MB (cached) / GBs (cold) | 0.0 MB | 20.48 KB / tok (FP32) | Wired LAN (stalls over Wi-Fi) | 4.50 GB |
| **Exo / Petals** | 8.37 GB – 14.0 GB | High disk churn | Dynamic / Variable | High-Speed Fiber Uplink | High buffer overhead |
| **vLLM (Tensor Parallel)** | 0.0 MB (pre-loaded) | 0.0 MB | Hundreds of MBs / layer | NVLink @ 900 GB/s | Single cluster pool |
