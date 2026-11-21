# AeroMESH Limitations & Trade-Offs

AeroMESH is built for consumer Windows PCs and gaming laptops (6–8 GB VRAM) linked over home Wi-Fi or Tailscale, not datacenter InfiniBand racks. The constraints below are deliberate trade-offs made to keep consumer-link throughput usable.

---

## Contents

1. [Sequential Pipeline Parallelism (Autoregressive Bubble)](#1-sequential-pipeline-parallelism-autoregressive-bubble)
2. [Local Base GGUF File on Each Node](#2-local-base-gguf-file-on-each-node)
3. [Windows-First Process Containment](#3-windows-first-process-containment)
4. [Stateless Context & Single Active Generation](#4-stateless-context--single-active-generation)
5. [Fixed Context Window Ceiling (`n_ctx`)](#5-fixed-context-window-ceiling-n_ctx)
6. [Supported Model Architectures](#6-supported-model-architectures)
7. [Trade-Off & Roadmap Summary](#7-trade-off--roadmap-summary)

---

## 1. Sequential Pipeline Parallelism (Autoregressive Bubble)

AeroMESH splits models along transformer layer boundaries: Stage 1 runs blocks `0..K`, sends the compressed hidden state (~3–5 KB), and Stage 2 runs blocks `K+1..N-1` plus the LM head and returns the sampled token in 1 hop.

- **Constraint:** At batch size 1 (`B = 1`), only one GPU computes at a time during decoding. While Laptop A runs layers `0..K`, Laptop B waits; while Laptop B runs `K+1..N-1`, Laptop A waits.
- **Why:** Tensor parallelism overlaps compute across GPUs, but requires all-reduce syncs inside every layer. Over 1–15 ms Wi-Fi or WireGuard links, that overhead drops throughput below 0.1 tok/s. Accepting the pipeline bubble keeps traffic to ~3–5 KB/token so 15–25 tok/s is achievable over Wi-Fi.

---

## 2. Local Base GGUF File on Each Node

Each node in the cluster must have the same base `.gguf` file on local storage before starting distributed inference.

- **Why:** Streaming 8–20 GB of weights over Wi-Fi or residential internet causes multi-minute startup delays and saturates home uplinks. Pre-positioning keeps runtime weight transfer at **0.0 MB**.
- **Built-in tooling:** You do not need to duplicate models if they already exist in LM Studio or Ollama—use the **Model Hub** tab to link existing folders (zero-copy) or pull matching quantizations from Hugging Face. Once paired via **One-Click Cluster Pairing**, the Cluster tab automatically highlights which GGUF files match on both machines, and `aeromesh model-check` verifies header hashes before execution.

---

## 3. Windows-First Process Containment

Child engine and worker processes are attached to a Win32 Job Object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`.

- **Why:** On Windows + CUDA, orphaned worker processes can hold GPU contexts open and leak VRAM until reboot. Job Objects ensure the NT kernel kills child processes and reclaims VRAM automatically if the coordinator crashes or exits.
- **Portability:** Native builds currently target Windows 10/11 x64. Linux/macOS support requires replacing `job_object.rs` with `prctl(PR_SET_PDEATHSIG, SIGKILL)` or `cgroups v2` and swapping the Windows `WSAIoctl` TCP socket tuning.

---

## 4. Stateless Context & Single Active Generation

The coordinator maintains an in-memory KV cache with no embedded database. Multi-turn history is supplied by the client via the OpenAI-compatible `messages` array.

- **Concurrency:** One active generation runs at a time per coordinator instance; concurrent requests queue sequentially.
- **Reset behavior:** Switching models (`/api/model/switch`), restarting a node, or clearing a chat flushes the transient KV cache.

---

## 5. Fixed Context Window Ceiling (`n_ctx`)

Default context length is `n_ctx = 4096`. Requests where `prompt_tokens + max_tokens > n_ctx` are rejected early with HTTP `422` before allocating GPU buffers.

- **Why:** On 6–8 GB GPUs hosting half of a 14B or 32B model, unbounded KV-cache growth triggers CUDA out-of-memory crashes. Enforcing a hard cap keeps VRAM usage predictable across long sessions.

---

## 6. Supported Model Architectures

The deterministic GGUF slicer (`crates/aeromesh-engine/src/gguf_slicer.rs`) targets dense Transformer models with uniform `blk.X.*` block naming:
- **Supported:** Llama 2 / 3 / 3.1 / 3.2, Qwen 2 / 2.5, DeepSeek-R1 Distill, Mistral, and Gemma.
- **Not yet supported:** Sparse Mixture-of-Experts (MoE) architectures with cross-layer routing or non-transformer state-space models (Mamba/RWKV), which require custom partition logic.

---

## 7. Trade-Off & Roadmap Summary

| Area | Current Behavior | Reason | Planned Direction |
|:---|:---|:---|:---|
| **Topology** | 2-node P2P & multi-stage chain | Lowest 1-hop token latency | Dynamic layer rebalancing |
| **OS Support** | Windows 10/11 x64 (Win32 Jobs) | Clean CUDA VRAM reclamation | Linux `cgroups v2` / POSIX port |
| **Model Distribution** | Local GGUF + Model Hub / folder linking | Zero runtime weight streaming | Optional LAN P2P model sync |
| **Session State** | In-memory KV cache | Low edge footprint | Optional SQLite persistence |
| **Context Limit** | Hard `n_ctx` cap (default 4096) | Prevents 8 GB GPU OOM | Dynamic RoPE & FlashAttention-2 |
| **Concurrency** | Single active generation | Predictable consumer VRAM | Continuous batching |
