# AeroMESH: Architectural Boundaries & Operational Constraints

**Document ID**: DOC-LIMIT-2026-V4  
**Target Engine**: AeroMESH Distributed LLM Engine (Protocol v4)  
**Scope**: Deliberate engineering boundaries, hardware trade-offs, and design constraints.

---

## Table of Contents

- [1. Executive Philosophy](#1-executive-philosophy)
- [2. Pipeline Parallelism & The Autoregressive Bubble](#2-pipeline-parallelism--the-autoregressive-bubble)
- [3. Pre-Shared Symmetric GGUF Weights Requirement](#3-pre-shared-symmetric-gguf-weights-requirement)
- [4. Windows-Centric Process Containment (Win32 Job Objects)](#4-windows-centric-process-containment-win32-job-objects)
- [5. Ephemeral Stateless Context & Concurrency Semantics](#5-ephemeral-stateless-context--concurrency-semantics)
- [6. Context Window & Memory Hard Limits](#6-context-window--memory-hard-limits)
- [7. Model Architecture Support & Slicing Boundaries](#7-model-architecture-support--slicing-boundaries)
- [8. Architectural Trade-Off & Roadmap Matrix](#8-architectural-trade-off--roadmap-matrix)

---

## 1. Executive Philosophy

AeroMESH is engineered specifically for **distributed edge inference across consumer laptops and gaming PCs** rather than multi-tenant cloud datacenters. Every constraint outlined in this document reflects a deliberate engineering decision made to optimize inference over consumer Wi-Fi, low-cost Tailscale WireGuard tunnels, and laptops equipped with 6 GB to 8 GB of VRAM.

---

## 2. Pipeline Parallelism & The Autoregressive Bubble

### Current Architecture
AeroMESH partitions deep neural networks across transformer layer boundaries:
- **Coordinator Node (Stage 1)**: Executes tokenization, prompt embedding, transformer blocks `0..K`, Identity RMSNorm injection, and dynamic Outlier-INT8 compression.
- **Worker Node (Stage 2+)**: Dequantizes activations, executes transformer blocks `K+1..N-1`, evaluates genuine output normalization, computes LM head logits, samples the next token ID, and returns a 1-hop token frame directly to the coordinator.

### Operational Constraint
In pure sequential pipeline parallelism with batch size 1 ($B=1$), only one node is actively executing GEMM tensor arithmetic at any given millisecond during autoregressive decoding:
- While Laptop A computes layers `0..K`, Laptop B is waiting for the activation frame.
- While Laptop B computes layers `K+1..N-1` and samples the next token, Laptop A is waiting for the token response.

### Why this boundary exists
Alternative schemes (such as Tensor Parallelism) achieve 100% compute overlap across nodes, but require multi-gigabyte all-reduce collective communications inside every single attention layer. Over consumer Wi-Fi or Tailscale WireGuard (where latency ranges between 1 ms and 15 ms), Tensor Parallelism grinds inference down to < 0.1 tokens per second. AeroMESH accepts the sequential pipeline bubble to keep network traffic down to a tiny **~3.05 KB to 5.05 KB per token step**, enabling smooth 15–25 tok/s inference over standard wireless links.

---

## 3. Pre-Shared Symmetric GGUF Weights Requirement

### Current Architecture
All participating cluster nodes must pre-position the identical base `.gguf` model file on their local NVMe SSDs inside the `models/` folder prior to launching the cluster.

### Why this boundary exists
Shipping 8 GB to 30 GB of model weights across consumer internet or Wi-Fi creates multi-minute cold starts, exhausts data caps, and saturates home uplinks. AeroMESH operates under a **Zero-Weight Network Transfer** philosophy. By pre-positioning the base model, runtime network traffic is strictly confined to intermediate activation vectors.

### Operational Mitigation
Users copy the base `.gguf` file across laptops before execution (e.g., via high-speed USB 3.2 flash drive, local Windows file share `\\LAPTOP\models`, or direct download). When the cluster launches, AeroMESH validates model header hashes (`aeromesh model-check`) to confirm mathematical consistency across all peers.

---

## 4. Windows-Centric Process Containment (Win32 Job Objects)

### Current Architecture
All native child processes (including worker compute instances and RPC helpers) are bound to a Win32 Job Object configured with:
```rust
info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
```

### Why this boundary exists
Consumer gaming laptops running Windows 11 with discrete NVIDIA GeForce RTX GPUs represent the primary target baseline. In Windows, crashed or orphaned CUDA processes frequently hold active device context handles, locking dedicated VRAM until a full system reboot. Win32 Job Objects guarantee that if the coordinator process exits, crashes, or is terminated by `Ctrl+C`, the Windows NT kernel immediately terminates all child processes and reclaims all allocated GPU VRAM.

### Portability Constraint
This containment mechanism is Windows-specific. Running AeroMESH on Linux or macOS requires abstracting `job_object.rs` using POSIX `prctl(PR_SET_PDEATHSIG, SIGKILL)` or systemd `cgroups v2` hierarchical containment.

---

## 5. Ephemeral Stateless Context & Concurrency Semantics

### Current Architecture
AeroMESH maintains an ephemeral, in-memory KV-cache context. The engine does not embed an external database (such as SQLite, PostgreSQL, or Redis). The coordinator exposes an OpenAI-compliant stateless HTTP API (`/v1/chat/completions`) that reconstructs multi-turn conversational context from the client-supplied `messages` array.

### Operational Constraint
- **Concurrency**: Generation runs on a single active inference context per coordinator instance. Concurrent incoming client requests are queued sequentially.
- **State Invalidation**: Switching models (`/api/model/switch`), restarting node processes, or refreshing the browser flushes the transient KV cache. Context persistence is delegated to client storage.

---

## 6. Context Window & Memory Hard Limits

### Current Architecture
The inference engine enforces strict upper bounds on context sequence lengths:
- Default context length: `n_ctx = 4096` tokens.
- Request validation: Prompts where `prompt_tokens + max_tokens > n_ctx` are rejected immediately with HTTP 422 before any GPU computation begins.

### Why this boundary exists
Consumer laptop GPUs (e.g., RTX 4060 8GB) have tight VRAM margins when hosting 14B or 32B model slices. Allowing unbounded context growth would trigger CUDA Out-Of-Memory (OOM) allocations, terminating the worker process. The hard context ceiling ensures predictable VRAM footprints throughout long conversations.

---

## 7. Model Architecture Support & Slicing Boundaries

### Current Architecture
AeroMESH's deterministic GGUF slicer (`crates/aeromesh-engine/src/gguf_slicer.rs`) is optimized for dense Transformer architectures where layers are indexed uniformly:
- Supported: LLaMA (Llama-2, Llama-3, Llama-3.1, Llama-3.2), Qwen (Qwen-2, Qwen-2.5, DeepSeek-R1 Distill), Mistral, and Gemma.
- Boundary Rules: Models must be sliced strictly along transformer block boundaries (`blk.X.*`). The embedding layer (`token_embd.weight`) is retained on Stage 1; the language model head (`output.weight`) and genuine RMSNorm are retained on the terminal stage.

### Operational Constraint
Non-standard architectures (such as Mixture-of-Experts with dynamic cross-layer token routing, or non-transformer state space models like Mamba) require custom stage-partitioning logic to avoid splitting inter-expert routing logic across network hops.

---

## 8. Architectural Trade-Off & Roadmap Matrix

| Constraint Area | Current AeroMESH Architecture | Underlying Engineering Reason | Future Milestone / Target |
|---|---|---|---|
| **Cluster Topology** | 2-Node Point-to-Point & Multi-Stage Ring | Lowest token feedback latency | Dynamic Pipeline Rebalancing |
| **Operating System** | Windows 10/11 x64 (Win32 Job Objects) | Prevents orphaned CUDA VRAM leaks | Linux `cgroups v2` / POSIX abstraction |
| **Model Weights** | Pre-positioned local base GGUF | Zero-weight streaming over consumer Wi-Fi | Local P2P BitTorrent model pre-seeding |
| **Context Memory** | Ephemeral in-memory KV-cache | Minimal edge footprint; client-driven state | Optional SQLite session persistence |
| **Context Length** | Bounded by `n_ctx` (default: 4096) | Prevents GPU Out-of-Memory crashes | Dynamic RoPE scaling & FlashAttention-2 |
| **Concurrency** | Single active generation session | Bounded VRAM on consumer GPUs | Multi-turn continuous batching |
