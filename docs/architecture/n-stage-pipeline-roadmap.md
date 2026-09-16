# N-Stage Pipeline Parallelism Roadmap (FIX-28)

> **Status**: Architectural Specification & Future Engineering Roadmap  
> **Component**: `aeromesh-core`, `aeromesh-engine`, `aeromesh-cli`  
> **Classification**: Architectural Decision Record (ADR) / Roadmap  
> **Related Documents**: `LIMITATIONS.md`, `ROADMAP.md`, `README.md`, `docs/architecture/pipeline-refactoring-plan.md`

---

## 1. Executive Summary & Current Scope

AeroMESH targets heterogeneous distributed edge inference across consumer hardware.

> **Current Supported Scope: Exactly 2 Nodes (1 Coordinator + 1 Worker)**  
> AeroMESH implements and benchmarks a 2-stage pipeline split: one coordinator node evaluating Stage 1 layers ($0 \dots K$) and one remote worker node evaluating Stage 2 layers ($K+1 \dots N-1$) plus the final RMSNorm, LM head, and token sampling.

Multi-node $N$-stage pipelines ($N \ge 3$) are architecturally planned but are **not** currently implemented, tested, or supported in the v0.1.x release. This document defines the architectural blueprint, protocol requirements, GGUF slicing generalizations, and failure semantics for future multi-node implementations.

---

## 2. Rationale: Why 2-Node Topology Was Prioritized

1. **Closed Autoregressive Feedback Loop**:
   - In a 2-node pipeline, intermediate activations flow forward ($A \to B$) and sampled tokens flow directly backward ($B \to A$).
   - This eliminates multi-hop serialization delays and removes the need for complex pipeline bubble scheduling (e.g., 1F1B or interleaved schedules) on latency-sensitive edge connections.
2. **Typical Consumer Edge Setup**:
   - Operators typically combine two laptops (e.g., an 8 GB VRAM primary laptop + a 4 GB or 6 GB secondary laptop) to run 14B parameter models like DeepSeek-R1-Distill-Qwen-14B that neither machine can fit alone.
   - P2P WireGuard between two nodes achieves direct 1–2 ms UDP hole-punched latencies.
3. **Deterministic Fault Containment**:
   - Win32 Job Objects cleanly encapsulate the worker daemon on Laptop B. If the coordinator drops, the worker terminates without lingering zombie processes or VRAM leaks.

---

## 3. Future Target Architecture: N-Stage Pipeline Ring ($N \ge 3$)

### 3.1 Stage Topologies & Node Roles

```
[ Client / Browser ]
        │ HTTP / SSE (127.0.0.1:7860)
        ▼
[ Head Stage / Coordinator (Node 0) ]
  • Tokenizes raw prompt
  • Evaluates token embeddings & Layers 0..K1
  • Applies Identity RMSNorm (weights = 1.0)
  • Quantizes activations (INT8 / FP8)
        │
        ▼ ActivationFrame (Port 50052, Direct WireGuard)
[ Intermediate Stage (Node 1) ]
  • Dequantizes activations
  • Evaluates Layers K1+1..K2 GEMM
  • Applies Identity RMSNorm (weights = 1.0)
  • Re-quantizes activations
        │
        ▼ ActivationFrame (Port 50052, Direct WireGuard)
[ Tail Stage (Node 2) ]
  • Dequantizes activations
  • Evaluates Layers K2+1..N-1 GEMM
  • Applies TRUE Output RMSNorm & LM Head (output.weight)
  • Evaluates logits & samples next token (T_next)
        │
        ▼ TokenResponseFrame (Direct TCP Return Channel)
[ Head Stage / Coordinator (Node 0) ]
  • Yields token text to client via SSE
  • Loops T_next into next autoregressive decode step
```

| Role | Responsibilities | Tensor Slices Required |
|---|---|---|
| **Head Stage** (Node 0) | Prompt tokenization, embedding lookup, Layers $0 \dots K_1$, client SSE streaming | `token_embd.weight`, `blk.0`..`blk.K1`, Identity RMSNorm |
| **Intermediate Stage(s)** (Node $1 \dots N-2$) | Ingests `ActivationFrame`, evaluates middle layers, forwards activations downstream | `blk.K1+1`..`blk.K2`, Identity RMSNorm (no LM head) |
| **Tail Stage** (Node $N-1$) | Ingests `ActivationFrame`, evaluates final layers, executes true output norm, evaluates logits, samples tokens | `blk.K2+1`..`blk.N-1`, `output_norm.weight`, `output.weight` |

---

## 4. Engineering Requirements for N-Stage Pipelines

### 4.1 Generalized Multi-Split GGUF Slicing
Current slicing produces two stages (`0..K` and `K+1..N-1`). Multi-stage execution requires the slicer (`crates/aeromesh-engine/src/gguf_slicer.rs`) to support arbitrary split sequences:

```powershell
# Future CLI invocation
cargo run --bin aeromesh -- slice `
    --model models/DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf `
    --splits 16,32 `
    --out-dir models/slices/
```

Output slices:
- `stage_0.gguf`: Embeddings + Layers $0 \dots 16$ + Identity RMSNorm.
- `stage_1.gguf`: Layers $17 \dots 32$ (re-indexed $0 \dots 15$) + Identity RMSNorm.
- `stage_2.gguf`: Layers $33 \dots 48$ (re-indexed $0 \dots 15$) + True Output Norm + LM Head.

### 4.2 Transport Forwarding & Return Channel
- **Downstream Forwarding**: Intermediate nodes do not maintain client connections. They accept an `ActivationFrame` on their upstream port, evaluate local layers, and write a fresh `ActivationFrame` to their downstream peer.
- **Direct Return Path**: The tail stage connects directly back to Node 0 to return `TokenResponseFrame` packets, avoiding unnecessary reverse-hop latency through intermediate nodes.
- **Sequence Integrity**: All frames retain the identical `session_id` and `sequence_id` to prevent stage desynchronization.

### 4.3 Multi-Hop WireGuard Admission Control
Each physical hop introduces network latency:
$$\text{Total Latency} = \text{RTT}_{\text{Hop 1}} + \text{RTT}_{\text{Hop 2}} + \dots + \text{RTT}_{\text{Hop } N-1} + \text{RTT}_{\text{Return}}$$
Admission criteria must verify that **every** inter-node link maintains:
- Direct WireGuard peering (no DERP relays).
- Per-hop $\text{RTT} \le 80\text{ ms}$ (cumulative round-trip $\le 200\text{ ms}$).

### 4.4 KV Cache Lockstep Synchronization
All stages must advance their KV cache pointers in lockstep:
- Prefill: Each stage processes $P$ prompt tokens.
- Decode: Each stage evaluates exactly 1 token position per autoregressive step.
- Reset: `FLAG_CLEAR_KV` must propagate through all stages before new tokens are evaluated.

### 4.5 Fail-Fast Failure Semantics
In an edge computing environment, self-healing multi-node reconfiguration introduces massive complexity. Initial $N$-stage support should adhere to **Fail-Fast** principles:
- If any node in the chain drops or encounters a socket timeout, the active generation aborts immediately.
- The coordinator returns an explicit error envelope to the client indicating the failed stage index.
- No dynamic re-slicing or hot-failover is attempted at runtime.

---

## 5. Implementation Phasing

1. **Phase 1 (Architecture & Deferral — Completed in FIX-28)**: Clearly document current 2-node scope and publish the N-stage architectural roadmap.
2. **Phase 2 (`pipeline.rs` Modularization — Documented in FIX-21)**: Extract coordinator, worker, and intermediate abstractions into separate files.
3. **Phase 3 (Multi-Split Slicing Engine)**: Extend `gguf_slicer.rs` to accept arbitrary split arrays and output intermediate identity norm slices.
4. **Phase 4 (Intermediate Service & Ring Transport)**: Build `PipelineIntermediateService` and direct tail-to-coordinator return socket channels.
5. **Phase 5 (Multi-Node HUD & Benchmarking)**: Update cluster telemetry HUD to visualize $N$-stage chains and publish latency/throughput comparisons against 2-node baselines.
