# AeroMESH: Architectural Boundaries & Operational Constraints (FIX-18)

This document explicitly outlines the deliberate architectural boundaries, hardware requirements, and design trade-offs of AeroMESH. These boundaries reflect conscious engineering decisions tailored for distributed edge inference across consumer laptops rather than multi-tenant cloud clusters.

---

## 1. Two-Node Heterogeneous Pipeline Maximum (FIX-28)

* **Current Architecture**: AeroMESH implements a point-to-point, two-stage pipeline split:
  * **Stage 1 (Coordinator / Laptop A)**: Tokenizes raw prompts, executes token embeddings and transformer blocks $0 \dots K$, applies an Identity RMSNorm, and dynamically quantizes intermediate activations.
  * **Stage 2 (Worker / Laptop B)**: Receives activation frames, executes blocks $K+1 \dots N-1$, applies the final RMSNorm, evaluates logits at the LM head, samples the next token, and loops back the token ID.
* **Why this boundary exists**: A two-stage pipeline establishes an immediate closed autoregressive feedback loop without intermediate tensor serialization hops or complex pipeline bubble scheduling (e.g., 1F1B or interleaved schedules).
* **Roadmap ($N \ge 3$)**: Supporting three or more compute nodes requires intermediate token-less relay stages and multi-peer TCP ring coordination.

---

## 2. Windows-Centric Process Isolation via Win32 Job Objects (FIX-29)

* **Current Architecture**: Child compute daemons (`ggml-rpc-server.exe` and local worker instances) are encapsulated in Win32 Job Objects configured with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` (`crates/aeromesh-engine/src/job_object.rs`).
* **Why this boundary exists**: Consumer gaming laptops running Windows 11 with discrete NVIDIA GeForce RTX GPUs are the primary presentation and operational target. Win32 Job Objects guarantee that if the coordinator crashes or is forcefully terminated, the Windows NT kernel forcefully reclaims all child processes, preventing zombie CUDA handles and VRAM leaks.
* **Cross-Platform Portability**: Deploying on Linux or macOS requires abstracting `job_object.rs` with `prctl(PR_SET_PDEATHSIG, SIGKILL)` or systemd cgroups v2 containment.

---

## 3. Pre-Shared Symmetric GGUF Weights (Zero-Weight Network Transfer)

* **Current Architecture**: Both Laptop A and Laptop B must possess the identical base `.gguf` file stored on local NVMe storage within their `models/` directories.
* **Why this boundary exists**: This is AeroMESH's foundational design principle. Paging 8 GB to 30 GB of model weights across standard Wi-Fi or VPN tunnels causes multi-minute cold starts and network saturation. AeroMESH performs deterministic local GGUF header slicing (`gguf_slicer.rs`), eliminating weight streaming and reducing runtime wire transfer to small activation vectors (~4 KB to 16 KB per step).
* **Mitigation**: Users must transfer the GGUF model to both machines prior to launching the cluster (e.g., via USB 3.0, LAN file share, or direct download).

---

## 4. Ephemeral In-Memory Context & Stateless Orchestration

* **Current Architecture**: The coordinator exposes an OpenAI-compliant HTTP API (`/v1/chat/completions`) that relies entirely on client-supplied conversation history (`messages: Vec<ChatMessage>`).
* **Why this boundary exists**: To maximize edge throughput and eliminate runtime overhead on resource-constrained laptops, the engine does not bundle a persistent SQLite, PostgreSQL, or Redis database. 
* **Operational Impact**: Model switching (`/api/model/switch`), process restarts, or reloading the Claymorphic web interface resets transient KV caches and session context. Context preservation is delegated to client storage.

---

## 5. Architectural Summary & Roadmap Matrix

| Constraint | Current Behavior | Underlying Reason | Target Milestone |
|---|---|---|---|
| **Cluster Topology** | Exactly 2 compute nodes (`FIX-28`) | Direct token feedback loop without bubble scheduling overhead | Multi-Node Activation Ring ($N \ge 3$) |
| **Operating System** | Windows 10/11 x64 (`FIX-29`) | Hardware baseline; Win32 Job Objects prevent VRAM leaks | POSIX `cgroups` / macOS abstraction |
| **Model Distribution** | Local NVMe model storage | Zero-weight streaming over low-bandwidth consumer Wi-Fi | Local P2P GGUF BitTorrent pre-seeding |
| **Conversation State** | In-memory ephemeral KV cache | Minimal edge footprint; OpenAI API client-driven state | Optional SQLite chat session persistence |
| **Context Length** | Bounded by local VRAM & `n_ctx` (`FIX-24`) | Hard limit enforced to prevent out-of-memory crashes | Dynamic RoPE scaling / FlashAttention-2 |
