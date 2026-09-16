# Stateless Conversation Policy & Persistence Architecture (FIX-20)

> **Status**: Intended Architecture (Deliberate Design Non-Goal)  
> **Component**: `aeromesh-core`, `aeromesh-engine`, `aeromesh-cli`, `app.py`  
> **Classification**: Architectural Decision Record (ADR)  
> **Related Standards**: OpenAI Chat Completions Schema, RFC 7230, GGUF Specification

---

## 1. Executive Summary & Policy Statement

AeroMESH adheres to the established runtime model of local high-performance LLM engines (such as `llama-server`, `vLLM`, and `Ollama`):

> **The core AeroMESH distributed inference engine is strictly stateless with respect to durable conversation history.**
> Active inference context, intermediate activation tensors, and KV cache buffers live exclusively in volatile memory (RAM/VRAM). Durable conversation history is intentionally **not** written to disk by `aeromesh-core`, `aeromesh-engine`, or the binary network pipeline.

Conversation persistence is explicitly categorized as an **application-layer concern** and is decoupled from the low-level inference cluster. The AeroMESH web dashboard manages conversation continuity locally within the browser's `localStorage` (`aeromesh_conversations_v2`), while the distributed engine remains clean, fast, and privacy-conservative.

---

## 2. Technical Rationale & Architectural Decision Drivers

Reviewers and contributors often question whether a distributed LLM cluster requires a centralized database (such as SQLite, PostgreSQL, or Redis) to store conversations. AeroMESH deliberately excludes a database layer based on four core architectural principles:

### 2.1 The Industry Standard for Local LLM Servers is In-Memory Statelessness
OpenAI-compatible inference engines (`llama-server`, `Ollama`, `vLLM`) are designed as stateless compute servers. Multi-turn dialogue is accomplished by having the client resend the accumulated conversation array (`messages: [{"role": "user", ...}, {"role": "assistant", ...}]`) with each turn. The inference coordinator evaluates the prompt sequence, re-evaluating or extending its in-memory KV cache, without querying a persistent database.

### 2.2 Absence of Database Dependencies Protects Engine Simplicity
AeroMESH is engineered to compile with zero external database drivers. The engine interacts directly with the operating system kernel:
- File-based GGUF weights on NVMe storage memory-mapped via Win32 / POSIX `mmap`.
- Low-latency activation vector streaming over TCP sockets or shared memory (`aeromesh-core::shm`).
- Process containment via Win32 Job Objects (`aeromesh-engine::job_object`).

Injecting a database into the Rust workspace would introduce substantial dependency overhead, schema migration burdens, lock contention, and foreign runtime failure modes completely unrelated to distributed matrix multiplication.

### 2.3 Distributed KV-Cache Deserialization is Fragile and Prohibitive
In a 2-node pipeline cluster, Stage 1 (Coordinator: layers $0 \dots K$) and Stage 2 (Worker: layers $K+1 \dots N-1$) advance their respective internal KV caches in lockstep across sequence steps. 
Attempting to serialize, persist to disk, and restore distributed KV caches across process restarts would require:
- Exact model quantization and tensor alignment matching.
- Exact layer partition boundaries ($K$).
- Context length ($n_{\text{ctx}}$) parity across both machines.
- Distributed synchronization primitives to ensure both caches resume at the exact same sequence token.

Because consumer SSD I/O bandwidth (~2–5 GB/s) is orders of magnitude slower than unified VRAM/DRAM bandwidth (~100–800 GB/s), cold prompt prefill across consumer GPUs is faster and vastly more reliable than deserializing multi-gigabyte KV cache states from disk.

### 2.4 Privacy-Conservative Defaults
AeroMESH is designed for edge and consumer deployments where users process sensitive personal, medical, or corporate prompts. By ensuring that prompts, completions, and activation vectors never touch the host filesystem:
- No residual plain-text chat transcripts remain on disk after process termination.
- Zero data leakage occurs if an edge node's NVMe drive is inspected or stolen.
- Compliance with strict ephemeral data-handling policies is achieved out-of-the-box.

---

## 3. Classification of State in AeroMESH

To eliminate ambiguity, AeroMESH explicitly distinguishes between three independent tiers of state:

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. Model Weights (Durable / Persistent on Disk)                         │
│ ─────────────────────────────────────────────────────────────────────── │
│ • Stored as GGUF files in models/ on local NVMe SSD                    │
│ • Symmetric: Stored locally on both Coordinator and Worker              │
│ • Read-only memory-mapped (mmap) during runtime execution               │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ 2. KV Cache & Activation State (Volatile / In-Memory Only)              │
│ ─────────────────────────────────────────────────────────────────────── │
│ • Held in GPU VRAM or host RAM during active inference                  │
│ • Advanced in lockstep between Coordinator and Worker                   │
│ • Cleared on model switch, engine restart, context overflow, or clear-kv│
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ 3. Conversation Transcripts (Application / Browser Layer Only)          │
│ ─────────────────────────────────────────────────────────────────────── │
│ • User prompts, assistant replies, reasoning traces, timestamps         │
│ • Stored exclusively in client browser localStorage (aeromesh_conv_v2)  │
│ • Exportable to Markdown via UI button; zero storage on server disk     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Protocol Clarification: `session_id` vs Database Keys

The AeroMESH binary wire protocol (`PROTOCOL_VERSION = 3`) defines explicit session headers:

```rust
pub struct ActivationHeader {
    pub magic: u32,
    pub version: u32,
    pub session_id: u64,
    pub sequence_id: u64,
    pub layer_start: u16,
    pub layer_end: u16,
    pub flags: u32,
    pub hidden_dim: u32,
    pub seq_len: u32,
    pub payload_bytes: u32,
}
```

### Critical Rule: `session_id` is a Wire Coordination Token, NOT a Database Key
- `session_id` is an ephemeral 64-bit random integer generated by the coordinator when initializing a streaming generation session.
- Its sole purpose is to pair activation frames with corresponding token response frames across network latency and detect out-of-order packet interleaving.
- It does **not** index a database record.
- It is discarded immediately upon generation completion (`FLAG_IS_PROMPT` or EOS).
- Client software must not treat `session_id` as a durable conversation identifier.

---

## 5. Multi-Turn Conversation Mechanics

Multi-turn dialogues operate cleanly without backend persistence:

1. **Client Submits Conversation History**:
   The web dashboard or API client maintains the message history array and transmits it in the request payload:
   ```json
   {
     "model": "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
     "messages": [
       {"role": "user", "content": "What is pipeline parallelism?"},
       {"role": "assistant", "content": "Pipeline parallelism partitions layers across nodes..."},
       {"role": "user", "content": "How does AeroMESH avoid transferring weights?"}
     ],
     "stream": true
   }
   ```
2. **Coordinator Evaluates Prompt**:
   The coordinator tokenizes the full prompt sequence ($S$). If `FLAG_CLEAR_KV` is set (or if the session is new), internal KV caches are reset, and the full sequence is evaluated through layers $0 \dots K$ before streaming activations to the worker.
3. **Worker Samples Token**:
   Stage 2 processes layers $K+1 \dots N-1$, samples the first token $T_1$, and streaming begins.
4. **Subsequent Generation**:
   The autoregressive loop advances token-by-token until generation completes.

---

## 6. Where Persistence Belongs: Application & Gateway Layer

If persistent chat logging is required for a specific business use case, it must be implemented outside the core engine:

### 6.1 Browser Client Storage (Currently Implemented)
The claymorphic Single Page Application (`static/js/app.js`) implements client-side persistence:
- Chat sessions are stored in `window.localStorage` under key `aeromesh_conversations_v2`.
- Conversations restore automatically upon browser reload.
- Full transcripts can be exported as structured Markdown files via the "Export Chat" button.
- All stored conversations can be wiped instantly using the "Clear All History" button.

### 6.2 Optional FastAPI Gateway Logging (Future Extensibility)
If server-side audit trails are desired, the Python gateway (`app.py`) can optionally append messages to JSONL files (e.g., `.aeromesh/conversations/YYYY-MM-DD.jsonl`):
- Must be strictly opt-in (`AEROMESH_ENABLE_CHAT_LOGGING=1`).
- Disabled by default.
- Completely decoupled from the Rust inference engine.

---

## 7. Architectural Guarantees & Summary

| Property | AeroMESH Core Engine | Web UI / Browser |
|---|:---:|:---:|
| **Persists Chat History to Disk** | ❌ No | ✅ Yes (`localStorage`) |
| **Persists Model Weights to Disk** | ✅ Yes (Local GGUF) | ❌ No |
| **Persists KV-Cache to Disk** | ❌ No (Volatile VRAM/RAM) | ❌ No |
| **Requires Database (SQL/NoSQL)** | ❌ No | ❌ No |
| **Supports Multi-Turn Dialogue** | ✅ Yes (Context Resend) | ✅ Yes |
| **Supports Markdown Chat Export** | ❌ No (API Only) | ✅ Yes |
| **Privacy-Safe Zero-Trace Operation** | ✅ Yes | Optional (Clear History) |
