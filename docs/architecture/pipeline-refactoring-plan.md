# Pipeline Module Refactoring Specification (FIX-21)

> **Status**: Deferred to Post-Demo Maintenance (Stability & Zero Regression Risk)  
> **Target Module**: `crates/aeromesh-engine/src/pipeline.rs`  
> **Current Size**: ~1,070 lines  
> **Post-Demo Target Structure**: `pipeline/coordinator_client.rs`, `pipeline/worker_service.rs`, `pipeline.rs`  
> **Current Quality Status**: 43 / 43 Workspace Unit Tests Passing, Zero Release Warnings

---

## 1. Executive Summary & Rationale for Deferral

`crates/aeromesh-engine/src/pipeline.rs` is the demo-critical core of the AeroMESH zero-weight distributed inference engine. It contains the complete distributed pipeline state machine:
- **`PipelineWorkerService`**: Remote worker server, mutual handshake verification (v3), activation frame receipt, tensor dequantization, Stage 2 GEMM evaluation, and penalty sampling.
- **`PipelineCoordinatorClient`**: Local Stage 1 prefill, intermediate activation quantization, transport frame dispatch, autoregressive token stream loopback, auto-reconnection, and shared-memory (SHM) IPC.
- **`tests_pipeline_safety`**: Comprehensive unit tests covering instance safety, error recovery, and context size reporting.

### Why Refactoring is Deferred to Post-Demo Maintenance
1. **Module Stability**: The module is fully tested, hardened against remote panics, and verified clean in release builds.
2. **Critical Path Risk**: `pipeline.rs` manages raw C++ FFI pointers, asynchronous TCP streams, and lockstep KV-cache synchronization. Modifying file boundaries immediately before live demonstrations introduces gratuitous regression risk for zero functional gain.
3. **Pure Code Organization**: This refactor is strictly an internal code hygiene improvement. It changes zero wire formats, zero algorithms, and zero user-facing APIs.

---

## 2. Target File Layout (Post-Demo)

The post-demo refactor will mechanically decompose `pipeline.rs` into focused submodules while keeping `pipeline.rs` as the thin root with 100% backward-compatible public re-exports:

```text
crates/aeromesh-engine/src/
  ├── pipeline.rs                          # Thin module root with documentation & re-exports
  └── pipeline/
        ├── coordinator_client.rs          # PipelineCoordinatorClient & generation loops
        ├── worker_service.rs              # PipelineWorkerService & TCP listener/dispatch
        └── tests.rs                       # Unit & safety integration tests
```

---

## 3. Submodule Responsibilities & Code Boundaries

### 3.1 `crates/aeromesh-engine/src/pipeline.rs` (Root Module)
Becomes a lightweight facade declaring submodules and re-exporting all existing public symbols:

```rust
//! Distributed pipeline-parallel execution primitives.
//!
//! Provides the coordinator client and worker service used to partition
//! transformer layers and stream intermediate activations across AeroMESH nodes.

mod coordinator_client;
mod worker_service;

#[cfg(test)]
mod tests;

pub use coordinator_client::PipelineCoordinatorClient;
pub use worker_service::PipelineWorkerService;
pub const SOCKET_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);
```

### 3.2 `crates/aeromesh-engine/src/pipeline/coordinator_client.rs` (~500 lines)
Contains the coordinator-side pipeline client:
- `pub struct PipelineCoordinatorClient`
- `PipelineCoordinatorClient::new()`
- `PipelineCoordinatorClient::generate()` (SSE generation stream)
- `PipelineCoordinatorClient::switch_model()` (transactional hot-swap)
- `PipelineCoordinatorClient::n_ctx()` and telemetry queries
- Prefill and autoregressive token decode loops
- Activation tensor quantization and frame serialization

### 3.3 `crates/aeromesh-engine/src/pipeline/worker_service.rs` (~350 lines)
Contains the remote worker server:
- `pub struct PipelineWorkerService`
- `PipelineWorkerService::new()`
- `PipelineWorkerService::run()` (async TCP listener loop)
- Handshake v3 mutual authentication verification
- Activation frame deserialization and shape invariant checks
- Stage 2 FFI model forward pass (`instance.decode()`)
- Penalty sampling and `TokenResponseFrame` transmission

### 3.4 `crates/aeromesh-engine/src/pipeline/tests.rs` (~200 lines)
Houses existing characterization tests:
- `test_coordinator_status_rejection_when_failed`
- `test_switch_model_invalid_path_preserves_old_model`
- `test_generate_pipeline_rejects_when_failed`
- `test_coordinator_n_ctx_reporting`

---

## 4. Public API Invariant Checklist

To prevent breaking callers (`server.rs`, `main.rs`, CLI binaries), the following exports must remain identical:

| Exported Symbol | Type | Current Import Path | Target Post-Refactor Path |
|---|---|---|---|
| `PipelineCoordinatorClient` | `struct` | `crate::pipeline::PipelineCoordinatorClient` | `crate::pipeline::PipelineCoordinatorClient` (via re-export) |
| `PipelineWorkerService` | `struct` | `crate::pipeline::PipelineWorkerService` | `crate::pipeline::PipelineWorkerService` (via re-export) |
| `SOCKET_TIMEOUT` | `const Duration` | `crate::pipeline::SOCKET_TIMEOUT` | `crate::pipeline::SOCKET_TIMEOUT` (via re-export) |

No caller outside `pipeline.rs` will require code changes.

---

## 5. Multi-PR Execution Strategy (Post-Demo)

To maintain safety and bisectability, the post-demo refactor must be executed across small, mechanical commits:

1. **PR 1: Test Baseline Verification**
   - Run `cargo test --workspace` and record clean test baseline.
2. **PR 2: Extract Coordinator Client**
   - Create `pipeline/coordinator_client.rs`.
   - Move `PipelineCoordinatorClient` implementation.
   - Add `pub use coordinator_client::PipelineCoordinatorClient;` to `pipeline.rs`.
   - Validate with `cargo check --workspace` and `cargo test --workspace`.
3. **PR 3: Extract Worker Service**
   - Create `pipeline/worker_service.rs`.
   - Move `PipelineWorkerService` implementation.
   - Add `pub use worker_service::PipelineWorkerService;` to `pipeline.rs`.
   - Validate with `cargo check --workspace` and `cargo test --workspace`.
4. **PR 4: Extract Tests & Clean Module Docs**
   - Move `#[cfg(test)] mod tests` to `pipeline/tests.rs`.
   - Verify zero compiler warnings via `cargo clippy --workspace --all-targets`.
5. **PR 5: Local Test Mesh Smoke Test**
   - Run full end-to-end inference via `.\start.ps1 all`.

---

## 6. Acceptance Criteria

The post-demo refactoring will be considered complete when:
- [ ] `crates/aeromesh-engine/src/pipeline.rs` is under 60 lines.
- [ ] `PipelineCoordinatorClient` resides in `pipeline/coordinator_client.rs`.
- [ ] `PipelineWorkerService` resides in `pipeline/worker_service.rs`.
- [ ] All 43 workspace tests pass without modification.
- [ ] `cargo check --workspace --release` passes with 0 warnings.
- [ ] Full local mesh (`.\start.ps1 all`) generates streaming tokens successfully.
