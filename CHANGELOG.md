# AeroMESH Changelog

All notable changes to the AeroMESH project are documented in this file.

## [v0.1.0] - 2026-10-10

### Security & Authentication
- **Gateway Loopback & Key Guard (SEC-01 / Tier 0)**: Secured `app.py` auth verification with fail-closed loopback detection and authorization checks.
- **Upstream Credential Injection**: Stripped incoming client `Authorization` header prior to injecting gateway credentials to prevent double-header injection vulnerabilities.
- **Bounded HTTP Payloads**: Enforced 2 MB request payload ceilings on gateway routes to prevent unbounded memory buffering.
- **XSS & Content Sanitization**: Pinned CDN assets with Subresource Integrity (SRI) hashes and configured secure markdown DOM sanitization in the web UI.

### Architecture & Engine
- **Runtime FFI Dynamic Resolution**: Switched all native llama.cpp symbols from static linking to runtime dynamic loading, allowing engine tests and builds to pass on runners without native DLLs.
- **Protocol v4 Wire Format**: Formalized 44-byte binary activation frame header supporting multi-node topologies, token positions, and explicit flags (`FLAG_CLEAR_KV`, `FLAG_IS_PROMPT`).
- **Identity-RMSNorm Injection**: Extended GGUF slicer identity normalization support to F32, F16, BF16, and Q8_0 with loud failures on unsupported quantization types.
- **Slice Cache Collision Prevention**: Encoded split points and stage layer counts into partition filenames to prevent stale slice reuse across differing partition runs.
- **Real-Time Generation Abort**: Threaded lock-free atomic cancellation flag into the coordinator decode loop, allowing `/api/chat/abort` to interrupt active generations immediately.
- **Full OpenAI SSE Streaming Compliance**: Emitted `role: "assistant"` on initial token chunks, added terminal chunks with `finish_reason: "stop"`, and transmitted standard `data: [DONE]` event streams.
- **Accurate Wire Metrics**: Replaced hardcoded client-side estimates with real server-reported telemetry based on active tensor hidden dimensions and quantization bytes.

### Build & Tooling
- **CI / CD Pipeline Integrity**: Added `pyproject.toml` pythonpath configuration, enabled `uv sync` lockfile installs, and validated clean execution of both Rust and Python test suites.
- **CLI & Setup Hardening**: Lazy initialized native compute libraries in `aeromesh-cli` to enable instant `--help` and status queries without GPU load.
