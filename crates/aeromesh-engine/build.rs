//! Build script for `aeromesh-engine`.
//!
//! All `llama.cpp` and `ggml` native symbols are resolved dynamically at
//! runtime via `LoadLibraryW` / `GetProcAddress` in `src/llama_ffi.rs`.
//!
//! We deliberately do **not** emit `cargo:rustc-link-lib=dylib=llama` or
//! `ggml`, because a load-time import would:
//!   1. require compile-time `.lib` / `.dll.a` import libraries (`dlltool`
//!      workarounds on MinGW/GNU),
//!   2. crash `cargo test` and read-only CLI subcommands with
//!      `STATUS_DLL_NOT_FOUND` (0xc0000135) whenever `llama.dll` is absent, and
//!   3. prevent the Windows installer from swapping CPU / CUDA 12 / Vulkan
//!      runtime backends cleanly at install time.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=LLAMA_LIB_DIR");
}
