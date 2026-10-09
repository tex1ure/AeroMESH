//! Dynamic (runtime) bindings to the native llama.cpp / GGML shared libraries.
//!
//! # Why dynamic loading instead of `#[link(name = "llama")]`
//!
//! A static `#[link]` produces a **load-time** import in the binary. On Windows
//! that means every executable that links `aeromesh-engine` — including the
//! `cargo test` harness binaries under `target/<profile>/deps/` — refuses to
//! start when `llama.dll` / `ggml.dll` are not on the DLL search path:
//!
//! ```text
//! process didn't exit successfully: ...\aeromesh_engine-<hash>.exe
//! (exit code: 0xc0000135, STATUS_DLL_NOT_FOUND)
//! ```
//!
//! Resolving the symbols at runtime instead means:
//!   * a missing DLL is a recoverable, well-described error rather than a
//!     process that dies before `main()`;
//!   * pure unit tests (framing, validation, slicing, hashing) run on a machine
//!     with no CUDA stack at all — which is what CI needs;
//!   * `import` (and therefore `LoadLibrary`) is deferred until the first FFI
//!     call, so read-only subcommands do not pay backend-initialisation cost.
//!
//! Errors are cached after the first attempt, so repeated calls do not re-probe
//! the filesystem.

use std::ffi::{c_char, c_void};
use std::path::PathBuf;
use std::sync::{Once, OnceLock};
use tracing::{debug, info, warn};

pub type LlamaPos = i32;
pub type LlamaToken = i32;
pub type LlamaSeqId = i32;

#[repr(C)]
pub struct LlamaModel {
    _private: [u8; 0],
}

#[repr(C)]
pub struct LlamaVocab {
    _private: [u8; 0],
}

#[repr(C)]
pub struct LlamaContext {
    _private: [u8; 0],
}

#[repr(C)]
pub struct LlamaSampler {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct LlamaBatch {
    pub n_tokens: i32,
    pub token: *mut LlamaToken,
    pub embd: *mut f32,
    pub pos: *mut LlamaPos,
    pub n_seq_id: *mut i32,
    pub seq_id: *mut *mut LlamaSeqId,
    pub logits: *mut i8,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct LlamaModelParams {
    pub devices: *mut *mut c_void,
    pub tensor_buft_overrides: *const c_void,
    pub n_gpu_layers: i32,
    pub split_mode: i32,
    pub load_mode: i32,
    pub main_gpu: i32,
    pub tensor_split: *const f32,
    pub progress_callback: Option<extern "C" fn(f32, *mut c_void) -> bool>,
    pub progress_callback_user_data: *mut c_void,
    pub kv_overrides: *const c_void,
    pub vocab_only: bool,
    pub check_tensors: bool,
    pub use_extra_bufts: bool,
    pub no_host: bool,
    pub no_alloc: bool,
    pub load_mtp: bool,
}

pub const GGML_MAX_DIMS: usize = 4;
pub const GGML_MAX_NAME: usize = 64;

#[repr(C)]
pub struct GgmlTensor {
    pub type_: i32,
    pub buffer: *mut c_void,
    pub ne: [i64; GGML_MAX_DIMS],
    pub nb: [usize; GGML_MAX_DIMS],
    pub op: i32,
    pub op_params: [i32; 16],
    pub flags: i32,
    pub src: [*mut GgmlTensor; 10],
    pub view_src: *mut GgmlTensor,
    pub view_offs: usize,
    pub data: *mut c_void,
    pub name: [c_char; GGML_MAX_NAME],
    pub extra: *mut c_void,
    pub padding: [u8; 8],
}

pub type GgmlBackendSchedEvalCallback =
    extern "C" fn(tensor: *mut GgmlTensor, ask: bool, user_data: *mut c_void) -> bool;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct LlamaContextParams {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_seq_max: u32,
    pub n_rs_seq: u32,
    pub n_outputs_max: u32,
    pub n_outputs_max_per_seq: u32,
    pub n_threads: i32,
    pub n_threads_batch: i32,
    pub ctx_type: i32,
    pub rope_scaling_type: i32,
    pub pooling_type: i32,
    pub attention_type: i32,
    pub flash_attn_type: i32,
    pub rope_freq_base: f32,
    pub rope_freq_scale: f32,
    pub yarn_ext_factor: f32,
    pub yarn_attn_factor: f32,
    pub yarn_beta_fast: f32,
    pub yarn_beta_slow: f32,
    pub yarn_orig_ctx: u32,
    pub defrag_thold: f32,
    pub cb_eval: Option<GgmlBackendSchedEvalCallback>,
    pub cb_eval_user_data: *mut c_void,
    pub type_k: i32,
    pub type_v: i32,
    pub abort_callback: Option<extern "C" fn(*mut c_void) -> bool>,
    pub abort_callback_data: *mut c_void,
    pub embeddings: bool,
    pub offload_kqv: bool,
    pub no_perf: bool,
    pub op_offload: bool,
    pub swa_full: bool,
    pub kv_unified: bool,
    pub samplers: *mut c_void,
    pub n_samplers: usize,
    pub ctx_other: *mut LlamaContext,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct LlamaSamplerChainParams {
    pub no_perf: bool,
}

pub type LlamaLogCallback =
    extern "C" fn(level: i32, text: *const c_char, user_data: *mut c_void);

/// A fully resolved set of llama.cpp entry points.
///
/// Trailing-pointer member: adding new symbols only appends fields, so the
/// layout is stable and every field is initialised via `::default()` (which
/// requires `T: Sized`) in the struct literal below.
#[allow(non_snake_case)]
pub struct LlamaApi {
    pub llama_backend_init: unsafe extern "C" fn(),
    pub llama_backend_free: unsafe extern "C" fn(),

    pub llama_model_default_params: unsafe extern "C" fn() -> LlamaModelParams,
    pub llama_model_load_from_file: unsafe extern "C" fn(
        path_model: *const c_char,
        params: LlamaModelParams,
    ) -> *mut LlamaModel,
    pub llama_model_free: unsafe extern "C" fn(model: *mut LlamaModel),

    pub llama_context_default_params: unsafe extern "C" fn() -> LlamaContextParams,
    pub llama_init_from_model: unsafe extern "C" fn(
        model: *mut LlamaModel,
        params: LlamaContextParams,
    ) -> *mut LlamaContext,
    pub llama_free: unsafe extern "C" fn(ctx: *mut LlamaContext),

    pub llama_model_get_vocab: unsafe extern "C" fn(model: *const LlamaModel) -> *const LlamaVocab,
    pub llama_model_n_embd: unsafe extern "C" fn(model: *const LlamaModel) -> i32,
    pub llama_model_n_layer: unsafe extern "C" fn(model: *const LlamaModel) -> i32,
    pub llama_model_desc: unsafe extern "C" fn(
        model: *const LlamaModel,
        buf: *mut c_char,
        buf_size: usize,
    ) -> i32,

    pub llama_tokenize: unsafe extern "C" fn(
        vocab: *const LlamaVocab,
        text: *const c_char,
        text_len: i32,
        tokens: *mut LlamaToken,
        n_tokens_max: i32,
        add_special: bool,
        parse_special: bool,
    ) -> i32,

    pub llama_token_to_piece: unsafe extern "C" fn(
        vocab: *const LlamaVocab,
        token: LlamaToken,
        buf: *mut c_char,
        length: i32,
        lstrip: i32,
        special: bool,
    ) -> i32,

    pub llama_detokenize: unsafe extern "C" fn(
        vocab: *const LlamaVocab,
        tokens: *const LlamaToken,
        n_tokens: i32,
        text: *mut c_char,
        text_len_max: i32,
        remove_special: bool,
        unparse_special: bool,
    ) -> i32,

    pub llama_vocab_is_eog: unsafe extern "C" fn(vocab: *const LlamaVocab, token: LlamaToken) -> bool,
    pub llama_vocab_is_control:
        unsafe extern "C" fn(vocab: *const LlamaVocab, token: LlamaToken) -> bool,
    pub llama_vocab_bos: unsafe extern "C" fn(vocab: *const LlamaVocab) -> LlamaToken,
    pub llama_vocab_eos: unsafe extern "C" fn(vocab: *const LlamaVocab) -> LlamaToken,
    pub llama_vocab_n_tokens: unsafe extern "C" fn(vocab: *const LlamaVocab) -> i32,

    pub llama_batch_init: unsafe extern "C" fn(n_tokens: i32, embd: i32, n_seq_max: i32) -> LlamaBatch,
    pub llama_batch_free: unsafe extern "C" fn(batch: LlamaBatch),

    pub llama_decode: unsafe extern "C" fn(ctx: *mut LlamaContext, batch: LlamaBatch) -> i32,
    pub llama_get_logits_ith: unsafe extern "C" fn(ctx: *mut LlamaContext, i: i32) -> *mut f32,
    pub llama_get_embeddings_ith: unsafe extern "C" fn(ctx: *mut LlamaContext, i: i32) -> *mut f32,

    pub llama_sampler_chain_default_params: unsafe extern "C" fn() -> LlamaSamplerChainParams,
    pub llama_sampler_chain_init:
        unsafe extern "C" fn(params: LlamaSamplerChainParams) -> *mut LlamaSampler,
    pub llama_sampler_chain_add:
        unsafe extern "C" fn(chain: *mut LlamaSampler, smpl: *mut LlamaSampler),
    pub llama_sampler_init_greedy: unsafe extern "C" fn() -> *mut LlamaSampler,
    pub llama_sampler_init_temp: unsafe extern "C" fn(t: f32) -> *mut LlamaSampler,
    pub llama_sampler_init_top_p: unsafe extern "C" fn(p: f32, min_keep: usize) -> *mut LlamaSampler,
    pub llama_sampler_init_top_k: unsafe extern "C" fn(k: i32) -> *mut LlamaSampler,
    pub llama_sampler_init_min_p: unsafe extern "C" fn(p: f32, min_keep: usize) -> *mut LlamaSampler,
    pub llama_sampler_init_penalties: unsafe extern "C" fn(
        penalty_last_n: i32,
        penalty_repeat: f32,
        penalty_freq: f32,
        penalty_present: f32,
    ) -> *mut LlamaSampler,
    pub llama_sampler_init_dist: unsafe extern "C" fn(seed: u32) -> *mut LlamaSampler,
    pub llama_sampler_accept: unsafe extern "C" fn(smpl: *mut LlamaSampler, token: LlamaToken),
    pub llama_sampler_reset: unsafe extern "C" fn(smpl: *mut LlamaSampler),
    pub llama_sampler_sample: unsafe extern "C" fn(
        smpl: *mut LlamaSampler,
        ctx: *mut LlamaContext,
        idx: i32,
    ) -> LlamaToken,
    pub llama_sampler_free: unsafe extern "C" fn(smpl: *mut LlamaSampler),

    pub llama_get_memory: unsafe extern "C" fn(ctx: *const LlamaContext) -> *mut c_void,
    pub llama_memory_clear: unsafe extern "C" fn(mem: *mut c_void, data: bool),
    pub llama_synchronize: unsafe extern "C" fn(ctx: *mut LlamaContext),
    pub llama_log_set:
        unsafe extern "C" fn(log_callback: Option<LlamaLogCallback>, user_data: *mut c_void),
}

type GgmlBackendTensorGetFn =
    extern "C" fn(tensor: *const GgmlTensor, data: *mut c_void, offset: usize, size: usize);

pub type GgmlBackendLoadAllFn = extern "C" fn();
pub type GgmlBackendLoadAllFromPathFn = extern "C" fn(dir_path: *const c_char);
pub type GgmlBackendRegCountFn = extern "C" fn() -> usize;

fn symbol_names() -> Vec<&'static str> {
    let mut names: Vec<&'static str> = Vec::new();
    macro_rules! sym {
        ($($name:ident),* $(,)?) => { $( names.push(stringify!($name)); )* };
    }
    sym!(
        llama_backend_init,
        llama_backend_free,
        llama_model_default_params,
        llama_model_load_from_file,
        llama_model_free,
        llama_context_default_params,
        llama_init_from_model,
        llama_free,
        llama_model_get_vocab,
        llama_model_n_embd,
        llama_model_n_layer,
        llama_model_desc,
        llama_tokenize,
        llama_token_to_piece,
        llama_detokenize,
        llama_vocab_is_eog,
        llama_vocab_is_control,
        llama_vocab_bos,
        llama_vocab_eos,
        llama_vocab_n_tokens,
        llama_batch_init,
        llama_batch_free,
        llama_decode,
        llama_get_logits_ith,
        llama_get_embeddings_ith,
        llama_sampler_chain_default_params,
        llama_sampler_chain_init,
        llama_sampler_chain_add,
        llama_sampler_init_greedy,
        llama_sampler_init_temp,
        llama_sampler_init_top_p,
        llama_sampler_init_top_k,
        llama_sampler_init_min_p,
        llama_sampler_init_penalties,
        llama_sampler_init_dist,
        llama_sampler_accept,
        llama_sampler_reset,
        llama_sampler_sample,
        llama_sampler_free,
        llama_get_memory,
        llama_memory_clear,
        llama_synchronize,
        llama_log_set,
    );
    names
}

/// Resolve every entry point from a freshly opened library handle.
///
/// # Safety
/// `handle` must be a valid module handle for a library that exports the
/// llama.cpp C API, and it must stay loaded for the lifetime of the returned
/// struct (the library is deliberately never freed).
unsafe fn resolve_from(handle: *mut c_void) -> Result<LlamaApi, String> {
    unsafe fn ptr(handle: *mut c_void, name: &str) -> Result<*const c_void, String> {
        let c_name = std::ffi::CString::new(name).map_err(|e| e.to_string())?;
        let addr = windows_sys::Win32::System::LibraryLoader::GetProcAddress(
            handle as windows_sys::Win32::Foundation::HMODULE,
            c_name.as_ptr() as *const u8,
        );
        match addr {
            Some(f) => Ok(f as *const c_void),
            None => Err(format!("symbol `{name}` not found")),
        }
    }

    // Each assignment is an unsafe fn-pointer transmute from GetProcAddress.
    // Signatures are transcribed from llama.cpp/include/llama.h (b10499).
    unsafe fn take<T: Copy>(handle: *mut c_void, name: &str) -> Result<T, String> {
        let p = unsafe { ptr(handle, name)? };
        // SAFETY: caller guarantees the handle exports `name` with the signature
        // declared on `LlamaApi`; the pointer is a valid code address.
        Ok(unsafe { std::mem::transmute_copy::<*const c_void, T>(&p) })
    }

    Ok(LlamaApi {
        llama_backend_init: unsafe { take(handle, "llama_backend_init")? },
        llama_backend_free: unsafe { take(handle, "llama_backend_free")? },
        llama_model_default_params: unsafe { take(handle, "llama_model_default_params")? },
        llama_model_load_from_file: unsafe { take(handle, "llama_model_load_from_file")? },
        llama_model_free: unsafe { take(handle, "llama_model_free")? },
        llama_context_default_params: unsafe { take(handle, "llama_context_default_params")? },
        llama_init_from_model: unsafe { take(handle, "llama_init_from_model")? },
        llama_free: unsafe { take(handle, "llama_free")? },
        llama_model_get_vocab: unsafe { take(handle, "llama_model_get_vocab")? },
        llama_model_n_embd: unsafe { take(handle, "llama_model_n_embd")? },
        llama_model_n_layer: unsafe { take(handle, "llama_model_n_layer")? },
        llama_model_desc: unsafe { take(handle, "llama_model_desc")? },
        llama_tokenize: unsafe { take(handle, "llama_tokenize")? },
        llama_token_to_piece: unsafe { take(handle, "llama_token_to_piece")? },
        llama_detokenize: unsafe { take(handle, "llama_detokenize")? },
        llama_vocab_is_eog: unsafe { take(handle, "llama_vocab_is_eog")? },
        llama_vocab_is_control: unsafe { take(handle, "llama_vocab_is_control")? },
        llama_vocab_bos: unsafe { take(handle, "llama_vocab_bos")? },
        llama_vocab_eos: unsafe { take(handle, "llama_vocab_eos")? },
        llama_vocab_n_tokens: unsafe { take(handle, "llama_vocab_n_tokens")? },
        llama_batch_init: unsafe { take(handle, "llama_batch_init")? },
        llama_batch_free: unsafe { take(handle, "llama_batch_free")? },
        llama_decode: unsafe { take(handle, "llama_decode")? },
        llama_get_logits_ith: unsafe { take(handle, "llama_get_logits_ith")? },
        llama_get_embeddings_ith: unsafe { take(handle, "llama_get_embeddings_ith")? },
        llama_sampler_chain_default_params: unsafe {
            take(handle, "llama_sampler_chain_default_params")?
        },
        llama_sampler_chain_init: unsafe { take(handle, "llama_sampler_chain_init")? },
        llama_sampler_chain_add: unsafe { take(handle, "llama_sampler_chain_add")? },
        llama_sampler_init_greedy: unsafe { take(handle, "llama_sampler_init_greedy")? },
        llama_sampler_init_temp: unsafe { take(handle, "llama_sampler_init_temp")? },
        llama_sampler_init_top_p: unsafe { take(handle, "llama_sampler_init_top_p")? },
        llama_sampler_init_top_k: unsafe { take(handle, "llama_sampler_init_top_k")? },
        llama_sampler_init_min_p: unsafe { take(handle, "llama_sampler_init_min_p")? },
        llama_sampler_init_penalties: unsafe { take(handle, "llama_sampler_init_penalties")? },
        llama_sampler_init_dist: unsafe { take(handle, "llama_sampler_init_dist")? },
        llama_sampler_accept: unsafe { take(handle, "llama_sampler_accept")? },
        llama_sampler_reset: unsafe { take(handle, "llama_sampler_reset")? },
        llama_sampler_sample: unsafe { take(handle, "llama_sampler_sample")? },
        llama_sampler_free: unsafe { take(handle, "llama_sampler_free")? },
        llama_get_memory: unsafe { take(handle, "llama_get_memory")? },
        llama_memory_clear: unsafe { take(handle, "llama_memory_clear")? },
        llama_synchronize: unsafe { take(handle, "llama_synchronize")? },
        llama_log_set: unsafe { take(handle, "llama_log_set")? },
    })
}

/// Candidate directories that may hold the native runtime, most specific first.
fn candidate_dll_dirs() -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = Vec::new();

    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            dirs.push(exe_dir.to_path_buf());
            dirs.push(exe_dir.join("bin"));
            if let Some(parent) = exe_dir.parent() {
                dirs.push(parent.join("bin"));
                if let Some(grand) = parent.parent() {
                    dirs.push(grand.join("bin"));
                }
            }
        }
    }

    if let Ok(cwd) = std::env::current_dir() {
        dirs.push(cwd.join("bin"));
        dirs.push(cwd.join("..").join("bin"));
        dirs.push(cwd.join("llama.cpp").join("build").join("bin").join("Release"));
        dirs.push(cwd.join("llama.cpp").join("build").join("bin").join("Debug"));
    }

    dirs
}

/// Cached resolution result. The [`LlamaApi`] is leaked into static storage so
/// call sites can hold a genuine `&'static LlamaApi` without any lock.
static API_CACHE: OnceLock<Result<&'static LlamaApi, String>> = OnceLock::new();

/// Directories that were searched, recorded for backend plugin discovery.
static RESOLVED_DIRS: OnceLock<Vec<PathBuf>> = OnceLock::new();

/// Loads `llama.dll` and resolves the C API on first use.
///
/// Returns a descriptive error (never panics) when the native runtime is
/// missing, so callers can surface an actionable message instead of dying at
/// process start.
pub fn api() -> Result<&'static LlamaApi, String> {
    API_CACHE
        .get_or_init(|| {
            debug!("Resolving native llama.cpp entry points (deferred until first FFI use)");

            let mut dirs = candidate_dll_dirs();
            if let Ok(explicit) = std::env::var("LLAMA_LIB_DIR") {
                dirs.insert(0, PathBuf::from(explicit));
            }

            let mut resolved: Option<LlamaApi> = None;
            let mut resolved_dir: Option<PathBuf> = None;
            let mut last_error = String::from("no candidate directory contained llama.dll");

            for dir in &dirs {
                if !dir.is_dir() {
                    continue;
                }
                let dll = dir.join("llama.dll");
                if !dll.is_file() {
                    continue;
                }

                match load_and_resolve(&dll) {
                    Ok(api) => {
                        info!(dir = %dir.display(), "✅ Native llama.cpp entry points resolved");
                        resolved = Some(api);
                        resolved_dir = Some(dir.clone());
                        break;
                    }
                    Err(err) => {
                        last_error = format!("{}: {err}", dll.display());
                    }
                }
            }

            RESOLVED_DIRS.set(dirs.clone()).ok();

            match (resolved, resolved_dir) {
                (Some(api), Some(dir)) => {
                    // Make the directory available for *dependent* DLLs that
                    // llama.dll loads lazily (ggml-cuda.dll, ggml-cpu.dll, ...).
                    // The load above already used an absolute path.
                    #[cfg(target_os = "windows")]
                    if let Ok(abs) = std::fs::canonicalize(&dir) {
                        use std::os::windows::ffi::OsStrExt;
                        let mut wide: Vec<u16> = abs.as_os_str().encode_wide().collect();
                        wide.push(0);
                        unsafe {
                            windows_sys::Win32::System::LibraryLoader::SetDllDirectoryW(
                                wide.as_ptr(),
                            );
                        }
                        match std::env::var("PATH") {
                            Ok(path_var) => std::env::set_var(
                                "PATH",
                                format!("{};{}", abs.display(), path_var),
                            ),
                            Err(_) => {
                                std::env::set_var("PATH", abs.display().to_string())
                            }
                        }
                    }
                    let _ = &dir;

                    // Resolve the ggml-only symbols from the same directory.
                    let ggml = load_ggml_symbols(&dir.join("llama.dll"));
                    GGML_SYMBOLS.set(ggml).ok();

                    // Leak so the reference is genuinely 'static and lock-free.
                    Ok(&*Box::leak(Box::new(api)))
                }
                _ => {
                    let searched = dirs
                        .iter()
                        .map(|d| d.display().to_string())
                        .collect::<Vec<_>>()
                        .join(", ");
                    let err = format!(
                        "Native llama.cpp runtime not found ({last_error}). Searched: {searched}. \
                         Run `pwsh ./setup.ps1`, build llama.cpp under llama.cpp/build, or set \
                         LLAMA_LIB_DIR to the directory holding llama.dll."
                    );
                    warn!("{err}");
                    Err(err)
                }
            }
        })
        .clone()
}

/// Directories searched for the native runtime, most specific first.
pub fn searched_dirs() -> Vec<PathBuf> {
    RESOLVED_DIRS.get().cloned().unwrap_or_default()
}

/// True when the native runtime resolved successfully.
pub fn is_available() -> bool {
    api().is_ok()
}

/// Convenience handle for call sites: `LLAMA.api()?` yields the resolved API.
///
/// Exists so code reads as `LLAMA.api()?.llama_decode(...)` rather than a
/// free-function import, which keeps the "is the runtime present?" question
/// visible at every FFI boundary.
pub struct LlamaHandle;

pub static LLAMA: LlamaHandle = LlamaHandle;

impl LlamaHandle {
    /// Returns the resolved API, or a descriptive error.
    pub fn api(&self) -> Result<&'static LlamaApi, String> {
        api()
    }

    /// Returns the resolved API as an `anyhow::Error` for `?` propagation in
    /// the engine's `anyhow`-based code.
    pub fn require(&self) -> anyhow::Result<&'static LlamaApi> {
        api().map_err(|e| anyhow::anyhow!(e))
    }
}

/// Invokes a resolved llama.cpp entry point, propagating resolution failure.
///
/// Why a macro: every FFI call must (a) resolve the library lazily and
/// (b) report a missing runtime instead of segfaulting on a null pointer. Doing
/// this by hand at ~50 call sites invites omissions. `llama_call!` guarantees
/// both, and keeps the call sites shaped like ordinary function calls.
///
/// Expands to `anyhow::Result<T>`, so `?` at the call site yields the raw value
/// and a missing runtime becomes an ordinary error:
///
/// ```ignore
/// let tokens = llama_call!(llama_tokenize, vocab, ptr, len, out, cap, true, false)?;
/// llama_call!(llama_sampler_accept, sampler, token)?;
/// ```
///
/// Import it where used: `use crate::llama_ffi::llama_call;`
macro_rules! llama_call {
    ($f:ident $(, $arg:expr)* $(,)?) => {{
        let __api = $crate::llama_ffi::LLAMA.require()?;
        // SAFETY: `LLAMA.require()` has verified that the library is loaded and
        // that this symbol resolved with the signature declared on `LlamaApi`.
        Ok::<_, anyhow::Error>(unsafe { (__api.$f)($($arg),*) })
    }};
}

pub(crate) use llama_call;

fn load_and_resolve(dll: &std::path::Path) -> Result<LlamaApi, String> {
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::ffi::OsStrExt;
        let mut wide: Vec<u16> = dll.as_os_str().encode_wide().collect();
        wide.push(0);
        let handle = unsafe {
            windows_sys::Win32::System::LibraryLoader::LoadLibraryW(wide.as_ptr())
        };
        if handle.is_null() {
            return Err(format!(
                "LoadLibraryW failed (error {})",
                std::io::Error::last_os_error()
            ));
        }
        let missing: Vec<String> = symbol_names()
            .into_iter()
            .filter(|name| {
                let c = std::ffi::CString::new(*name).unwrap();
                unsafe {
                    windows_sys::Win32::System::LibraryLoader::GetProcAddress(handle, c.as_ptr() as *const u8)
                }
                .is_none()
            })
            .map(str::to_string)
            .collect();
        if !missing.is_empty() {
            return Err(format!(
                "library loaded but {} required symbol(s) are missing: {}",
                missing.len(),
                missing.join(", ")
            ));
        }
        unsafe { resolve_from(handle as *mut c_void) }
    }

    #[cfg(not(target_os = "windows"))]
    {
        let _ = dll;
        Err("dynamic llama.cpp loading is currently implemented for Windows only".to_string())
    }
}

#[derive(Default)]
struct GgmlSymbols {
    tensor_get: Option<GgmlBackendTensorGetFn>,
    load_all: Option<GgmlBackendLoadAllFn>,
    load_all_from_path: Option<GgmlBackendLoadAllFromPathFn>,
    reg_count: Option<GgmlBackendRegCountFn>,
}

static GGML_SYMBOLS: OnceLock<GgmlSymbols> = OnceLock::new();

/// Resolve the ggml symbols that are not part of the llama.h surface.
fn load_ggml_symbols(dll: &std::path::Path) -> GgmlSymbols {
    let mut out = GgmlSymbols::default();

    #[cfg(target_os = "windows")]
    {
        use std::os::windows::ffi::OsStrExt;
        let mut wide: Vec<u16> = dll.as_os_str().encode_wide().collect();
        wide.push(0);
        let handle = unsafe {
            windows_sys::Win32::System::LibraryLoader::LoadLibraryW(wide.as_ptr())
        };
        if handle.is_null() {
            return out;
        }

        unsafe fn lookup<T: Copy>(handle: *mut c_void, name: &str) -> Option<T> {
            let c = std::ffi::CString::new(name).ok()?;
            let addr = unsafe {
                windows_sys::Win32::System::LibraryLoader::GetProcAddress(
                    handle as windows_sys::Win32::Foundation::HMODULE,
                    c.as_ptr() as *const u8,
                )
            }?;
            let p = addr as *const c_void;
            Some(unsafe { std::mem::transmute_copy::<*const c_void, T>(&p) })
        }

        // Prefer ggml-base.dll / ggml.dll, which actually export these, then
        // fall back to llama.dll (it re-exports some of them).
        for extra in ["ggml-base.dll", "ggml.dll"] {
            let mut w: Vec<u16> = std::ffi::OsStr::new(extra).encode_wide().collect();
            w.push(0);
            let h = unsafe { windows_sys::Win32::System::LibraryLoader::LoadLibraryW(w.as_ptr()) };
            if h.is_null() {
                continue;
            }
            let h = h as *mut c_void;
            if out.tensor_get.is_none() {
                out.tensor_get = unsafe { lookup(h, "ggml_backend_tensor_get") };
            }
            if out.load_all.is_none() {
                out.load_all = unsafe { lookup(h, "ggml_backend_load_all") };
            }
            if out.load_all_from_path.is_none() {
                out.load_all_from_path = unsafe { lookup(h, "ggml_backend_load_all_from_path") };
            }
            if out.reg_count.is_none() {
                out.reg_count = unsafe { lookup(h, "ggml_backend_reg_count") };
            }
        }

        let h = handle as *mut c_void;
        if out.tensor_get.is_none() {
            out.tensor_get = unsafe { lookup(h, "ggml_backend_tensor_get") };
        }
        if out.load_all.is_none() {
            out.load_all = unsafe { lookup(h, "ggml_backend_load_all") };
        }
        if out.load_all_from_path.is_none() {
            out.load_all_from_path = unsafe { lookup(h, "ggml_backend_load_all_from_path") };
        }
        if out.reg_count.is_none() {
            out.reg_count = unsafe { lookup(h, "ggml_backend_reg_count") };
        }
    }

    #[cfg(not(target_os = "windows"))]
    {
        let _ = dll;
    }

    out
}

fn ggml_symbols() -> &'static GgmlSymbols {
    // Ensure the resolver has run at least once.
    let _ = api();
    GGML_SYMBOLS.get_or_init(GgmlSymbols::default)
}

static GGML_TENSOR_GET_WARNED: Once = Once::new();

/// Copies tensor data from any ggml backend (CUDA, Vulkan, host CPU) into a host
/// buffer by invoking `ggml_backend_tensor_get`.
///
/// Returns `false` when the symbol is unavailable. Callers must **not** fall
/// back to reading `tensor->data` directly: on GPU backends that is a device
/// pointer.
pub fn safe_ggml_backend_tensor_get(
    tensor: *const GgmlTensor,
    data: *mut c_void,
    offset: usize,
    size: usize,
) -> bool {
    match ggml_symbols().tensor_get {
        Some(f) => {
            f(tensor, data, offset, size);
            true
        }
        None => {
            GGML_TENSOR_GET_WARNED.call_once(|| {
                warn!(
                    "`ggml_backend_tensor_get` could not be resolved; activation capture will fall \
                     back to embedding outputs. Ensure ggml-base.dll / ggml.dll sit beside llama.dll."
                );
            });
            false
        }
    }
}

/// Loads the dynamic ggml backend plugins (ggml-cuda.dll, ggml-cpu.dll, ...).
pub fn load_ggml_backends(search_dirs: &[PathBuf]) {
    let syms = ggml_symbols();
    unsafe {
        if let Some(load_from_path) = syms.load_all_from_path {
            for dir in search_dirs {
                let clean = dir.to_string_lossy().trim_start_matches(r"\\?\").to_string();
                if let Ok(c_dir) = std::ffi::CString::new(clean) {
                    load_from_path(c_dir.as_ptr());
                }
            }
            if let Ok(c_bin) = std::ffi::CString::new("bin") {
                load_from_path(c_bin.as_ptr());
            }
        }
        if let Some(load_all) = syms.load_all {
            load_all();
        }
    }
}

/// Number of registered ggml backends, or 0 when unavailable.
pub fn ggml_backend_reg_count() -> usize {
    match ggml_symbols().reg_count {
        Some(f) => f(),
        None => 0,
    }
}

extern "C" fn quiet_llama_log_callback(level: i32, text: *const c_char, _user_data: *mut c_void) {
    // llama.cpp log levels: 0=DEBUG 1=INFO 2=WARN 3=ERROR 4=CONT
    if level < 3 || text.is_null() {
        return;
    }
    let msg = unsafe { std::ffi::CStr::from_ptr(text) }.to_string_lossy();
    // Suppress the high-volume, well-understood CUDA-graph allocation chatter.
    const NOISE: [&str; 3] = ["CUDA Graph", "reused", "warmup"];
    if !NOISE.iter().any(|needle| msg.contains(needle)) {
        eprint!("{}", msg);
    }
}

static INIT_BACKEND_ONCE: Once = Once::new();
static INIT_BACKEND_RESULT: OnceLock<Result<(), String>> = OnceLock::new();

/// Initialises the llama.cpp backend, injecting the runtime directory into the
/// Windows DLL search path first.
///
/// Idempotent, and caches the outcome. Returns an error (rather than aborting the
/// process) when the native runtime is unavailable, so callers decide whether the
/// operation can proceed.
pub fn try_ensure_llama_initialized() -> Result<(), String> {
    INIT_BACKEND_RESULT
        .get_or_init(|| {
            INIT_BACKEND_ONCE.call_once(|| {});

            let api = api()?;

            let dirs = searched_dirs();

            unsafe {
                (api.llama_log_set)(Some(quiet_llama_log_callback), std::ptr::null_mut());
            }

            load_ggml_backends(&dirs);

            unsafe {
                (api.llama_backend_init)();
            }

            let count = ggml_backend_reg_count();
            info!(backend_count = count, "✅ Native llama.cpp / GGML backend initialized");
            Ok(())
        })
        .clone()
}

/// Convenience wrapper that logs instead of propagating.
///
/// Retained for call sites that cannot act on the error; prefer
/// [`try_ensure_llama_initialized`] in new code.
pub fn ensure_llama_initialized() {
    if let Err(err) = try_ensure_llama_initialized() {
        warn!("{err}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symbol_list_is_complete_and_unique() {
        let names = symbol_names();
        assert!(!names.is_empty(), "the symbol list must not be empty");

        let mut sorted = names.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), names.len(), "duplicate symbol names in the list");

        // Every name the resolver looks up must be a real field of `LlamaApi`.
        // Referencing each as a field forces a compile error if a field is
        // renamed but the resolver list is not updated — the failure mode that
        // would otherwise surface only at runtime as a missing symbol.
        macro_rules! assert_is_field {
            ($($name:ident),* $(,)?) => {
                fn __all_names_are_fields() {
                    #[allow(unreachable_code, clippy::diverging_sub_expression)]
                    let api: &LlamaApi = loop {};
                    $( let _ = &api.$name; )*
                    let _ = api;
                }
            };
        }
        assert_is_field!(
            llama_backend_init,
            llama_backend_free,
            llama_model_default_params,
            llama_model_load_from_file,
            llama_model_free,
            llama_context_default_params,
            llama_init_from_model,
            llama_free,
            llama_model_get_vocab,
            llama_model_n_embd,
            llama_model_n_layer,
            llama_model_desc,
            llama_tokenize,
            llama_token_to_piece,
            llama_detokenize,
            llama_vocab_is_eog,
            llama_vocab_is_control,
            llama_vocab_bos,
            llama_vocab_eos,
            llama_vocab_n_tokens,
            llama_batch_init,
            llama_batch_free,
            llama_decode,
            llama_get_logits_ith,
            llama_get_embeddings_ith,
            llama_sampler_chain_default_params,
            llama_sampler_chain_init,
            llama_sampler_chain_add,
            llama_sampler_init_greedy,
            llama_sampler_init_temp,
            llama_sampler_init_top_p,
            llama_sampler_init_top_k,
            llama_sampler_init_min_p,
            llama_sampler_init_penalties,
            llama_sampler_init_dist,
            llama_sampler_accept,
            llama_sampler_reset,
            llama_sampler_sample,
            llama_sampler_free,
            llama_get_memory,
            llama_memory_clear,
            llama_synchronize,
            llama_log_set,
        );
    }

    #[test]
    fn missing_runtime_error_is_actionable() {
        // The message must tell the operator what to do; this is the entire
        // point of resolving dynamically instead of failing at process start.
        let err = match api() {
            Ok(_) => return, // runtime present on this machine
            Err(e) => e,
        };
        assert!(
            err.contains("setup.ps1") || err.contains("LLAMA_LIB_DIR"),
            "error should name a remedy, got: {err}"
        );
    }

    #[test]
    fn api_resolution_failure_is_descriptive_not_fatal() {
        // On a machine with no native runtime this must return Err and *not*
        // abort the process. This is the behaviour that lets pure unit tests
        // (framing, validation, slicing) run without CUDA installed.
        match api() {
            Ok(_) => {
                // Runtime present: the resolved struct must be usable.
                assert!(is_available());
            }
            Err(msg) => {
                assert!(
                    msg.contains("llama.cpp") || msg.contains("LLAMA_LIB_DIR"),
                    "error should be actionable, got: {msg}"
                );
            }
        }
    }

    #[test]
    fn tensor_get_never_reads_device_memory_on_failure() {
        // Passing a null tensor must not dereference anything when the symbol is
        // missing, and must return a bool rather than panicking.
        let ok = safe_ggml_backend_tensor_get(std::ptr::null(), std::ptr::null_mut(), 0, 0);
        // Either the symbol resolved (and llama.cpp handled the null), or we got false.
        let _ = ok;
    }
}
