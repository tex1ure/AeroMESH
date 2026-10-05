use std::ffi::{c_char, c_void};
use std::path::PathBuf;
use std::sync::Once;
use tracing::info;

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

pub type GgmlBackendSchedEvalCallback = extern "C" fn(
    tensor: *mut GgmlTensor,
    ask: bool,
    user_data: *mut c_void,
) -> bool;

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

#[link(name = "llama")]
extern "C" {
    pub fn llama_backend_init();
    pub fn llama_backend_free();

    pub fn llama_model_default_params() -> LlamaModelParams;
    pub fn llama_model_load_from_file(
        path_model: *const c_char,
        params: LlamaModelParams,
    ) -> *mut LlamaModel;
    pub fn llama_model_free(model: *mut LlamaModel);

    pub fn llama_context_default_params() -> LlamaContextParams;
    pub fn llama_init_from_model(
        model: *mut LlamaModel,
        params: LlamaContextParams,
    ) -> *mut LlamaContext;
    pub fn llama_free(ctx: *mut LlamaContext);

    pub fn llama_model_get_vocab(model: *const LlamaModel) -> *const LlamaVocab;
    pub fn llama_model_n_embd(model: *const LlamaModel) -> i32;
    pub fn llama_model_n_layer(model: *const LlamaModel) -> i32;
    pub fn llama_model_desc(model: *const LlamaModel, buf: *mut c_char, buf_size: usize) -> i32;

    pub fn llama_tokenize(
        vocab: *const LlamaVocab,
        text: *const c_char,
        text_len: i32,
        tokens: *mut LlamaToken,
        n_tokens_max: i32,
        add_special: bool,
        parse_special: bool,
    ) -> i32;

    pub fn llama_token_to_piece(
        vocab: *const LlamaVocab,
        token: LlamaToken,
        buf: *mut c_char,
        length: i32,
        lstrip: i32,
        special: bool,
    ) -> i32;

    pub fn llama_detokenize(
        vocab: *const LlamaVocab,
        tokens: *const LlamaToken,
        n_tokens: i32,
        text: *mut c_char,
        text_len_max: i32,
        remove_special: bool,
        unparse_special: bool,
    ) -> i32;

    pub fn llama_vocab_is_eog(vocab: *const LlamaVocab, token: LlamaToken) -> bool;
    pub fn llama_vocab_is_control(vocab: *const LlamaVocab, token: LlamaToken) -> bool;
    pub fn llama_vocab_bos(vocab: *const LlamaVocab) -> LlamaToken;
    pub fn llama_vocab_eos(vocab: *const LlamaVocab) -> LlamaToken;
    pub fn llama_vocab_n_tokens(vocab: *const LlamaVocab) -> i32;

    pub fn llama_batch_init(n_tokens: i32, embd: i32, n_seq_max: i32) -> LlamaBatch;
    pub fn llama_batch_free(batch: LlamaBatch);

    pub fn llama_decode(ctx: *mut LlamaContext, batch: LlamaBatch) -> i32;
    pub fn llama_get_logits_ith(ctx: *mut LlamaContext, i: i32) -> *mut f32;
    pub fn llama_get_embeddings_ith(ctx: *mut LlamaContext, i: i32) -> *mut f32;

    pub fn llama_sampler_chain_default_params() -> LlamaSamplerChainParams;
    pub fn llama_sampler_chain_init(params: LlamaSamplerChainParams) -> *mut LlamaSampler;
    pub fn llama_sampler_chain_add(chain: *mut LlamaSampler, smpl: *mut LlamaSampler);
    pub fn llama_sampler_init_greedy() -> *mut LlamaSampler;
    pub fn llama_sampler_init_temp(t: f32) -> *mut LlamaSampler;
    pub fn llama_sampler_init_top_p(p: f32, min_keep: usize) -> *mut LlamaSampler;
    pub fn llama_sampler_init_top_k(k: i32) -> *mut LlamaSampler;
    pub fn llama_sampler_init_min_p(p: f32, min_keep: usize) -> *mut LlamaSampler;
    pub fn llama_sampler_init_penalties(
        penalty_last_n: i32,
        penalty_repeat: f32,
        penalty_freq: f32,
        penalty_present: f32,
    ) -> *mut LlamaSampler;
    pub fn llama_sampler_init_dist(seed: u32) -> *mut LlamaSampler;
    pub fn llama_sampler_accept(smpl: *mut LlamaSampler, token: LlamaToken);
    pub fn llama_sampler_reset(smpl: *mut LlamaSampler);
    pub fn llama_sampler_sample(
        smpl: *mut LlamaSampler,
        ctx: *mut LlamaContext,
        idx: i32,
    ) -> LlamaToken;
    pub fn llama_sampler_free(smpl: *mut LlamaSampler);
    pub fn llama_get_memory(ctx: *const LlamaContext) -> *mut c_void;
    pub fn llama_memory_clear(mem: *mut c_void, data: bool);
    pub fn llama_synchronize(ctx: *mut LlamaContext);
    pub fn llama_log_set(
        log_callback: Option<extern "C" fn(level: i32, text: *const c_char, user_data: *mut c_void)>,
        user_data: *mut c_void,
    );
}

type GgmlBackendTensorGetFn = extern "C" fn(
    tensor: *const GgmlTensor,
    data: *mut c_void,
    offset: usize,
    size: usize,
);

static GGML_BACKEND_TENSOR_GET: std::sync::OnceLock<Option<GgmlBackendTensorGetFn>> =
    std::sync::OnceLock::new();

/// Dynamically resolves and invokes `ggml_backend_tensor_get` from loaded GGML DLLs.
/// Copies tensor data from any device backend (CUDA, Vulkan, host CPU) into host destination buffer.
pub fn safe_ggml_backend_tensor_get(
    tensor: *const GgmlTensor,
    data: *mut c_void,
    offset: usize,
    size: usize,
) -> bool {
    let func = GGML_BACKEND_TENSOR_GET.get_or_init(|| {
        #[cfg(target_os = "windows")]
        unsafe {
            use windows_sys::Win32::System::LibraryLoader::{
                GetModuleHandleA, GetProcAddress, LoadLibraryA,
            };
            let proc_name = b"ggml_backend_tensor_get\0".as_ptr();

            for mod_name in [
                b"ggml-base.dll\0".as_ptr(),
                b"llama.dll\0".as_ptr(),
                b"ggml.dll\0".as_ptr(),
                b"mtmd.dll\0".as_ptr(),
            ] {
                let h_mod = GetModuleHandleA(mod_name);
                if !h_mod.is_null() {
                    let addr = GetProcAddress(h_mod, proc_name);
                    if let Some(f) = addr {
                        return Some(std::mem::transmute::<_, GgmlBackendTensorGetFn>(f));
                    }
                }
            }

            for mod_name in [b"ggml-base.dll\0".as_ptr(), b"llama.dll\0".as_ptr()] {
                let h_mod = LoadLibraryA(mod_name);
                if !h_mod.is_null() {
                    let addr = GetProcAddress(h_mod, proc_name);
                    if let Some(f) = addr {
                        return Some(std::mem::transmute::<_, GgmlBackendTensorGetFn>(f));
                    }
                }
            }
        }
        None
    });

    if let Some(f) = func {
        f(tensor, data, offset, size);
        true
    } else {
        unsafe {
            if !tensor.is_null() && !(*tensor).data.is_null() && !data.is_null() {
                std::ptr::copy_nonoverlapping(
                    ((*tensor).data as *const u8).add(offset),
                    data as *mut u8,
                    size,
                );
                return true;
            }
        }
        false
    }
}

#[link(name = "ggml")]
extern "C" {
    pub fn ggml_backend_load_all();
    pub fn ggml_backend_load_all_from_path(dir_path: *const c_char);
    pub fn ggml_backend_reg_count() -> usize;
}

extern "C" fn quiet_llama_log_callback(level: i32, text: *const c_char, _user_data: *mut c_void) {
    if level >= 3 && !text.is_null() {
        let msg = unsafe { std::ffi::CStr::from_ptr(text) }.to_string_lossy();
        if !msg.contains("CUDA Graph") && !msg.contains("reused") && !msg.contains("warmup") {
            eprint!("{}", msg);
        }
    }
}

static INIT_BACKEND_ONCE: Once = Once::new();

/// Automatically injects `bin/` directory into Windows DLL search path and initializes llama backend.
pub fn ensure_llama_initialized() {
    INIT_BACKEND_ONCE.call_once(|| {
        let mut loaded_bin_dirs: Vec<PathBuf> = Vec::new();

        #[cfg(target_os = "windows")]
        {
            use std::os::windows::ffi::OsStrExt;
            let mut candidate_dirs: Vec<PathBuf> = vec![
                PathBuf::from("bin"),
                PathBuf::from("../bin"),
                PathBuf::from("../../bin"),
                PathBuf::from("llama.cpp/build/bin/Release"),
            ];

            if let Ok(exe_path) = std::env::current_exe() {
                if let Some(exe_dir) = exe_path.parent() {
                    candidate_dirs.insert(0, exe_dir.to_path_buf());
                    candidate_dirs.push(exe_dir.join("bin"));
                    if let Some(parent) = exe_dir.parent() {
                        candidate_dirs.push(parent.join("bin"));
                        if let Some(grand) = parent.parent() {
                            candidate_dirs.push(grand.join("bin"));
                        }
                    }
                }
            }

            for dir in &candidate_dirs {
                if dir.exists() && dir.join("llama.dll").exists() {
                    if let Ok(abs) = std::fs::canonicalize(dir) {
                        let mut wide: Vec<u16> = abs.as_os_str().encode_wide().collect();
                        wide.push(0);
                        unsafe {
                            windows_sys::Win32::System::LibraryLoader::SetDllDirectoryW(wide.as_ptr());
                        }
                        if let Ok(path_var) = std::env::var("PATH") {
                            let new_path = format!("{};{}", abs.display(), path_var);
                            std::env::set_var("PATH", new_path);
                        }
                        info!(dll_dir = %abs.display(), "Injected llama.dll runtime directory into Windows search path");
                        loaded_bin_dirs.push(abs);
                        break;
                    }
                }
            }
        }

        unsafe {
            llama_log_set(Some(quiet_llama_log_callback), std::ptr::null_mut());

            // Explicitly load dynamic backends (ggml-cuda.dll, ggml-cpu.dll, etc.)
            for bin_dir in &loaded_bin_dirs {
                let clean_str = bin_dir.to_string_lossy().trim_start_matches(r"\\?\").to_string();
                if let Ok(c_dir) = std::ffi::CString::new(clean_str) {
                    ggml_backend_load_all_from_path(c_dir.as_ptr());
                }
            }
            if let Ok(c_bin) = std::ffi::CString::new("bin") {
                ggml_backend_load_all_from_path(c_bin.as_ptr());
            }
            ggml_backend_load_all();

            llama_backend_init();

            let count = ggml_backend_reg_count();
            info!(backend_count = count, "✅ Native llama.cpp / GGML Backend Initialized");
        }
    });
}
