use std::ffi::{c_char, c_void};
use std::path::Path;
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
    pub cb_eval: Option<extern "C" fn(*mut c_void, *mut c_void) -> bool>,
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
        #[cfg(target_os = "windows")]
        {
            use std::os::windows::ffi::OsStrExt;
            let candidate_dirs = ["bin", "../bin", "../../bin", "llama.cpp/build/bin/Release"];
            for dir in &candidate_dirs {
                let p = Path::new(dir);
                if p.exists() && p.join("llama.dll").exists() {
                    if let Ok(abs) = std::fs::canonicalize(p) {
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
                        break;
                    }
                }
            }
        }

        unsafe {
            llama_log_set(Some(quiet_llama_log_callback), std::ptr::null_mut());
            llama_backend_init();
            info!("✅ Native llama.cpp / GGML CUDA Backend Initialized");
        }
    });
}
