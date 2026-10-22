pub mod cache_prime;
pub mod gguf;
pub mod gguf_slicer;
pub mod job_object;
pub mod llama_ffi;
pub mod pipeline;
pub mod process;
pub mod server;
pub mod slice_loader;

pub use cache_prime::{check_rpc_cache_status, fnv_hash, get_rpc_cache_dir, prime_rpc_cache};
pub use gguf::{inspect_gguf_file, resolve_model_path, GgufMetadata};
pub use gguf_slicer::{slice_gguf_for_multi_stage, slice_gguf_for_pipeline, slice_gguf_stage};
pub use job_object::SafeProcessJob;
pub use llama_ffi::ensure_llama_initialized;
pub use pipeline::{PipelineCoordinatorClient, PipelineWorkerService};
pub use process::EngineSupervisor;
pub use server::{
    constant_time_eq_str, load_dotenv_if_present, ApiErrorDetail, ApiErrorResponse,
    PipelineHttpServer, Utf8StreamAccumulator,
};
pub use slice_loader::{
    ChatTemplateFormat, GgufSliceLoader, LayerSliceConfig, LayerSliceReport, LlamaPipelineInstance,
};
