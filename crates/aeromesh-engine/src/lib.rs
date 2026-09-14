pub mod gguf;
pub mod job_object;
pub mod process;
pub mod slice_loader;
pub mod pipeline;
pub mod cache_prime;
pub mod llama_ffi;
pub mod server;
pub mod gguf_slicer;

pub use gguf::{inspect_gguf_file, resolve_model_path, GgufMetadata};
pub use gguf_slicer::slice_gguf_for_pipeline;
pub use job_object::SafeProcessJob;
pub use process::EngineSupervisor;
pub use slice_loader::{GgufSliceLoader, LayerSliceConfig, LayerSliceReport, LlamaPipelineInstance};
pub use pipeline::{PipelineCoordinatorClient, PipelineWorkerService};
pub use server::{
    load_dotenv_if_present, constant_time_eq_str, ApiErrorDetail, ApiErrorResponse,
    PipelineHttpServer, Utf8StreamAccumulator,
};
pub use llama_ffi::ensure_llama_initialized;
pub use cache_prime::{check_rpc_cache_status, fnv_hash, get_rpc_cache_dir, prime_rpc_cache};
