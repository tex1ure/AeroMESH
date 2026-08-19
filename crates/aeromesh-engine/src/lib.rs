pub mod gguf;
pub mod job_object;
pub mod process;
pub mod slice_loader;
pub mod pipeline;
pub mod cache_prime;

pub use gguf::{inspect_gguf_file, resolve_model_path, GgufMetadata};
pub use job_object::SafeProcessJob;
pub use process::EngineSupervisor;
pub use slice_loader::{GgufSliceLoader, LayerSliceConfig, LayerSliceReport};
pub use pipeline::{PipelineCoordinatorClient, PipelineWorkerService};
pub use cache_prime::{fnv_hash, get_rpc_cache_dir, prime_rpc_cache};
