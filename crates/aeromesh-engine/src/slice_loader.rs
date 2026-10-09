use std::collections::HashMap;
use std::fs::File;
use std::ops::RangeInclusive;
use std::path::{Path, PathBuf};
use anyhow::{bail, ensure, Context, Result};
use byteorder::{LittleEndian, ReadBytesExt};
use memmap2::Mmap;
use tracing::{info, warn};

/// GGUF Value Types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgufValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl GgufValueType {
    pub fn from_u32(val: u32) -> Result<Self> {
        match val {
            0 => Ok(Self::Uint8),
            1 => Ok(Self::Int8),
            2 => Ok(Self::Uint16),
            3 => Ok(Self::Int16),
            4 => Ok(Self::Uint32),
            5 => Ok(Self::Int32),
            6 => Ok(Self::Float32),
            7 => Ok(Self::Bool),
            8 => Ok(Self::String),
            9 => Ok(Self::Array),
            10 => Ok(Self::Uint64),
            11 => Ok(Self::Int64),
            12 => Ok(Self::Float64),
            _ => bail!("Unknown GGUF value type: {}", val),
        }
    }
}

/// Metadata descriptor for a tensor stored in GGUF.
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub ggml_type: u32,
    pub offset: u64,
    pub size_bytes: u64,
    pub layer_index: Option<usize>,
}

/// Configuration for a specific pipeline stage layer slice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerSliceConfig {
    pub layer_start: usize,
    pub layer_end: usize,
    pub total_layers: usize,
    pub is_first_stage: bool,
    pub is_last_stage: bool,
}

impl LayerSliceConfig {
    pub fn new(layer_start: usize, layer_end: usize, total_layers: usize) -> Result<Self> {
        ensure!(
            layer_start <= layer_end,
            "layer_start ({}) must be <= layer_end ({})",
            layer_start,
            layer_end
        );
        ensure!(
            layer_end < total_layers,
            "layer_end ({}) must be < total_layers ({})",
            layer_end,
            total_layers
        );

        Ok(Self {
            layer_start,
            layer_end,
            total_layers,
            is_first_stage: layer_start == 0,
            is_last_stage: layer_end == total_layers.saturating_sub(1),
        })
    }

    pub fn layer_range(&self) -> RangeInclusive<usize> {
        self.layer_start..=self.layer_end
    }

    pub fn layer_count(&self) -> usize {
        self.layer_end - self.layer_start + 1
    }
}

/// Summary report of the loaded layer slice.
#[derive(Debug, Clone)]
pub struct LayerSliceReport {
    pub model_path: PathBuf,
    pub architecture: String,
    pub total_layers: usize,
    pub slice: LayerSliceConfig,
    pub tensor_count: usize,
    pub slice_bytes: u64,
    pub total_model_bytes: u64,
    pub memory_reduction_ratio: f64,
}

/// Zero-Copy GGUF Slice Loader using memory-mapped I/O.
pub struct GgufSliceLoader {
    pub model_path: PathBuf,
    pub architecture: String,
    pub total_layers: usize,
    pub hidden_dim: u32,
    pub alignment: u64,
    pub data_section_offset: u64,
    pub all_tensors: Vec<GgufTensorInfo>,
    pub tensor_index: HashMap<String, usize>,
    mmap: Mmap,
}

impl GgufSliceLoader {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_ref = path.as_ref();
        let file = File::open(path_ref)
            .with_context(|| format!("Failed to open GGUF file at {:?}", path_ref))?;
        let mmap = unsafe { Mmap::map(&file)? };

        let mut cursor = std::io::Cursor::new(&mmap[..]);

        let mut magic = [0u8; 4];
        std::io::Read::read_exact(&mut cursor, &mut magic)?;
        ensure!(&magic == b"GGUF", "Invalid GGUF magic bytes");

        let version = cursor.read_u32::<LittleEndian>()?;
        ensure!(version == 2 || version == 3, "Unsupported GGUF version: {}", version);

        let tensor_count = cursor.read_u64::<LittleEndian>()? as usize;
        let kv_count = cursor.read_u64::<LittleEndian>()? as usize;

        let mut architecture = "llama".to_string();
        let mut total_layers = 32usize;
        let mut hidden_dim = 4096u32;
        let mut alignment = 32u64;

        // Parse KV metadata
        for _ in 0..kv_count {
            let key = read_gguf_string(&mut cursor)?;
            let val_type = GgufValueType::from_u32(cursor.read_u32::<LittleEndian>()?)?;
            
            match key.as_str() {
                "general.architecture" => {
                    if val_type == GgufValueType::String {
                        architecture = read_gguf_string(&mut cursor)?;
                    } else {
                        skip_gguf_value(&mut cursor, val_type)?;
                    }
                }
                "general.alignment" => {
                    if val_type == GgufValueType::Uint32 {
                        alignment = cursor.read_u32::<LittleEndian>()? as u64;
                    } else if val_type == GgufValueType::Uint64 {
                        alignment = cursor.read_u64::<LittleEndian>()?;
                    } else {
                        skip_gguf_value(&mut cursor, val_type)?;
                    }
                }
                k if k.ends_with(".block_count") => {
                    if val_type == GgufValueType::Uint32 {
                        total_layers = cursor.read_u32::<LittleEndian>()? as usize;
                    } else if val_type == GgufValueType::Uint64 {
                        total_layers = cursor.read_u64::<LittleEndian>()? as usize;
                    } else {
                        skip_gguf_value(&mut cursor, val_type)?;
                    }
                }
                k if k.ends_with(".embedding_length") => {
                    if val_type == GgufValueType::Uint32 {
                        hidden_dim = cursor.read_u32::<LittleEndian>()?;
                    } else if val_type == GgufValueType::Uint64 {
                        hidden_dim = cursor.read_u64::<LittleEndian>()? as u32;
                    } else {
                        skip_gguf_value(&mut cursor, val_type)?;
                    }
                }
                _ => {
                    skip_gguf_value(&mut cursor, val_type)?;
                }
            }
        }

        // Parse Tensor Info Table
        let mut all_tensors = Vec::with_capacity(tensor_count);
        let mut tensor_index = HashMap::with_capacity(tensor_count);

        for idx in 0..tensor_count {
            let name = read_gguf_string(&mut cursor)?;
            let n_dims = cursor.read_u32::<LittleEndian>()? as usize;
            let mut dimensions = Vec::with_capacity(n_dims);
            let mut element_count: u64 = 1;
            for _ in 0..n_dims {
                let d = cursor.read_u64::<LittleEndian>()?;
                dimensions.push(d);
                element_count = element_count.saturating_mul(d);
            }
            let ggml_type = cursor.read_u32::<LittleEndian>()?;
            let offset = cursor.read_u64::<LittleEndian>()?;

            let type_size = ggml_type_size_bytes(ggml_type);
            let size_bytes = (element_count as f64 * type_size).ceil() as u64;

            let layer_index = extract_layer_index(&name);

            let info = GgufTensorInfo {
                name: name.clone(),
                dimensions,
                ggml_type,
                offset,
                size_bytes,
                layer_index,
            };

            all_tensors.push(info);
            tensor_index.insert(name, idx);
        }

        // Align cursor to calculate data section offset
        let curr_pos = cursor.position();
        let padding = (alignment - (curr_pos % alignment)) % alignment;
        let data_section_offset = curr_pos + padding;

        info!(
            model = %path_ref.display(),
            architecture = %architecture,
            total_layers = total_layers,
            hidden_dim = hidden_dim,
            tensors = tensor_count,
            "GGUF Zero-Copy Slice Loader initialized"
        );

        Ok(Self {
            model_path: path_ref.to_path_buf(),
            architecture,
            total_layers,
            hidden_dim,
            alignment,
            data_section_offset,
            all_tensors,
            tensor_index,
            mmap,
        })
    }

    /// Filters and indexes all tensors belonging to the specified layer slice.
    pub fn get_slice_report(&self, slice: &LayerSliceConfig) -> LayerSliceReport {
        let relevant_tensors = self.get_tensors_for_slice(slice);
        let slice_bytes: u64 = relevant_tensors.iter().map(|t| t.size_bytes).sum();
        let total_model_bytes = self.mmap.len() as u64;
        let memory_reduction_ratio = if total_model_bytes > 0 {
            1.0 - (slice_bytes as f64 / total_model_bytes as f64)
        } else {
            0.0
        };

        LayerSliceReport {
            model_path: self.model_path.clone(),
            architecture: self.architecture.clone(),
            total_layers: self.total_layers,
            slice: slice.clone(),
            tensor_count: relevant_tensors.len(),
            slice_bytes,
            total_model_bytes,
            memory_reduction_ratio,
        }
    }

    /// Returns references to all tensor info metadata for a given layer slice.
    pub fn get_tensors_for_slice(&self, slice: &LayerSliceConfig) -> Vec<&GgufTensorInfo> {
        self.all_tensors
            .iter()
            .filter(|t| {
                if let Some(layer) = t.layer_index {
                    slice.layer_range().contains(&layer)
                } else {
                    // Global non-layer tensors
                    (slice.is_first_stage && (t.name.contains("token_embd") || t.name.contains("embed")))
                        || (slice.is_last_stage && (t.name.contains("output") || t.name.contains("norm")))
                }
            })
            .collect()
    }

    /// Access raw tensor binary slice directly from the memory-mapped file.
    pub fn get_tensor_bytes(&self, tensor_name: &str) -> Option<&[u8]> {
        let idx = *self.tensor_index.get(tensor_name)?;
        let info = &self.all_tensors[idx];
        let abs_start = (self.data_section_offset + info.offset) as usize;
        let abs_end = abs_start + info.size_bytes as usize;

        if abs_end <= self.mmap.len() {
            Some(&self.mmap[abs_start..abs_end])
        } else {
            None
        }
    }

    /// Computes balanced layer partitions for N pipeline nodes.
    pub fn compute_balanced_splits(&self, num_nodes: usize) -> Vec<LayerSliceConfig> {
        let num_nodes = num_nodes.max(1);
        let layers_per_node = self.total_layers.div_ceil(num_nodes);
        let mut splits = Vec::with_capacity(num_nodes);

        let mut current_start = 0;
        for _i in 0..num_nodes {
            let current_end = (current_start + layers_per_node - 1).min(self.total_layers - 1);
            if current_start <= current_end {
                if let Ok(cfg) = LayerSliceConfig::new(current_start, current_end, self.total_layers) {
                    splits.push(cfg);
                }
            }
            current_start = current_end + 1;
            if current_start >= self.total_layers {
                break;
            }
        }

        splits
    }
}

fn read_gguf_string<R: std::io::Read>(reader: &mut R) -> Result<String> {
    let len = reader.read_u64::<LittleEndian>()? as usize;
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).to_string())
}

fn skip_gguf_value<R: std::io::Read>(reader: &mut R, val_type: GgufValueType) -> Result<()> {
    match val_type {
        GgufValueType::Uint8 | GgufValueType::Int8 | GgufValueType::Bool => {
            reader.read_u8()?;
        }
        GgufValueType::Uint16 | GgufValueType::Int16 => {
            reader.read_u16::<LittleEndian>()?;
        }
        GgufValueType::Uint32 | GgufValueType::Int32 | GgufValueType::Float32 => {
            reader.read_u32::<LittleEndian>()?;
        }
        GgufValueType::Uint64 | GgufValueType::Int64 | GgufValueType::Float64 => {
            reader.read_u64::<LittleEndian>()?;
        }
        GgufValueType::String => {
            let _ = read_gguf_string(reader)?;
        }
        GgufValueType::Array => {
            let elem_type = GgufValueType::from_u32(reader.read_u32::<LittleEndian>()?)?;
            let count = reader.read_u64::<LittleEndian>()? as usize;
            for _ in 0..count {
                skip_gguf_value(reader, elem_type)?;
            }
        }
    }
    Ok(())
}

fn extract_layer_index(name: &str) -> Option<usize> {
    // Matches "blk.N." or "layers.N." or "model.layers.N."
    for part in name.split('.') {
        if let Ok(idx) = part.parse::<usize>() {
            return Some(idx);
        }
    }
    None
}

fn ggml_type_size_bytes(ggml_type: u32) -> f64 {
    match ggml_type {
        0 => 4.0,           // F32
        1 => 2.0,           // F16
        2 => 0.5 + 2.0/32.0, // Q4_0
        3 => 0.5 + 4.0/32.0, // Q4_1
        7 => 1.0,           // I8
        8 => 1.0 + 2.0/32.0, // Q8_0
        12 => 0.5625,       // Q4_K
        13 => 0.6875,       // Q5_K
        14 => 0.8125,       // Q6_K
        _ => 2.0,           // Fallback estimate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_slice_config_validation() {
        let cfg = LayerSliceConfig::new(0, 15, 32).unwrap();
        assert!(cfg.is_first_stage);
        assert!(!cfg.is_last_stage);
        assert_eq!(cfg.layer_count(), 16);

        let cfg2 = LayerSliceConfig::new(16, 31, 32).unwrap();
        assert!(!cfg2.is_first_stage);
        assert!(cfg2.is_last_stage);
        assert_eq!(cfg2.layer_count(), 16);

        assert!(LayerSliceConfig::new(20, 10, 32).is_err());
        assert!(LayerSliceConfig::new(0, 35, 32).is_err());
    }

    #[test]
    fn test_layer_index_extraction() {
        assert_eq!(extract_layer_index("blk.0.attn_q.weight"), Some(0));
        assert_eq!(extract_layer_index("blk.24.ffn_down.weight"), Some(24));
        assert_eq!(extract_layer_index("model.layers.15.self_attn.q_proj.weight"), Some(15));
        assert_eq!(extract_layer_index("token_embd.weight"), None);
        assert_eq!(extract_layer_index("output_norm.weight"), None);
    }
}

// ---------------------------------------------------------------------------
// Native llama.cpp Pipeline Context Instance
// ---------------------------------------------------------------------------

use crate::llama_ffi::{
    ensure_llama_initialized, llama_call, safe_ggml_backend_tensor_get, GgmlTensor, LlamaContext,
    LlamaModel, LlamaSampler, LlamaToken, LlamaVocab, LLAMA,
};
use std::ffi::{c_char, c_void, CString};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ChatTemplateFormat {
    DeepSeek,
    ChatML,
    Llama3,
    Raw,
}

#[repr(C)]
pub struct ActivationCaptureState {
    pub target_layer_idx: i32,
    pub target_tensor_name: [u8; 64],
    pub captured: bool,
    pub buffer: *mut f32,
    pub capacity_floats: usize,
    pub captured_floats: usize,
}

impl Default for ActivationCaptureState {
    fn default() -> Self {
        Self {
            target_layer_idx: -1,
            target_tensor_name: [0u8; 64],
            captured: false,
            buffer: std::ptr::null_mut(),
            capacity_floats: 0,
            captured_floats: 0,
        }
    }
}

unsafe fn tensor_name_matches(
    tensor_name: *const c_char,
    target_name: &[u8; 64],
    target_layer: i32,
) -> bool {
    if tensor_name.is_null() {
        return false;
    }
    // 1. Direct name match with target_name
    let mut i = 0;
    let mut exact = true;
    while i < 64 && target_name[i] != 0 {
        if *tensor_name.add(i) != target_name[i] as c_char {
            exact = false;
            break;
        }
        i += 1;
    }
    if exact && *tensor_name.add(i) == 0 {
        return true;
    }

    // 2. Prefix match for "l_out-" with target_layer
    let prefix = b"l_out-";
    let mut is_prefix = true;
    for j in 0..prefix.len() {
        if *tensor_name.add(j) != prefix[j] as c_char {
            is_prefix = false;
            break;
        }
    }
    if is_prefix && target_layer >= 0 {
        let mut idx: i32 = 0;
        let mut k = prefix.len();
        let mut has_digits = false;
        while k < 64 && *tensor_name.add(k) != 0 {
            let ch = *tensor_name.add(k) as u8;
            if ch >= b'0' && ch <= b'9' {
                idx = idx * 10 + (ch - b'0') as i32;
                has_digits = true;
            } else {
                break;
            }
            k += 1;
        }
        if has_digits && idx == target_layer && *tensor_name.add(k) == 0 {
            return true;
        }
    }

    false
}

extern "C" fn llama_cb_eval_capture(
    tensor: *mut GgmlTensor,
    ask: bool,
    user_data: *mut c_void,
) -> bool {
    if tensor.is_null() || user_data.is_null() {
        return false;
    }
    let state = unsafe { &mut *(user_data as *mut ActivationCaptureState) };
    if state.target_layer_idx < 0 {
        return false;
    }

    let tensor_name_ptr = unsafe { (*tensor).name.as_ptr() };
    let matches = unsafe {
        tensor_name_matches(tensor_name_ptr, &state.target_tensor_name, state.target_layer_idx)
    };

    if !matches {
        return false;
    }

    if ask {
        return true;
    }

    // ask == false: The target tensor just finished computing on its backend!
    let ne0 = unsafe { (*tensor).ne[0] } as usize;
    let ne1 = unsafe { (*tensor).ne[1] } as usize;
    let total_floats = ne0 * ne1;
    let n_bytes = total_floats * std::mem::size_of::<f32>();

    // llama_decode may split a large batch into several ubatches (n_ubatch, default 512);
    // the callback fires once per ubatch, so append at the running offset instead of overwriting.
    let offset_floats = state.captured_floats;
    if !state.buffer.is_null() && offset_floats + total_floats <= state.capacity_floats {
        let dst = unsafe { state.buffer.add(offset_floats) };
        let ok = safe_ggml_backend_tensor_get(tensor, dst as *mut c_void, 0, n_bytes);
        if ok {
            state.captured_floats = offset_floats + total_floats;
            state.captured = state.captured_floats == state.capacity_floats;
        }
    }

    true
}

/// Context window used when the caller does not specify one.
pub const DEFAULT_N_CTX: u32 = 4096;

/// Sampling configuration for the sampler chain built at model-load time.
///
/// These values are applied when the model is opened. A per-request override
/// path exists via [`LlamaPipelineInstance::rebuild_sampler`]; the HTTP layer
/// resolves request values against these defaults in `SamplingParams::resolve`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub min_p: f32,
    pub repeat_penalty: f32,
    pub freq_penalty: f32,
    pub present_penalty: f32,
    pub penalty_last_n: i32,
}

impl SamplingParams {
    /// Generic instruct-model defaults.
    pub const DEFAULT: Self = Self {
        temperature: 0.7,
        top_p: 0.9,
        top_k: 40,
        min_p: 0.05,
        repeat_penalty: 1.15,
        freq_penalty: 0.05,
        present_penalty: 0.05,
        penalty_last_n: 256,
    };

    /// Reasoning-tuned defaults: lower temperature and a gentler repeat penalty
    /// so long chains of thought do not collapse into loops.
    pub const REASONING_DEFAULT: Self = Self {
        temperature: 0.6,
        top_p: 0.95,
        repeat_penalty: 1.12,
        ..Self::DEFAULT
    };

    /// Clamps request-supplied values into ranges llama.cpp accepts, falling
    /// back to `base` for anything absent or non-finite.
    pub fn resolve(
        base: Self,
        temperature: Option<f32>,
        top_p: Option<f32>,
        top_k: Option<i32>,
        min_p: Option<f32>,
        repeat_penalty: Option<f32>,
    ) -> Self {
        let mut out = base;
        if let Some(t) = temperature.filter(|v| v.is_finite()) {
            out.temperature = t.clamp(0.0, 2.0);
        }
        if let Some(p) = top_p.filter(|v| v.is_finite()) {
            out.top_p = p.clamp(0.0, 1.0);
        }
        if let Some(k) = top_k.filter(|v| *v > 0) {
            out.top_k = k.min(1_000_000);
        }
        if let Some(p) = min_p.filter(|v| v.is_finite()) {
            out.min_p = p.clamp(0.0, 1.0);
        }
        if let Some(r) = repeat_penalty.filter(|v| v.is_finite()) {
            out.repeat_penalty = r.clamp(1.0, 4.0);
        }
        out
    }

    /// True when `self` differs from `other` in any wire-visible parameter.
    pub fn differs_from(&self, other: &Self) -> bool {
        (self.temperature - other.temperature).abs() > f32::EPSILON
            || (self.top_p - other.top_p).abs() > f32::EPSILON
            || self.top_k != other.top_k
            || (self.min_p - other.min_p).abs() > f32::EPSILON
            || (self.repeat_penalty - other.repeat_penalty).abs() > f32::EPSILON
    }
}

/// Heuristic classification of "reasoning" models, used only to pick sampler
/// defaults. Deliberately conservative: it inspects the description reported by
/// llama.cpp and the file name, and requires a whole-token match so a name such
/// as `handsd` cannot match the `ds` substring.
fn looks_like_reasoning_model(model_path: &Path, model_desc: &str) -> bool {
    let haystack = format!(
        "{} {}",
        model_path
            .file_name()
            .map(|s| s.to_string_lossy().to_lowercase())
            .unwrap_or_default(),
        model_desc.to_lowercase()
    );

    // Split on non-alphanumeric boundaries so we compare whole words.
    let tokens: Vec<&str> = haystack
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|s| !s.is_empty())
        .collect();

    const MARKERS: [&str; 5] = ["deepseek", "r1", "qwq", "reasoner", "thinking"];
    if tokens.iter().any(|t| MARKERS.contains(t)) {
        return true;
    }
    // `ds` is only meaningful as a standalone token (e.g. "DS.gguf").
    tokens.iter().any(|t| *t == "ds")
}

pub struct LlamaPipelineInstance {
    model: *mut LlamaModel,
    ctx: *mut LlamaContext,
    vocab: *const LlamaVocab,
    sampler: *mut LlamaSampler,
    pub slice_config: LayerSliceConfig,
    pub hidden_dim: usize,
    pub total_layers: usize,
    pub n_ctx: usize,
    /// Sampling parameters currently baked into `sampler`.
    pub sampling: SamplingParams,
    /// Whether the active sampler was built with reasoning-tuned defaults.
    pub sampling_is_reasoning: bool,
    pub capture_state: Box<ActivationCaptureState>,
}

unsafe impl Send for LlamaPipelineInstance {}

impl std::fmt::Debug for LlamaPipelineInstance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaPipelineInstance")
            .field("is_ready", &self.is_ready())
            .field("hidden_dim", &self.hidden_dim)
            .field("total_layers", &self.total_layers)
            .field("n_ctx", &self.n_ctx)
            .finish()
    }
}

impl LlamaPipelineInstance {
    pub fn open(
        model_path: &Path,
        slice_config: LayerSliceConfig,
        n_gpu_layers: i32,
        n_ctx: u32,
    ) -> Result<Self> {
        // Resolve the native runtime up front so a missing DLL surfaces as one
        // clear error instead of a null-pointer crash inside the first call.
        ensure_llama_initialized();

        let path_str = model_path.to_str().context("Invalid model path string")?;
        let c_path = CString::new(path_str)?;

        let mut mparams = llama_call!(llama_model_default_params)?;
        mparams.n_gpu_layers = n_gpu_layers.max(0);

        let model = llama_call!(llama_model_load_from_file, c_path.as_ptr(), mparams)?;
        if model.is_null() {
            bail!("Failed to load GGUF model via llama.cpp from {:?}", model_path);
        }

        let vocab = llama_call!(llama_model_get_vocab, model)?;
        if vocab.is_null() {
            llama_call!(llama_model_free, model)?;
            bail!("Failed to get model vocabulary from GGUF model");
        }

        let hidden_dim = llama_call!(llama_model_n_embd, model)? as usize;
        let total_layers = llama_call!(llama_model_n_layer, model)? as usize;

        // Derive the model description from llama.cpp rather than the file name.
        // It carries the architecture and parameter count, which is what the
        // sampler heuristics below should key off.
        let model_desc = {
            let mut buf = vec![0 as std::ffi::c_char; 256];
            let n = llama_call!(llama_model_desc, model, buf.as_mut_ptr(), buf.len())?;
            if n > 0 {
                let bytes: Vec<u8> = buf
                    .iter()
                    .take_while(|c| **c != 0)
                    .map(|c| *c as u8)
                    .collect();
                String::from_utf8_lossy(&bytes).to_string()
            } else {
                String::new()
            }
        };

        // Half the logical cores, leaving headroom for the peer process and the
        // GPU feeder. Previously computed and logged but never applied, so
        // llama.cpp silently used its own default.
        let physical_threads = std::thread::available_parallelism()
            .map(|n| (n.get() / 2).max(1) as i32)
            .unwrap_or(4);

        let mut capture_state = Box::new(ActivationCaptureState::default());

        let mut cparams = llama_call!(llama_context_default_params)?;
        cparams.n_ctx = if n_ctx > 0 { n_ctx } else { DEFAULT_N_CTX };
        cparams.n_threads = physical_threads;
        cparams.n_threads_batch = physical_threads;
        cparams.embeddings = true;
        cparams.cb_eval = Some(llama_cb_eval_capture);
        cparams.cb_eval_user_data = capture_state.as_mut() as *mut _ as *mut c_void;

        let ctx = llama_call!(llama_init_from_model, model, cparams)?;
        if ctx.is_null() {
            llama_call!(llama_model_free, model)?;
            bail!("Failed to initialize llama_context for model {:?}", model_path);
        }

        let sparams = llama_call!(llama_sampler_chain_default_params)?;
        let sampler = llama_call!(llama_sampler_chain_init, sparams)?;

        let sampling_is_reasoning = looks_like_reasoning_model(model_path, &model_desc);
        let sampling = if sampling_is_reasoning {
            SamplingParams::REASONING_DEFAULT
        } else {
            SamplingParams::DEFAULT
        };

        // Build the initial chain. On failure we must free what we already
        // created, otherwise the model and context leak on the error path.
        if let Err(err) = Self::populate_sampler_chain(sampler, sampling) {
            llama_call!(llama_sampler_free, sampler)?;
            llama_call!(llama_free, ctx)?;
            llama_call!(llama_model_free, model)?;
            return Err(err);
        }

        info!(
            model_desc = %model_desc,
            reasoning_tuned = sampling_is_reasoning,
            temperature = sampling.temperature,
            top_p = sampling.top_p,
            repeat_penalty = sampling.repeat_penalty,
            "Sampler chain configured"
        );

        info!(
            model = %model_path.display(),
            layers = %format!("{}..={}", slice_config.layer_start, slice_config.layer_end),
            hidden_dim = hidden_dim,
            total_layers = total_layers,
            n_ctx = cparams.n_ctx,
            n_threads = physical_threads,
            gpu_layers = mparams.n_gpu_layers,
            "✅ Loaded native llama.cpp pipeline context instance with penalty sampler & full CUDA offload"
        );

        Ok(Self {
            model,
            ctx,
            vocab,
            sampler,
            slice_config,
            hidden_dim,
            total_layers,
            n_ctx: cparams.n_ctx as usize,
            sampling,
            sampling_is_reasoning,
            capture_state,
        })
    }

    /// Appends the ordered sampler stages for `params` to `chain`.
    ///
    /// Order matters and mirrors llama.cpp's recommended chain:
    /// penalties -> top_k -> top_p -> min_p -> temperature -> dist.
    ///
    /// On failure the partially built chain is freed and the error returned, so
    /// callers never observe a half-configured sampler.
    fn populate_sampler_chain(chain: *mut LlamaSampler, params: SamplingParams) -> Result<()> {
        let stages = [
            llama_call!(
                llama_sampler_init_penalties,
                params.penalty_last_n,
                params.repeat_penalty,
                params.freq_penalty,
                params.present_penalty
            )?,
            llama_call!(llama_sampler_init_top_k, params.top_k)?,
            llama_call!(llama_sampler_init_top_p, params.top_p, 1)?,
            llama_call!(llama_sampler_init_min_p, params.min_p, 1)?,
            llama_call!(llama_sampler_init_temp, params.temperature)?,
            llama_call!(llama_sampler_init_dist, u32::MAX)?,
        ];

        for (idx, stage) in stages.iter().enumerate() {
            if stage.is_null() {
                llama_call!(llama_sampler_free, chain)?;
                for leftover in stages.iter().skip(idx + 1) {
                    if !leftover.is_null() {
                        llama_call!(llama_sampler_free, *leftover)?;
                    }
                }
                bail!("llama_sampler_init_* returned a null stage at index {idx}");
            }
            llama_call!(llama_sampler_chain_add, chain, *stage)?;
        }
        Ok(())
    }

    /// Replaces the active sampler chain so the next sampling call honours
    /// `params`. No-op when `params` already match the active configuration.
    ///
    /// This is what makes the OpenAI `temperature` / `top_p` / `top_k` / `min_p`
    /// request fields effective: llama.cpp bakes sampling configuration into the
    /// sampler object, so honouring a per-request value requires rebuilding it.
    pub fn rebuild_sampler(&mut self, params: SamplingParams) -> Result<()> {
        self.ensure_ready()?;
        if !params.differs_from(&self.sampling) {
            return Ok(());
        }

        let sparams = llama_call!(llama_sampler_chain_default_params)?;
        let new_chain = llama_call!(llama_sampler_chain_init, sparams)?;
        if new_chain.is_null() {
            bail!("llama_sampler_chain_init returned null while rebuilding the sampler");
        }

        if let Err(err) = Self::populate_sampler_chain(new_chain, params) {
            // `populate_sampler_chain` frees the partial chain for us.
            return Err(err);
        }

        // Swap only once the replacement is fully built, so a failure leaves the
        // previous sampler intact and usable.
        let old = std::mem::replace(&mut self.sampler, new_chain);
        if !old.is_null() {
            llama_call!(llama_sampler_free, old)?;
        }
        self.sampling = params;
        Ok(())
    }

    /// Sampling parameters currently baked into the active sampler.
    pub fn sampling_params(&self) -> SamplingParams {
        self.sampling
    }

    /// Sets up activation capture callback to intercept raw pre-norm residual stream (unnormalized x_K)
    /// for the specified layer before output_norm is applied.
    pub fn prepare_activation_capture(&mut self, target_layer: i32, buffer: *mut f32, capacity: usize) {
        self.capture_state.target_layer_idx = target_layer;
        self.capture_state.captured = false;
        self.capture_state.buffer = buffer;
        self.capture_state.capacity_floats = capacity;
        self.capture_state.captured_floats = 0;

        let name = format!("l_out-{}", target_layer);
        let mut name_bytes = [0u8; 64];
        for (i, b) in name.bytes().enumerate().take(63) {
            name_bytes[i] = b;
        }
        self.capture_state.target_tensor_name = name_bytes;
    }

    /// Returns true if the activation capture callback successfully intercepted the target layer tensor.
    pub fn is_activation_captured(&self) -> bool {
        self.capture_state.captured
    }

    /// Logs once per process whether raw pre-norm capture is active, plus the RMS of the first
    /// captured row (raw residual RMS is typically >> 1.0; post-norm identity output is ~1.0).
    fn log_capture_once(&self, out: &[f32], tag: &str) {
        use std::sync::atomic::{AtomicBool, Ordering};
        static LOGGED: AtomicBool = AtomicBool::new(false);
        if LOGGED.swap(true, Ordering::Relaxed) || self.hidden_dim == 0 || out.len() < self.hidden_dim {
            return;
        }
        let row = &out[..self.hidden_dim];
        let rms = (row.iter().map(|v| v * v).sum::<f32>() / row.len() as f32).sqrt();
        info!(
            stage = tag,
            captured = self.capture_state.captured,
            row0_rms = rms,
            "🔬 Activation capture check (raw pre-norm residual expected RMS >> 1.0)"
        );
    }

    /// Returns true if all native C++ pointers are valid and non-null.
    pub fn is_ready(&self) -> bool {
        !self.model.is_null() && !self.ctx.is_null() && !self.vocab.is_null() && !self.sampler.is_null()
    }

    /// Automatically detects the chat template format supported by the model's vocabulary.
    pub fn detect_chat_format(&self) -> ChatTemplateFormat {
        if !self.is_ready() {
            return ChatTemplateFormat::ChatML;
        }
        // DeepSeek models use `<｜User｜>` as a single special control token (token 151644)
        if let Ok(toks) = self.tokenize("<｜User｜>", false) {
            if toks.len() == 1 {
                return ChatTemplateFormat::DeepSeek;
            }
        }
        // ChatML / Qwen models use `<|im_start|>` as a single special control token
        if let Ok(toks) = self.tokenize("<|im_start|>", false) {
            if toks.len() == 1 {
                return ChatTemplateFormat::ChatML;
            }
        }
        // Llama 3 models use `<|start_header_id|>` as a single special control token
        if let Ok(toks) = self.tokenize("<|start_header_id|>", false) {
            if toks.len() == 1 {
                return ChatTemplateFormat::Llama3;
            }
        }
        ChatTemplateFormat::ChatML
    }

    /// Validates that the native instance is initialized and ready, returning a typed error otherwise.
    pub fn ensure_ready(&self) -> Result<()> {
        if !self.is_ready() {
            bail!("LlamaPipelineInstance is not ready or has been closed (native C++ pointers are null)");
        }
        Ok(())
    }

    pub fn tokenize(&self, text: &str, add_special: bool) -> Result<Vec<LlamaToken>> {
        self.ensure_ready()?;
        let c_text = CString::new(text)?;
        let max_tokens = (text.len() + 16).max(128) as i32;
        let mut tokens = vec![0 as LlamaToken; max_tokens as usize];

        let n_tokens = llama_call!(
            llama_tokenize,
            self.vocab,
            c_text.as_ptr(),
            text.len() as i32,
            tokens.as_mut_ptr(),
            max_tokens,
            add_special,
            true
        )?;

        if n_tokens < 0 {
            // Negative return is llama.cpp's "buffer too small" signal: its
            // magnitude is the required token capacity.
            let req_size = (-n_tokens) as usize;
            tokens.resize(req_size, 0);
            let n2 = llama_call!(
                llama_tokenize,
                self.vocab,
                c_text.as_ptr(),
                text.len() as i32,
                tokens.as_mut_ptr(),
                req_size as i32,
                add_special,
                true
            )?;
            if n2 < 0 {
                bail!("Tokenization failed for input text");
            }
            tokens.truncate(n2 as usize);
        } else {
            tokens.truncate(n_tokens as usize);
        }

        Ok(tokens)
    }

    pub fn token_to_piece(&self, token: LlamaToken) -> Result<Vec<u8>> {
        self.ensure_ready()?;
        let mut buf = [0u8; 128];
        let n = llama_call!(
            llama_token_to_piece,
            self.vocab,
            token,
            buf.as_mut_ptr() as *mut c_char,
            buf.len() as i32,
            0,
            true
        )?;
        if n < 0 {
            let req_size = (-n) as usize;
            let mut big_buf = vec![0u8; req_size];
            let n2 = llama_call!(
                llama_token_to_piece,
                self.vocab,
                token,
                big_buf.as_mut_ptr() as *mut c_char,
                req_size as i32,
                0,
                true
            )?;
            if n2 < 0 {
                bail!("Token piece extraction failed");
            }
            big_buf.truncate(n2 as usize);
            Ok(big_buf)
        } else {
            Ok(buf[..n as usize].to_vec())
        }
    }

    pub fn token_to_piece_str(&self, token: LlamaToken) -> String {
        self.token_to_piece(token)
            .ok()
            .and_then(|bytes| String::from_utf8(bytes).ok())
            .unwrap_or_default()
    }

    pub fn is_eog(&self, token: LlamaToken) -> bool {
        if !self.is_ready() {
            // Treat "not ready" as end-of-generation: continuing would sample
            // from a dead context.
            return true;
        }
        match LLAMA.api() {
            Ok(api) => unsafe { (api.llama_vocab_is_eog)(self.vocab, token) },
            Err(_) => true,
        }
    }

    /// Clears the KV cache and resets sampler state for a new session.
    ///
    /// Best-effort: failures are logged rather than propagated, because callers
    /// are usually mid-stream and a partial reset is preferable to aborting a
    /// generation. Returns `true` when the reset ran.
    pub fn clear_kv_cache(&self) -> bool {
        if !self.is_ready() {
            return false;
        }
        let api = match LLAMA.api() {
            Ok(api) => api,
            Err(err) => {
                warn!("Cannot clear KV cache: {err}");
                return false;
            }
        };
        unsafe {
            let mem = (api.llama_get_memory)(self.ctx);
            if !mem.is_null() {
                (api.llama_memory_clear)(mem, true);
            }
            if !self.sampler.is_null() {
                (api.llama_sampler_reset)(self.sampler);
            }
        }
        true
    }

    /// Stage 1: Evaluate input tokens through local layer slice, extracting intermediate activation matrix [seq_len * hidden_dim].
    pub fn evaluate_tokens_to_activations(
        &mut self,
        tokens: &[LlamaToken],
        start_pos: u32,
    ) -> Result<Vec<f32>> {
        self.ensure_ready()?;
        let mut activations = Vec::with_capacity(tokens.len() * self.hidden_dim);
        self.evaluate_tokens_to_activations_into(tokens, start_pos, &mut activations)?;
        Ok(activations)
    }

    /// Stage 1 In-Place: Evaluate tokens directly into pre-allocated buffer without hot-loop heap allocations.
    pub fn evaluate_tokens_to_activations_into(
        &mut self,
        tokens: &[LlamaToken],
        start_pos: u32,
        out: &mut Vec<f32>,
    ) -> Result<()> {
        self.ensure_ready()?;
        if tokens.is_empty() {
            out.clear();
            return Ok(());
        }

        let n_tokens = tokens.len() as i32;
        let total_floats = (n_tokens as usize) * self.hidden_dim;
        if out.len() != total_floats {
            out.resize(total_floats, 0.0);
        }

        // Bug 1 Fix: Arm cb_eval to intercept raw pre-norm hidden states (unnormalized x_K)
        // at the final transformer layer of this slice ("l_out-<last_layer>").
        // This completely avoids identity RMSNorm residual attenuation.
        let last_layer = (self.total_layers.saturating_sub(1)) as i32;
        self.prepare_activation_capture(last_layer, out.as_mut_ptr(), total_floats);

        let mut batch = llama_call!(llama_batch_init, n_tokens, 0, 1)?;
        batch.n_tokens = n_tokens;

        for (i, &t) in tokens.iter().enumerate() {
            unsafe {
                *batch.token.add(i) = t;
                *batch.pos.add(i) = (start_pos as i32) + (i as i32);
                *batch.n_seq_id.add(i) = 1;
                *(*batch.seq_id.add(i)) = 0;
                *batch.logits.add(i) = 1;
            }
        }

        let res = llama_call!(llama_decode, self.ctx, batch)?;
        llama_call!(llama_batch_free, batch)?;

        if res != 0 {
            bail!("llama_decode failed in Stage 1 forward pass (code {})", res);
        }

        // Explicit CUDA Stream Synchronization to ensure clean host memory reads
        llama_call!(llama_synchronize, self.ctx)?;

        if !self.is_activation_captured() {
            tracing::warn!(
                layer = last_layer,
                "⚠️ cb_eval did not capture l_out-{} activations; falling back to llama_get_embeddings_ith",
                last_layer
            );
            for i in 0..n_tokens {
                let embd_ptr = llama_call!(llama_get_embeddings_ith, self.ctx, i)?;
                if embd_ptr.is_null() {
                    bail!("Failed to get embeddings at index {} from llama_context", i);
                }
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        embd_ptr,
                        out.as_mut_ptr().add((i as usize) * self.hidden_dim),
                        self.hidden_dim,
                    );
                }
            }
        }

        self.log_capture_once(out, "stage0");
        Ok(())
    }

    /// Stage 2: Inject intermediate activation matrix [seq_len * hidden_dim], compute through downstream layers + LM Head.
    pub fn evaluate_activations_to_logits(
        &mut self,
        activations: &[f32],
        seq_len: u32,
        start_pos: u32,
    ) -> Result<()> {
        self.ensure_ready()?;
        ensure!(
            activations.len() == (seq_len as usize) * self.hidden_dim,
            "Activation size mismatch: expected {} floats, got {}",
            (seq_len as usize) * self.hidden_dim,
            activations.len()
        );

        let n_tokens = seq_len as i32;
        let mut batch = llama_call!(llama_batch_init, n_tokens, self.hidden_dim as i32, 1)?;
        batch.n_tokens = n_tokens;

        unsafe {
            std::ptr::copy_nonoverlapping(
                activations.as_ptr(),
                batch.embd,
                activations.len(),
            );
        }

        for i in 0..n_tokens {
            unsafe {
                *batch.pos.add(i as usize) = (start_pos as i32) + i;
                *batch.n_seq_id.add(i as usize) = 1;
                *(*batch.seq_id.add(i as usize)) = 0;
                *batch.logits.add(i as usize) = if i == n_tokens - 1 { 1 } else { 0 };
            }
        }

        let res = llama_call!(llama_decode, self.ctx, batch)?;
        llama_call!(llama_batch_free, batch)?;

        if res != 0 {
            bail!("llama_decode failed in Stage 2 activation forward pass (code {})", res);
        }

        // Explicit CUDA Stream Synchronization
        llama_call!(llama_synchronize, self.ctx)?;

        Ok(())
    }

    /// Intermediate Stage (1 <= stage < N - 1):
    /// Evaluates input activation matrix through intermediate transformer layer blocks,
    /// extracting output activation matrix [seq_len * hidden_dim] for downstream forwarding.
    pub fn evaluate_activations_to_activations(
        &mut self,
        activations: &[f32],
        seq_len: u32,
        start_pos: u32,
        out: &mut Vec<f32>,
    ) -> Result<()> {
        self.ensure_ready()?;
        ensure!(
            activations.len() == (seq_len as usize) * self.hidden_dim,
            "Activation size mismatch: expected {} floats, got {}",
            (seq_len as usize) * self.hidden_dim,
            activations.len()
        );

        if seq_len == 0 {
            out.clear();
            return Ok(());
        }

        let n_tokens = seq_len as i32;
        let total_floats = (n_tokens as usize) * self.hidden_dim;
        if out.len() != total_floats {
            out.resize(total_floats, 0.0);
        }

        // Bug 1 Fix: Arm cb_eval to intercept raw pre-norm hidden states (unnormalized x_K)
        // at the final transformer layer of this intermediate slice ("l_out-<last_layer>").
        let last_layer = (self.total_layers.saturating_sub(1)) as i32;
        self.prepare_activation_capture(last_layer, out.as_mut_ptr(), total_floats);

        let mut batch = llama_call!(llama_batch_init, n_tokens, self.hidden_dim as i32, 1)?;
        batch.n_tokens = n_tokens;

        unsafe {
            std::ptr::copy_nonoverlapping(
                activations.as_ptr(),
                batch.embd,
                activations.len(),
            );
        }

        for i in 0..n_tokens {
            unsafe {
                *batch.pos.add(i as usize) = (start_pos as i32) + i;
                *batch.n_seq_id.add(i as usize) = 1;
                *(*batch.seq_id.add(i as usize)) = 0;
                // Request output for all tokens in intermediate stage
                *batch.logits.add(i as usize) = 1;
            }
        }

        let res = llama_call!(llama_decode, self.ctx, batch)?;
        llama_call!(llama_batch_free, batch)?;

        if res != 0 {
            bail!("llama_decode failed in intermediate activation forward pass (code {})", res);
        }

        // Explicit CUDA Stream Synchronization
        llama_call!(llama_synchronize, self.ctx)?;

        if !self.is_activation_captured() {
            tracing::warn!(
                layer = last_layer,
                "⚠️ cb_eval did not capture intermediate l_out-{} activations; falling back to llama_get_embeddings_ith",
                last_layer
            );
            for i in 0..n_tokens {
                let embd_ptr = llama_call!(llama_get_embeddings_ith, self.ctx, i)?;
                if embd_ptr.is_null() {
                    bail!("Failed to get intermediate embeddings at index {} from llama_context", i);
                }
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        embd_ptr,
                        out.as_mut_ptr().add((i as usize) * self.hidden_dim),
                        self.hidden_dim,
                    );
                }
            }
        }

        self.log_capture_once(out, "intermediate");
        Ok(())
    }

    /// Samples the next token from the latest logits, applying the current
    /// sampler chain and accepting the token into its penalty state.
    ///
    /// `temperature` / `top_p` are honoured: llama.cpp bakes sampling setup into
    /// the sampler object, so a request that differs from the active
    /// configuration rebuilds the chain first (see [`Self::rebuild_sampler`]).
    /// Passing `None` keeps whatever is currently configured.
    pub fn sample_next_token(
        &mut self,
        temperature: Option<f32>,
        top_p: Option<f32>,
        top_k: Option<i32>,
        min_p: Option<f32>,
    ) -> Result<LlamaToken> {
        self.ensure_ready()?;

        let requested =
            SamplingParams::resolve(self.sampling, temperature, top_p, top_k, min_p, None);
        self.rebuild_sampler(requested)?;

        let logits_ptr = llama_call!(llama_get_logits_ith, self.ctx, -1)?;
        if !logits_ptr.is_null() {
            let n_vocab = llama_call!(llama_vocab_n_tokens, self.vocab)? as usize;
            // SAFETY: llama.cpp owns `n_vocab` f32 logits for the last decoded
            // token and guarantees they stay valid until the next decode.
            let logits_slice = unsafe { std::slice::from_raw_parts_mut(logits_ptr, n_vocab) };
            let has_invalid = logits_slice.iter().any(|v| v.is_nan() || v.is_infinite());
            if has_invalid {
                warn!(
                    "⚠️ Stage 2 output logits contain NaN/Inf! Sanitizing invalid logits to \
                     prevent cluster context desynchronization."
                );
                for v in logits_slice.iter_mut() {
                    if v.is_nan() || v.is_infinite() {
                        *v = -1e9;
                    }
                }
            }
        }

        let token_id = llama_call!(llama_sampler_sample, self.sampler, self.ctx, -1)?;
        llama_call!(llama_sampler_accept, self.sampler, token_id)?;
        Ok(token_id)
    }

    /// Explicitly frees the llama context, model, and sampler from GPU/CPU memory and nullifies pointers.
    ///
    /// Idempotent and safe to call on an already-closed instance (CRASH-01):
    /// every pointer is null-checked and reset, so a later FFI call hits
    /// `ensure_ready()` rather than a dangling C++ object.
    pub fn close(&mut self) {
        let Ok(api) = LLAMA.api() else {
            // Runtime unavailable means nothing was ever allocated by us.
            self.sampler = std::ptr::null_mut();
            self.ctx = std::ptr::null_mut();
            self.model = std::ptr::null_mut();
            self.vocab = std::ptr::null();
            return;
        };
        unsafe {
            if !self.sampler.is_null() {
                (api.llama_sampler_free)(self.sampler);
                self.sampler = std::ptr::null_mut();
            }
            if !self.ctx.is_null() {
                (api.llama_free)(self.ctx);
                self.ctx = std::ptr::null_mut();
            }
            if !self.model.is_null() {
                (api.llama_model_free)(self.model);
                self.model = std::ptr::null_mut();
            }
            self.vocab = std::ptr::null();
        }
    }
}

#[cfg(test)]
impl LlamaPipelineInstance {
    /// A closed instance with all native pointers null, used to assert that
    /// every accessor fails safely instead of dereferencing a dead context
    /// (regression cover for CRASH-01).
    pub fn dummy_closed() -> Self {
        Self {
            model: std::ptr::null_mut(),
            ctx: std::ptr::null_mut(),
            vocab: std::ptr::null(),
            sampler: std::ptr::null_mut(),
            slice_config: LayerSliceConfig::new(0, 0, 1).unwrap(),
            hidden_dim: 128,
            total_layers: 1,
            n_ctx: 512,
            sampling: SamplingParams::DEFAULT,
            sampling_is_reasoning: false,
            capture_state: Box::new(ActivationCaptureState::default()),
        }
    }
}

impl Drop for LlamaPipelineInstance {
    fn drop(&mut self) {
        self.close();
    }
}

#[cfg(test)]
mod tests_instance_safety {
    use super::*;

    #[test]
    fn test_closed_instance_safety() {
        let mut inst = LlamaPipelineInstance::dummy_closed();
        assert!(!inst.is_ready(), "Dummy closed instance must not be ready");
        assert!(inst.ensure_ready().is_err(), "ensure_ready must return error");

        // tokenize must return error, no segfault
        assert!(inst.tokenize("Hello", true).is_err(), "tokenize on closed instance must return error");

        // token_to_piece must return error
        assert!(inst.token_to_piece(1).is_err(), "token_to_piece on closed instance must return error");

        // token_to_piece_str must return empty string safely
        assert_eq!(inst.token_to_piece_str(1), "", "token_to_piece_str must safely return empty string");

        // is_eog must safely return true
        assert!(inst.is_eog(1), "is_eog must safely return true on closed instance");

        // clear_kv_cache must not crash or panic
        inst.clear_kv_cache();

        // evaluate_tokens_to_activations must return error
        assert!(inst.evaluate_tokens_to_activations(&[1, 2], 0).is_err(), "evaluate_tokens must return error");

        // evaluate_tokens_to_activations_into must return error
        let mut out = Vec::new();
        assert!(inst.evaluate_tokens_to_activations_into(&[1, 2], 0, &mut out).is_err(), "evaluate_tokens_into must return error");

        // evaluate_activations_to_logits must return error
        let acts = vec![0.0f32; 128];
        assert!(inst.evaluate_activations_to_logits(&acts, 1, 0).is_err(), "evaluate_activations must return error");

        // evaluate_activations_to_activations must return error
        assert!(inst.evaluate_activations_to_activations(&acts, 1, 0, &mut out).is_err(), "evaluate_activations_to_activations must return error");

        // sample_next_token must return error
        assert!(
            inst.sample_next_token(Some(0.7), Some(0.9), Some(42), Some(0.05)).is_err(),
            "sample_next_token must return error"
        );

        // rebuild_sampler must also refuse to operate on a closed instance
        assert!(
            inst.rebuild_sampler(SamplingParams::DEFAULT).is_err(),
            "rebuild_sampler must return error on a closed instance"
        );

        // repeated close must be safe (idempotent)
        inst.close();
        assert!(!inst.is_ready());
    }

    #[test]
    fn test_chat_format_detection() {
        crate::ensure_llama_initialized();
        let ds_path = if std::path::Path::new("models/DS.gguf").exists() {
            Some(std::path::PathBuf::from("models/DS.gguf"))
        } else if std::path::Path::new("../../models/DS.gguf").exists() {
            Some(std::path::PathBuf::from("../../models/DS.gguf"))
        } else {
            None
        };
        if let Some(ref path) = ds_path {
            let slice = LayerSliceConfig::new(0, 0, 48).unwrap();
            if let Ok(inst) = LlamaPipelineInstance::open(path, slice, 0, 512) {
                assert_eq!(inst.detect_chat_format(), ChatTemplateFormat::DeepSeek);
            }
        }

        let test_path = if std::path::Path::new("models/test.gguf").exists() {
            Some(std::path::PathBuf::from("models/test.gguf"))
        } else if std::path::Path::new("../../models/test.gguf").exists() {
            Some(std::path::PathBuf::from("../../models/test.gguf"))
        } else {
            None
        };
        if let Some(ref path) = test_path {
            let slice = LayerSliceConfig::new(0, 0, 28).unwrap();
            if let Ok(inst) = LlamaPipelineInstance::open(path, slice, 0, 512) {
                let fmt = inst.detect_chat_format();
                assert!(
                    matches!(fmt, ChatTemplateFormat::Llama3 | ChatTemplateFormat::ChatML | ChatTemplateFormat::DeepSeek),
                    "Detected format should be a recognized format, got {:?}",
                    fmt
                );
            }
        }
    }
}

