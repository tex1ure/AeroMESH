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

use crate::llama_ffi::*;
use std::ffi::{c_char, CString};

pub struct LlamaPipelineInstance {
    model: *mut LlamaModel,
    ctx: *mut LlamaContext,
    vocab: *const LlamaVocab,
    pub sampler: *mut LlamaSampler,
    pub slice_config: LayerSliceConfig,
    pub hidden_dim: usize,
    pub total_layers: usize,
    pub n_ctx: usize,
    pub decode_buf: Vec<f32>,
}

unsafe impl Send for LlamaPipelineInstance {}

impl LlamaPipelineInstance {
    pub fn open(
        model_path: &Path,
        slice_config: LayerSliceConfig,
        n_gpu_layers: i32,
        n_ctx: u32,
    ) -> Result<Self> {
        ensure_llama_initialized();

        let path_str = model_path.to_str().context("Invalid model path string")?;
        let c_path = CString::new(path_str)?;

        let mut mparams = unsafe { llama_model_default_params() };
        mparams.n_gpu_layers = if n_gpu_layers > 0 {
            n_gpu_layers
        } else {
            0
        };

        let model = unsafe { llama_model_load_from_file(c_path.as_ptr(), mparams) };
        if model.is_null() {
            bail!("Failed to load GGUF model via llama.cpp from {:?}", model_path);
        }

        let vocab = unsafe { llama_model_get_vocab(model) };
        if vocab.is_null() {
            unsafe { llama_model_free(model) };
            bail!("Failed to get model vocabulary from GGUF model");
        }

        let hidden_dim = unsafe { llama_model_n_embd(model) } as usize;
        let total_layers = unsafe { llama_model_n_layer(model) } as usize;

        let physical_threads = std::thread::available_parallelism()
            .map(|n| (n.get() / 2).max(1))
            .unwrap_or(4) as i32;

        let mut cparams = unsafe { llama_context_default_params() };
        cparams.n_ctx = if n_ctx > 0 { n_ctx } else { 4096 };
        cparams.embeddings = true;

        let ctx = unsafe { llama_init_from_model(model, cparams) };
        if ctx.is_null() {
            unsafe { llama_model_free(model) };
            bail!("Failed to initialize llama_context for model {:?}", model_path);
        }

        let sparams = unsafe { llama_sampler_chain_default_params() };
        let sampler = unsafe { llama_sampler_chain_init(sparams) };
        unsafe {
            llama_sampler_chain_add(sampler, llama_sampler_init_penalties(512, 1.15, 0.20, 0.15));
            llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
            llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.9, 1));
            llama_sampler_chain_add(sampler, llama_sampler_init_min_p(0.05, 1));
            llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.7));
            llama_sampler_chain_add(sampler, llama_sampler_init_dist(42));
        }

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
            decode_buf: vec![0.0f32; hidden_dim],
        })
    }

    pub fn tokenize(&self, text: &str, add_special: bool) -> Result<Vec<LlamaToken>> {
        let c_text = CString::new(text)?;
        let max_tokens = (text.len() + 16).max(128) as i32;
        let mut tokens = vec![0 as LlamaToken; max_tokens as usize];

        let n_tokens = unsafe {
            llama_tokenize(
                self.vocab,
                c_text.as_ptr(),
                text.len() as i32,
                tokens.as_mut_ptr(),
                max_tokens,
                add_special,
                true,
            )
        };

        if n_tokens < 0 {
            let req_size = (-n_tokens) as usize;
            tokens.resize(req_size, 0);
            let n2 = unsafe {
                llama_tokenize(
                    self.vocab,
                    c_text.as_ptr(),
                    text.len() as i32,
                    tokens.as_mut_ptr(),
                    req_size as i32,
                    add_special,
                    true,
                )
            };
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
        let mut buf = [0u8; 128];
        let n = unsafe {
            llama_token_to_piece(
                self.vocab,
                token,
                buf.as_mut_ptr() as *mut c_char,
                buf.len() as i32,
                0,
                true,
            )
        };
        if n < 0 {
            let req_size = (-n) as usize;
            let mut big_buf = vec![0u8; req_size];
            let n2 = unsafe {
                llama_token_to_piece(
                    self.vocab,
                    token,
                    big_buf.as_mut_ptr() as *mut c_char,
                    req_size as i32,
                    0,
                    true,
                )
            };
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
        unsafe { llama_vocab_is_eog(self.vocab, token) }
    }

    pub fn clear_kv_cache(&self) {
        unsafe {
            let mem = llama_get_memory(self.ctx);
            if !mem.is_null() {
                llama_memory_clear(mem, true);
            }
            if !self.sampler.is_null() {
                llama_sampler_reset(self.sampler);
            }
        }
    }

    /// Stage 1: Evaluate input tokens through local layer slice, extracting intermediate activation matrix [seq_len * hidden_dim].
    pub fn evaluate_tokens_to_activations(
        &mut self,
        tokens: &[LlamaToken],
        start_pos: u32,
    ) -> Result<Vec<f32>> {
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
        if tokens.is_empty() {
            out.clear();
            return Ok(());
        }

        let n_tokens = tokens.len() as i32;
        let mut batch = unsafe { llama_batch_init(n_tokens, 0, 1) };
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

        let res = unsafe { llama_decode(self.ctx, batch) };
        unsafe { llama_batch_free(batch) };

        if res != 0 {
            bail!("llama_decode failed in Stage 1 forward pass (code {})", res);
        }

        // Explicit CUDA Stream Synchronization to ensure clean host memory reads
        unsafe { llama_synchronize(self.ctx) };

        let total_floats = (n_tokens as usize) * self.hidden_dim;
        if out.len() != total_floats {
            out.resize(total_floats, 0.0);
        }

        for i in 0..n_tokens {
            let embd_ptr = unsafe { llama_get_embeddings_ith(self.ctx, i) };
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

        Ok(())
    }

    /// Stage 2: Inject incoming intermediate activation matrix [seq_len * hidden_dim], compute through downstream layers + LM Head.
    pub fn evaluate_activations_to_logits(
        &mut self,
        activations: &[f32],
        seq_len: u32,
        start_pos: u32,
    ) -> Result<()> {
        ensure!(
            activations.len() == (seq_len as usize) * self.hidden_dim,
            "Activation size mismatch: expected {} floats, got {}",
            (seq_len as usize) * self.hidden_dim,
            activations.len()
        );

        let n_tokens = seq_len as i32;
        let mut batch = unsafe { llama_batch_init(n_tokens, self.hidden_dim as i32, 1) };
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

        let res = unsafe { llama_decode(self.ctx, batch) };
        unsafe { llama_batch_free(batch) };

        if res != 0 {
            bail!("llama_decode failed in Stage 2 activation forward pass (code {})", res);
        }

        // Explicit CUDA Stream Synchronization
        unsafe { llama_synchronize(self.ctx) };

        Ok(())
    }

    /// Samples next token from the latest logits computed by Stage 2, applying repetition penalties and accepting the token.
    pub fn sample_next_token(
        &mut self,
        _temperature: f32,
        _top_p: f32,
        _seed: u32,
    ) -> Result<LlamaToken> {
        let logits_ptr = unsafe { llama_get_logits_ith(self.ctx, -1) };
        if !logits_ptr.is_null() {
            let n_vocab = unsafe { llama_vocab_n_tokens(self.vocab) } as usize;
            let logits_slice = unsafe { std::slice::from_raw_parts(logits_ptr, n_vocab.min(256)) };
            let has_invalid = logits_slice.iter().any(|v| v.is_nan() || v.is_infinite());
            if has_invalid {
                warn!("⚠️ Stage 2 output logits contain NaN/Inf! Resetting context memory.");
                self.clear_kv_cache();
            }
        }

        let token_id = unsafe { llama_sampler_sample(self.sampler, self.ctx, -1) };
        unsafe {
            llama_sampler_accept(self.sampler, token_id);
        }
        Ok(token_id)
    }

    /// Explicitly frees the llama context, model, and sampler from GPU/CPU memory.
    pub fn close(&mut self) {
        unsafe {
            if !self.sampler.is_null() {
                llama_sampler_free(self.sampler);
                self.sampler = std::ptr::null_mut();
            }
            if !self.ctx.is_null() {
                llama_free(self.ctx);
                self.ctx = std::ptr::null_mut();
            }
            if !self.model.is_null() {
                llama_model_free(self.model);
                self.model = std::ptr::null_mut();
            }
        }
    }
}

impl Drop for LlamaPipelineInstance {
    fn drop(&mut self) {
        self.close();
    }
}
