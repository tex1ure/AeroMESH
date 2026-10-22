use crate::slice_loader::LayerSliceConfig;
use anyhow::{bail, ensure, Context, Result};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use memmap2::Mmap;
use std::fs::File;
use std::io::{BufWriter, Cursor, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use tracing::info;

pub const GGUF_MAGIC: [u8; 4] = *b"GGUF";
pub const DEFAULT_ALIGNMENT: u64 = 32;

/// Builds the on-disk name for a stage slice.
///
/// The name encodes the layer span of the partition:
///   * 2-stage:  `{stem}_layers{start}-{end}_stage{K}.gguf`
///   * N-stage:  `{stem}_stage{K}_of_{M}.gguf`
///
/// Encoding the span matters. The two-stage names used to be
/// `{stem}_stage1.gguf` / `{stem}_stage2.gguf`, which carry no split
/// information, and the reuse check only compared layer *counts*. Slicing the
/// same model with `--split 14` and then `--split 20` therefore reused the
/// stale 14-layer partition while reporting success.
pub fn get_stage_slice_path(
    out_dir: &Path,
    stem: &str,
    stage_index: usize,
    total_stages: usize,
    layer_start: usize,
    layer_end: usize,
) -> PathBuf {
    if total_stages == 2 {
        out_dir.join(format!(
            "{}_layers{}-{}_stage{}.gguf",
            stem,
            layer_start,
            layer_end,
            stage_index + 1
        ))
    } else {
        out_dir.join(format!(
            "{}_stage{}_of_{}.gguf",
            stem,
            stage_index + 1,
            total_stages
        ))
    }
}

/// Resolves the path of a stage slice.
///
/// A partition written by an older build is never silently substituted: the
/// legacy name (`{stem}_stage1.gguf`) encodes no layer span, so it cannot be
/// validated against a request. See [`legacy_stage_slice_path`] for detecting
/// one when reporting to the user.
pub fn resolve_stage_slice_path(
    out_dir: &Path,
    stem: &str,
    stage_index: usize,
    total_stages: usize,
    layer_start: usize,
    layer_end: usize,
) -> PathBuf {
    get_stage_slice_path(
        out_dir,
        stem,
        stage_index,
        total_stages,
        layer_start,
        layer_end,
    )
}

/// Path of a slice written by a build predating span-encoded names.
///
/// Used only to *report* a reusable legacy artefact, never to claim it satisfies
/// a new request.
pub fn legacy_stage_slice_path(
    out_dir: &Path,
    stem: &str,
    stage_index: usize,
    total_stages: usize,
) -> PathBuf {
    if total_stages == 2 {
        out_dir.join(format!("{}_stage{}.gguf", stem, stage_index + 1))
    } else {
        out_dir.join(format!(
            "{}_stage{}_of_{}.gguf",
            stem,
            stage_index + 1,
            total_stages
        ))
    }
}

/// Parses the split encoded in a slice file name, if present.
fn split_from_stage_name(name: &str) -> Option<usize> {
    let rest = name.split("_split").nth(1)?;
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse().ok()
}

/// True when `name` looks like a stage-1 partition produced by this slicer.
///
/// Recognises both naming schemes:
///   * `{stem}_split{N}_stage1.gguf` (current, 2-stage)
///   * `{stem}_stage1.gguf`         (legacy, 2-stage, no split encoded)
///   * `{stem}_stage1_of_{M}.gguf`  (N-stage)
pub fn is_stage1_slice_name(name: &str) -> bool {
    let lowered = name.to_ascii_lowercase();
    if !lowered.ends_with(".gguf") {
        return false;
    }
    lowered.ends_with("_stage1.gguf")
        || lowered.contains("_split") && lowered.contains("_stage1.gguf")
        || lowered.contains("_stage1_of_")
}

/// True when `name` looks like any non-terminal or terminal stage partition.
pub fn is_stage_slice_name(name: &str) -> bool {
    let lowered = name.to_ascii_lowercase();
    if !lowered.ends_with(".gguf") {
        return false;
    }
    if lowered.contains("_stage") && lowered.contains("_of_") {
        return true;
    }
    // `_split{N}_stage{K}.gguf`
    if lowered.contains("_split") && lowered.contains("_stage") {
        return true;
    }
    // Legacy `_stage1.gguf` / `_stage2.gguf`, but not e.g. "stagehand.gguf".
    for suffix in ["_stage1.gguf", "_stage2.gguf"] {
        if lowered.ends_with(suffix) {
            return true;
        }
    }
    false
}

/// Slices a single stage GGUF model file for a given node's assigned layer slice:
/// - Stages 0..total_stages-2: Layers slice.layer_start..=slice.layer_end with Identity RMSNorm (output_norm = 1.0)
/// - Stage total_stages-1 (Final stage): Layers slice.layer_start..=slice.layer_end with authentic output_norm and true LM Head
pub fn slice_gguf_stage<P: AsRef<Path>, Q: AsRef<Path>>(
    source_model: P,
    output_dir: Q,
    slice: &LayerSliceConfig,
    stage_index: usize,
    total_stages: usize,
) -> Result<PathBuf> {
    let src_path = source_model.as_ref();
    let out_dir = output_dir.as_ref();
    std::fs::create_dir_all(out_dir)?;

    ensure!(
        total_stages >= 2,
        "total_stages must be >= 2 (got {})",
        total_stages
    );
    ensure!(
        stage_index < total_stages,
        "stage_index ({}) must be < total_stages ({})",
        stage_index,
        total_stages
    );

    if std::fs::metadata(src_path).map(|m| m.len() > 10 * 1024 * 1024).unwrap_or(false) {
        info!(
            source = %src_path.display(),
            stage_index = stage_index,
            total_stages = total_stages,
            layers = ?slice.layer_range(),
            "Deferring stage slice to Ephemeral VRAM Loader (0.0 MB persistent disk overhead)"
        );
        return Ok(src_path.to_path_buf());
    }

    let stem = src_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model");
    let stage_path = resolve_stage_slice_path(
        out_dir,
        stem,
        stage_index,
        total_stages,
        slice.layer_start,
        slice.layer_end,
    );

    // Reuse an existing slice when it is structurally the artefact we were
    // asked for.
    //
    // The old check also required the file to exceed an arbitrary 1000 bytes.
    // That constant was a proxy for "not truncated", but it is wrong for small
    // models and tests, which were silently re-sliced on every run. Structural
    // validity is the real requirement, and it is already established by
    // successfully opening the file as a GGUF and reading its layer count: a
    // truncated tensor table or data section fails to parse.
    if stage_path.exists() {
        // A legacy (split-less) name cannot be trusted on its own: the encoded
        // split must match the request.
        let name_matches = match split_from_stage_name(
            stage_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default(),
        ) {
            Some(encoded_split) => encoded_split == slice.layer_end + 1,
            None => true, // legacy name: fall through to the structural check
        };

        if name_matches {
            if let Ok(loader) = crate::GgufSliceLoader::open(&stage_path) {
                if loader.total_layers == slice.layer_count() {
                    if let (Ok(src_meta), Ok(s_meta)) =
                        (std::fs::metadata(src_path), std::fs::metadata(&stage_path))
                    {
                        if let (Ok(src_time), Ok(s_time)) = (src_meta.modified(), s_meta.modified())
                        {
                            if s_time >= src_time {
                                info!(
                                    stage = %stage_path.display(),
                                    stage_index = stage_index,
                                    total_stages = total_stages,
                                    layers = ?slice.layer_range(),
                                    bytes = s_meta.len(),
                                    "⚡ Reusing existing partitioned sliced GGUF file"
                                );
                                return Ok(stage_path);
                            }
                        }
                    }
                }
            }
        } else {
            info!(
                stage = %stage_path.display(),
                requested_split = slice.layer_end + 1,
                "Existing slice encodes a different split point; re-slicing"
            );
        }
    }

    info!(
        source = %src_path.display(),
        stage_index = stage_index,
        total_stages = total_stages,
        layers = ?slice.layer_range(),
        "🔪 Slicing GGUF for Stage {}/{} (layers {}..={})...",
        stage_index + 1,
        total_stages,
        slice.layer_start,
        slice.layer_end
    );

    let file = File::open(src_path)
        .with_context(|| format!("Failed to open source GGUF at {:?}", src_path))?;
    let mmap = unsafe { Mmap::map(&file)? };
    let mut cursor = Cursor::new(&mmap[..]);

    let mut magic = [0u8; 4];
    cursor.read_exact(&mut magic)?;
    ensure!(magic == GGUF_MAGIC, "Invalid GGUF magic bytes");

    let version = cursor.read_u32::<LittleEndian>()?;
    ensure!(
        version == 2 || version == 3,
        "Unsupported GGUF version: {}",
        version
    );

    let tensor_count = cursor.read_u64::<LittleEndian>()? as usize;
    let kv_count = cursor.read_u64::<LittleEndian>()? as usize;

    let mut total_layers = 0usize;
    let mut alignment = DEFAULT_ALIGNMENT;

    // Parse all raw KV metadata
    let mut raw_kvs = Vec::with_capacity(kv_count);
    for _ in 0..kv_count {
        let key = read_gguf_string(&mut cursor)?;
        let val_type = cursor.read_u32::<LittleEndian>()?;
        let start_pos = cursor.position();
        skip_gguf_val(&mut cursor, val_type)?;
        let end_pos = cursor.position();
        let val_bytes = mmap[start_pos as usize..end_pos as usize].to_vec();

        if key == "general.alignment" {
            if val_type == 4 && val_bytes.len() >= 4 {
                alignment = (&val_bytes[..4]).read_u32::<LittleEndian>()? as u64;
            } else if val_type == 10 && val_bytes.len() >= 8 {
                alignment = (&val_bytes[..8]).read_u64::<LittleEndian>()?;
            }
        } else if key.ends_with(".block_count") {
            if val_type == 4 && val_bytes.len() >= 4 {
                total_layers = (&val_bytes[..4]).read_u32::<LittleEndian>()? as usize;
            } else if val_type == 10 && val_bytes.len() >= 8 {
                total_layers = (&val_bytes[..8]).read_u64::<LittleEndian>()? as usize;
            }
        }

        raw_kvs.push(RawKvEntry {
            key,
            val_type,
            val_bytes,
        });
    }

    ensure!(
        slice.layer_end < total_layers,
        "slice layer_end ({}) must be < model total_layers ({})",
        slice.layer_end,
        total_layers
    );

    // Parse Tensor Info Table
    let mut tensors = Vec::with_capacity(tensor_count);
    for _ in 0..tensor_count {
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

        let size_bytes = ggml_tensor_size_bytes(ggml_type, element_count);
        let layer_index = extract_layer_index(&name);

        tensors.push(SourceTensor {
            name,
            dimensions,
            ggml_type,
            offset,
            size_bytes,
            layer_index,
        });
    }

    let header_end = cursor.position();
    let pad = (alignment - (header_end % alignment)) % alignment;
    let data_section_offset = header_end + pad;

    write_stage_gguf(
        &stage_path,
        version,
        alignment,
        data_section_offset,
        &raw_kvs,
        &tensors,
        &mmap,
        slice.layer_start,
        slice.layer_end,
        stage_index,
        total_stages,
    )?;

    info!(stage = %stage_path.display(), "✅ Stage GGUF slicing complete");
    Ok(stage_path)
}

/// Slices a monolithic GGUF model file into N standalone partitioned GGUF models according to stage_slices.
pub fn slice_gguf_for_multi_stage<P: AsRef<Path>, Q: AsRef<Path>>(
    source_model: P,
    output_dir: Q,
    stage_slices: &[LayerSliceConfig],
) -> Result<Vec<PathBuf>> {
    ensure!(!stage_slices.is_empty(), "stage_slices must not be empty");
    let total_stages = stage_slices.len();
    ensure!(
        total_stages >= 2,
        "Multi-stage pipeline requires at least 2 stages"
    );

    let src_path = source_model.as_ref();
    let out_dir = output_dir.as_ref();
    std::fs::create_dir_all(out_dir)?;

    let stem = src_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model");
    let mut stage_paths = Vec::with_capacity(total_stages);

    let mut all_cached = true;
    for (i, slice) in stage_slices.iter().enumerate() {
        let p = resolve_stage_slice_path(
            out_dir,
            stem,
            i,
            total_stages,
            slice.layer_start,
            slice.layer_end,
        );
        if !p.exists() {
            all_cached = false;
        } else if let Ok(loader) = crate::GgufSliceLoader::open(&p) {
            if loader.total_layers != slice.layer_count() {
                all_cached = false;
            }
        } else {
            all_cached = false;
        }
        stage_paths.push(p);
    }

    if all_cached {
        if let Ok(src_meta) = std::fs::metadata(src_path) {
            if let Ok(src_time) = src_meta.modified() {
                let mut valid_timestamps = true;
                for p in &stage_paths {
                    if let Ok(meta) = std::fs::metadata(p) {
                        if let Ok(mtime) = meta.modified() {
                            if mtime < src_time {
                                valid_timestamps = false;
                                break;
                            }
                        } else {
                            valid_timestamps = false;
                            break;
                        }
                    } else {
                        valid_timestamps = false;
                        break;
                    }
                }
                if valid_timestamps {
                    info!("⚡ Reusing all existing partitioned sliced GGUF files for {}-stage pipeline", total_stages);
                    return Ok(stage_paths);
                }
            }
        }
    }

    for (i, slice) in stage_slices.iter().enumerate() {
        slice_gguf_stage(src_path, out_dir, slice, i, total_stages)?;
    }

    Ok(stage_paths)
}

/// Slices a monolithic GGUF model file into two standalone partitioned GGUF models:
/// - Stage 1: Layers 0..split_at with Identity RMSNorm (output_norm = 1.0)
/// - Stage 2: Layers split_at..total_layers re-indexed to blk.0..blk.(total-split-1) with true LM Head
pub fn slice_gguf_for_pipeline<P: AsRef<Path>, Q: AsRef<Path>>(
    source_model: P,
    output_dir: Q,
    split_at: usize,
) -> Result<(PathBuf, PathBuf)> {
    let src_path = source_model.as_ref();
    let file = File::open(src_path)
        .with_context(|| format!("Failed to open source GGUF at {:?}", src_path))?;
    let mmap = unsafe { Mmap::map(&file)? };
    let mut cursor = Cursor::new(&mmap[..]);

    let mut magic = [0u8; 4];
    cursor.read_exact(&mut magic)?;
    ensure!(magic == GGUF_MAGIC, "Invalid GGUF magic bytes");

    let version = cursor.read_u32::<LittleEndian>()?;
    ensure!(
        version == 2 || version == 3,
        "Unsupported GGUF version: {}",
        version
    );

    let _tensor_count = cursor.read_u64::<LittleEndian>()?;
    let kv_count = cursor.read_u64::<LittleEndian>()? as usize;

    let mut total_layers = 0usize;
    for _ in 0..kv_count {
        let key = read_gguf_string(&mut cursor)?;
        let val_type = cursor.read_u32::<LittleEndian>()?;
        let start_pos = cursor.position();
        skip_gguf_val(&mut cursor, val_type)?;
        let end_pos = cursor.position();
        let val_bytes = &mmap[start_pos as usize..end_pos as usize];

        if key.ends_with(".block_count") {
            if val_type == 4 && val_bytes.len() >= 4 {
                total_layers = (&val_bytes[..4]).read_u32::<LittleEndian>()? as usize;
            } else if val_type == 10 && val_bytes.len() >= 8 {
                total_layers = (&val_bytes[..8]).read_u64::<LittleEndian>()? as usize;
            }
        }
    }

    ensure!(
        total_layers > 0,
        "Could not determine model block_count from GGUF metadata"
    );
    ensure!(
        split_at > 0 && split_at < total_layers,
        "split_at ({}) must be > 0 and < total_layers ({})",
        split_at,
        total_layers
    );

    let s1 = LayerSliceConfig::new(0, split_at - 1, total_layers)?;
    let s2 = LayerSliceConfig::new(split_at, total_layers - 1, total_layers)?;

    let paths = slice_gguf_for_multi_stage(source_model, output_dir, &[s1, s2])?;
    Ok((paths[0].clone(), paths[1].clone()))
}

#[derive(Clone)]
struct RawKvEntry {
    key: String,
    val_type: u32,
    val_bytes: Vec<u8>,
}

struct SourceTensor {
    name: String,
    dimensions: Vec<u64>,
    ggml_type: u32,
    offset: u64,
    size_bytes: u64,
    layer_index: Option<usize>,
}

struct SlicedTensor {
    name: String,
    dimensions: Vec<u64>,
    ggml_type: u32,
    src_offset: u64,
    size_bytes: u64,
    is_identity_norm: bool,
}

#[allow(clippy::too_many_arguments)]
fn write_stage_gguf(
    dest_path: &Path,
    version: u32,
    alignment: u64,
    src_data_offset: u64,
    raw_kvs: &[RawKvEntry],
    tensors: &[SourceTensor],
    src_mmap: &Mmap,
    layer_start: usize,
    layer_end: usize,
    stage_index: usize,
    total_stages: usize,
) -> Result<()> {
    let block_count = layer_end - layer_start + 1;
    let is_final_stage = stage_index == total_stages - 1;
    let mut selected_tensors: Vec<SlicedTensor> = Vec::new();

    for t in tensors {
        if let Some(idx) = t.layer_index {
            if idx >= layer_start && idx <= layer_end {
                let new_idx = idx - layer_start;
                let new_name = if layer_start > 0 {
                    reindex_tensor_name(&t.name, idx, new_idx)
                } else {
                    t.name.clone()
                };
                selected_tensors.push(SlicedTensor {
                    name: new_name,
                    dimensions: t.dimensions.clone(),
                    ggml_type: t.ggml_type,
                    src_offset: t.offset,
                    size_bytes: t.size_bytes,
                    is_identity_norm: false,
                });
            }
        } else if t.name.contains("output_norm") || t.name.contains("norm.weight") {
            // Identity RMSNorm: weights filled with 1.0 for all stages EXCEPT the final stage!
            selected_tensors.push(SlicedTensor {
                name: t.name.clone(),
                dimensions: t.dimensions.clone(),
                ggml_type: t.ggml_type,
                src_offset: t.offset,
                size_bytes: t.size_bytes,
                is_identity_norm: !is_final_stage,
            });
        } else if !is_final_stage && (t.name == "output.weight" || t.name == "output.bias") {
            continue;
        } else {
            // Global tensors (token_embd, output, etc.)
            selected_tensors.push(SlicedTensor {
                name: t.name.clone(),
                dimensions: t.dimensions.clone(),
                ggml_type: t.ggml_type,
                src_offset: t.offset,
                size_bytes: t.size_bytes,
                is_identity_norm: false,
            });
        }
    }

    let file = File::create(dest_path)
        .with_context(|| format!("Failed to create destination slice GGUF at {:?}", dest_path))?;
    let mut writer = BufWriter::with_capacity(16 * 1024 * 1024, file);

    // 1. Header
    writer.write_all(&GGUF_MAGIC)?;
    writer.write_u32::<LittleEndian>(version)?;
    writer.write_u64::<LittleEndian>(selected_tensors.len() as u64)?;
    writer.write_u64::<LittleEndian>(raw_kvs.len() as u64)?;

    // 2. Metadata KV Table (with block_count overridden)
    for kv in raw_kvs {
        write_gguf_string(&mut writer, &kv.key)?;
        writer.write_u32::<LittleEndian>(kv.val_type)?;

        if kv.key.ends_with(".block_count") {
            if kv.val_type == 4 {
                writer.write_u32::<LittleEndian>(block_count as u32)?;
            } else if kv.val_type == 10 {
                writer.write_u64::<LittleEndian>(block_count as u64)?;
            } else {
                writer.write_all(&kv.val_bytes)?;
            }
        } else {
            writer.write_all(&kv.val_bytes)?;
        }
    }

    // 3. Compute aligned offsets for new tensor table (relative to the start of the data section)
    let mut current_offset = 0u64;
    let mut aligned_offsets = Vec::with_capacity(selected_tensors.len());

    for t in &selected_tensors {
        let pad = (alignment - (current_offset % alignment)) % alignment;
        current_offset += pad;
        aligned_offsets.push(current_offset);
        current_offset += t.size_bytes;
    }

    // Write Tensor Info Table with exact computed offsets
    for (i, t) in selected_tensors.iter().enumerate() {
        write_gguf_string(&mut writer, &t.name)?;
        writer.write_u32::<LittleEndian>(t.dimensions.len() as u32)?;
        for &d in &t.dimensions {
            writer.write_u64::<LittleEndian>(d)?;
        }
        writer.write_u32::<LittleEndian>(t.ggml_type)?;
        writer.write_u64::<LittleEndian>(aligned_offsets[i])?;
    }

    // Pad before data section to alignment boundary
    const ZERO_BUF: [u8; 4096] = [0u8; 4096];
    let curr_pos = writer.stream_position()?;
    let data_pad = ((alignment - (curr_pos % alignment)) % alignment) as usize;
    let mut remaining_pad = data_pad;
    while remaining_pad > 0 {
        let chunk = remaining_pad.min(ZERO_BUF.len());
        writer.write_all(&ZERO_BUF[..chunk])?;
        remaining_pad -= chunk;
    }

    let data_section_start = writer.stream_position()?;

    // 4. Write Tensor Binary Payloads
    for (i, t) in selected_tensors.iter().enumerate() {
        let target_pos = data_section_start + aligned_offsets[i];
        let current_pos = writer.stream_position()?;
        if target_pos > current_pos {
            let padding_needed = (target_pos - current_pos) as usize;
            ensure!(
                padding_needed <= (alignment as usize),
                "Tensor alignment padding {} exceeds alignment {}",
                padding_needed,
                alignment
            );
            let mut remaining = padding_needed;
            while remaining > 0 {
                let chunk = remaining.min(ZERO_BUF.len());
                writer.write_all(&ZERO_BUF[..chunk])?;
                remaining -= chunk;
            }
        }

        if t.is_identity_norm {
            // Identity RMSNorm: fill with 1.0
            if t.ggml_type == 0 {
                // F32: 1.0f32
                let count = t.size_bytes as usize / 4;
                let one_bytes = 1.0f32.to_le_bytes();
                for _ in 0..count {
                    writer.write_all(&one_bytes)?;
                }
            } else if t.ggml_type == 1 {
                // F16: 0x3C00 (1.0 in IEEE 754 half-precision)
                let count = t.size_bytes as usize / 2;
                let one_f16 = 0x3C00u16.to_le_bytes();
                for _ in 0..count {
                    writer.write_all(&one_f16)?;
                }
            } else if t.ggml_type == 30 {
                // BF16: 0x3F80 (1.0 in bfloat16)
                let count = t.size_bytes as usize / 2;
                let one_bf16 = 0x3F80u16.to_le_bytes();
                for _ in 0..count {
                    writer.write_all(&one_bf16)?;
                }
            } else if t.ggml_type == 8 {
                // Q8_0: block of 32 values with f16 scale d.
                // For all 1.0, d = 1.0 (0x3C00) and each qs byte = 1 (1 * 1.0 = 1.0).
                // block size is 34 bytes (2 bytes f16 scale + 32 bytes i8).
                let num_blocks = t.size_bytes as usize / 34;
                for _ in 0..num_blocks {
                    writer.write_all(&0x3C00u16.to_le_bytes())?;
                    writer.write_all(&[1u8; 32])?;
                }
            } else {
                bail!(
                    "Cannot inject Identity-RMSNorm for unsupported tensor ggml_type {} on tensor '{}'. Supported types: F32 (0), F16 (1), BF16 (30), Q8_0 (8).",
                    t.ggml_type, t.name
                );
            }
        } else {
            let src_start = (src_data_offset + t.src_offset) as usize;
            let src_end = src_start + t.size_bytes as usize;
            ensure!(
                src_end <= src_mmap.len(),
                "Source tensor offset exceeds mmap length"
            );
            writer.write_all(&src_mmap[src_start..src_end])?;
        }
    }

    writer.flush()?;
    Ok(())
}

fn write_gguf_string<W: Write>(writer: &mut W, s: &str) -> Result<()> {
    writer.write_u64::<LittleEndian>(s.len() as u64)?;
    writer.write_all(s.as_bytes())?;
    Ok(())
}

fn read_gguf_string<R: Read>(reader: &mut R) -> Result<String> {
    let len = reader.read_u64::<LittleEndian>()? as usize;
    ensure!(
        len <= 64 * 1024,
        "GGUF string length ({}) exceeds 64KB safety limit",
        len
    );
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).to_string())
}

fn skip_gguf_val<R: Read + Seek>(reader: &mut R, val_type: u32) -> Result<()> {
    match val_type {
        0 | 1 | 7 => {
            reader.seek(SeekFrom::Current(1))?;
        }
        2 | 3 => {
            reader.seek(SeekFrom::Current(2))?;
        }
        4..=6 => {
            reader.seek(SeekFrom::Current(4))?;
        }
        8 => {
            let len = reader.read_u64::<LittleEndian>()?;
            ensure!(
                len <= (i64::MAX as u64),
                "GGUF string length exceeds seek bounds"
            );
            reader.seek(SeekFrom::Current(len as i64))?;
        }
        9 => {
            let elem_type = reader.read_u32::<LittleEndian>()?;
            let count = reader.read_u64::<LittleEndian>()?;
            ensure!(
                count <= 1_000_000,
                "GGUF array count ({}) exceeds maximum safe limit",
                count
            );
            for _ in 0..count {
                skip_gguf_val(reader, elem_type)?;
            }
        }
        10..=12 => {
            reader.seek(SeekFrom::Current(8))?;
        }
        _ => bail!("Unknown GGUF val_type: {}", val_type),
    }
    Ok(())
}

fn extract_layer_index(name: &str) -> Option<usize> {
    let parts: Vec<&str> = name.split('.').collect();
    for i in 0..parts.len().saturating_sub(1) {
        if parts[i] == "blk" || parts[i] == "layers" {
            if let Ok(idx) = parts[i + 1].parse::<usize>() {
                return Some(idx);
            }
        }
    }
    None
}

fn reindex_tensor_name(name: &str, old_idx: usize, new_idx: usize) -> String {
    let old_blk = format!("blk.{}.", old_idx);
    let new_blk = format!("blk.{}.", new_idx);
    if name.contains(&old_blk) {
        return name.replace(&old_blk, &new_blk);
    }

    let old_layer = format!("layers.{}.", old_idx);
    let new_layer = format!("layers.{}.", new_idx);
    if name.contains(&old_layer) {
        return name.replace(&old_layer, &new_layer);
    }

    name.to_string()
}

pub fn ggml_tensor_size_bytes(ggml_type: u32, element_count: u64) -> u64 {
    let (blck_size, type_size): (u64, u64) = match ggml_type {
        0 => (1, 4),      // F32
        1 => (1, 2),      // F16
        2 => (32, 18),    // Q4_0
        3 => (32, 20),    // Q4_1
        6 => (32, 22),    // Q5_0
        7 => (32, 24),    // Q5_1
        8 => (32, 34),    // Q8_0
        9 => (32, 36),    // Q8_1
        10 => (256, 84),  // Q2_K
        11 => (256, 110), // Q3_K
        12 => (256, 144), // Q4_K
        13 => (256, 176), // Q5_K
        14 => (256, 210), // Q6_K
        15 => (256, 292), // Q8_K
        16 => (256, 66),  // IQ2_XXS
        17 => (256, 74),  // IQ2_XS
        18 => (256, 98),  // IQ3_XXS
        19 => (256, 48),  // IQ1_S
        20 => (32, 18),   // IQ4_NL
        21 => (256, 110), // IQ3_S
        22 => (256, 82),  // IQ2_S
        23 => (256, 136), // IQ4_XS
        24 => (1, 1),     // I8
        25 => (1, 2),     // I16
        26 => (1, 4),     // I32
        27 => (1, 8),     // I64
        28 => (1, 8),     // F64
        29 => (256, 70),  // IQ1_M
        30 => (1, 2),     // BF16
        39 => (32, 18),   // MXFP4
        40 => (32, 18),   // NVFP4
        41 => (32, 6),    // Q1_0
        42 => (32, 10),   // Q2_0
        _ => (1, 2),
    };

    if blck_size == 1 {
        element_count * type_size
    } else {
        element_count.div_ceil(blck_size) * type_size
    }
}


pub struct InPlaceStageGuard {
    pub effective_path: PathBuf,
    ephemeral_path: Option<PathBuf>,
}

impl InPlaceStageGuard {
    pub fn try_new(model_path: &Path, slice: &LayerSliceConfig) -> Result<Self> {
        let loader = crate::slice_loader::GgufSliceLoader::open(model_path)?;
        if slice.layer_count() >= loader.total_layers || loader.total_layers == 0 {
            return Ok(Self {
                effective_path: model_path.to_path_buf(),
                ephemeral_path: None,
            });
        }
        drop(loader);

        let parent = model_path.parent().unwrap_or_else(|| Path::new("models"));
        let stage_num = if slice.is_last_stage { 2 } else { 1 };
        let tmp_path = parent.join(format!(
            ".aeromesh_ephemeral_s{}_{}_stage{}.gguf",
            slice.layer_start, slice.layer_end, stage_num
        ));

        let file = File::open(model_path)
            .with_context(|| format!("Failed to open source GGUF at {:?}", model_path))?;
        let mmap = unsafe { Mmap::map(&file)? };
        let mut cursor = Cursor::new(&mmap[..]);

        let mut magic = [0u8; 4];
        cursor.read_exact(&mut magic)?;
        ensure!(magic == GGUF_MAGIC, "Invalid GGUF magic bytes");

        let version = cursor.read_u32::<LittleEndian>()?;
        let tensor_count = cursor.read_u64::<LittleEndian>()? as usize;
        let kv_count = cursor.read_u64::<LittleEndian>()? as usize;

        let mut alignment = DEFAULT_ALIGNMENT;
        let mut raw_kvs = Vec::with_capacity(kv_count);
        for _ in 0..kv_count {
            let key = read_gguf_string(&mut cursor)?;
            let val_type = cursor.read_u32::<LittleEndian>()?;
            let start_pos = cursor.position();
            skip_gguf_val(&mut cursor, val_type)?;
            let end_pos = cursor.position();
            let val_bytes = mmap[start_pos as usize..end_pos as usize].to_vec();
            if key == "general.alignment" {
                if val_type == 4 && val_bytes.len() >= 4 {
                    alignment = (&val_bytes[..4]).read_u32::<LittleEndian>()? as u64;
                } else if val_type == 10 && val_bytes.len() >= 8 {
                    alignment = (&val_bytes[..8]).read_u64::<LittleEndian>()?;
                }
            }
            raw_kvs.push(RawKvEntry { key, val_type, val_bytes });
        }

        let mut tensors = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
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
            let size_bytes = ggml_tensor_size_bytes(ggml_type, element_count);
            let layer_index = extract_layer_index(&name);
            tensors.push(SourceTensor {
                name,
                dimensions,
                ggml_type,
                offset,
                size_bytes,
                layer_index,
            });
        }

        let header_end = cursor.position();
        let pad = (alignment - (header_end % alignment)) % alignment;
        let data_section_offset = header_end + pad;

        let stage_idx = if slice.is_last_stage { 1 } else { 0 };
        write_stage_gguf(
            &tmp_path,
            version,
            alignment,
            data_section_offset,
            &raw_kvs,
            &tensors,
            &mmap,
            slice.layer_start,
            slice.layer_end,
            stage_idx,
            2,
        )?;

        Ok(Self {
            effective_path: tmp_path.clone(),
            ephemeral_path: Some(tmp_path),
        })
    }
}

impl Drop for InPlaceStageGuard {
    fn drop(&mut self) {
        if let Some(ref p) = self.ephemeral_path {
            if std::fs::remove_file(p).is_ok() {
                info!(
                    ephemeral_file = %p.display(),
                    "Auto-deleted ephemeral stage GGUF after VRAM upload (0.0 MB disk overhead)"
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_synthetic_gguf(path: &Path) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Header: Magic, version 3, tensor_count 3, kv_count 3
        writer.write_all(&GGUF_MAGIC)?;
        writer.write_u32::<LittleEndian>(3)?;
        writer.write_u64::<LittleEndian>(3)?; // 3 tensors: blk.0, blk.1, output_norm
        writer.write_u64::<LittleEndian>(3)?; // 3 KVs: arch, alignment, block_count

        // KV 1: general.architecture (type 8 = string, "llama")
        write_gguf_string(&mut writer, "general.architecture")?;
        writer.write_u32::<LittleEndian>(8)?;
        write_gguf_string(&mut writer, "llama")?;

        // KV 2: general.alignment (type 4 = uint32, 32)
        write_gguf_string(&mut writer, "general.alignment")?;
        writer.write_u32::<LittleEndian>(4)?;
        writer.write_u32::<LittleEndian>(32)?;

        // KV 3: llama.block_count (type 4 = uint32, 2)
        write_gguf_string(&mut writer, "llama.block_count")?;
        writer.write_u32::<LittleEndian>(4)?;
        writer.write_u32::<LittleEndian>(2)?;

        // Tensor 1: "blk.0.attn_q.weight", dims [4, 4], type 0 (F32), offset 0
        write_gguf_string(&mut writer, "blk.0.attn_q.weight")?;
        writer.write_u32::<LittleEndian>(2)?;
        writer.write_u64::<LittleEndian>(4)?;
        writer.write_u64::<LittleEndian>(4)?;
        writer.write_u32::<LittleEndian>(0)?;
        writer.write_u64::<LittleEndian>(0)?;

        // Tensor 2: "blk.1.attn_q.weight", dims [4, 4], type 0 (F32), offset 64
        write_gguf_string(&mut writer, "blk.1.attn_q.weight")?;
        writer.write_u32::<LittleEndian>(2)?;
        writer.write_u64::<LittleEndian>(4)?;
        writer.write_u64::<LittleEndian>(4)?;
        writer.write_u32::<LittleEndian>(0)?;
        writer.write_u64::<LittleEndian>(64)?;

        // Tensor 3: "output_norm.weight", dims [4], type 0 (F32), offset 128
        write_gguf_string(&mut writer, "output_norm.weight")?;
        writer.write_u32::<LittleEndian>(1)?;
        writer.write_u64::<LittleEndian>(4)?;
        writer.write_u32::<LittleEndian>(0)?;
        writer.write_u64::<LittleEndian>(128)?;

        // Alignment pad to 32 bytes before data section
        let pos = writer.stream_position()?;
        let pad = ((32 - (pos % 32)) % 32) as usize;
        for _ in 0..pad {
            writer.write_u8(0)?;
        }

        // Data section:
        // Tensor 1 data (64 bytes)
        let t1_data = vec![0.5f32; 16];
        for val in t1_data {
            writer.write_all(&val.to_le_bytes())?;
        }

        // Tensor 2 data (64 bytes)
        let t2_data = vec![1.5f32; 16];
        for val in t2_data {
            writer.write_all(&val.to_le_bytes())?;
        }

        // Tensor 3 data (16 bytes)
        let t3_data = vec![1.0f32; 4];
        for val in t3_data {
            writer.write_all(&val.to_le_bytes())?;
        }

        writer.flush()?;
        Ok(())
    }

    #[test]
    fn test_slice_gguf_no_underflow_panic() {
        let temp_dir = std::env::temp_dir().join(format!(
            "aeromesh_test_slicer_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&temp_dir).unwrap();

        let src_model = temp_dir.join("test_model.gguf");
        create_synthetic_gguf(&src_model).expect("Failed to create synthetic GGUF");

        let out_dir = temp_dir.join("slices");
        let result = slice_gguf_for_pipeline(&src_model, &out_dir, 1);
        assert!(
            result.is_ok(),
            "slice_gguf_for_pipeline failed: {:?}",
            result.err()
        );

        let (s1, s2) = result.unwrap();
        assert!(s1.exists(), "Stage 1 slice file must exist");
        assert!(s2.exists(), "Stage 2 slice file must exist");

        let s1_meta = std::fs::metadata(&s1).unwrap();
        let s2_meta = std::fs::metadata(&s2).unwrap();
        assert!(s1_meta.len() > 0, "Stage 1 file size must be non-zero");
        assert!(s2_meta.len() > 0, "Stage 2 file size must be non-zero");

        // Verify that Stage 1 and Stage 2 have valid GGUF headers and can be opened
        let s1_file = File::open(&s1).unwrap();
        let s1_mmap = unsafe { memmap2::Mmap::map(&s1_file).unwrap() };
        assert_eq!(&s1_mmap[0..4], &GGUF_MAGIC);

        let s2_file = File::open(&s2).unwrap();
        let s2_mmap = unsafe { memmap2::Mmap::map(&s2_file).unwrap() };
        assert_eq!(&s2_mmap[0..4], &GGUF_MAGIC);

        // Clean up
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    fn create_synthetic_3layer_gguf(path: &Path) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Header: Magic, version 3, tensor_count 4, kv_count 3
        writer.write_all(&GGUF_MAGIC)?;
        writer.write_u32::<LittleEndian>(3)?;
        writer.write_u64::<LittleEndian>(4)?; // 4 tensors: blk.0, blk.1, blk.2, output_norm
        writer.write_u64::<LittleEndian>(3)?; // 3 KVs: arch, alignment, block_count

        write_gguf_string(&mut writer, "general.architecture")?;
        writer.write_u32::<LittleEndian>(8)?;
        write_gguf_string(&mut writer, "llama")?;

        write_gguf_string(&mut writer, "general.alignment")?;
        writer.write_u32::<LittleEndian>(4)?;
        writer.write_u32::<LittleEndian>(32)?;

        write_gguf_string(&mut writer, "llama.block_count")?;
        writer.write_u32::<LittleEndian>(4)?;
        writer.write_u32::<LittleEndian>(3)?;

        // Tensor 0: blk.0.attn_q.weight
        write_gguf_string(&mut writer, "blk.0.attn_q.weight")?;
        writer.write_u32::<LittleEndian>(2)?;
        writer.write_u64::<LittleEndian>(4)?;
        writer.write_u64::<LittleEndian>(4)?;
        writer.write_u32::<LittleEndian>(0)?;
        writer.write_u64::<LittleEndian>(0)?;

        // Tensor 1: blk.1.attn_q.weight
        write_gguf_string(&mut writer, "blk.1.attn_q.weight")?;
        writer.write_u32::<LittleEndian>(2)?;
        writer.write_u64::<LittleEndian>(4)?;
        writer.write_u64::<LittleEndian>(4)?;
        writer.write_u32::<LittleEndian>(0)?;
        writer.write_u64::<LittleEndian>(64)?;

        // Tensor 2: blk.2.attn_q.weight
        write_gguf_string(&mut writer, "blk.2.attn_q.weight")?;
        writer.write_u32::<LittleEndian>(2)?;
        writer.write_u64::<LittleEndian>(4)?;
        writer.write_u64::<LittleEndian>(4)?;
        writer.write_u32::<LittleEndian>(0)?;
        writer.write_u64::<LittleEndian>(128)?;

        // Tensor 3: output_norm.weight
        write_gguf_string(&mut writer, "output_norm.weight")?;
        writer.write_u32::<LittleEndian>(1)?;
        writer.write_u64::<LittleEndian>(4)?;
        writer.write_u32::<LittleEndian>(0)?;
        writer.write_u64::<LittleEndian>(192)?;

        let pos = writer.stream_position()?;
        let pad = ((32 - (pos % 32)) % 32) as usize;
        for _ in 0..pad {
            writer.write_u8(0)?;
        }

        // 16 floats for blk.0
        for _ in 0..16 {
            writer.write_all(&0.5f32.to_le_bytes())?;
        }
        // 16 floats for blk.1
        for _ in 0..16 {
            writer.write_all(&1.5f32.to_le_bytes())?;
        }
        // 16 floats for blk.2
        for _ in 0..16 {
            writer.write_all(&2.5f32.to_le_bytes())?;
        }
        // 4 floats for output_norm (original weight = 2.75)
        for _ in 0..4 {
            writer.write_all(&2.75f32.to_le_bytes())?;
        }

        writer.flush()?;
        Ok(())
    }

    #[test]
    fn test_multi_stage_slicing_3_nodes() {
        let temp_dir = std::env::temp_dir().join(format!(
            "aeromesh_test_slicer_3stage_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&temp_dir).unwrap();

        let src_model = temp_dir.join("test_model_3l.gguf");
        create_synthetic_3layer_gguf(&src_model).expect("Failed to create synthetic 3-layer GGUF");

        let out_dir = temp_dir.join("slices_3stage");
        let s0 = LayerSliceConfig::new(0, 0, 3).unwrap();
        let s1 = LayerSliceConfig::new(1, 1, 3).unwrap();
        let s2 = LayerSliceConfig::new(2, 2, 3).unwrap();

        let stages = slice_gguf_for_multi_stage(&src_model, &out_dir, &[s0, s1, s2])
            .expect("Multi-stage slicing failed");
        assert_eq!(stages.len(), 3);

        for p in &stages {
            assert!(p.exists());
            let meta = std::fs::metadata(p).unwrap();
            assert!(meta.len() > 0);
            let f = File::open(p).unwrap();
            let m = unsafe { memmap2::Mmap::map(&f).unwrap() };
            assert_eq!(&m[0..4], &GGUF_MAGIC);
        }

        assert!(stages[0].to_str().unwrap().contains("stage1_of_3.gguf"));
        assert!(stages[1].to_str().unwrap().contains("stage2_of_3.gguf"));
        assert!(stages[2].to_str().unwrap().contains("stage3_of_3.gguf"));

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    // -----------------------------------------------------------------------
    // Semantic assertions on the sliced output.
    //
    // The tests above only check that files exist and start with the GGUF
    // magic. These verify the mechanism the whole distributed design rests on:
    // intermediate stages must apply an Identity RMSNorm, the terminal stage
    // must keep the genuine norm, and block counts must be rewritten.
    // -----------------------------------------------------------------------

    fn unique_temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "aeromesh_slicer_{tag}_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Reads a F32 tensor's raw bytes out of a sliced GGUF.
    fn read_tensor_f32(path: &Path, name: &str) -> Vec<f32> {
        let loader = crate::slice_loader::GgufSliceLoader::open(path)
            .unwrap_or_else(|e| panic!("failed to open {}: {e}", path.display()));
        let bytes = loader
            .get_tensor_bytes(name)
            .unwrap_or_else(|| panic!("tensor {name} missing from {}", path.display()));
        assert_eq!(
            bytes.len() % 4,
            0,
            "F32 tensor should be a multiple of 4 bytes"
        );
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn read_block_count(path: &Path) -> u32 {
        // Parse the KV table directly: GgufSliceLoader does not expose the value,
        // and the rewritten block count is exactly what stage 2 depends on.
        let file = File::open(path).unwrap();
        let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };
        let mut cursor = std::io::Cursor::new(&mmap[..]);
        let mut magic = [0u8; 4];
        std::io::Read::read_exact(&mut cursor, &mut magic).unwrap();
        assert_eq!(&magic, &GGUF_MAGIC);
        let _version = cursor.read_u32::<LittleEndian>().unwrap();
        let _tensor_count = cursor.read_u64::<LittleEndian>().unwrap();
        let kv_count = cursor.read_u64::<LittleEndian>().unwrap();

        fn read_string(cursor: &mut std::io::Cursor<&[u8]>) -> String {
            let len = cursor.read_u64::<LittleEndian>().unwrap() as usize;
            let mut buf = vec![0u8; len];
            std::io::Read::read_exact(cursor, &mut buf).unwrap();
            String::from_utf8(buf).unwrap()
        }

        for _ in 0..kv_count {
            let key = read_string(&mut cursor);
            let value_type = cursor.read_u32::<LittleEndian>().unwrap();
            match value_type {
                // UINT32
                4 => {
                    let value = cursor.read_u32::<LittleEndian>().unwrap();
                    if key.ends_with(".block_count") {
                        return value;
                    }
                }
                // UINT64
                10 => {
                    let value = cursor.read_u64::<LittleEndian>().unwrap();
                    if key.ends_with(".block_count") {
                        return value as u32;
                    }
                }
                // STRING
                8 => {
                    let _ = read_string(&mut cursor);
                }
                // FLOAT32
                6 => {
                    let _ = cursor.read_u32::<LittleEndian>().unwrap();
                }
                other => panic!("unexpected GGUF value type {other} for key {key}"),
            }
        }
        panic!("no .block_count key found in {}", path.display());
    }

    #[test]
    fn stage1_replaces_output_norm_with_identity() {
        let dir = unique_temp_dir("identity");
        let src = dir.join("model.gguf");
        create_synthetic_3layer_gguf(&src).unwrap();

        // Stage 1 of 3 covers layer 0 only, so it is an intermediate stage and
        // must neutralise the final norm.
        let stages = slice_gguf_for_multi_stage(
            &src,
            dir.join("out"),
            &[
                LayerSliceConfig::new(0, 0, 3).unwrap(),
                LayerSliceConfig::new(1, 1, 3).unwrap(),
                LayerSliceConfig::new(2, 2, 3).unwrap(),
            ],
        )
        .expect("slicing failed");

        let stage1_norm = read_tensor_f32(&stages[0], "output_norm.weight");
        assert_eq!(stage1_norm.len(), 4, "synthetic norm has 4 elements");
        assert!(
            stage1_norm.iter().all(|v| (*v - 1.0).abs() < f32::EPSILON),
            "stage 1 output_norm must be all 1.0 (identity), got {stage1_norm:?}"
        );

        // Stage 3 is terminal: it must keep the model's real weights (3.14).
        let stage3_norm = read_tensor_f32(&stages[2], "output_norm.weight");
        assert!(
            stage3_norm.iter().all(|v| (*v - 2.75).abs() < 1e-5),
            "terminal stage must preserve the genuine output_norm, got {stage3_norm:?}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn two_stage_split_identity_and_genuine_norm() {
        let dir = unique_temp_dir("identity2");
        let src = dir.join("model.gguf");
        create_synthetic_3layer_gguf(&src).unwrap();

        let (s1, s2) = slice_gguf_for_pipeline(&src, dir.join("out"), 1).expect("slicing failed");

        let s1_norm = read_tensor_f32(&s1, "output_norm.weight");
        assert!(
            s1_norm.iter().all(|v| (*v - 1.0).abs() < f32::EPSILON),
            "stage 1 of a 2-way split must use an identity norm, got {s1_norm:?}"
        );

        let s2_norm = read_tensor_f32(&s2, "output_norm.weight");
        assert!(
            s2_norm.iter().all(|v| (*v - 2.75).abs() < 1e-5),
            "stage 2 of a 2-way split is terminal and must keep the real norm, got {s2_norm:?}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn block_count_is_rewritten_per_stage() {
        let dir = unique_temp_dir("blockcount");
        let src = dir.join("model.gguf");
        create_synthetic_3layer_gguf(&src).unwrap();

        assert_eq!(read_block_count(&src), 3, "source declares 3 layers");

        let stages = slice_gguf_for_multi_stage(
            &src,
            dir.join("out"),
            &[
                LayerSliceConfig::new(0, 0, 3).unwrap(),
                LayerSliceConfig::new(1, 1, 3).unwrap(),
                LayerSliceConfig::new(2, 2, 3).unwrap(),
            ],
        )
        .expect("slicing failed");

        // Each stage holds exactly one layer, so llama.cpp must be told 1.
        for (idx, path) in stages.iter().enumerate() {
            assert_eq!(
                read_block_count(path),
                1,
                "stage {} should declare 1 block, not the source total",
                idx + 1
            );
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn layer_tensors_are_reindexed_from_zero() {
        let dir = unique_temp_dir("reindex");
        let src = dir.join("model.gguf");
        create_synthetic_3layer_gguf(&src).unwrap();

        // Stage 2 covers layer 1, which must be re-indexed to blk.0.
        let stages = slice_gguf_for_multi_stage(
            &src,
            dir.join("out"),
            &[
                LayerSliceConfig::new(0, 0, 3).unwrap(),
                LayerSliceConfig::new(1, 2, 3).unwrap(),
            ],
        )
        .expect("slicing failed");

        let loader = crate::slice_loader::GgufSliceLoader::open(&stages[1]).unwrap();
        let names: Vec<String> = loader.all_tensors.iter().map(|t| t.name.clone()).collect();

        assert!(
            names.iter().any(|n| n == "blk.0.attn_q.weight"),
            "source layer 1 must become blk.0 in stage 2; got {names:?}"
        );
        assert!(
            names.iter().any(|n| n == "blk.1.attn_q.weight"),
            "source layer 2 must become blk.1 in stage 2; got {names:?}"
        );
        assert!(
            !names.iter().any(|n| n.starts_with("blk.2.")),
            "no tensor should keep a global index; got {names:?}"
        );

        // The re-indexed block must appear exactly once, with its own data.
        let data = read_tensor_f32(&stages[1], "blk.0.attn_q.weight");
        assert!(
            data.iter().all(|v| (*v - 1.5).abs() < 1e-6),
            "blk.0 of stage 2 should carry source layer 1's data (1.5), got {data:?}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn slicing_is_idempotent_and_reuses_output() {
        let dir = unique_temp_dir("reuse");
        let src = dir.join("model.gguf");
        create_synthetic_3layer_gguf(&src).unwrap();
        let out = dir.join("out");

        let first = slice_gguf_for_pipeline(&src, &out, 1).expect("first slice failed");
        let first_len = std::fs::metadata(&first.0).unwrap().len();
        let first_modified = std::fs::metadata(&first.0).unwrap().modified().unwrap();

        // A second identical request must reuse the existing artefacts.
        let second = slice_gguf_for_pipeline(&src, &out, 1).expect("second slice failed");
        assert_eq!(first.0, second.0);
        assert_eq!(first_len, std::fs::metadata(&second.0).unwrap().len());
        assert_eq!(
            first_modified,
            std::fs::metadata(&second.0).unwrap().modified().unwrap(),
            "the second call should not rewrite the file"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn different_split_points_do_not_collide() {
        // Regression test: with split-less names and a layer-count-only reuse
        // check, slicing at 1 and then at 2 silently returned the first
        // partition while reporting success.
        let dir = unique_temp_dir("splitcollision");
        let src = dir.join("model.gguf");
        create_synthetic_3layer_gguf(&src).unwrap();
        let out = dir.join("out");

        let (s1_at_1, s2_at_1) = slice_gguf_for_pipeline(&src, &out, 1).expect("split at 1");
        let (s1_at_2, s2_at_2) = slice_gguf_for_pipeline(&src, &out, 2).expect("split at 2");

        assert_ne!(
            s1_at_1, s1_at_2,
            "a different split point must produce a different stage-1 file"
        );
        assert_ne!(s2_at_1, s2_at_2);

        // Split at 1: stage 1 = 1 layer, stage 2 = 2 layers.
        let l1 = crate::slice_loader::GgufSliceLoader::open(&s1_at_1).unwrap();
        let r1 = crate::slice_loader::GgufSliceLoader::open(&s2_at_1).unwrap();
        assert_eq!(l1.total_layers, 1);
        assert_eq!(r1.total_layers, 2);

        // Split at 2: stage 1 = 2 layers, stage 2 = 1 layer.
        let l2 = crate::slice_loader::GgufSliceLoader::open(&s1_at_2).unwrap();
        let r2 = crate::slice_loader::GgufSliceLoader::open(&s2_at_2).unwrap();
        assert_eq!(l2.total_layers, 2);
        assert_eq!(r2.total_layers, 1);

        // And the encoders agree with the requests.
        assert_eq!(read_block_count(&s1_at_1), 1);
        assert_eq!(read_block_count(&s1_at_2), 2);
        assert_eq!(read_block_count(&s2_at_1), 2);
        assert_eq!(read_block_count(&s2_at_2), 1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn stage_slice_name_predicates() {
        // Current naming
        assert!(is_stage_slice_name("DS_layers0-23_stage1.gguf"));
        assert!(is_stage_slice_name("DS_layers24-47_stage2.gguf"));
        assert!(is_stage1_slice_name("DS_layers0-23_stage1.gguf"));
        assert!(!is_stage1_slice_name("DS_layers24-47_stage2.gguf"));

        // N-stage naming
        assert!(is_stage_slice_name("DS_stage1_of_3.gguf"));
        assert!(is_stage_slice_name("DS_stage3_of_3.gguf"));
        assert!(is_stage1_slice_name("DS_stage1_of_3.gguf"));
        assert!(!is_stage1_slice_name("DS_stage2_of_3.gguf"));

        // Legacy naming
        assert!(is_stage_slice_name("DS_stage1.gguf"));
        assert!(is_stage_slice_name("DS_stage2.gguf"));
        assert!(is_stage1_slice_name("DS_stage1.gguf"));
        assert!(!is_stage1_slice_name("DS_stage2.gguf"));

        // Ordinary models must never be mistaken for partitions
        assert!(!is_stage_slice_name("DS.gguf"));
        assert!(!is_stage_slice_name("qwen2.5-0.5b-instruct.gguf"));
        assert!(!is_stage_slice_name("stagehand-model.gguf"));
        assert!(!is_stage1_slice_name("stagehand-model.gguf"));
        assert!(!is_stage_slice_name("deepstage_2.gguf"));
    }

    #[test]
    fn split_parsing_from_name() {
        assert_eq!(split_from_stage_name("DS_split24_stage1.gguf"), Some(24));
        assert_eq!(split_from_stage_name("DS_split1_stage2.gguf"), Some(1));
        assert_eq!(split_from_stage_name("DS_stage1.gguf"), None);
        assert_eq!(split_from_stage_name("DS.gguf"), None);
    }
}
