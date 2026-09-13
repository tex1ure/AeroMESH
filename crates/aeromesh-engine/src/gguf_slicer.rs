use std::fs::File;
use std::io::{BufWriter, Cursor, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use anyhow::{bail, ensure, Context, Result};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use memmap2::Mmap;
use tracing::info;

pub const GGUF_MAGIC: [u8; 4] = *b"GGUF";
pub const DEFAULT_ALIGNMENT: u64 = 32;

/// Slices a monolithic GGUF model file into two standalone partitioned GGUF models:
/// - Stage 1: Layers 0..split_at with Identity RMSNorm (output_norm = 1.0)
/// - Stage 2: Layers split_at..total_layers re-indexed to blk.0..blk.(total-split-1) with true LM Head
pub fn slice_gguf_for_pipeline<P: AsRef<Path>, Q: AsRef<Path>>(
    source_model: P,
    output_dir: Q,
    split_at: usize,
) -> Result<(PathBuf, PathBuf)> {
    let src_path = source_model.as_ref();
    let out_dir = output_dir.as_ref();
    std::fs::create_dir_all(out_dir)?;

    let stem = src_path.file_stem().and_then(|s| s.to_str()).unwrap_or("model");
    let stage1_path = out_dir.join(format!("{}_stage1.gguf", stem));
    let stage2_path = out_dir.join(format!("{}_stage2.gguf", stem));

    // If both slice files exist and are newer than source, reuse them
    if stage1_path.exists() && stage2_path.exists() {
        if let (Ok(src_meta), Ok(s1_meta), Ok(s2_meta)) = (
            std::fs::metadata(src_path),
            std::fs::metadata(&stage1_path),
            std::fs::metadata(&stage2_path),
        ) {
            if let (Ok(src_time), Ok(s1_time), Ok(s2_time)) = (
                src_meta.modified(),
                s1_meta.modified(),
                s2_meta.modified(),
            ) {
                if s1_time >= src_time && s2_time >= src_time && s1_meta.len() > 1000 && s2_meta.len() > 1000 {
                    info!(
                        stage1 = %stage1_path.display(),
                        stage2 = %stage2_path.display(),
                        "⚡ Reusing existing partitioned sliced GGUF files"
                    );
                    return Ok((stage1_path, stage2_path));
                }
            }
        }
    }

    info!(
        source = %src_path.display(),
        split_at = split_at,
        "🔪 Slicing monolithic GGUF into Stage 1 (0..{}) and Stage 2 ({}..end)...",
        split_at,
        split_at
    );

    let file = File::open(src_path)
        .with_context(|| format!("Failed to open source GGUF at {:?}", src_path))?;
    let mmap = unsafe { Mmap::map(&file)? };

    let mut cursor = Cursor::new(&mmap[..]);

    let mut magic = [0u8; 4];
    cursor.read_exact(&mut magic)?;
    ensure!(&magic == &GGUF_MAGIC, "Invalid GGUF magic bytes");

    let version = cursor.read_u32::<LittleEndian>()?;
    ensure!(version == 2 || version == 3, "Unsupported GGUF version: {}", version);

    let tensor_count = cursor.read_u64::<LittleEndian>()? as usize;
    let kv_count = cursor.read_u64::<LittleEndian>()? as usize;

    let mut architecture = "llama".to_string();
    let mut total_layers = 28usize;
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

        if key == "general.architecture" {
            if let Ok(arch) = std::str::from_utf8(&val_bytes[8..]) {
                architecture = arch.to_string();
            }
        } else if key == "general.alignment" {
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

    ensure!(split_at < total_layers, "split_at ({}) must be < total_layers ({})", split_at, total_layers);
    let stage1_layers = split_at;
    let stage2_layers = total_layers - split_at;

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

        let type_size = ggml_type_size_bytes(ggml_type);
        let size_bytes = (element_count as f64 * type_size).ceil() as u64;
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

    // Write Stage 1 GGUF
    write_sliced_gguf(
        &stage1_path,
        version,
        alignment,
        data_section_offset,
        &raw_kvs,
        &tensors,
        &mmap,
        &architecture,
        stage1_layers,
        true,   // is_stage1
        split_at,
    )?;

    // Write Stage 2 GGUF
    write_sliced_gguf(
        &stage2_path,
        version,
        alignment,
        data_section_offset,
        &raw_kvs,
        &tensors,
        &mmap,
        &architecture,
        stage2_layers,
        false,  // is_stage1
        split_at,
    )?;

    info!(
        stage1 = %stage1_path.display(),
        stage2 = %stage2_path.display(),
        "✅ Partitioned GGUF slicing complete"
    );

    Ok((stage1_path, stage2_path))
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

fn write_sliced_gguf(
    dest_path: &Path,
    version: u32,
    alignment: u64,
    src_data_offset: u64,
    raw_kvs: &[RawKvEntry],
    tensors: &[SourceTensor],
    src_mmap: &Mmap,
    _architecture: &str,
    block_count: usize,
    is_stage1: bool,
    split_at: usize,
) -> Result<()> {
    let mut selected_tensors: Vec<SlicedTensor> = Vec::new();

    if is_stage1 {
        // Stage 1: layers 0..split_at
        for t in tensors {
            if let Some(idx) = t.layer_index {
                if idx < split_at {
                    selected_tensors.push(SlicedTensor {
                        name: t.name.clone(),
                        dimensions: t.dimensions.clone(),
                        ggml_type: t.ggml_type,
                        src_offset: t.offset,
                        size_bytes: t.size_bytes,
                        is_identity_norm: false,
                    });
                }
            } else if t.name.contains("token_embd") || t.name.contains("embed") {
                selected_tensors.push(SlicedTensor {
                    name: t.name.clone(),
                    dimensions: t.dimensions.clone(),
                    ggml_type: t.ggml_type,
                    src_offset: t.offset,
                    size_bytes: t.size_bytes,
                    is_identity_norm: false,
                });
            } else if t.name.contains("output_norm") || t.name.contains("norm.weight") {
                // Identity RMSNorm: weights filled with 1.0
                selected_tensors.push(SlicedTensor {
                    name: t.name.clone(),
                    dimensions: t.dimensions.clone(),
                    ggml_type: t.ggml_type,
                    src_offset: t.offset,
                    size_bytes: t.size_bytes,
                    is_identity_norm: true,
                });
            } else if t.name.contains("output") {
                // Keep output.weight so llama.cpp initializes graph without complaining
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
    } else {
        // Stage 2: layers split_at..total_layers (re-indexed to 0..stage2_layers-1)
        for t in tensors {
            if let Some(idx) = t.layer_index {
                if idx >= split_at {
                    let new_idx = idx - split_at;
                    let new_name = reindex_tensor_name(&t.name, idx, new_idx);
                    selected_tensors.push(SlicedTensor {
                        name: new_name,
                        dimensions: t.dimensions.clone(),
                        ggml_type: t.ggml_type,
                        src_offset: t.offset,
                        size_bytes: t.size_bytes,
                        is_identity_norm: false,
                    });
                }
            } else {
                // Non-layer global tensors (token_embd, output_norm, output)
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
            } else {
                // Fallback: copy original
                let src_start = (src_data_offset + t.src_offset) as usize;
                let src_end = src_start + t.size_bytes as usize;
                writer.write_all(&src_mmap[src_start..src_end])?;
            }
        } else {
            let src_start = (src_data_offset + t.src_offset) as usize;
            let src_end = src_start + t.size_bytes as usize;
            ensure!(src_end <= src_mmap.len(), "Source tensor offset exceeds mmap length");
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
    ensure!(len <= 64 * 1024, "GGUF string length ({}) exceeds 64KB safety limit", len);
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).to_string())
}

fn skip_gguf_val<R: Read + Seek>(reader: &mut R, val_type: u32) -> Result<()> {
    match val_type {
        0 | 1 | 7 => { reader.seek(SeekFrom::Current(1))?; }
        2 | 3 => { reader.seek(SeekFrom::Current(2))?; }
        4 | 5 | 6 => { reader.seek(SeekFrom::Current(4))?; }
        8 => {
            let len = reader.read_u64::<LittleEndian>()?;
            ensure!(len <= (i64::MAX as u64), "GGUF string length exceeds seek bounds");
            reader.seek(SeekFrom::Current(len as i64))?;
        }
        9 => {
            let elem_type = reader.read_u32::<LittleEndian>()?;
            let count = reader.read_u64::<LittleEndian>()?;
            ensure!(count <= 1_000_000, "GGUF array count ({}) exceeds maximum safe limit", count);
            for _ in 0..count {
                skip_gguf_val(reader, elem_type)?;
            }
        }
        10 | 11 | 12 => { reader.seek(SeekFrom::Current(8))?; }
        _ => bail!("Unknown GGUF val_type: {}", val_type),
    }
    Ok(())
}

fn extract_layer_index(name: &str) -> Option<usize> {
    for part in name.split('.') {
        if let Ok(idx) = part.parse::<usize>() {
            return Some(idx);
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
        _ => 2.0,
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
}
