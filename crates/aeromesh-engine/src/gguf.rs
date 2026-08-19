use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};
use tracing::info;

#[derive(Debug, Clone)]
pub struct GgufMetadata {
    pub file_path: String,
    pub version: u32,
    pub tensor_count: u64,
    pub kv_count: u64,
    pub file_size_bytes: u64,
    pub fast_checksum: String,
}

pub fn resolve_model_path<P: AsRef<Path>>(input_path: Option<P>) -> Result<PathBuf> {
    if let Some(p) = input_path {
        let path = p.as_ref();
        if path.exists() {
            return Ok(path.to_path_buf());
        }

        // Try with .gguf extension appended
        let with_gguf = path.with_extension("gguf");
        if with_gguf.exists() {
            return Ok(with_gguf);
        }

        // Try inside models/ folder
        let in_models = Path::new("models").join(path);
        if in_models.exists() {
            return Ok(in_models);
        }

        let in_models_gguf = Path::new("models").join(&with_gguf);
        if in_models_gguf.exists() {
            return Ok(in_models_gguf);
        }

        // Try stripping leading "models/" if already inside models/ directory
        if let Ok(stripped) = path.strip_prefix("models") {
            if stripped.exists() {
                return Ok(stripped.to_path_buf());
            }
            let stripped_gguf = stripped.with_extension("gguf");
            if stripped_gguf.exists() {
                return Ok(stripped_gguf);
            }
        }

        // Try just the file_name in current dir or models dir
        if let Some(file_name) = path.file_name() {
            let in_curr = Path::new(file_name);
            if in_curr.exists() {
                return Ok(in_curr.to_path_buf());
            }
            let in_parent_models = Path::new("..").join("models").join(file_name);
            if in_parent_models.exists() {
                return Ok(in_parent_models);
            }
        }

        bail!("Could not find model file at {:?} (also checked current directory, models/ and ../models/)", path);
    }

    // Auto-discover in current directory or models/ folder
    for candidate_dir in &[Path::new("models"), Path::new("."), Path::new("..").join("models").as_path()] {
        if candidate_dir.exists() {
            if let Ok(entries) = std::fs::read_dir(candidate_dir) {
                for entry in entries.flatten() {
                    let p = entry.path();
                    if p.is_file() && p.extension().map_or(false, |ext| ext == "gguf") {
                        info!(found = %p.display(), "Auto-discovered model file");
                        return Ok(p);
                    }
                }
            }
        }
    }

    bail!("No model specified and no .gguf models found in models/ or current directory");
}

pub fn inspect_gguf_file<P: AsRef<Path>>(path: P) -> Result<GgufMetadata> {
    let resolved = resolve_model_path(Some(path))?;
    let mut file = File::open(&resolved).context(format!("Failed to open GGUF model file at {:?}", resolved))?;
    let file_size_bytes = file.metadata()?.len();

    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        bail!("Invalid GGUF header magic. Expected 'GGUF', got {:?}", magic);
    }

    let mut ver_bytes = [0u8; 4];
    file.read_exact(&mut ver_bytes)?;
    let version = u32::from_le_bytes(ver_bytes);
    if !(2..=3).contains(&version) {
        bail!("Unsupported GGUF version: {}", version);
    }

    let mut count_bytes = [0u8; 8];
    file.read_exact(&mut count_bytes)?;
    let tensor_count = u64::from_le_bytes(count_bytes);

    file.read_exact(&mut count_bytes)?;
    let kv_count = u64::from_le_bytes(count_bytes);

    // Compute fast 64MB block hash
    file.seek(SeekFrom::Start(0))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 64 * 1024 * 1024];
    let n = file.read(&mut buffer)?;
    hasher.update(&buffer[..n]);
    let fast_checksum = hex::encode(&hasher.finalize()[..8]);

    info!(
        path = %resolved.display(),
        version = version,
        tensors = tensor_count,
        kv_pairs = kv_count,
        size_gb = (file_size_bytes as f64) / 1024.0 / 1024.0 / 1024.0,
        checksum = %fast_checksum,
        "✅ GGUF Model Header Verified"
    );

    Ok(GgufMetadata {
        file_path: resolved.to_string_lossy().to_string(),
        version,
        tensor_count,
        kv_count,
        file_size_bytes,
        fast_checksum,
    })
}
