use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use anyhow::{Context, Result};
use tracing::info;

use crate::slice_loader::GgufSliceLoader;

const HASH_THRESHOLD: u64 = 10 * 1024 * 1024; // 10 MB

/// 64-bit FNV-1a hash matching llama.cpp's ggml-rpc hashing algorithm
pub fn fnv_hash(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Returns the standard llama.cpp RPC cache directory on this platform
pub fn get_rpc_cache_dir() -> PathBuf {
    if let Ok(override_dir) = std::env::var("LLAMA_CACHE") {
        PathBuf::from(override_dir).join("rpc")
    } else {
        #[cfg(target_os = "windows")]
        {
            let local_app_data = std::env::var("LOCALAPPDATA")
                .unwrap_or_else(|_| "C:\\Users\\Default\\AppData\\Local".to_string());
            PathBuf::from(local_app_data).join("llama.cpp").join("rpc")
        }
        #[cfg(not(target_os = "windows"))]
        {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
            PathBuf::from(home).join(".cache").join("llama.cpp").join("rpc")
        }
    }
}

/// Pre-populates the local RPC disk cache from a local GGUF model file.
/// This guarantees 0.0 MB of network transfer even on the very first cluster startup.
pub fn prime_rpc_cache<P: AsRef<Path>>(model_path: P) -> Result<(usize, u64)> {
    let path_ref = model_path.as_ref();
    let cache_dir = get_rpc_cache_dir();
    fs::create_dir_all(&cache_dir)
        .with_context(|| format!("Failed to create RPC cache directory at {:?}", cache_dir))?;

    info!(cache_dir = %cache_dir.display(), model = %path_ref.display(), "Priming local RPC disk cache from SSD");

    let loader = GgufSliceLoader::open(path_ref)?;
    let mut primed_count = 0;
    let mut primed_bytes = 0u64;

    for info in &loader.all_tensors {
        if info.size_bytes > HASH_THRESHOLD {
            if let Some(bytes) = loader.get_tensor_bytes(&info.name) {
                let hash = fnv_hash(bytes);
                let hash_file_name = format!("{:016x}", hash);
                let target_file = cache_dir.join(&hash_file_name);

                // If already cached with correct size, skip
                if target_file.exists() {
                    if let Ok(meta) = fs::metadata(&target_file) {
                        if meta.len() == info.size_bytes {
                            primed_count += 1;
                            primed_bytes += info.size_bytes;
                            continue;
                        }
                    }
                }

                // Write tensor block to cache
                let mut f = File::create(&target_file)
                    .with_context(|| format!("Failed to write cache block {:?}", target_file))?;
                f.write_all(bytes)?;
                primed_count += 1;
                primed_bytes += info.size_bytes;
            }
        }
    }

    info!(
        cached_tensors = primed_count,
        cached_mb = primed_bytes / 1024 / 1024,
        "✅ Local RPC Disk Cache primed successfully"
    );

    Ok((primed_count, primed_bytes))
}

/// Checks the status of the local RPC disk cache
pub fn check_rpc_cache_status() -> Result<(usize, u64)> {
    let cache_dir = get_rpc_cache_dir();
    if !cache_dir.exists() {
        return Ok((0, 0));
    }

    let mut count = 0;
    let mut total_bytes = 0u64;

    for entry in fs::read_dir(&cache_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            if let Ok(meta) = fs::metadata(&path) {
                count += 1;
                total_bytes += meta.len();
            }
        }
    }

    Ok((count, total_bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fnv_hash_deterministic() {
        let data = b"AeroMesh Distributed Inference Engine";
        let h1 = fnv_hash(data);
        let h2 = fnv_hash(data);
        assert_eq!(h1, h2);
        assert_ne!(h1, 0);
    }
}
