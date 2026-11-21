//! Cross-platform install vs. writable user-data path resolution.
//!
//! When AeroMESH is installed into a read-only location such as
//! `C:\Program Files\AeroMesh`, runtime state (`.env`, downloaded models,
//! ephemeral GGUF stage slices, browser profiles, and logs) MUST be stored in
//! a user-writable directory (`%LOCALAPPDATA%\AeroMesh`).
//!
//! During local development (when `Cargo.toml` or `.aeromesh_portable` sits
//! next to the installation root), data continues to live inside the checkout
//! root so existing developer workflows keep working seamlessly.

use anyhow::{bail, Result};
use std::env;
use std::path::{Component, Path, PathBuf};

/// Returns the read-only installation root (containing `static/`, `bin/`, etc.).
pub fn install_dir() -> PathBuf {
    if let Ok(explicit) = env::var("AEROMESH_INSTALL_DIR") {
        let p = PathBuf::from(explicit.trim());
        if !p.as_os_str().is_empty() && p.is_dir() {
            return p;
        }
    }

    if let Ok(exe) = env::current_exe() {
        let mut cur = exe.parent();
        while let Some(dir) = cur {
            if dir.join("static").join("index.html").is_file()
                || dir.join("Cargo.toml").is_file()
                || dir.join("bin").join("llama.dll").is_file()
                || dir.join("llama.dll").is_file()
            {
                return dir.to_path_buf();
            }
            cur = dir.parent();
        }
        if let Some(parent) = exe.parent() {
            return parent.to_path_buf();
        }
    }

    env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

/// Returns true when running from a source checkout or an explicit portable bundle.
pub fn is_portable_mode() -> bool {
    let root = install_dir();
    root.join("Cargo.toml").is_file() || root.join(".aeromesh_portable").is_file()
}

/// Returns the writable application data directory.
///
/// Resolution order:
/// 1. `AEROMESH_DATA_DIR` environment variable
/// 2. Installation root when running in portable/source-checkout mode
/// 3. `%LOCALAPPDATA%\AeroMesh` on Windows (or `$XDG_DATA_HOME/aeromesh` /
///    `$HOME/.local/share/aeromesh` on Unix)
pub fn data_dir() -> PathBuf {
    if let Ok(explicit) = env::var("AEROMESH_DATA_DIR") {
        let p = PathBuf::from(explicit.trim());
        if !p.as_os_str().is_empty() {
            return p;
        }
    }

    if is_portable_mode() {
        return install_dir();
    }

    #[cfg(target_os = "windows")]
    {
        if let Ok(local_app_data) = env::var("LOCALAPPDATA") {
            if !local_app_data.trim().is_empty() {
                return PathBuf::from(local_app_data).join("AeroMesh");
            }
        }
        if let Ok(user_profile) = env::var("USERPROFILE") {
            if !user_profile.trim().is_empty() {
                return PathBuf::from(user_profile)
                    .join("AppData")
                    .join("Local")
                    .join("AeroMesh");
            }
        }
    }

    #[cfg(not(target_os = "windows"))]
    {
        if let Ok(xdg) = env::var("XDG_DATA_HOME") {
            if !xdg.trim().is_empty() {
                return PathBuf::from(xdg).join("aeromesh");
            }
        }
        if let Ok(home) = env::var("HOME") {
            if !home.trim().is_empty() {
                return PathBuf::from(home)
                    .join(".local")
                    .join("share")
                    .join("aeromesh");
            }
        }
    }

    install_dir()
}

/// Writable directory storing user GGUF and Stable Diffusion models.
pub fn models_dir() -> PathBuf {
    if let Ok(explicit) = env::var("AEROMESH_MODELS_DIR") {
        let p = PathBuf::from(explicit.trim());
        if !p.as_os_str().is_empty() {
            return p;
        }
    }
    data_dir().join("models")
}

/// All directories searched when discovering installed models (writable first,
/// read-only install fallback second).
pub fn model_search_dirs() -> Vec<PathBuf> {
    let mut dirs = vec![models_dir()];
    let install_models = install_dir().join("models");
    if !dirs.contains(&install_models) {
        dirs.push(install_models);
    }
    let cwd_models = PathBuf::from("models");
    if cwd_models.is_dir() && !dirs.contains(&cwd_models) {
        dirs.push(cwd_models);
    }
    dirs
}

/// Writable cache directory for ephemeral stage slices and temporary artifacts.
pub fn cache_dir() -> PathBuf {
    data_dir().join("cache")
}

/// Writable directory for runtime logs.
pub fn logs_dir() -> PathBuf {
    data_dir().join("logs")
}

/// Writable `.env` secrets file location.
pub fn env_file_path() -> PathBuf {
    data_dir().join(".env")
}

/// Candidate `.env` paths to read from in priority order.
pub fn env_search_paths() -> Vec<PathBuf> {
    let mut paths = vec![env_file_path()];
    let cwd_env = PathBuf::from(".env");
    if !paths.contains(&cwd_env) {
        paths.push(cwd_env);
    }
    let install_env = install_dir().join(".env");
    if !paths.contains(&install_env) {
        paths.push(install_env);
    }
    paths
}

/// Creates the writable data directory hierarchy if it does not already exist.
pub fn ensure_data_dirs() -> std::io::Result<()> {
    std::fs::create_dir_all(data_dir())?;
    std::fs::create_dir_all(models_dir())?;
    std::fs::create_dir_all(cache_dir())?;
    std::fs::create_dir_all(logs_dir())?;
    Ok(())
}

/// Validates a model filename supplied over HTTP/UI to prevent path traversal
/// and CLI flag injection (e.g., `--something` or `../secret`).
pub fn sanitize_model_filename(raw: &str) -> Result<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        bail!("Model name cannot be empty");
    }
    if trimmed.contains('\0') {
        bail!("Model name contains invalid null byte");
    }
    // Strip an optional benign "models/" or "models\" prefix sent by legacy UI clients.
    let stripped = trimmed
        .strip_prefix("models/")
        .or_else(|| trimmed.strip_prefix("models\\"))
        .unwrap_or(trimmed);

    if stripped.is_empty() || stripped.starts_with('-') {
        bail!("Invalid model name '{}'", trimmed);
    }

    let p = Path::new(stripped);
    let mut comps = p.components();
    match (comps.next(), comps.next()) {
        (Some(Component::Normal(name)), None) => {
            let clean = name.to_string_lossy().to_string();
            if clean.starts_with('-') || clean.contains("..") {
                bail!("Invalid model filename '{}'", trimmed);
            }
            Ok(clean)
        }
        _ => bail!(
            "Model name '{}' must be a single filename inside the models directory",
            trimmed
        ),
    }
}

/// Resolves a user-requested model strictly within one of the configured models
/// directories, verifying canonical path confinement.
pub fn resolve_confined_model_path(raw: &str) -> Result<PathBuf> {
    let filename = sanitize_model_filename(raw)?;
    let with_ext = if filename.to_ascii_lowercase().ends_with(".gguf") {
        filename.clone()
    } else {
        format!("{filename}.gguf")
    };

    for dir in model_search_dirs() {
        if !dir.is_dir() {
            continue;
        }
        let Ok(canon_dir) = std::fs::canonicalize(&dir) else {
            continue;
        };
        for candidate_name in [&filename, &with_ext] {
            let candidate = dir.join(candidate_name);
            if candidate.is_file() {
                let canon_file = std::fs::canonicalize(&candidate)?;
                if canon_file.starts_with(&canon_dir) {
                    return Ok(canon_file);
                }
            }
        }
    }

    bail!("Model '{}' not found in configured models directory", raw)
}

/// Resolves a relative static asset path strictly within `static_root`.
/// Returns `None` if the path attempts directory traversal or does not exist.
pub fn resolve_confined_static_path(static_root: &Path, rel: &str) -> Option<PathBuf> {
    if rel.contains('\0') {
        return None;
    }
    let trimmed = rel.trim_start_matches('/').trim_start_matches('\\');
    if trimmed.is_empty() {
        return None;
    }

    let rel_path = Path::new(trimmed);
    for comp in rel_path.components() {
        match comp {
            Component::Normal(_) => {}
            _ => return None,
        }
    }

    let canon_root = std::fs::canonicalize(static_root).ok()?;
    let candidate = canon_root.join(rel_path);
    let canon_file = std::fs::canonicalize(&candidate).ok()?;
    if canon_file.starts_with(&canon_root) && canon_file.is_file() {
        Some(canon_file)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_rejects_traversal_and_flags() {
        assert!(sanitize_model_filename("qwen2.5-7b.gguf").is_ok());
        assert_eq!(
            sanitize_model_filename("models/qwen2.5-7b.gguf").unwrap(),
            "qwen2.5-7b.gguf"
        );
        assert!(sanitize_model_filename("../secret.gguf").is_err());
        assert!(sanitize_model_filename("....//secret.gguf").is_err());
        assert!(sanitize_model_filename("C:\\Windows\\system.ini").is_err());
        assert!(sanitize_model_filename("/etc/passwd").is_err());
        assert!(sanitize_model_filename("--malicious-flag").is_err());
        assert!(sanitize_model_filename("").is_err());
    }
}
