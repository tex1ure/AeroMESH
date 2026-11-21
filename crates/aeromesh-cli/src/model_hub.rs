//! Native Model Hub backend for AeroMesh-Desktop.exe.
//!
//! Provides the exact same LM Studio / Ollama-style model management API as
//! `app.py`, allowing installed Windows users to browse Hugging Face GGUF
//! repos, stream-download models with progress/cancellation, drop `.gguf`
//! files directly into the app, or register existing local model folders and
//! exact `.gguf` paths without Python.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
#[cfg(target_os = "windows")]
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, UNIX_EPOCH};

use axum::{
    body::Body,
    extract::{Query, State},
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    Json,
};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::io::AsyncWriteExt;

#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x08000000;

const ALLOWED_DOWNLOAD_HOSTS: &[&str] = &[
    "huggingface.co",
    "www.huggingface.co",
    "hf.co",
    "cdn-lfs.huggingface.co",
    "cdn-lfs-us-1.huggingface.co",
    "cdn-lfs-eu-1.huggingface.co",
    "cas-bridge.xethub.hf.co",
];

#[derive(Clone, Debug, Serialize)]
pub struct DownloadProgress {
    pub id: String,
    pub filename: String,
    pub repo: String,
    pub status: String,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub percent: f64,
    pub speed_mbps: f64,
    pub eta_seconds: u64,
    pub error: Option<String>,
    pub path: Option<String>,
}

#[derive(Clone, Default)]
pub struct ModelHubState {
    pub downloads: Arc<Mutex<HashMap<String, DownloadProgress>>>,
    pub cancel_flags: Arc<Mutex<HashMap<String, Arc<AtomicBool>>>>,
}

fn format_bytes(bytes: u64) -> String {
    const GB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MB: f64 = 1024.0 * 1024.0;
    const KB: f64 = 1024.0;
    let b = bytes as f64;
    if b >= GB {
        format!("{:.2} GB", b / GB)
    } else if b >= MB {
        format!("{:.1} MB", b / MB)
    } else if b >= KB {
        format!("{:.0} KB", b / KB)
    } else {
        format!("{} B", bytes)
    }
}

fn extract_quant_tag(filename: &str) -> String {
    let upper = filename.to_ascii_uppercase();
    for tag in [
        "IQ4_XS", "IQ4_NL", "IQ3_XXS", "IQ3_XS", "IQ3_S", "IQ3_M", "IQ2_XXS", "IQ2_XS", "IQ2_S",
        "IQ2_M", "Q8_0", "Q6_K_L", "Q6_K", "Q5_K_L", "Q5_K_M", "Q5_K_S", "Q5_0", "Q4_K_L",
        "Q4_K_M", "Q4_K_S", "Q4_0", "Q3_K_XL", "Q3_K_L", "Q3_K_M", "Q3_K_S", "Q2_K_L", "Q2_K",
        "BF16", "FP16", "F16", "F32",
    ] {
        if upper.contains(tag) {
            return tag.to_string();
        }
    }
    if upper.ends_with(".SAFETENSORS") {
        "FP16".to_string()
    } else {
        "GGUF".to_string()
    }
}

fn is_allowed_hf_host(host: &str) -> bool {
    let low = host.to_ascii_lowercase();
    ALLOWED_DOWNLOAD_HOSTS.contains(&low.as_str())
        || low.ends_with(".hf.co")
        || low.ends_with(".huggingface.co")
}

fn validate_hf_url(raw: &str) -> Result<reqwest::Url, String> {
    let url = reqwest::Url::parse(raw.trim()).map_err(|_| "Invalid download URL.".to_string())?;
    if url.scheme() != "https" {
        return Err("Only HTTPS download URLs are permitted.".to_string());
    }
    let host = url.host_str().unwrap_or("");
    if !is_allowed_hf_host(host) {
        return Err(format!(
            "Download host '{}' is not on the Hugging Face allowlist.",
            host
        ));
    }
    if url.path().contains("..") {
        return Err("Invalid download URL path.".to_string());
    }
    Ok(url)
}

fn validate_hf_repo_id(raw: &str) -> Result<String, String> {
    let cleaned = raw.trim().trim_matches('/');
    let mut parts = cleaned.split('/');
    let (Some(owner), Some(repo), None) = (parts.next(), parts.next(), parts.next()) else {
        return Err("Invalid Hugging Face repository ID (expected 'owner/repo').".to_string());
    };
    let valid_part = |s: &str| {
        !s.is_empty()
            && !s.contains("..")
            && s.chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-'))
    };
    if !valid_part(owner) || !valid_part(repo) {
        return Err("Invalid Hugging Face repository ID.".to_string());
    }
    Ok(format!("{}/{}", owner, repo))
}

fn curated_catalog() -> Value {
    json!([
        {
            "id": "qwen2.5-1.5b",
            "name": "Qwen 2.5 1.5B Instruct",
            "repo": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
            "recommended_file": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
            "params": "1.5B",
            "quant": "Q4_K_M",
            "size_hint": "~1.1 GB",
            "vram_hint": "2 GB VRAM / CPU",
            "category": "Fast & Compact",
            "description": "Ultra-fast starter model for quick testing, laptops, and CPU fallback."
        },
        {
            "id": "llama-3.2-3b",
            "name": "Llama 3.2 3B Instruct",
            "repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
            "recommended_file": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            "params": "3B",
            "quant": "Q4_K_M",
            "size_hint": "~2.0 GB",
            "vram_hint": "3 GB VRAM",
            "category": "General Chat",
            "description": "Balanced everyday assistant with high token speed on single GPUs and Wi-Fi meshes."
        },
        {
            "id": "qwen2.5-7b",
            "name": "Qwen 2.5 7B Instruct",
            "repo": "bartowski/Qwen2.5-7B-Instruct-GGUF",
            "recommended_file": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
            "params": "7B",
            "quant": "Q4_K_M",
            "size_hint": "~4.7 GB",
            "vram_hint": "6 GB VRAM",
            "category": "General & Coding",
            "description": "Strong instruction following and coding quality that fits comfortably in 6-8 GB VRAM."
        },
        {
            "id": "deepseek-r1-7b",
            "name": "DeepSeek-R1 Distill Qwen 7B",
            "repo": "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF",
            "recommended_file": "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
            "params": "7B",
            "quant": "Q4_K_M",
            "size_hint": "~4.7 GB",
            "vram_hint": "6 GB VRAM",
            "category": "Reasoning",
            "description": "Step-by-step chain-of-thought reasoning with native <think> foldout support."
        },
        {
            "id": "deepseek-r1-14b",
            "name": "DeepSeek-R1 Distill Qwen 14B",
            "repo": "bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF",
            "recommended_file": "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
            "params": "14B",
            "quant": "Q4_K_M",
            "size_hint": "~9.0 GB",
            "vram_hint": "2x 8 GB Cluster",
            "category": "Cluster Flagship",
            "description": "Built for 2-node AeroMESH layer pipelining across two 6-8 GB gaming laptops."
        },
        {
            "id": "qwen2.5-coder-14b",
            "name": "Qwen 2.5 Coder 14B Instruct",
            "repo": "bartowski/Qwen2.5-Coder-14B-Instruct-GGUF",
            "recommended_file": "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf",
            "params": "14B",
            "quant": "Q4_K_M",
            "size_hint": "~9.0 GB",
            "vram_hint": "2x 8 GB Cluster",
            "category": "Coding Cluster",
            "description": "High-accuracy local software engineering model designed for distributed VRAM pooling."
        }
    ])
}

pub fn build_library_payload(hub: &ModelHubState) -> Value {
    let primary_dir = aeromesh_core::models_dir();
    let _ = std::fs::create_dir_all(&primary_dir);
    let primary_canon = std::fs::canonicalize(&primary_dir).unwrap_or_else(|_| primary_dir.clone());

    let custom_dirs: HashSet<String> = aeromesh_core::load_custom_model_dirs()
        .into_iter()
        .map(|d| {
            std::fs::canonicalize(&d)
                .unwrap_or(d)
                .to_string_lossy()
                .to_ascii_lowercase()
        })
        .collect();

    let mut directories = Vec::new();
    let mut models = Vec::new();
    let mut seen_files = HashSet::new();

    for (idx, dir) in aeromesh_core::model_search_dirs().into_iter().enumerate() {
        let exists = dir.is_dir();
        let dir_canon = std::fs::canonicalize(&dir).unwrap_or_else(|_| dir.clone());
        let dir_low = dir_canon.to_string_lossy().to_ascii_lowercase();
        let mut count = 0usize;

        if exists {
            if let Ok(entries) = std::fs::read_dir(&dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if !path.is_file() {
                        continue;
                    }
                    let fname = entry.file_name().to_string_lossy().to_string();
                    let lower = fname.to_ascii_lowercase();
                    let kind = if lower.ends_with(".gguf") {
                        if fname.contains("_stage") || fname.starts_with(".aeromesh_") {
                            continue;
                        }
                        "llm"
                    } else if lower.ends_with(".safetensors") {
                        "diffusion"
                    } else {
                        continue;
                    };

                    let canon_file = std::fs::canonicalize(&path).unwrap_or_else(|_| path.clone());
                    let key = canon_file.to_string_lossy().to_ascii_lowercase();
                    if !seen_files.insert(key) {
                        continue;
                    }
                    let meta = entry.metadata().ok();
                    let size_bytes = meta.as_ref().map(|m| m.len()).unwrap_or(0);
                    let modified_ts = meta
                        .and_then(|m| m.modified().ok())
                        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                        .map(|d| d.as_secs())
                        .unwrap_or(0);
                    count += 1;
                    models.push(json!({
                        "name": fname,
                        "path": path.to_string_lossy(),
                        "directory": dir.to_string_lossy(),
                        "size_bytes": size_bytes,
                        "size_label": format_bytes(size_bytes),
                        "kind": kind,
                        "quant": extract_quant_tag(&fname),
                        "managed": canon_file.starts_with(&primary_canon),
                        "modified_ts": modified_ts
                    }));
                }
            }
        }

        let dir_kind = if idx == 0 && dir_canon == primary_canon {
            "primary"
        } else if custom_dirs.contains(&dir_low) {
            "custom"
        } else {
            "default"
        };
        directories.push(json!({
            "path": dir.to_string_lossy(),
            "kind": dir_kind,
            "exists": exists,
            "model_count": count
        }));
    }

    models.sort_by(|a, b| {
        let ka = if a["kind"] == "llm" { 0 } else { 1 };
        let kb = if b["kind"] == "llm" { 0 } else { 1 };
        ka.cmp(&kb).then_with(|| {
            a["name"]
                .as_str()
                .unwrap_or("")
                .to_ascii_lowercase()
                .cmp(&b["name"].as_str().unwrap_or("").to_ascii_lowercase())
        })
    });

    let downloads: Vec<DownloadProgress> = hub
        .downloads
        .lock()
        .map(|g| g.values().cloned().collect())
        .unwrap_or_default();

    json!({
        "primary_dir": primary_dir.to_string_lossy(),
        "disk_free_bytes": 0,
        "disk_free_label": "",
        "directories": directories,
        "models": models,
        "catalog": curated_catalog(),
        "downloads": downloads
    })
}

pub async fn get_models_library(State(hub): State<ModelHubState>) -> impl IntoResponse {
    (StatusCode::OK, Json(build_library_payload(&hub)))
}

#[derive(Deserialize)]
pub struct ManagePathReq {
    pub action: Option<String>,
    pub path: String,
}

pub async fn post_model_paths(
    State(hub): State<ModelHubState>,
    Json(req): Json<ManagePathReq>,
) -> impl IntoResponse {
    let action = req.action.unwrap_or_else(|| "add".to_string()).to_ascii_lowercase();
    let trimmed = req.path.trim().trim_matches('"');
    if trimmed.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"detail": "Directory path is required."})),
        );
    }
    let dir = PathBuf::from(trimmed);
    let res = match action.as_str() {
        "add" => aeromesh_core::add_custom_model_dir(&dir),
        "remove" => aeromesh_core::remove_custom_model_dir(&dir),
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"detail": "Unsupported action (expected 'add' or 'remove')."})),
            )
        }
    };
    match res {
        Ok(_) => (
            StatusCode::OK,
            Json(json!({"status": "ok", "library": build_library_payload(&hub)})),
        ),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(json!({"detail": e.to_string()})),
        ),
    }
}

#[derive(Deserialize)]
pub struct ImportPathReq {
    pub path: String,
    pub copy_to_library: Option<bool>,
}

pub async fn post_import_model_path(
    State(hub): State<ModelHubState>,
    Json(req): Json<ImportPathReq>,
) -> impl IntoResponse {
    let trimmed = req.path.trim().trim_matches('"');
    if trimmed.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"detail": "Model file path is required."})),
        );
    }
    let p = PathBuf::from(trimmed);
    if p.is_dir() {
        if let Err(e) = aeromesh_core::add_custom_model_dir(&p) {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"detail": e.to_string()})),
            );
        }
        return (
            StatusCode::OK,
            Json(json!({
                "status": "ok",
                "imported_type": "directory",
                "path": p.to_string_lossy(),
                "library": build_library_payload(&hub)
            })),
        );
    }
    if !p.is_file() {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({"detail": format!("File not found: {}", p.display())})),
        );
    }
    let fname = p
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    let lower = fname.to_ascii_lowercase();
    if !(lower.ends_with(".gguf") || lower.ends_with(".safetensors")) {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"detail": "Only .gguf and .safetensors model files are supported."})),
        );
    }
    if lower.ends_with(".gguf") {
        if let Err(e) = aeromesh_core::verify_gguf_header(&p) {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"detail": e.to_string()})),
            );
        }
    }

    let final_path = if req.copy_to_library.unwrap_or(false) {
        let safe_name = match aeromesh_core::sanitize_model_filename(&fname) {
            Ok(s) => s,
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(json!({"detail": e.to_string()})),
                )
            }
        };
        let primary = aeromesh_core::models_dir();
        let _ = std::fs::create_dir_all(&primary);
        let dest = primary.join(&safe_name);
        let part = primary.join(format!("{}.part", safe_name));
        let src_clone = p.clone();
        let part_clone = part.clone();
        let copy_res = tokio::task::spawn_blocking(move || std::fs::copy(&src_clone, &part_clone)).await;
        match copy_res {
            Ok(Ok(_)) => {
                if let Err(e) = std::fs::rename(&part, &dest) {
                    let _ = std::fs::remove_file(&part);
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(json!({"detail": format!("Failed to finalize copy: {}", e)})),
                    );
                }
                dest
            }
            _ => {
                let _ = std::fs::remove_file(&part);
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({"detail": "Failed to copy model file into library."})),
                );
            }
        }
    } else {
        if let Some(parent) = p.parent() {
            let _ = aeromesh_core::add_custom_model_dir(parent);
        }
        p
    };

    (
        StatusCode::OK,
        Json(json!({
            "status": "ok",
            "imported_type": "file",
            "model": fname,
            "path": final_path.to_string_lossy(),
            "library": build_library_payload(&hub)
        })),
    )
}

#[derive(Deserialize)]
pub struct BrowseReq {
    pub mode: Option<String>,
}

pub async fn post_browse_path(Json(req): Json<BrowseReq>) -> impl IntoResponse {
    let mode = req.mode.unwrap_or_else(|| "file".to_string()).to_ascii_lowercase();
    #[cfg(target_os = "windows")]
    {
        let selected = tokio::task::spawn_blocking(move || {
            let script = if mode == "folder" {
                "Add-Type -AssemblyName System.Windows.Forms; $d = New-Object System.Windows.Forms.FolderBrowserDialog; $d.Description = 'Select a folder containing GGUF or Safetensors models'; $d.UseDescriptionForTitle = $true; if ($d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { Write-Output $d.SelectedPath }"
            } else {
                "Add-Type -AssemblyName System.Windows.Forms; $d = New-Object System.Windows.Forms.OpenFileDialog; $d.Title = 'Select a GGUF or Safetensors Model File'; $d.Filter = 'AI Models (*.gguf;*.safetensors)|*.gguf;*.safetensors|All Files (*.*)|*.*'; if ($d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { Write-Output $d.FileName }"
            };
            Command::new("powershell")
                .creation_flags(CREATE_NO_WINDOW)
                .args(["-NoProfile", "-STA", "-Command", script])
                .output()
                .ok()
                .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
                .filter(|s| !s.is_empty())
        })
        .await
        .ok()
        .flatten();
        (StatusCode::OK, Json(json!({"supported": true, "path": selected})))
    }
    #[cfg(not(target_os = "windows"))]
    {
        let _ = mode;
        (
            StatusCode::OK,
            Json(json!({"supported": false, "path": Value::Null})),
        )
    }
}

#[derive(Deserialize)]
pub struct UploadQuery {
    pub filename: Option<String>,
}

pub async fn post_upload_model(
    State(hub): State<ModelHubState>,
    Query(q): Query<UploadQuery>,
    headers: HeaderMap,
    body: Body,
) -> impl IntoResponse {
    let raw_name = q
        .filename
        .or_else(|| {
            headers
                .get("x-model-filename")
                .and_then(|v| v.to_str().ok())
                .map(|s| s.to_string())
        })
        .unwrap_or_default();

    let filename = match aeromesh_core::sanitize_model_filename(&raw_name) {
        Ok(f) => f,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"detail": e.to_string()})),
            )
        }
    };
    let lower = filename.to_ascii_lowercase();
    if !(lower.ends_with(".gguf") || lower.ends_with(".safetensors")) {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"detail": "Only .gguf and .safetensors files can be uploaded."})),
        );
    }

    let primary = aeromesh_core::models_dir();
    let _ = std::fs::create_dir_all(&primary);
    let dest = primary.join(&filename);
    let part = primary.join(format!("{}.part", filename));

    let mut file = match tokio::fs::File::create(&part).await {
        Ok(f) => f,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"detail": format!("Failed to create destination file: {}", e)})),
            )
        }
    };

    let mut stream = body.into_data_stream();
    let mut written: u64 = 0;
    let mut header_buf: Vec<u8> = Vec::with_capacity(4);
    let is_gguf = lower.ends_with(".gguf");

    while let Some(chunk_res) = stream.next().await {
        match chunk_res {
            Ok(chunk) => {
                if chunk.is_empty() {
                    continue;
                }
                if header_buf.len() < 4 {
                    let take = (4 - header_buf.len()).min(chunk.len());
                    header_buf.extend_from_slice(&chunk[..take]);
                    if header_buf.len() == 4 && is_gguf && header_buf.as_slice() != b"GGUF" {
                        drop(file);
                        let _ = tokio::fs::remove_file(&part).await;
                        return (
                            StatusCode::BAD_REQUEST,
                            Json(json!({"detail": "Dropped file is not a valid GGUF model (missing GGUF magic header)."})),
                        );
                    }
                }
                if let Err(e) = file.write_all(&chunk).await {
                    drop(file);
                    let _ = tokio::fs::remove_file(&part).await;
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(json!({"detail": format!("Disk write error: {}", e)})),
                    );
                }
                written += chunk.len() as u64;
            }
            Err(e) => {
                drop(file);
                let _ = tokio::fs::remove_file(&part).await;
                return (
                    StatusCode::BAD_REQUEST,
                    Json(json!({"detail": format!("Upload stream interrupted: {}", e)})),
                );
            }
        }
    }

    let _ = file.flush().await;
    drop(file);

    if written == 0 {
        let _ = tokio::fs::remove_file(&part).await;
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"detail": "Uploaded file was empty."})),
        );
    }

    if let Err(e) = tokio::fs::rename(&part, &dest).await {
        let _ = tokio::fs::remove_file(&part).await;
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"detail": format!("Failed to finalize upload: {}", e)})),
        );
    }

    (
        StatusCode::OK,
        Json(json!({
            "status": "ok",
            "model": filename,
            "size_bytes": written,
            "path": dest.to_string_lossy(),
            "library": build_library_payload(&hub)
        })),
    )
}

#[derive(Deserialize)]
pub struct DeleteModelReq {
    pub name: String,
}

pub async fn post_delete_model(
    State(hub): State<ModelHubState>,
    Json(req): Json<DeleteModelReq>,
) -> impl IntoResponse {
    let filename = match aeromesh_core::sanitize_model_filename(&req.name) {
        Ok(f) => f,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"detail": e.to_string()})),
            )
        }
    };
    let primary = aeromesh_core::models_dir();
    let Ok(primary_canon) = std::fs::canonicalize(&primary) else {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({"detail": "Managed models directory not found."})),
        );
    };
    let candidate = primary.join(&filename);
    let Ok(cand_canon) = std::fs::canonicalize(&candidate) else {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({"detail": "Only models stored in AeroMESH's managed folder can be deleted here."})),
        );
    };
    if !cand_canon.starts_with(&primary_canon) || !cand_canon.is_file() {
        return (
            StatusCode::FORBIDDEN,
            Json(json!({"detail": "Cannot delete models outside AeroMESH's managed folder."})),
        );
    }
    if let Err(e) = std::fs::remove_file(&cand_canon) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"detail": format!("Failed to delete model: {}", e)})),
        );
    }
    (
        StatusCode::OK,
        Json(json!({"status": "ok", "deleted": filename, "library": build_library_payload(&hub)})),
    )
}

#[derive(Deserialize)]
pub struct HfSearchQuery {
    pub q: Option<String>,
    pub limit: Option<usize>,
}

pub async fn get_hf_search(Query(query): Query<HfSearchQuery>) -> impl IntoResponse {
    let limit = query.limit.unwrap_or(12).clamp(1, 25);
    let mut url = match reqwest::Url::parse("https://huggingface.co/api/models") {
        Ok(u) => u,
        Err(_) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"detail": "URL error"})),
            )
        }
    };
    {
        let mut pairs = url.query_pairs_mut();
        pairs.append_pair("filter", "gguf");
        pairs.append_pair("sort", "downloads");
        pairs.append_pair("direction", "-1");
        pairs.append_pair("limit", &limit.to_string());
        if let Some(q) = query.q.as_deref() {
            let trimmed = q.trim();
            if !trimmed.is_empty() {
                pairs.append_pair("search", trimmed);
            }
        }
    }

    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(12))
        .user_agent("AeroMESH-Studio/0.3")
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"detail": e.to_string()})),
            )
        }
    };

    let resp = match client.get(url).send().await {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_GATEWAY,
                Json(json!({"detail": format!("Could not reach Hugging Face Hub: {}", e)})),
            )
        }
    };
    if !resp.status().is_success() {
        return (
            StatusCode::BAD_GATEWAY,
            Json(json!({"detail": format!("Hugging Face search returned HTTP {}", resp.status())})),
        );
    }
    let raw_items: Vec<Value> = resp.json().await.unwrap_or_default();
    let results: Vec<Value> = raw_items
        .into_iter()
        .filter_map(|item| {
            let repo_id = item
                .get("id")
                .or_else(|| item.get("modelId"))
                .and_then(|v| v.as_str())?
                .to_string();
            let author = item
                .get("author")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| repo_id.split('/').next().unwrap_or("").to_string());
            Some(json!({
                "repo": repo_id,
                "author": author,
                "downloads": item.get("downloads").and_then(|v| v.as_u64()).unwrap_or(0),
                "likes": item.get("likes").and_then(|v| v.as_u64()).unwrap_or(0),
                "gated": item.get("gated").and_then(|v| v.as_bool()).unwrap_or(false),
                "last_modified": item.get("lastModified").and_then(|v| v.as_str()).unwrap_or("")
            }))
        })
        .collect();

    (StatusCode::OK, Json(json!({"results": results})))
}

#[derive(Deserialize)]
pub struct HfFilesQuery {
    pub repo: String,
}

pub async fn get_hf_files(Query(query): Query<HfFilesQuery>) -> impl IntoResponse {
    let repo_id = match validate_hf_repo_id(&query.repo) {
        Ok(r) => r,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(json!({"detail": e}))),
    };

    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(12))
        .user_agent("AeroMESH-Studio/0.3")
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"detail": e.to_string()})),
            )
        }
    };

    let url = format!("https://huggingface.co/api/models/{}/tree/main", repo_id);
    let resp = match client.get(&url).send().await {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_GATEWAY,
                Json(json!({"detail": format!("Could not inspect {}: {}", repo_id, e)})),
            )
        }
    };
    if !resp.status().is_success() {
        return (
            StatusCode::BAD_GATEWAY,
            Json(json!({"detail": format!("Hugging Face returned HTTP {} for {}", resp.status(), repo_id)})),
        );
    }

    let entries: Vec<Value> = resp.json().await.unwrap_or_default();
    let mut files: Vec<Value> = entries
        .into_iter()
        .filter_map(|entry| {
            let path = entry.get("path").and_then(|v| v.as_str())?.to_string();
            if !path.to_ascii_lowercase().ends_with(".gguf") {
                return None;
            }
            let size = entry
                .get("size")
                .and_then(|v| v.as_u64())
                .or_else(|| entry.pointer("/lfs/size").and_then(|v| v.as_u64()))
                .unwrap_or(0);
            let fname = Path::new(&path)
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| path.clone());
            let est_vram_gb = if size > 0 {
                ((((size as f64 / (1024.0 * 1024.0 * 1024.0)) * 1.18 + 0.5) * 10.0).round()) / 10.0
            } else {
                0.0
            };
            let recommended = fname.to_ascii_uppercase().contains("Q4_K_M");
            Some(json!({
                "filename": fname,
                "path": path,
                "size_bytes": size,
                "size_label": if size > 0 { format_bytes(size) } else { "Unknown".to_string() },
                "quant": extract_quant_tag(&fname),
                "est_vram_gb": est_vram_gb,
                "download_url": format!("https://huggingface.co/{}/resolve/main/{}", repo_id, path),
                "recommended": recommended
            }))
        })
        .collect();

    files.sort_by(|a, b| {
        let ra = if a["recommended"].as_bool().unwrap_or(false) { 0 } else { 1 };
        let rb = if b["recommended"].as_bool().unwrap_or(false) { 0 } else { 1 };
        ra.cmp(&rb).then_with(|| {
            a["size_bytes"]
                .as_u64()
                .unwrap_or(0)
                .cmp(&b["size_bytes"].as_u64().unwrap_or(0))
        })
    });

    (
        StatusCode::OK,
        Json(json!({"repo": repo_id, "files": files})),
    )
}

#[derive(Deserialize)]
pub struct StartDownloadReq {
    pub url: Option<String>,
    pub repo: Option<String>,
    pub filename: Option<String>,
    pub hf_token: Option<String>,
}

pub async fn post_start_download(
    State(hub): State<ModelHubState>,
    Json(req): Json<StartDownloadReq>,
) -> impl IntoResponse {
    let repo_raw = req.repo.unwrap_or_default().trim().to_string();
    let mut raw_url = req.url.unwrap_or_default().trim().to_string();
    let mut raw_filename = req.filename.unwrap_or_default().trim().to_string();
    let hf_token = req.hf_token.unwrap_or_default().trim().to_string();

    if raw_url.is_empty() && !repo_raw.is_empty() && !raw_filename.is_empty() {
        let repo_id = match validate_hf_repo_id(&repo_raw) {
            Ok(r) => r,
            Err(e) => return (StatusCode::BAD_REQUEST, Json(json!({"detail": e}))),
        };
        raw_url = format!("https://huggingface.co/{}/resolve/main/{}", repo_id, raw_filename);
    }

    let url = match validate_hf_url(&raw_url) {
        Ok(u) => u,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(json!({"detail": e}))),
    };

    if raw_filename.is_empty() {
        raw_filename = url
            .path_segments()
            .and_then(|mut s| s.next_back())
            .unwrap_or("")
            .to_string();
    } else {
        raw_filename = Path::new(&raw_filename)
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or(raw_filename);
    }

    let filename = match aeromesh_core::sanitize_model_filename(&raw_filename) {
        Ok(f) => f,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"detail": e.to_string()})),
            )
        }
    };
    let lower = filename.to_ascii_lowercase();
    if !(lower.ends_with(".gguf") || lower.ends_with(".safetensors")) {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"detail": "Only .gguf and .safetensors files can be downloaded."})),
        );
    }

    let mut id_bytes = [0u8; 6];
    let _ = getrandom::getrandom(&mut id_bytes);
    let download_id = hex::encode(id_bytes);

    let progress = DownloadProgress {
        id: download_id.clone(),
        filename: filename.clone(),
        repo: if repo_raw.is_empty() {
            url.path().trim_matches('/').to_string()
        } else {
            repo_raw
        },
        status: "starting".to_string(),
        downloaded_bytes: 0,
        total_bytes: 0,
        percent: 0.0,
        speed_mbps: 0.0,
        eta_seconds: 0,
        error: None,
        path: None,
    };

    let cancel_flag = Arc::new(AtomicBool::new(false));
    if let Ok(mut d) = hub.downloads.lock() {
        d.insert(download_id.clone(), progress.clone());
    }
    if let Ok(mut c) = hub.cancel_flags.lock() {
        c.insert(download_id.clone(), cancel_flag.clone());
    }

    let hub_clone = hub.clone();
    let dl_id = download_id.clone();
    tokio::spawn(async move {
        run_download_task(hub_clone, dl_id, url, filename, hf_token, cancel_flag).await;
    });

    (
        StatusCode::ACCEPTED,
        Json(json!({"status": "started", "download": progress})),
    )
}

async fn run_download_task(
    hub: ModelHubState,
    download_id: String,
    url: reqwest::Url,
    filename: String,
    hf_token: String,
    cancel_flag: Arc<AtomicBool>,
) {
    let primary = aeromesh_core::models_dir();
    let _ = std::fs::create_dir_all(&primary);
    let dest = primary.join(&filename);
    let part = primary.join(format!("{}.part", filename));
    let is_gguf = filename.to_ascii_lowercase().ends_with(".gguf");

    let set_err = |hub: &ModelHubState, id: &str, status: &str, msg: String| {
        if let Ok(mut g) = hub.downloads.lock() {
            if let Some(t) = g.get_mut(id) {
                t.status = status.to_string();
                t.error = Some(msg);
            }
        }
    };

    let client = match reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(15))
        .user_agent("AeroMESH-Studio/0.3")
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            set_err(&hub, &download_id, "error", e.to_string());
            return;
        }
    };

    let mut req = client.get(url);
    if !hf_token.is_empty() {
        req = req.bearer_auth(hf_token);
    }

    let resp = match req.send().await {
        Ok(r) => r,
        Err(e) => {
            set_err(&hub, &download_id, "error", format!("Network error: {}", e));
            return;
        }
    };

    let final_host = resp.url().host_str().unwrap_or("");
    if resp.url().scheme() != "https" || !is_allowed_hf_host(final_host) {
        set_err(
            &hub,
            &download_id,
            "error",
            format!("Blocked redirect to non-allowlisted host '{}'", final_host),
        );
        return;
    }

    if !resp.status().is_success() {
        set_err(
            &hub,
            &download_id,
            "error",
            format!("Hugging Face returned HTTP {}", resp.status()),
        );
        return;
    }

    let total_bytes = resp.content_length().unwrap_or(0);
    if let Ok(mut g) = hub.downloads.lock() {
        if let Some(t) = g.get_mut(&download_id) {
            t.total_bytes = total_bytes;
            t.status = "downloading".to_string();
        }
    }

    let mut file = match tokio::fs::File::create(&part).await {
        Ok(f) => f,
        Err(e) => {
            set_err(&hub, &download_id, "error", format!("Cannot write file: {}", e));
            return;
        }
    };

    let t0 = Instant::now();
    let mut downloaded: u64 = 0;
    let mut header_buf: Vec<u8> = Vec::with_capacity(4);
    let mut stream = resp.bytes_stream();

    while let Some(chunk_res) = stream.next().await {
        if cancel_flag.load(Ordering::Relaxed) {
            drop(file);
            let _ = tokio::fs::remove_file(&part).await;
            set_err(&hub, &download_id, "cancelled", "Cancelled by user".to_string());
            return;
        }
        let chunk = match chunk_res {
            Ok(c) => c,
            Err(e) => {
                drop(file);
                let _ = tokio::fs::remove_file(&part).await;
                set_err(&hub, &download_id, "error", format!("Stream error: {}", e));
                return;
            }
        };
        if header_buf.len() < 4 {
            let take = (4 - header_buf.len()).min(chunk.len());
            header_buf.extend_from_slice(&chunk[..take]);
            if header_buf.len() == 4 && is_gguf && header_buf.as_slice() != b"GGUF" {
                drop(file);
                let _ = tokio::fs::remove_file(&part).await;
                set_err(
                    &hub,
                    &download_id,
                    "error",
                    "Downloaded file is not a valid GGUF model (missing GGUF magic header).".to_string(),
                );
                return;
            }
        }
        if let Err(e) = file.write_all(&chunk).await {
            drop(file);
            let _ = tokio::fs::remove_file(&part).await;
            set_err(&hub, &download_id, "error", format!("Disk write error: {}", e));
            return;
        }
        downloaded += chunk.len() as u64;
        let elapsed = t0.elapsed().as_secs_f64().max(0.05);
        let speed_mbps = (((downloaded as f64) / (1024.0 * 1024.0)) / elapsed * 100.0).round() / 100.0;
        let pct = if total_bytes > 0 {
            (((downloaded as f64) / (total_bytes as f64)) * 1000.0).round() / 10.0
        } else {
            0.0
        };
        let eta = if total_bytes > downloaded && downloaded > 0 {
            ((total_bytes - downloaded) as f64 / (downloaded as f64 / elapsed)) as u64
        } else {
            0
        };
        if let Ok(mut g) = hub.downloads.lock() {
            if let Some(t) = g.get_mut(&download_id) {
                t.downloaded_bytes = downloaded;
                t.percent = pct;
                t.speed_mbps = speed_mbps;
                t.eta_seconds = eta;
            }
        }
    }

    let _ = file.flush().await;
    drop(file);

    if is_gguf {
        if let Err(e) = aeromesh_core::verify_gguf_header(&part) {
            let _ = tokio::fs::remove_file(&part).await;
            set_err(&hub, &download_id, "error", e.to_string());
            return;
        }
    }

    if let Err(e) = tokio::fs::rename(&part, &dest).await {
        let _ = tokio::fs::remove_file(&part).await;
        set_err(&hub, &download_id, "error", format!("Finalize failed: {}", e));
        return;
    }

    if let Ok(mut g) = hub.downloads.lock() {
        if let Some(t) = g.get_mut(&download_id) {
            t.status = "completed".to_string();
            t.percent = 100.0;
            t.downloaded_bytes = downloaded;
            t.total_bytes = downloaded;
            t.eta_seconds = 0;
            t.path = Some(dest.to_string_lossy().to_string());
        }
    }
    if let Ok(mut c) = hub.cancel_flags.lock() {
        c.remove(&download_id);
    }
}

pub async fn get_downloads(State(hub): State<ModelHubState>) -> impl IntoResponse {
    let items: Vec<DownloadProgress> = hub
        .downloads
        .lock()
        .map(|g| g.values().cloned().collect())
        .unwrap_or_default();
    (StatusCode::OK, Json(json!({"downloads": items})))
}

#[derive(Deserialize)]
pub struct CancelReq {
    pub id: String,
}

pub async fn post_cancel_download(
    State(hub): State<ModelHubState>,
    Json(req): Json<CancelReq>,
) -> impl IntoResponse {
    let flag = hub
        .cancel_flags
        .lock()
        .ok()
        .and_then(|g| g.get(req.id.trim()).cloned());
    if let Some(f) = flag {
        f.store(true, Ordering::Relaxed);
        (
            StatusCode::OK,
            Json(json!({"status": "cancelling", "id": req.id})),
        )
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(json!({"detail": "Download task not found."})),
        )
    }
}
