#![windows_subsystem = "windows"]
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::io::{BufRead, BufReader as StdBufReader};
use std::net::{IpAddr, Ipv4Addr, SocketAddr, UdpSocket};
use std::os::windows::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::{
    body::Body,
    extract::{Path as AxumPath, State},
    http::{header, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use bytes::Bytes;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;
use tower_http::cors::{Any, CorsLayer};

use aeromesh_engine::SafeProcessJob;

const CREATE_NO_WINDOW: u32 = 0x08000000;

struct RuntimeProcess {
    role: String,
    model: String,
    peers: String,
    child: Option<Child>,
    job: Option<SafeProcessJob>,
    logs: Arc<Mutex<Vec<String>>>,
}

#[derive(Clone)]
struct DesktopState {
    install_dir: PathBuf,
    data_dir: PathBuf,
    api_key: String,
    port: u16,
    runtime: Arc<Mutex<RuntimeProcess>>,
}

fn generate_csprng_hex(num_bytes: usize) -> String {
    let mut buf = vec![0u8; num_bytes];
    if getrandom::getrandom(&mut buf).is_ok() {
        return hex::encode(buf);
    }
    // Fallback mixing high-resolution time + PID + stack address if OS RNG fails unexpectedly
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let mut h = Sha256::new();
    h.update(now.to_le_bytes());
    h.update(std::process::id().to_le_bytes());
    h.update((&buf as *const Vec<u8> as usize).to_le_bytes());
    hex::encode(h.finalize())
}

fn base64_encode(data: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;
        out.push(TABLE[((triple >> 18) & 0x3F) as usize] as char);
        out.push(TABLE[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            out.push(TABLE[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(TABLE[(triple & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

fn ensure_env_secrets() -> (String, String) {
    let mut api_key = env::var("AEROMESH_API_KEY").unwrap_or_default();
    let mut worker_secret = env::var("AEROMESH_WORKER_SECRET").unwrap_or_default();

    for candidate in aeromesh_core::env_search_paths() {
        if candidate.is_file() {
            if let Ok(content) = fs::read_to_string(&candidate) {
                for line in content.lines() {
                    let line = line.trim();
                    if line.is_empty() || line.starts_with('#') {
                        continue;
                    }
                    if let Some((k, v)) = line.split_once('=') {
                        let k = k.trim();
                        let v = v.trim().trim_matches('"').trim_matches('\'');
                        if env::var(k).is_err() {
                            env::set_var(k, v);
                        }
                        if k == "AEROMESH_API_KEY" && api_key.is_empty() {
                            api_key = v.to_string();
                        }
                        if k == "AEROMESH_WORKER_SECRET" && worker_secret.is_empty() {
                            worker_secret = v.to_string();
                        }
                    }
                }
            }
            break;
        }
    }

    if api_key.trim().is_empty() {
        api_key = generate_csprng_hex(32);
    }
    if worker_secret.trim().is_empty() {
        worker_secret = generate_csprng_hex(32);
    }

    env::set_var("AEROMESH_API_KEY", &api_key);
    env::set_var("AEROMESH_WORKER_SECRET", &worker_secret);
    if env::var("AEROMESH_ACTIVATION_DTYPE").is_err() {
        env::set_var("AEROMESH_ACTIVATION_DTYPE", "int8");
    }

    let env_path = aeromesh_core::env_file_path();
    if !env_path.exists() {
        if let Some(parent) = env_path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let _ = fs::write(
            &env_path,
            format!(
                "AEROMESH_API_KEY={}\nAEROMESH_WORKER_SECRET={}\nAEROMESH_ACTIVATION_DTYPE=int8\n",
                api_key, worker_secret
            ),
        );
    }
    (api_key, worker_secret)
}

fn get_local_wifi_ip() -> String {
    if let Ok(sock) = UdpSocket::bind("0.0.0.0:0") {
        if sock.connect("8.8.8.8:80").is_ok() {
            if let Ok(addr) = sock.local_addr() {
                return addr.ip().to_string();
            }
        }
    }
    "127.0.0.1".to_string()
}

fn get_tailscale_ip() -> String {
    if let Ok(out) = Command::new("tailscale")
        .creation_flags(CREATE_NO_WINDOW)
        .args(["ip", "-4"])
        .output()
    {
        if out.status.success() {
            if let Some(line) = String::from_utf8_lossy(&out.stdout).lines().next() {
                let trimmed = line.trim();
                if !trimmed.is_empty() {
                    return trimmed.to_string();
                }
            }
        }
    }
    "Not Connected".to_string()
}

fn get_gpu_summary() -> String {
    let candidates = [
        "nvidia-smi",
        r"C:\Windows\System32\nvidia-smi.exe",
        r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
    ];
    for p in candidates {
        if let Ok(out) = Command::new(p)
            .creation_flags(CREATE_NO_WINDOW)
            .args([
                "--query-gpu=name,memory.total,memory.used",
                "--format=csv,noheader",
            ])
            .output()
        {
            if out.status.success() {
                if let Some(line) = String::from_utf8_lossy(&out.stdout).lines().next() {
                    let trimmed = line.trim();
                    if !trimmed.is_empty() {
                        return trimmed.to_string();
                    }
                }
            }
        }
    }
    "CPU / Integrated GPU".to_string()
}

async fn discover_workers(port: u16) -> Vec<Value> {
    let mut candidates: HashMap<String, (String, String)> = HashMap::new();
    let my_wifi = get_local_wifi_ip();
    let my_ts = get_tailscale_ip();

    // 1. Tailscale peers
    if let Ok(out) = Command::new("tailscale")
        .creation_flags(CREATE_NO_WINDOW)
        .args(["status", "--json"])
        .output()
    {
        if out.status.success() {
            if let Ok(v) = serde_json::from_slice::<Value>(&out.stdout) {
                if let Some(peers) = v.get("Peer").and_then(|p| p.as_object()) {
                    for peer in peers.values() {
                        if peer.get("Online").and_then(|o| o.as_bool()) == Some(true) {
                            let host = peer
                                .get("HostName")
                                .and_then(|h| h.as_str())
                                .unwrap_or("tailscale-peer")
                                .to_string();
                            if let Some(ips) = peer.get("TailscaleIPs").and_then(|i| i.as_array()) {
                                for ip in ips.iter().filter_map(|x| x.as_str()) {
                                    if ip.contains('.') && ip != my_ts {
                                        candidates.insert(
                                            ip.to_string(),
                                            ("Tailscale Direct WireGuard".to_string(), host.clone()),
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // 2. Local Wi-Fi / LAN ARP table (passive, avoids noisy /24 port sweeps)
    if let Ok(out) = Command::new("arp")
        .creation_flags(CREATE_NO_WINDOW)
        .arg("-a")
        .output()
    {
        let text = String::from_utf8_lossy(&out.stdout);
        for line in text.lines() {
            if line.contains("dynamic") {
                if let Some(ip_tok) = line.split_whitespace().next() {
                    if ip_tok.chars().filter(|&c| c == '.').count() == 3
                        && !ip_tok.ends_with(".255")
                        && !ip_tok.starts_with("224.")
                        && ip_tok != "127.0.0.1"
                        && ip_tok != my_wifi
                    {
                        candidates.entry(ip_tok.to_string()).or_insert_with(|| {
                            (
                                "Wi-Fi / LAN Direct TCP".to_string(),
                                format!("lan-{}", ip_tok),
                            )
                        });
                    }
                }
            }
        }
    }

    let mut handles = Vec::new();
    for (ip, (net_type, host)) in candidates {
        handles.push(tokio::spawn(async move {
            let addr_str = format!("{}:{}", ip, port);
            let sock_addr: SocketAddr = addr_str.parse().ok()?;
            let t0 = Instant::now();
            match tokio::time::timeout(Duration::from_millis(350), TcpStream::connect(sock_addr))
                .await
            {
                Ok(Ok(_)) => {
                    let rtt = (t0.elapsed().as_secs_f64() * 1000.0 * 100.0).round() / 100.0;
                    Some(json!({
                        "endpoint": addr_str,
                        "ip": ip,
                        "port": port,
                        "type": net_type,
                        "hostname": host,
                        "rtt_ms": rtt
                    }))
                }
                _ => None,
            }
        }));
    }

    let mut found = Vec::new();
    for h in handles {
        if let Ok(Some(hit)) = h.await {
            found.push(hit);
        }
    }
    found
}

fn push_log(logs: &Arc<Mutex<Vec<String>>>, line: String) {
    if let Ok(mut guard) = logs.lock() {
        guard.push(line);
        if guard.len() > 120 {
            let drain_count = guard.len() - 120;
            guard.drain(0..drain_count);
        }
    }
}

fn stop_child_runtime(rt: &mut RuntimeProcess) {
    if let Some(mut old) = rt.child.take() {
        let _ = Command::new("taskkill")
            .creation_flags(CREATE_NO_WINDOW)
            .args(["/F", "/T", "/PID", &old.id().to_string()])
            .output();
        let _ = old.kill();
        let _ = old.wait();
    }
    // Dropping the Job Object immediately guarantees kernel reclamation of any nested processes/VRAM
    rt.job.take();
}

fn validate_peers_list(raw_peers: &str) -> Result<String, String> {
    let trimmed = raw_peers.trim();
    if trimmed.is_empty() {
        return Ok(String::new());
    }
    let mut valid = Vec::new();
    for part in trimmed.split(',') {
        let p = part.trim();
        if p.is_empty() {
            continue;
        }
        let addr: SocketAddr = p
            .parse()
            .map_err(|_| format!("Invalid peer socket address '{}'. Expected IP:PORT.", p))?;
        valid.push(addr.to_string());
    }
    Ok(valid.join(","))
}

fn spawn_role_process(
    install_dir: &Path,
    data_dir: &Path,
    rt: &mut RuntimeProcess,
    role: &str,
    model: &str,
    peers: &str,
) -> Result<(), String> {
    if !matches!(role, "worker" | "coordinator" | "all") {
        return Err(format!("Unsupported role '{}'", role));
    }

    let safe_peers = validate_peers_list(peers)?;
    let model_path = aeromesh_core::resolve_confined_model_path(model)
        .map_err(|e| e.to_string())?;
    let model_arg = model_path.to_string_lossy().to_string();

    stop_child_runtime(rt);

    let engine_exe = [
        install_dir.join("aeromesh.exe"),
        install_dir.join("bin").join("aeromesh.exe"),
        install_dir.join("target").join("release").join("aeromesh.exe"),
    ]
    .into_iter()
    .find(|p| p.is_file())
    .unwrap_or_else(|| PathBuf::from("aeromesh.exe"));

    let mut cmd = Command::new(&engine_exe);
    cmd.creation_flags(CREATE_NO_WINDOW);
    cmd.current_dir(data_dir);
    if role == "worker" {
        cmd.args([
            "start",
            "--role",
            "worker",
            "--port",
            "50052",
            "--model",
            &model_arg,
        ]);
    } else if role == "coordinator" {
        if safe_peers.is_empty() {
            cmd.args([
                "start",
                "--role",
                "coordinator",
                "--model",
                &model_arg,
                "--port",
                "8080",
            ]);
        } else {
            cmd.args([
                "start",
                "--role",
                "coordinator",
                "--model",
                &model_arg,
                "--port",
                "8080",
                "--peers",
                &safe_peers,
            ]);
        }
    } else {
        cmd.args([
            "start",
            "--role",
            "all",
            "--model",
            &model_arg,
            "--port",
            "8080",
        ]);
    }

    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("Failed to spawn {}: {}", engine_exe.display(), e))?;

    // Assign child process to a Win32 Job Object so closing or crashing the desktop app
    // immediately terminates aeromesh.exe and frees GPU VRAM.
    let job = match SafeProcessJob::new() {
        Ok(j) => {
            let _ = j.assign_child(&child);
            Some(j)
        }
        Err(_) => None,
    };

    let logs_ref = rt.logs.clone();
    if let Ok(mut g) = logs_ref.lock() {
        g.clear();
        g.push(format!(
            "[AeroMesh Studio] Started role '{}' with model '{}'",
            role.to_uppercase(),
            model_path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| model.to_string())
        ));
    }

    if let Some(stdout) = child.stdout.take() {
        let logs_out = logs_ref.clone();
        std::thread::spawn(move || {
            let reader = StdBufReader::new(stdout);
            for line in reader.lines().map_while(Result::ok) {
                println!("{}", line);
                push_log(&logs_out, line);
            }
        });
    }
    if let Some(stderr) = child.stderr.take() {
        let logs_err = logs_ref.clone();
        std::thread::spawn(move || {
            let reader = StdBufReader::new(stderr);
            for line in reader.lines().map_while(Result::ok) {
                eprintln!("{}", line);
                push_log(&logs_err, line);
            }
        });
    }

    rt.role = role.to_string();
    rt.model = model_path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or(model_arg);
    rt.peers = safe_peers;
    rt.child = Some(child);
    rt.job = job;
    Ok(())
}

fn list_discovered_models() -> (Vec<String>, Vec<String>) {
    let mut gguf_seen = HashSet::new();
    let mut img_seen = HashSet::new();
    let mut gguf_models = Vec::new();
    let mut image_models = Vec::new();

    for dir in aeromesh_core::model_search_dirs() {
        if let Ok(entries) = fs::read_dir(&dir) {
            for e in entries.flatten() {
                let name = e.file_name().to_string_lossy().to_string();
                if name.ends_with(".gguf")
                    && !name.contains("_stage")
                    && !name.starts_with(".aeromesh_")
                {
                    if gguf_seen.insert(name.clone()) {
                        gguf_models.push(name);
                    }
                } else if name.ends_with(".safetensors") && img_seen.insert(name.clone()) {
                    image_models.push(name);
                }
            }
        }
    }

    gguf_models.sort();
    image_models.sort();
    (gguf_models, image_models)
}

async fn get_node_state(State(st): State<Arc<DesktopState>>) -> impl IntoResponse {
    let (role, model, peers, running, logs) = {
        let mut rt = st.runtime.lock().unwrap();
        let is_alive = if let Some(ref mut c) = rt.child {
            matches!(c.try_wait(), Ok(None))
        } else {
            false
        };
        if !is_alive && rt.role != "idle" {
            rt.role = "idle".to_string();
        }
        let log_snapshot = rt.logs.lock().map(|g| g.clone()).unwrap_or_default();
        (
            rt.role.clone(),
            rt.model.clone(),
            rt.peers.clone(),
            is_alive,
            log_snapshot,
        )
    };

    let (gguf_models, image_models) = list_discovered_models();

    Json(json!({
        "role": role,
        "running": running,
        "model": model,
        "peers": peers,
        "gpu": get_gpu_summary(),
        "tailscale_ip": get_tailscale_ip(),
        "wifi_ip": get_local_wifi_ip(),
        "models_dir": aeromesh_core::models_dir().to_string_lossy(),
        "models": gguf_models,
        "image_models": image_models,
        "logs": logs
    }))
}

async fn post_node_start(
    State(st): State<Arc<DesktopState>>,
    Json(req): Json<Value>,
) -> impl IntoResponse {
    let mut role = req
        .get("role")
        .and_then(|r| r.as_str())
        .unwrap_or("all")
        .trim()
        .to_lowercase();
    let model = req
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("")
        .to_string();
    let mut peers = req
        .get("peers")
        .and_then(|p| p.as_str())
        .unwrap_or("")
        .to_string();

    if role == "auto" {
        let discovered = discover_workers(50052).await;
        if !discovered.is_empty() {
            let eps: Vec<String> = discovered
                .iter()
                .filter_map(|d| {
                    d.get("endpoint")
                        .and_then(|e| e.as_str())
                        .map(|s| s.to_string())
                })
                .collect();
            peers = eps.join(",");
            role = "coordinator".to_string();
        } else {
            role = "all".to_string();
        }
    }

    let res = {
        let mut rt = st.runtime.lock().unwrap();
        spawn_role_process(&st.install_dir, &st.data_dir, &mut rt, &role, &model, &peers)
    };

    match res {
        Ok(()) => Json(json!({ "status": "ok", "role": role, "model": model, "peers": peers })),
        Err(e) => Json(json!({ "status": "error", "message": e })),
    }
}

async fn post_node_stop(State(st): State<Arc<DesktopState>>) -> impl IntoResponse {
    let mut rt = st.runtime.lock().unwrap();
    stop_child_runtime(&mut rt);
    rt.role = "idle".to_string();
    push_log(
        &rt.logs,
        "[AeroMesh Studio] Node stopped by user.".to_string(),
    );
    Json(json!({ "status": "stopped", "role": "idle" }))
}

fn decode_chunked_body(raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos < raw.len() {
        let Some(rel_crlf) = raw[pos..].windows(2).position(|w| w == b"\r\n") else {
            break;
        };
        let size_line = String::from_utf8_lossy(&raw[pos..pos + rel_crlf]);
        let size_hex = size_line.split(';').next().unwrap_or("").trim();
        let Ok(chunk_len) = usize::from_str_radix(size_hex, 16) else {
            break;
        };
        pos += rel_crlf + 2;
        if chunk_len == 0 {
            break;
        }
        let end = (pos + chunk_len).min(raw.len());
        out.extend_from_slice(&raw[pos..end]);
        pos = end;
        if pos + 2 <= raw.len() && &raw[pos..pos + 2] == b"\r\n" {
            pos += 2;
        }
    }
    out
}

async fn backend_http_json(
    method: &str,
    path: &str,
    api_key: &str,
    body: Option<&[u8]>,
    read_timeout: Duration,
) -> Result<(u16, Vec<u8>), String> {
    let mut stream = tokio::time::timeout(
        Duration::from_millis(1500),
        TcpStream::connect("127.0.0.1:8080"),
    )
    .await
    .map_err(|_| "Timeout connecting to 127.0.0.1:8080".to_string())?
    .map_err(|e| e.to_string())?;

    let payload = body.unwrap_or(&[]);
    let req_head = format!(
        "{} {} HTTP/1.1\r\nHost: 127.0.0.1:8080\r\nAuthorization: Bearer {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        method,
        path,
        api_key,
        payload.len()
    );
    stream
        .write_all(req_head.as_bytes())
        .await
        .map_err(|e| e.to_string())?;
    if !payload.is_empty() {
        stream
            .write_all(payload)
            .await
            .map_err(|e| e.to_string())?;
    }

    let mut raw = Vec::new();
    tokio::time::timeout(read_timeout, stream.read_to_end(&mut raw))
        .await
        .map_err(|_| "Timed out waiting for coordinator response".to_string())?
        .map_err(|e| e.to_string())?;

    let Some(sep) = raw.windows(4).position(|w| w == b"\r\n\r\n") else {
        return Err("Malformed HTTP response from coordinator".to_string());
    };
    let head_str = String::from_utf8_lossy(&raw[..sep]);
    let status = head_str
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|s| s.parse::<u16>().ok())
        .unwrap_or(502);
    let is_chunked = head_str.lines().any(|l| {
        let lower = l.to_ascii_lowercase();
        lower.starts_with("transfer-encoding:") && lower.contains("chunked")
    });
    let raw_body = &raw[sep + 4..];
    let body_bytes = if is_chunked {
        decode_chunked_body(raw_body)
    } else {
        raw_body.to_vec()
    };
    Ok((status, body_bytes))
}

async fn serve_index(State(st): State<Arc<DesktopState>>) -> impl IntoResponse {
    let p = st.install_dir.join("static").join("index.html");
    let mut html = fs::read_to_string(p)
        .unwrap_or_else(|_| "<h1>AeroMesh UI missing static/index.html</h1>".to_string());
    let inject = format!(
        "<script>window.AEROMESH_API_BASE = 'http://127.0.0.1:{}';</script><base href=\"http://127.0.0.1:{}/\">",
        st.port, st.port
    );
    if let Some(pos) = html.find("<head>") {
        html.insert_str(pos + 6, &inject);
    }
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/html; charset=utf-8")
        .header(header::CACHE_CONTROL, "no-cache, no-store, must-revalidate")
        .body(Body::from(html))
        .unwrap()
}

async fn serve_static(
    State(st): State<Arc<DesktopState>>,
    AxumPath(rel): AxumPath<String>,
) -> impl IntoResponse {
    let static_root = st.install_dir.join("static");
    let Some(full) = aeromesh_core::resolve_confined_static_path(&static_root, &rel) else {
        return Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(Body::from("Not Found"))
            .unwrap();
    };
    let ext = full
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let mime = match ext.as_str() {
        "css" => "text/css; charset=utf-8",
        "js" | "mjs" => "application/javascript; charset=utf-8",
        "json" => "application/json; charset=utf-8",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "svg" => "image/svg+xml",
        "ico" => "image/x-icon",
        "woff2" => "font/woff2",
        "woff" => "font/woff",
        "ttf" => "font/ttf",
        _ => "application/octet-stream",
    };
    match fs::read(&full) {
        Ok(bytes) => Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, mime)
            .header(header::CACHE_CONTROL, "no-cache")
            .body(Body::from(bytes))
            .unwrap(),
        Err(_) => Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(Body::from("Not Found"))
            .unwrap(),
    }
}

async fn get_models(State(st): State<Arc<DesktopState>>) -> impl IntoResponse {
    if let Ok((200, bytes)) = backend_http_json(
        "GET",
        "/v1/models",
        &st.api_key,
        None,
        Duration::from_secs(5),
    )
    .await
    {
        if let Ok(v) = serde_json::from_slice::<Value>(&bytes) {
            return Json(v);
        }
    }
    let (gguf_models, _) = list_discovered_models();
    let now_secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let list: Vec<Value> = gguf_models
        .into_iter()
        .enumerate()
        .map(|(idx, name)| {
            json!({
                "id": name,
                "object": "model",
                "created": now_secs,
                "owned_by": "local_disk",
                "is_active": idx == 0
            })
        })
        .collect();
    Json(json!({ "object": "list", "data": list }))
}

async fn get_cluster_status(State(st): State<Arc<DesktopState>>) -> impl IntoResponse {
    if let Ok((200, bytes)) = backend_http_json(
        "GET",
        "/api/cluster/status",
        &st.api_key,
        None,
        Duration::from_secs(5),
    )
    .await
    {
        if let Ok(v) = serde_json::from_slice::<Value>(&bytes) {
            return Json(v);
        }
    }
    let role = st
        .runtime
        .lock()
        .map(|r| r.role.clone())
        .unwrap_or_else(|_| "idle".to_string());
    Json(json!({
        "state": if role == "worker" { "worker_mode" } else { "offline" },
        "status": "offline",
        "connected": false,
        "role": role,
        "coordinator_endpoint": "http://127.0.0.1:8080"
    }))
}

async fn get_cluster_discover(State(st): State<Arc<DesktopState>>) -> impl IntoResponse {
    let peers = discover_workers(50052).await;
    let (sd_bin, sd_model) = find_sd_assets(&st.install_dir);
    Json(json!({
        "discovered_workers": peers,
        "count": peers.len(),
        "diffusion_engine_ready": sd_bin.is_some() && sd_model.is_some(),
        "diffusion_model": sd_model.and_then(|p| p.file_name().map(|n| n.to_string_lossy().to_string()))
    }))
}

async fn post_chat_abort(State(st): State<Arc<DesktopState>>) -> impl IntoResponse {
    let _ = backend_http_json(
        "POST",
        "/api/chat/abort",
        &st.api_key,
        Some(b"{}"),
        Duration::from_secs(5),
    )
    .await;
    Json(json!({ "status": "aborted" }))
}

async fn post_model_switch(State(st): State<Arc<DesktopState>>, body: Bytes) -> impl IntoResponse {
    match backend_http_json(
        "POST",
        "/api/model/switch",
        &st.api_key,
        Some(&body),
        Duration::from_secs(180),
    )
    .await
    {
        Ok((code, resp)) => Response::builder()
            .status(StatusCode::from_u16(code).unwrap_or(StatusCode::OK))
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(resp))
            .unwrap(),
        Err(e) => Response::builder()
            .status(StatusCode::BAD_GATEWAY)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(json!({ "error": e }).to_string()))
            .unwrap(),
    }
}

async fn post_chat_stream(State(st): State<Arc<DesktopState>>, body: Bytes) -> impl IntoResponse {
    let api_key = st.api_key.clone();
    let (tx, rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(32);

    tokio::spawn(async move {
        let conn = tokio::time::timeout(
            Duration::from_secs(5),
            TcpStream::connect("127.0.0.1:8080"),
        )
        .await;
        let mut stream = match conn {
            Ok(Ok(s)) => s,
            _ => {
                let msg = json!({"choices":[{"index":0,"delta":{"content":"Coordinator is warming up or stopped. Open Node Control in the top bar to start Coordinator or Local Mesh."}}]}).to_string();
                let _ = tx.send(Ok(Bytes::from(format!("data: {}\n\n", msg)))).await;
                let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n"))).await;
                return;
            }
        };

        let req_head = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:8080\r\nAuthorization: Bearer {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            api_key,
            body.len()
        );
        if stream.write_all(req_head.as_bytes()).await.is_err()
            || stream.write_all(&body).await.is_err()
        {
            return;
        }

        let mut reader = BufReader::new(stream);
        let mut status_line = String::new();
        if reader.read_line(&mut status_line).await.unwrap_or(0) == 0 {
            return;
        }
        let status_code = status_line
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse::<u16>().ok())
            .unwrap_or(502);

        let mut is_chunked = false;
        let mut line = String::new();
        loop {
            line.clear();
            if reader.read_line(&mut line).await.unwrap_or(0) == 0 || line == "\r\n" || line == "\n"
            {
                break;
            }
            let lower = line.to_ascii_lowercase();
            if lower.starts_with("transfer-encoding:") && lower.contains("chunked") {
                is_chunked = true;
            }
        }

        if status_code != 200 {
            let mut err_buf = Vec::new();
            let _ = reader.read_to_end(&mut err_buf).await;
            let err_bytes = if is_chunked {
                decode_chunked_body(&err_buf)
            } else {
                err_buf
            };
            let err_text = String::from_utf8_lossy(&err_bytes).trim().to_string();
            let msg = json!({"choices":[{"index":0,"delta":{"content": format!("Coordinator returned HTTP {}: {}", status_code, err_text)}}]}).to_string();
            let _ = tx.send(Ok(Bytes::from(format!("data: {}\n\n", msg)))).await;
            let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n"))).await;
            return;
        }

        if is_chunked {
            let mut pending = String::new();
            loop {
                let mut size_line = String::new();
                if reader.read_line(&mut size_line).await.unwrap_or(0) == 0 {
                    break;
                }
                let size_hex = size_line.split(';').next().unwrap_or("").trim();
                let Ok(chunk_len) = usize::from_str_radix(size_hex, 16) else {
                    break;
                };
                if chunk_len == 0 {
                    break;
                }
                let mut chunk = vec![0u8; chunk_len];
                if reader.read_exact(&mut chunk).await.is_err() {
                    break;
                }
                let mut crlf = [0u8; 2];
                let _ = reader.read_exact(&mut crlf).await;

                pending.push_str(&String::from_utf8_lossy(&chunk));
                while let Some(idx) = pending.find('\n') {
                    let row = pending[..idx].trim().to_string();
                    pending.drain(..=idx);
                    if row.starts_with("data:")
                        && tx
                            .send(Ok(Bytes::from(format!("{}\n\n", row))))
                            .await
                            .is_err()
                    {
                        return;
                    }
                }
            }
        } else {
            loop {
                line.clear();
                match reader.read_line(&mut line).await {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {
                        let trimmed = line.trim();
                        if trimmed.starts_with("data:")
                            && tx
                                .send(Ok(Bytes::from(format!("{}\n\n", trimmed))))
                                .await
                                .is_err()
                        {
                            break;
                        }
                    }
                }
            }
        }
    });

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(ReceiverStream::new(rx)))
        .unwrap()
}

fn find_sd_assets(install_dir: &Path) -> (Option<PathBuf>, Option<PathBuf>) {
    let sd_candidates = [
        install_dir.join("bin").join("diffusion").join("sd-cli.exe"),
        install_dir.join("bin").join("sd-cli.exe"),
        install_dir.join("bin").join("sd.exe"),
    ];
    let sd_bin = sd_candidates.into_iter().find(|p| p.is_file());
    let mut safetensors: Vec<PathBuf> = Vec::new();
    for dir in aeromesh_core::model_search_dirs() {
        if let Ok(entries) = fs::read_dir(&dir) {
            for e in entries.flatten() {
                let p = e.path();
                if p.is_file() && p.extension().and_then(|x| x.to_str()) == Some("safetensors") {
                    safetensors.push(p);
                }
            }
        }
    }
    safetensors.sort();
    let sd_model = safetensors
        .iter()
        .find(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_lowercase()
                .contains("dreamshaper")
        })
        .cloned()
        .or_else(|| safetensors.into_iter().next());
    (sd_bin, sd_model)
}

async fn post_image_stream(
    State(st): State<Arc<DesktopState>>,
    Json(req): Json<Value>,
) -> impl IntoResponse {
    let prompt = req
        .get("prompt")
        .and_then(|p| p.as_str())
        .unwrap_or("cyberpunk city")
        .to_string();
    let steps = req
        .get("steps")
        .and_then(|v| v.as_u64())
        .unwrap_or(12)
        .clamp(1, 50) as usize;
    let width = ((req.get("width").and_then(|v| v.as_u64()).unwrap_or(512).clamp(256, 1024) / 64) * 64) as usize;
    let height = ((req.get("height").and_then(|v| v.as_u64()).unwrap_or(512).clamp(256, 1024) / 64) * 64) as usize;

    let install_dir = st.install_dir.clone();
    let api_key = st.api_key.clone();
    let (tx, rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(32);

    tokio::spawn(async move {
        let t0 = Instant::now();
        let mut mesh_active = false;
        let mut transport = "Local GPU/CPU Stable Diffusion (sd-cli)".to_string();

        if let Ok((200, bytes)) = backend_http_json(
            "GET",
            "/api/cluster/status",
            &api_key,
            None,
            Duration::from_secs(3),
        )
        .await
        {
            if let Ok(v) = serde_json::from_slice::<Value>(&bytes) {
                mesh_active = v.get("connected").and_then(|c| c.as_bool()).unwrap_or(false);
                if let Some(t) = v.get("transport").and_then(|t| t.as_str()) {
                    transport = format!("Local sd-cli + LLM Prompt Conditioning ({})", t);
                }
            }
        }

        let mut enhanced_prompt = prompt.clone();
        if mesh_active {
            let cond_body = json!({
                "messages": [{"role": "user", "content": format!("Write a 12-word vivid visual lighting and lens description for: {}", prompt)}],
                "max_tokens": 24,
                "temperature": 0.6,
                "stream": false
            })
            .to_string();
            if let Ok((200, resp_bytes)) = backend_http_json(
                "POST",
                "/v1/chat/completions",
                &api_key,
                Some(cond_body.as_bytes()),
                Duration::from_secs(20),
            )
            .await
            {
                if let Ok(v) = serde_json::from_slice::<Value>(&resp_bytes) {
                    if let Some(ext) = v
                        .pointer("/choices/0/message/content")
                        .and_then(|c| c.as_str())
                    {
                        let clean = ext.trim();
                        if !clean.is_empty() {
                            enhanced_prompt = format!("{}, {}", prompt, clean);
                        }
                    }
                }
            }
        }

        let (sd_bin, sd_model) = find_sd_assets(&install_dir);
        let model_label = sd_model
            .as_ref()
            .and_then(|p| p.file_name())
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "Not Installed".to_string());

        let cache_dir = aeromesh_core::cache_dir();
        let _ = fs::create_dir_all(&cache_dir);
        let out_png = cache_dir.join(format!(
            ".aeromesh_sd_{}_{}.png",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis()
        ));

        let Some((bin, model)) = sd_bin.zip(sd_model) else {
            let final_evt = json!({
                "type": "dit_complete",
                "prompt": prompt,
                "conditioning": enhanced_prompt,
                "mesh_active": mesh_active,
                "model_name": model_label,
                "steps": steps,
                "width": width,
                "height": height,
                "accum_wire_mb": 0.0,
                "weights_on_wire_mb": 0.0,
                "elapsed_s": (t0.elapsed().as_secs_f64() * 100.0).round() / 100.0,
                "b64_png": Value::Null,
                "sd_error": "Stable Diffusion engine (sd-cli.exe) or a .safetensors checkpoint is not installed. Select the Stable Diffusion component in the AeroMESH installer or place a checkpoint in %LOCALAPPDATA%\\AeroMesh\\models.",
                "transport": "Local sd-cli (Not Installed)"
            });
            let _ = tx
                .send(Ok(Bytes::from(format!("data: {}\n\n", final_evt))))
                .await;
            let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n"))).await;
            return;
        };

        let start_evt = json!({
            "type": "dit_step",
            "step": 1,
            "total_steps": steps,
            "mesh_active": mesh_active,
            "stage1_blocks": "Local UNet + CLIP Conditioning",
            "stage2_blocks": format!("Local FP16 VAE ({})", model_label),
            "step_wire_mb": 0.0,
            "accum_wire_mb": 0.0,
            "elapsed_s": (t0.elapsed().as_secs_f64() * 100.0).round() / 100.0,
            "transport": transport,
            "conditioning": enhanced_prompt,
            "model_name": model_label
        });
        let _ = tx
            .send(Ok(Bytes::from(format!("data: {}\n\n", start_evt))))
            .await;

        let p_clone = enhanced_prompt.clone();
        let out_clone = out_png.clone();
        let steps_str = steps.to_string();
        let width_str = width.to_string();
        let height_str = height.to_string();

        let sd_res = tokio::task::spawn_blocking(move || {
            let mut cmd = Command::new(&bin);
            cmd.creation_flags(CREATE_NO_WINDOW);
            cmd.current_dir(bin.parent().unwrap_or_else(|| Path::new(".")))
                .args([
                    "-m",
                    &model.to_string_lossy(),
                    "--type",
                    "f16",
                    "-p",
                    &p_clone,
                    "-n",
                    "blurry, out of focus, low quality, deformed, watermark, text",
                    "--cfg-scale",
                    "7.0",
                    "--sampling-method",
                    "euler_a",
                    "--steps",
                    &steps_str,
                    "-W",
                    &width_str,
                    "-H",
                    &height_str,
                    "-o",
                    &out_clone.to_string_lossy(),
                ]);
            cmd.output()
        })
        .await;

        let mut b64_png = None;
        let mut sd_error = None;
        match sd_res {
            Ok(Ok(out)) => {
                if out_png.exists() {
                    if let Ok(bytes) = fs::read(&out_png) {
                        b64_png = Some(base64_encode(&bytes));
                    }
                    let _ = fs::remove_file(&out_png);
                } else {
                    let msg = format!(
                        "{}{}",
                        String::from_utf8_lossy(&out.stdout),
                        String::from_utf8_lossy(&out.stderr)
                    );
                    let tail: String = msg
                        .chars()
                        .rev()
                        .take(240)
                        .collect::<String>()
                        .chars()
                        .rev()
                        .collect();
                    sd_error = Some(tail.trim().to_string());
                }
            }
            Ok(Err(e)) => sd_error = Some(format!("Failed to launch sd-cli.exe: {}", e)),
            Err(e) => sd_error = Some(format!("Diffusion task failed: {}", e)),
        }

        let mut hasher = Sha256::new();
        hasher.update(enhanced_prompt.as_bytes());
        let seed_hex = hex::encode(hasher.finalize())[..12].to_string();

        let final_evt = json!({
            "type": "dit_complete",
            "prompt": prompt,
            "conditioning": enhanced_prompt,
            "mesh_active": mesh_active,
            "model_name": model_label,
            "steps": steps,
            "width": width,
            "height": height,
            "accum_wire_mb": 0.0,
            "weights_on_wire_mb": 0.0,
            "elapsed_s": (t0.elapsed().as_secs_f64() * 100.0).round() / 100.0,
            "seed": seed_hex,
            "b64_png": b64_png,
            "sd_error": sd_error,
            "transport": transport
        });
        let _ = tx
            .send(Ok(Bytes::from(format!("data: {}\n\n", final_evt))))
            .await;
        let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n"))).await;
    });

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(ReceiverStream::new(rx)))
        .unwrap()
}

fn spawn_desktop_window(data_dir: &Path, url: &str) -> Option<Child> {
    let profile_dir = data_dir.join("app_profile");
    let _ = fs::create_dir_all(&profile_dir);

    let mut browser_paths: Vec<PathBuf> = vec![
        PathBuf::from(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        PathBuf::from(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
        PathBuf::from(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        PathBuf::from(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    ];
    if let Ok(local_app) = env::var("LOCALAPPDATA") {
        let base = PathBuf::from(local_app);
        browser_paths.push(base.join(r"Microsoft\Edge\Application\msedge.exe"));
        browser_paths.push(base.join(r"Google\Chrome\Application\chrome.exe"));
    }

    for browser in browser_paths {
        if browser.is_file() {
            if let Ok(child) = Command::new(&browser)
                .creation_flags(CREATE_NO_WINDOW)
                .args([
                    &format!("--user-data-dir={}", profile_dir.display()),
                    "--no-first-run",
                    "--no-default-browser-check",
                    &format!("--app={}", url),
                    "--window-size=1460,940",
                ])
                .spawn()
            {
                return Some(child);
            }
        }
    }
    let _ = Command::new("cmd")
        .creation_flags(CREATE_NO_WINDOW)
        .args(["/C", "start", "", url])
        .spawn();
    None
}

fn build_local_cors_layer(port: u16) -> CorsLayer {
    let mut allowed_origins = vec![
        format!("http://127.0.0.1:{}", port)
            .parse::<HeaderValue>()
            .expect("valid loopback origin"),
        format!("http://localhost:{}", port)
            .parse::<HeaderValue>()
            .expect("valid localhost origin"),
    ];
    if let Ok(custom) = env::var("AEROMESH_ALLOWED_ORIGIN") {
        for origin in custom.split(',') {
            let trimmed = origin.trim();
            if !trimmed.is_empty() {
                if let Ok(val) = trimmed.parse::<HeaderValue>() {
                    allowed_origins.push(val);
                }
            }
        }
    }
    CorsLayer::new()
        .allow_origin(allowed_origins)
        .allow_methods(Any)
        .allow_headers(Any)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let install_dir = aeromesh_core::install_dir();
    let data_dir = aeromesh_core::data_dir();
    let _ = aeromesh_core::ensure_data_dirs();

    // Prepend installed runtime DLL directories cleanly (no hardcoded dev toolchains)
    let bin_path = install_dir.join("bin");
    let diff_path = bin_path.join("diffusion");
    let current_path = env::var("PATH").unwrap_or_default();
    env::set_var(
        "PATH",
        format!(
            "{};{};{};{}",
            install_dir.display(),
            bin_path.display(),
            diff_path.display(),
            current_path
        ),
    );

    let (api_key, _) = ensure_env_secrets();

    println!("========================================================================");
    println!("   AEROMESH UNIFIED DESKTOP STUDIO (COORDINATOR + WORKER + UI)          ");
    println!("========================================================================");

    let (initial_models, _) = list_discovered_models();
    let default_model = initial_models
        .into_iter()
        .next()
        .unwrap_or_else(|| "test.gguf".to_string());

    let runtime = Arc::new(Mutex::new(RuntimeProcess {
        role: "idle".to_string(),
        model: default_model,
        peers: String::new(),
        child: None,
        job: None,
        logs: Arc::new(Mutex::new(vec![
            "[AeroMesh Studio] Ready. Select your node role (Coordinator, Worker, or Local Mesh) in the window.".to_string(),
        ])),
    }));

    // Find available loopback bind port (7860..=7875)
    let mut bound_listener = None;
    let mut bound_port = 7860;
    for test_port in 7860..=7875 {
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), test_port);
        if let Ok(listener) = tokio::net::TcpListener::bind(addr).await {
            bound_listener = Some(listener);
            bound_port = test_port;
            break;
        }
    }

    let state = Arc::new(DesktopState {
        install_dir: install_dir.clone(),
        data_dir: data_dir.clone(),
        api_key: api_key.clone(),
        port: bound_port,
        runtime: runtime.clone(),
    });

    let cors = build_local_cors_layer(bound_port);

    let app = Router::new()
        .route("/", get(serve_index))
        .route("/static/*path", get(serve_static))
        .route("/health", get(|| async { Json(json!({"gateway": "ok"})) }))
        .route("/v1/models", get(get_models))
        .route("/api/cluster/status", get(get_cluster_status))
        .route("/api/cluster/discover", get(get_cluster_discover))
        .route("/api/node/state", get(get_node_state))
        .route("/api/node/start", post(post_node_start))
        .route("/api/node/stop", post(post_node_stop))
        .route("/api/chat/abort", post(post_chat_abort))
        .route("/api/model/switch", post(post_model_switch))
        .route("/v1/models/load", post(post_model_switch))
        .route("/api/chat/stream", post(post_chat_stream))
        .route("/v1/chat/completions", post(post_chat_stream))
        .route("/api/chat", post(post_chat_stream))
        .route("/api/image/stream", post(post_image_stream))
        .route("/v1/images/generations", post(post_image_stream))
        .layer(cors)
        .with_state(state);

    let target_url = format!("http://127.0.0.1:{}", bound_port);
    if let Some(listener) = bound_listener {
        println!("[AeroMesh Studio] Running unified server on {}", target_url);
        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();

        if let Some(mut window_child) = spawn_desktop_window(&data_dir, &target_url) {
            // Monitor desktop window exit so closing the window cleanly shuts down the
            // HTTP server and terminates aeromesh.exe (freeing GPU VRAM instead of orphaning it).
            std::thread::spawn(move || {
                let _ = window_child.wait();
                let _ = shutdown_tx.send(());
            });
            let _ = axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    let _ = shutdown_rx.await;
                })
                .await;
        } else {
            // Fallback to OS default browser: keep server alive until Ctrl-C / termination signal.
            let _ = axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    let _ = tokio::signal::ctrl_c().await;
                })
                .await;
        }
    } else {
        let _ = spawn_desktop_window(&data_dir, &target_url);
    }

    if let Ok(mut rt) = runtime.lock() {
        stop_child_runtime(&mut rt);
    }
    Ok(())
}
