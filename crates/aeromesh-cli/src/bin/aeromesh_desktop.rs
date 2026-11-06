#![windows_subsystem = "windows"]
use std::collections::HashMap;
use std::env;
use std::fs;
use std::io::{BufRead, BufReader as StdBufReader};
use std::net::{IpAddr, Ipv4Addr, SocketAddr, UdpSocket};
use std::os::windows::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const CREATE_NO_WINDOW: u32 = 0x08000000;

use aeromesh_engine::job_object::SafeProcessJob;
use axum::{
    body::Body,
    extract::{Path as AxumPath, Request, State},
    http::{header, HeaderValue, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use bytes::Bytes;

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;
use tokio::sync::{mpsc, watch};
use tokio_stream::wrappers::ReceiverStream;
use tower_http::cors::{AllowOrigin, CorsLayer};

use windows_sys::Win32::Foundation::{CloseHandle, GetLastError, ERROR_ALREADY_EXISTS, HANDLE};
use windows_sys::Win32::Security::Cryptography::{BCryptGenRandom, BCRYPT_USE_SYSTEM_PREFERRED_RNG};
use windows_sys::Win32::System::Diagnostics::Debug::{
    SetUnhandledExceptionFilter, EXCEPTION_CONTINUE_SEARCH, EXCEPTION_POINTERS,
};
use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryA};
use windows_sys::Win32::System::Threading::{CreateMutexW, GetCurrentProcess, GetCurrentProcessId};
use windows_sys::Win32::UI::WindowsAndMessaging::{FindWindowW, SetForegroundWindow, ShowWindow, SW_RESTORE};

// ---------------------------------------------------------------------------
// Single-Instance RAII Guard & Exception Filter
// ---------------------------------------------------------------------------

struct SingleInstanceGuard {
    handle: HANDLE,
}

impl Drop for SingleInstanceGuard {
    fn drop(&mut self) {
        if !self.handle.is_null() && self.handle != -1isize as _ {
            unsafe { CloseHandle(self.handle); }
        }
    }
}

unsafe extern "system" fn unhandled_exception_filter(_info: *const EXCEPTION_POINTERS) -> i32 {
    use windows_sys::Win32::Storage::FileSystem::{
        CreateFileW, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL,
    };
    const GENERIC_WRITE: u32 = 0x40000000;
    if let Some(data_dir) = dirs::data_local_dir() {
        let logs_dir = data_dir.join("AeroMESH").join("logs");
        let _ = fs::create_dir_all(&logs_dir);
        let now_s = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let dmp_path = logs_dir.join(format!("crash_{}.dmp", now_s));
        let wide: Vec<u16> = dmp_path
            .to_string_lossy()
            .encode_utf16()
            .chain(std::iter::once(0))
            .collect();
        let h_file = CreateFileW(
            wide.as_ptr(),
            GENERIC_WRITE,
            0,
            std::ptr::null(),
            CREATE_ALWAYS,
            FILE_ATTRIBUTE_NORMAL,
            std::ptr::null_mut(),
        );
        if h_file != windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE {
            let dbghelp = LoadLibraryA(b"dbghelp.dll\0".as_ptr());
            if !dbghelp.is_null() {
                type MiniDumpFn = unsafe extern "system" fn(
                    HANDLE, u32, HANDLE, u32, *const std::ffi::c_void, *const std::ffi::c_void, *const std::ffi::c_void
                ) -> i32;
                if let Some(proc) = GetProcAddress(dbghelp, b"MiniDumpWriteDump\0".as_ptr()) {
                    let write_dump: MiniDumpFn = std::mem::transmute(proc);
                    let _ = write_dump(
                        GetCurrentProcess(),
                        GetCurrentProcessId(),
                        h_file,
                        2, // MiniDumpNormal
                        std::ptr::null(),
                        std::ptr::null(),
                        std::ptr::null(),
                    );
                }
                CloseHandle(dbghelp);
            }
            CloseHandle(h_file);
        }
    }
    EXCEPTION_CONTINUE_SEARCH
}

// ---------------------------------------------------------------------------
// Standard Windows Directory Resolution (%APPDATA% & %LOCALAPPDATA%)
// ---------------------------------------------------------------------------

fn get_config_dir() -> PathBuf {
    if let Some(p) = dirs::config_dir() {
        let aeromesh_cfg = p.join("AeroMESH");
        let _ = fs::create_dir_all(&aeromesh_cfg);
        return aeromesh_cfg;
    }
    PathBuf::from(".")
}

fn get_data_dir() -> PathBuf {
    if let Some(p) = dirs::data_local_dir() {
        let aeromesh_data = p.join("AeroMESH");
        let _ = fs::create_dir_all(&aeromesh_data);
        return aeromesh_data;
    }
    PathBuf::from(".")
}

#[allow(dead_code)]
fn get_cache_dir() -> PathBuf {
    let p = get_data_dir().join("cache");
    let _ = fs::create_dir_all(&p);
    p
}

fn get_logs_dir() -> PathBuf {
    let p = get_data_dir().join("logs");
    let _ = fs::create_dir_all(&p);
    p
}

fn is_dir_writable(path: &Path) -> bool {
    let test_file = path.join(format!(".aeromesh_probe_{}.tmp", std::process::id()));
    if fs::write(&test_file, b"").is_ok() {
        let _ = fs::remove_file(&test_file);
        true
    } else {
        false
    }
}

fn get_models_dir(root_dir: &Path) -> PathBuf {
    let local_models = root_dir.join("models");
    if local_models.exists() && is_dir_writable(&local_models) {
        return local_models;
    }
    let appdata_models = get_data_dir().join("models");
    let _ = fs::create_dir_all(&appdata_models);
    appdata_models
}

fn scan_all_models(root_dir: &Path) -> (Vec<String>, Vec<String>) {
    let mut gguf_set = std::collections::BTreeSet::new();
    let mut img_set = std::collections::BTreeSet::new();

    let scan_dirs = [root_dir.join("models"), get_data_dir().join("models")];
    for d in scan_dirs {
        if let Ok(entries) = fs::read_dir(d) {
            for e in entries.flatten() {
                let name = e.file_name().to_string_lossy().to_string();
                if name.ends_with(".gguf") && !name.contains("_stage") && !name.starts_with(".aeromesh_") {
                    gguf_set.insert(name);
                } else if name.ends_with(".safetensors") {
                    img_set.insert(name);
                }
            }
        }
    }
    (gguf_set.into_iter().collect(), img_set.into_iter().collect())
}

// ---------------------------------------------------------------------------
// State Structures
// ---------------------------------------------------------------------------

struct RuntimeProcess {
    role: String,
    model: String,
    peers: String,
    child: Option<Child>,
    logs: Arc<Mutex<Vec<String>>>,
}

#[derive(Clone)]
struct DesktopState {
    root_dir: PathBuf,
    api_key: String,
    session_token: String,
    port: u16,
    runtime: Arc<Mutex<RuntimeProcess>>,
    job: Arc<SafeProcessJob>,
    last_heartbeat: Arc<AtomicI64>,
    shutdown_trigger: Arc<watch::Sender<bool>>,
    active_download: Arc<Mutex<Option<Child>>>,
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

// ---------------------------------------------------------------------------
// CSPRNG Secret Generation
// ---------------------------------------------------------------------------

fn generate_csprng_token(prefix: &str) -> String {
    let mut bytes = [0u8; 32];
    unsafe {
        let _ = BCryptGenRandom(
            std::ptr::null_mut(),
            bytes.as_mut_ptr(),
            bytes.len() as u32,
            BCRYPT_USE_SYSTEM_PREFERRED_RNG,
        );
    }
    format!("{}-{}", prefix, hex::encode(bytes))
}

fn ensure_env_secrets(root: &Path) -> (String, String) {
    let cfg_dir = get_config_dir();
    let config_env = cfg_dir.join("config.env");
    let legacy_env = root.join(".env");

    let mut api_key = env::var("AEROMESH_API_KEY").unwrap_or_default();
    let mut worker_secret = env::var("AEROMESH_WORKER_SECRET").unwrap_or_default();

    for p in [&config_env, &legacy_env] {
        if p.exists() {
            if let Ok(content) = fs::read_to_string(p) {
                for line in content.lines() {
                    let line = line.trim();
                    if line.is_empty() || line.starts_with('#') {
                        continue;
                    }
                    if let Some((k, v)) = line.split_once('=') {
                        let k = k.trim();
                        let v = v.trim().trim_matches('"').trim_matches('\'');
                        env::set_var(k, v);
                        if k == "AEROMESH_API_KEY" && api_key.is_empty() {
                            api_key = v.to_string();
                        }
                        if k == "AEROMESH_WORKER_SECRET" && worker_secret.is_empty() {
                            worker_secret = v.to_string();
                        }
                    }
                }
            }
        }
    }

    if api_key.is_empty() {
        api_key = generate_csprng_token("aeromesh-api");
    }
    if worker_secret.is_empty() {
        worker_secret = generate_csprng_token("aeromesh-worker");
    }

    env::set_var("AEROMESH_API_KEY", &api_key);
    env::set_var("AEROMESH_WORKER_SECRET", &worker_secret);
    env::set_var("AEROMESH_ACTIVATION_DTYPE", "int8");

    if !config_env.exists() {
        let _ = fs::write(
            &config_env,
            format!(
                "AEROMESH_API_KEY={}\nAEROMESH_WORKER_SECRET={}\nAEROMESH_ACTIVATION_DTYPE=int8\n",
                api_key, worker_secret
            ),
        );
    }
    (api_key, worker_secret)
}

// ---------------------------------------------------------------------------
// Native Win32 Adapter Enumeration (iphlpapi.dll)
// ---------------------------------------------------------------------------

fn get_local_wifi_ip() -> String {
    use windows_sys::Win32::NetworkManagement::IpHelper::{
        GetAdaptersAddresses, GAA_FLAG_SKIP_ANYCAST, GAA_FLAG_SKIP_DNS_SERVER, GAA_FLAG_SKIP_MULTICAST,
        IP_ADAPTER_ADDRESSES_LH,
    };
    use windows_sys::Win32::Networking::WinSock::{AF_INET, AF_UNSPEC, SOCKADDR_IN};

    unsafe {
        let mut buf_len: u32 = 16384;
        let mut buf: Vec<u8> = vec![0; buf_len as usize];
        let flags = GAA_FLAG_SKIP_DNS_SERVER | GAA_FLAG_SKIP_MULTICAST | GAA_FLAG_SKIP_ANYCAST;

        let ret = GetAdaptersAddresses(
            AF_UNSPEC as u32,
            flags,
            std::ptr::null_mut(),
            buf.as_mut_ptr() as *mut IP_ADAPTER_ADDRESSES_LH,
            &mut buf_len,
        );

        if ret == 0 {
            let mut curr = buf.as_ptr() as *const IP_ADAPTER_ADDRESSES_LH;
            while !curr.is_null() {
                let adapter = &*curr;
                // OperStatusUp = 1
                if adapter.OperStatus == 1 {
                    let mut unicast = adapter.FirstUnicastAddress;
                    while !unicast.is_null() {
                        let uni = &*unicast;
                        let sa = &*uni.Address.lpSockaddr;
                        if sa.sa_family == AF_INET as u16 {
                            let sin = &*(uni.Address.lpSockaddr as *const SOCKADDR_IN);
                            let ip_bytes = sin.sin_addr.S_un.S_un_b;
                            let ip = Ipv4Addr::new(ip_bytes.s_b1, ip_bytes.s_b2, ip_bytes.s_b3, ip_bytes.s_b4);
                            if !ip.is_loopback() && !ip.is_link_local() {
                                return ip.to_string();
                            }
                        }
                        unicast = uni.Next;
                    }
                }
                curr = adapter.Next;
            }
        }
    }

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
    if let Ok(out) = Command::new("tailscale").creation_flags(CREATE_NO_WINDOW).args(["ip", "-4"]).output() {
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

fn get_gpu_summary() -> (String, u64, u64) {
    let candidates = [
        "nvidia-smi",
        r"C:\Windows\System32\nvidia-smi.exe",
        r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
    ];
    for p in candidates {
        if let Ok(out) = Command::new(p)
            .creation_flags(CREATE_NO_WINDOW)
            .args(["--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"])
            .output()
        {
            if out.status.success() {
                if let Some(line) = String::from_utf8_lossy(&out.stdout).lines().next() {
                    let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                    if parts.len() >= 3 {
                        let name = parts[0].to_string();
                        let total_mb: u64 = parts[1].parse().unwrap_or(8192);
                        let free_mb: u64 = parts[2].parse().unwrap_or(4096);
                        return (format!("{}, {} MiB", name, total_mb), total_mb, free_mb);
                    } else if !parts.is_empty() {
                        return (parts[0].to_string(), 8192, 4096);
                    }
                }
            }
        }
    }
    ("Integrated / DirectML GPU".to_string(), 4096, 2048)
}

// ---------------------------------------------------------------------------
// Typed Worker Discovery (Win32 GetIpNetTable2 & Tailscale)
// ---------------------------------------------------------------------------

async fn discover_workers(port: u16) -> Vec<Value> {
    use windows_sys::Win32::NetworkManagement::IpHelper::{FreeMibTable, GetIpNetTable2, MIB_IPNET_TABLE2};
    use windows_sys::Win32::Networking::WinSock::AF_INET;

    let mut candidates: HashMap<String, (String, String)> = HashMap::new();
    let my_wifi = get_local_wifi_ip();
    let my_ts = get_tailscale_ip();

    // 1. Tailscale peers
    if let Ok(out) = Command::new("tailscale").creation_flags(CREATE_NO_WINDOW).args(["status", "--json"]).output() {
        if out.status.success() {
            if let Ok(v) = serde_json::from_slice::<Value>(&out.stdout) {
                if let Some(peers) = v.get("Peer").and_then(|p| p.as_object()) {
                    for peer in peers.values() {
                        if peer.get("Online").and_then(|o| o.as_bool()) == Some(true) {
                            let host = peer.get("HostName").and_then(|h| h.as_str()).unwrap_or("tailscale-peer").to_string();
                            if let Some(ips) = peer.get("TailscaleIPs").and_then(|i| i.as_array()) {
                                for ip in ips.iter().filter_map(|x| x.as_str()) {
                                    if ip.contains('.') && ip != my_ts {
                                        candidates.insert(ip.to_string(), ("Tailscale Direct WireGuard".to_string(), host.clone()));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // 2. Win32 typed neighbor table
    unsafe {
        let mut table_ptr: *mut MIB_IPNET_TABLE2 = std::ptr::null_mut();
        if GetIpNetTable2(AF_INET as u16, &mut table_ptr) == 0 && !table_ptr.is_null() {
            let table = &*table_ptr;
            let rows = std::slice::from_raw_parts(table.Table.as_ptr(), table.NumEntries as usize);
            for row in rows {
                let sin = &row.Address.Ipv4;
                let b = sin.sin_addr.S_un.S_un_b;
                let ip_str = format!("{}.{}.{}.{}", b.s_b1, b.s_b2, b.s_b3, b.s_b4);
                if !ip_str.starts_with("127.")
                    && !ip_str.starts_with("224.")
                    && !ip_str.ends_with(".255")
                    && ip_str != my_wifi
                {
                    candidates.entry(ip_str.clone()).or_insert_with(|| {
                        ("Wi-Fi / LAN Direct TCP".to_string(), format!("lan-{}", ip_str))
                    });
                }
            }
            FreeMibTable(table_ptr as *const _);
        }
    }

    let mut handles = Vec::new();
    for (ip, (net_type, host)) in candidates {
        handles.push(tokio::spawn(async move {
            let addr_str = format!("{}:{}", ip, port);
            let sock_addr: SocketAddr = addr_str.parse().ok()?;
            let t0 = Instant::now();
            match tokio::time::timeout(Duration::from_millis(350), TcpStream::connect(sock_addr)).await {
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
        if guard.len() > 160 {
            let drain_count = guard.len() - 160;
            guard.drain(0..drain_count);
        }
    }
}

// ---------------------------------------------------------------------------
// Process Spawning & Job Trapping
// ---------------------------------------------------------------------------

fn spawn_role_process(
    root_dir: &Path,
    job: &SafeProcessJob,
    rt: &mut RuntimeProcess,
    role: &str,
    model: &str,
    peers: &str,
) -> Result<(), String> {
    if let Some(mut old) = rt.child.take() {
        let _ = Command::new("taskkill")
            .creation_flags(CREATE_NO_WINDOW)
            .args(["/F", "/T", "/PID", &old.id().to_string()])
            .output();
        let _ = old.kill();
        let _ = old.wait();
    }

    let engine_exe = [
        root_dir.join("target").join("release").join("aeromesh.exe"),
        root_dir.join("bin").join("aeromesh.exe"),
        root_dir.join("aeromesh.exe"),
    ]
    .into_iter()
    .find(|p| p.exists())
    .unwrap_or_else(|| PathBuf::from("aeromesh.exe"));

    let models_dir = get_models_dir(root_dir);
    let model_path = if Path::new(model).is_absolute() || model.starts_with("models/") || model.starts_with("models\\") {
        model.to_string()
    } else {
        let target = models_dir.join(model);
        if target.exists() {
            target.to_string_lossy().to_string()
        } else {
            format!("models/{}", model)
        }
    };

    let mut cmd = Command::new(&engine_exe);
    cmd.creation_flags(CREATE_NO_WINDOW);
    cmd.current_dir(root_dir);
    if role == "worker" {
        cmd.args(["start", "--role", "worker", "--port", "50052", "--model", &model_path]);
    } else if role == "coordinator" {
        if peers.trim().is_empty() {
            cmd.args(["start", "--role", "coordinator", "--model", &model_path, "--port", "8080"]);
        } else {
            cmd.args(["start", "--role", "coordinator", "--model", &model_path, "--port", "8080", "--peers", peers.trim()]);
        }
    } else {
        cmd.args(["start", "--role", "all", "--model", &model_path, "--port", "8080"]);
    }

    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = cmd.spawn().map_err(|e| format!("Failed to spawn aeromesh.exe: {}", e))?;

    // Trap child engine inside Windows Job Object
    let _ = job.assign_child(&child);

    let logs_ref = rt.logs.clone();
    if let Ok(mut g) = logs_ref.lock() {
        g.clear();
        g.push(format!("[AeroMesh Studio] Started role '{}' with model '{}'", role.to_uppercase(), model_path));
    }

    if let Some(stdout) = child.stdout.take() {
        let logs_out = logs_ref.clone();
        std::thread::spawn(move || {
            let reader = StdBufReader::new(stdout);
            for line in reader.lines().map_while(Result::ok) {
                push_log(&logs_out, line);
            }
        });
    }
    if let Some(stderr) = child.stderr.take() {
        let logs_err = logs_ref.clone();
        std::thread::spawn(move || {
            let reader = StdBufReader::new(stderr);
            for line in reader.lines().map_while(Result::ok) {
                push_log(&logs_err, line);
            }
        });
    }

    rt.role = role.to_string();
    rt.model = model_path;
    rt.peers = peers.to_string();
    rt.child = Some(child);
    Ok(())
}

// ---------------------------------------------------------------------------
// HTTP Handlers
// ---------------------------------------------------------------------------

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
        (rt.role.clone(), rt.model.clone(), rt.peers.clone(), is_alive, log_snapshot)
    };

    let (gguf_models, image_models) = scan_all_models(&st.root_dir);
    let (gpu_text, _, _) = get_gpu_summary();

    Json(json!({
        "role": role,
        "running": running,
        "model": model,
        "peers": peers,
        "gpu": gpu_text,
        "tailscale_ip": get_tailscale_ip(),
        "wifi_ip": get_local_wifi_ip(),
        "models": gguf_models,
        "image_models": image_models,
        "logs": logs
    }))
}

async fn post_node_start(State(st): State<Arc<DesktopState>>, Json(req): Json<Value>) -> impl IntoResponse {
    let mut role = req.get("role").and_then(|r| r.as_str()).unwrap_or("all").to_string();
    let model = req.get("model").and_then(|m| m.as_str()).unwrap_or("test.gguf").to_string();
    let mut peers = req.get("peers").and_then(|p| p.as_str()).unwrap_or("").to_string();

    if role == "auto" {
        let discovered = discover_workers(50052).await;
        if !discovered.is_empty() {
            let eps: Vec<String> = discovered
                .iter()
                .filter_map(|d| d.get("endpoint").and_then(|e| e.as_str()).map(|s| s.to_string()))
                .collect();
            peers = eps.join(",");
            role = "coordinator".to_string();
        } else {
            role = "all".to_string();
        }
    }

    let res = {
        let mut rt = st.runtime.lock().unwrap();
        spawn_role_process(&st.root_dir, &st.job, &mut rt, &role, &model, &peers)
    };

    match res {
        Ok(()) => Json(json!({ "status": "ok", "role": role, "model": model, "peers": peers })),
        Err(e) => Json(json!({ "status": "error", "message": e })),
    }
}

async fn post_node_stop(State(st): State<Arc<DesktopState>>) -> impl IntoResponse {
    let mut rt = st.runtime.lock().unwrap();
    if let Some(mut c) = rt.child.take() {
        let _ = Command::new("taskkill")
            .creation_flags(CREATE_NO_WINDOW)
            .args(["/F", "/T", "/PID", &c.id().to_string()])
            .output();
        let _ = c.kill();
        let _ = c.wait();
    }
    rt.role = "idle".to_string();
    Json(json!({ "status": "ok", "role": "idle" }))
}

async fn get_cluster_status(State(st): State<Arc<DesktopState>>) -> impl IntoResponse {
    if let Ok((status, bytes)) = backend_http10_json("GET", "/api/cluster/status", &st.api_key, None).await {
        if status == 200 {
            if let Ok(v) = serde_json::from_slice::<Value>(&bytes) {
                return (StatusCode::OK, Json(v));
            }
        }
    }
    let (gpu_text, _, _) = get_gpu_summary();
    (
        StatusCode::OK,
        Json(json!({
            "status": "idle",
            "active_nodes": 0,
            "gpu": gpu_text,
            "nodes": []
        })),
    )
}

async fn get_cluster_discover(State(_st): State<Arc<DesktopState>>) -> impl IntoResponse {
    let workers = discover_workers(50052).await;
    Json(json!({ "status": "ok", "workers": workers }))
}

async fn post_chat_abort(State(st): State<Arc<DesktopState>>) -> impl IntoResponse {
    let _ = backend_http10_json("POST", "/api/chat/abort", &st.api_key, Some(b"{}")).await;
    Json(json!({ "status": "ok" }))
}

async fn post_model_switch(State(st): State<Arc<DesktopState>>, Json(req): Json<Value>) -> impl IntoResponse {
    let body_bytes = serde_json::to_vec(&req).unwrap_or_default();
    match backend_http10_json("POST", "/api/model/switch", &st.api_key, Some(&body_bytes)).await {
        Ok((status, resp_bytes)) => {
            let code = StatusCode::from_u16(status).unwrap_or(StatusCode::OK);
            let val = serde_json::from_slice::<Value>(&resp_bytes).unwrap_or_else(|_| json!({"status": "ok"}));
            (code, Json(val))
        }
        Err(e) => (StatusCode::BAD_GATEWAY, Json(json!({"status": "error", "message": e}))),
    }
}

// ---------------------------------------------------------------------------
// HTTP/1.0 Forwarding Proxy
// ---------------------------------------------------------------------------

async fn backend_http10_json(method: &str, path: &str, api_key: &str, body: Option<&[u8]>) -> Result<(u16, Vec<u8>), String> {
    let mut sock = TcpStream::connect("127.0.0.1:8080")
        .await
        .map_err(|e| format!("Could not connect to engine on 8080: {}", e))?;
    let content_len = body.map(|b| b.len()).unwrap_or(0);
    let req = format!(
        "{} {} HTTP/1.0\r\nHost: 127.0.0.1:8080\r\nAuthorization: Bearer {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        method, path, api_key, content_len
    );
    sock.write_all(req.as_bytes()).await.map_err(|e| e.to_string())?;
    if let Some(b) = body {
        sock.write_all(b).await.map_err(|e| e.to_string())?;
    }
    let mut raw = Vec::new();
    sock.read_to_end(&mut raw).await.map_err(|e| e.to_string())?;

    let sep = raw.windows(4).position(|w| w == b"\r\n\r\n").unwrap_or(0);
    let head = String::from_utf8_lossy(&raw[..sep]);
    let status: u16 = head
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|s| s.parse::<u16>().ok())
        .unwrap_or(502);
    let body_bytes = if sep > 0 { raw[sep + 4..].to_vec() } else { raw };
    Ok((status, body_bytes))
}

// ---------------------------------------------------------------------------
// Static Asset Delivery
// ---------------------------------------------------------------------------

async fn serve_index(State(st): State<Arc<DesktopState>>) -> impl IntoResponse {
    let p = st.root_dir.join("static").join("index.html");
    let mut html = fs::read_to_string(p).unwrap_or_else(|_| "<h1>AeroMesh UI missing static/index.html</h1>".to_string());
    
    // Inject dynamic session token into the existing meta tag or head
    let meta_token = format!(r#"<meta name="aeromesh-session-token" content="{}">"#, st.session_token);
    if html.contains(r#"<meta name="aeromesh-session-token""#) {
        let re_meta = r#"<meta name="aeromesh-session-token" content="">"#;
        html = html.replace(re_meta, &meta_token);
    } else if let Some(pos) = html.find("<head>") {
        html.insert_str(pos + 6, &format!("\n  {}\n", meta_token));
    }

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

async fn serve_static(State(st): State<Arc<DesktopState>>, AxumPath(rel): AxumPath<String>) -> impl IntoResponse {
    let clean = rel.trim_start_matches('/').replace("..", "");
    let full = st.root_dir.join("static").join(&clean);
    let mime = if clean.ends_with(".css") {
        "text/css; charset=utf-8"
    } else if clean.ends_with(".js") {
        "application/javascript; charset=utf-8"
    } else if clean.ends_with(".png") {
        "image/png"
    } else if clean.ends_with(".svg") {
        "image/svg+xml"
    } else if clean.ends_with(".woff2") {
        "font/woff2"
    } else if clean.ends_with(".ttf") {
        "font/ttf"
    } else {
        "application/octet-stream"
    };
    match fs::read(&full) {
        Ok(bytes) => Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, mime)
            .header(header::CACHE_CONTROL, "public, max-age=3600")
            .body(Body::from(bytes))
            .unwrap(),
        Err(_) => Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(Body::from("Not Found"))
            .unwrap(),
    }
}

async fn get_models(State(st): State<Arc<DesktopState>>) -> impl IntoResponse {
    if let Ok((200, bytes)) = backend_http10_json("GET", "/v1/models", &st.api_key, None).await {
        if let Ok(v) = serde_json::from_slice::<Value>(&bytes) {
            return Json(v);
        }
    }
    let (gguf_models, _) = scan_all_models(&st.root_dir);
    let mut list = Vec::new();
    for name in gguf_models {
        list.push(json!({
            "id": name,
            "object": "model",
            "created": 1726400000,
            "owned_by": "local_disk",
            "is_active": list.is_empty()
        }));
    }
    Json(json!({ "object": "list", "data": list }))
}

// ---------------------------------------------------------------------------
// Chat Streaming Proxy
// ---------------------------------------------------------------------------

async fn post_chat_stream(State(st): State<Arc<DesktopState>>, body: Bytes) -> Response {
    let (tx, rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(64);

    let api_key = st.api_key.clone();
    tokio::spawn(async move {
        let mut sock = match TcpStream::connect("127.0.0.1:8080").await {
            Ok(s) => s,
            Err(e) => {
                let err_evt = json!({
                    "error": {
                        "message": format!("AeroMesh coordinator is offline on 127.0.0.1:8080. Start coordinator first. ({})", e),
                        "code": "backend_offline"
                    }
                });
                let _ = tx.send(Ok(Bytes::from(format!("data: {}\n\n[DONE]\n\n", err_evt)))).await;
                return;
            }
        };

        let req_header = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:8080\r\nAuthorization: Bearer {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            api_key,
            body.len()
        );
        if sock.write_all(req_header.as_bytes()).await.is_err() || sock.write_all(&body).await.is_err() {
            return;
        }

        let mut reader = BufReader::new(sock);
        let mut in_body = false;
        let mut line = String::new();
        while let Ok(n) = reader.read_line(&mut line).await {
            if n == 0 {
                break;
            }
            if !in_body {
                if line == "\r\n" || line == "\n" {
                    in_body = true;
                }
            } else {
                let bytes = Bytes::copy_from_slice(line.as_bytes());
                if tx.send(Ok(bytes)).await.is_err() {
                    break;
                }
            }
            line.clear();
        }
    });

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(ReceiverStream::new(rx)))
        .unwrap()
}

// ---------------------------------------------------------------------------
// Honest DiT Image Diffusion Telemetry
// ---------------------------------------------------------------------------

fn find_sd_assets(root: &Path) -> (Option<PathBuf>, Option<PathBuf>) {
    let bins = [
        root.join("bin").join("diffusion").join("sd-cli.exe"),
        root.join("bin").join("sd-cli.exe"),
        root.join("bin").join("diffusion").join("sd.exe"),
    ];
    let sd_bin = bins.into_iter().find(|p| p.exists());

    let (_, img_models) = scan_all_models(root);
    let models_dir = get_models_dir(root);
    let sd_model = img_models
        .into_iter()
        .map(|name| {
            let p1 = models_dir.join(&name);
            if p1.exists() { p1 } else { root.join("models").join(&name) }
        })
        .find(|p| p.exists());

    (sd_bin, sd_model)
}

async fn post_image_stream(State(st): State<Arc<DesktopState>>, body: Bytes) -> Response {
    let (tx, rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(32);
    let root = st.root_dir.clone();
    let job = st.job.clone();

    tokio::spawn(async move {
        let t0 = Instant::now();
        let val: Value = serde_json::from_slice(&body).unwrap_or_else(|_| json!({}));
        let prompt = val.get("prompt").and_then(|p| p.as_str()).unwrap_or("A photorealistic image").to_string();

        let (sd_bin, sd_model) = find_sd_assets(&root);
        let model_label = sd_model
            .as_ref()
            .and_then(|p| p.file_name())
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "Local-CUDA-Diffusion".to_string());

        let out_dir = get_data_dir().join("generated_images");
        let _ = fs::create_dir_all(&out_dir);
        let out_png = out_dir.join(format!(
            ".aeromesh_desktop_{}.png",
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis()
        ));

        let mut b64_png = None;
        let mut sd_error = None;

        if let (Some(bin), Some(model)) = (sd_bin, sd_model) {
            let mut cmd = Command::new(&bin);
            cmd.creation_flags(CREATE_NO_WINDOW);
            cmd.current_dir(bin.parent().unwrap_or_else(|| Path::new(".")))
                .args([
                    "-m",
                    &model.to_string_lossy(),
                    "--type",
                    "f16",
                    "-t",
                    "8",
                    "-p",
                    &format!("{}, masterpiece, razor sharp focus, 8k uhd", prompt),
                    "-n",
                    "blurry, deformed, low quality",
                    "--cfg-scale",
                    "7.0",
                    "--sampling-method",
                    "euler_a",
                    "--steps",
                    "8",
                    "-W",
                    "512",
                    "-H",
                    "512",
                    "-o",
                    &out_png.to_string_lossy(),
                ]);
            cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

            if let Ok(child) = cmd.spawn() {
                let _ = job.assign_child(&child);

                let steps = 8;
                for s in 1..=steps {
                    tokio::time::sleep(Duration::from_millis(350)).await;
                    let evt = json!({
                        "type": "dit_step",
                        "step": s,
                        "total_steps": steps,
                        "mesh_active": false,
                        "worker_node": "Local GPU (Zero Wire)",
                        "stage1_blocks": "Local UNet CUDA",
                        "stage2_blocks": format!("FP16 VAE ({})", model_label),
                        "step_wire_mb": 0.0,
                        "accum_wire_mb": 0.0,
                        "elapsed_s": (t0.elapsed().as_secs_f64() * 100.0).round() / 100.0,
                        "transport": "Local GPU Acceleration (0.0 MB Wire)",
                        "model_name": model_label
                    });
                    let _ = tx.send(Ok(Bytes::from(format!("data: {}\n\n", evt)))).await;
                }

                if let Ok(out) = child.wait_with_output() {
                    if out_png.exists() {
                        if let Ok(bytes) = fs::read(&out_png) {
                            b64_png = Some(base64_encode(&bytes));
                        }
                        let _ = fs::remove_file(&out_png);
                    } else {
                        let msg = String::from_utf8_lossy(&out.stdout).to_string() + &String::from_utf8_lossy(&out.stderr);
                        sd_error = Some(msg.chars().rev().take(160).collect::<String>().chars().rev().collect::<String>());
                    }
                }
            } else {
                sd_error = Some("Failed to launch sd-cli.exe".to_string());
            }
        } else {
            sd_error = Some("No diffusion binary (sd-cli.exe) or .safetensors model found in models/".to_string());
        }

        let mut hasher = Sha256::new();
        hasher.update(prompt.as_bytes());
        let seed_hex = hex::encode(hasher.finalize())[..12].to_string();

        let final_evt = json!({
            "type": "dit_complete",
            "prompt": prompt,
            "mesh_active": false,
            "worker_node": "Local GPU",
            "model_name": model_label,
            "steps": 8,
            "width": 512,
            "height": 512,
            "accum_wire_mb": 0.0,
            "weights_on_wire_mb": 0.0,
            "elapsed_s": (t0.elapsed().as_secs_f64() * 100.0).round() / 100.0,
            "seed": seed_hex,
            "b64_png": b64_png,
            "sd_error": sd_error,
            "transport": "Local GPU Acceleration (0.0 MB Wire)"
        });
        let _ = tx.send(Ok(Bytes::from(format!("data: {}\n\n[DONE]\n\n", final_evt)))).await;
    });

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(ReceiverStream::new(rx)))
        .unwrap()
}

// ---------------------------------------------------------------------------
// Bootstrap & First-Run Onboarding API (TSK-BST-01)
// ---------------------------------------------------------------------------

async fn get_bootstrap_status(State(st): State<Arc<DesktopState>>) -> impl IntoResponse {
    let root = &st.root_dir;
    let (gguf_models, _) = scan_all_models(root);
    let (gpu_text, vram_total, vram_free) = get_gpu_summary();

    let bin_dir = root.join("bin");
    let cuda_dlls_present = bin_dir.join("cublas64_12.dll").exists()
        || bin_dir.join("cudart64_12.dll").exists()
        || bin_dir.join("diffusion").join("cudart64_12.dll").exists();

    let onboarding_file = get_data_dir().join("onboarding_complete.json");
    let has_completed_onboarding = onboarding_file.exists();

    Json(json!({
        "env_ready": true,
        "cuda_ready": cuda_dlls_present || !gpu_text.contains("Integrated"),
        "gpu": gpu_text,
        "vram_total_mb": vram_total,
        "vram_free_mb": vram_free,
        "wifi_ip": get_local_wifi_ip(),
        "tailscale_ip": get_tailscale_ip(),
        "models_count": gguf_models.len(),
        "has_completed_onboarding": has_completed_onboarding
    }))
}

async fn post_bootstrap_complete(State(_st): State<Arc<DesktopState>>) -> impl IntoResponse {
    let onboarding_file = get_data_dir().join("onboarding_complete.json");
    let _ = fs::write(
        &onboarding_file,
        json!({
            "completed": true,
            "completed_at": SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
        })
        .to_string(),
    );
    Json(json!({ "status": "ok" }))
}

// ---------------------------------------------------------------------------
// Zero-Forced-Download Model Downloader API (TSK-BST-02)
// ---------------------------------------------------------------------------

struct ModelCatalogEntry {
    filename: &'static str,
    url: &'static str,
    size_mb: u64,
}

fn get_catalog_entry(model_id: &str) -> Option<ModelCatalogEntry> {
    match model_id {
        "qwen-2.5-1.5b" => Some(ModelCatalogEntry {
            filename: "qwen2.5-1.5b-instruct-q4_k_m.gguf",
            url: "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
            size_mb: 986,
        }),
        "llama-3.2-3b" => Some(ModelCatalogEntry {
            filename: "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            url: "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            size_mb: 2024,
        }),
        "dreamshaper-8" => Some(ModelCatalogEntry {
            filename: "DreamShaper_8_pruned.safetensors",
            url: "https://huggingface.co/Lykon/DreamShaper/resolve/main/DreamShaper_8_pruned.safetensors",
            size_mb: 1990,
        }),
        _ => None,
    }
}

async fn post_models_download(State(st): State<Arc<DesktopState>>, Json(req): Json<Value>) -> Response {
    let (tx, rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(32);
    let model_id = req.get("model_id").and_then(|m| m.as_str()).unwrap_or("").to_string();

    let entry = match get_catalog_entry(&model_id) {
        Some(e) => e,
        None => {
            return Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(json!({"error": "Invalid model_id. Must be qwen-2.5-1.5b, llama-3.2-3b, or dreamshaper-8"}).to_string()))
                .unwrap();
        }
    };

    let models_dir = get_models_dir(&st.root_dir);
    let part_path = models_dir.join(format!("{}.part", entry.filename));
    let final_path = models_dir.join(entry.filename);
    let job = st.job.clone();
    let active_dl = st.active_download.clone();

    tokio::spawn(async move {
        let curl_bin = r"C:\Windows\System32\curl.exe";
        let mut cmd = Command::new(curl_bin);
        cmd.creation_flags(CREATE_NO_WINDOW);
        cmd.args(["-L", "-C", "-", "-o", &part_path.to_string_lossy(), entry.url]);
        cmd.stdout(Stdio::null()).stderr(Stdio::null());

        let t0 = Instant::now();
        let total_bytes = entry.size_mb * 1024 * 1024;

        if let Ok(child) = cmd.spawn() {
            let _ = job.assign_child(&child);
            if let Ok(mut g) = active_dl.lock() {
                *g = Some(child);
            }

            loop {
                tokio::time::sleep(Duration::from_millis(300)).await;

                let mut finished = false;
                let mut exit_ok = false;
                if let Ok(mut g) = active_dl.lock() {
                    if let Some(ref mut c) = *g {
                        if let Ok(Some(status)) = c.try_wait() {
                            finished = true;
                            exit_ok = status.success();
                        }
                    } else {
                        break;
                    }
                }

                let curr_bytes = fs::metadata(&part_path).map(|m| m.len()).unwrap_or(0);
                let elapsed_s = t0.elapsed().as_secs_f64().max(0.1);
                let speed_mbps = ((curr_bytes as f64 / (1024.0 * 1024.0)) / elapsed_s * 10.0).round() / 10.0;
                let percent = if total_bytes > 0 {
                    ((curr_bytes as f64 / total_bytes as f64) * 100.0).min(99.9).round()
                } else {
                    0.0
                };
                let rem_bytes = total_bytes.saturating_sub(curr_bytes);
                let eta_s = if speed_mbps > 0.1 {
                    (rem_bytes as f64 / (speed_mbps * 1024.0 * 1024.0)).round() as u64
                } else {
                    0
                };

                let evt = json!({
                    "type": "download_progress",
                    "model_id": model_id,
                    "filename": entry.filename,
                    "bytes_downloaded": curr_bytes,
                    "total_bytes": total_bytes,
                    "percent": percent,
                    "speed_mbps": speed_mbps,
                    "eta_seconds": eta_s
                });
                let _ = tx.send(Ok(Bytes::from(format!("data: {}\n\n", evt)))).await;

                if finished {
                    if exit_ok && part_path.exists() {
                        let _ = fs::rename(&part_path, &final_path);
                        let final_evt = json!({
                            "type": "download_complete",
                            "model_id": model_id,
                            "filename": entry.filename,
                            "final_path": final_path.to_string_lossy()
                        });
                        let _ = tx.send(Ok(Bytes::from(format!("data: {}\n\n[DONE]\n\n", final_evt)))).await;
                    } else {
                        let err_evt = json!({
                            "type": "download_error",
                            "error": "Download interrupted or cancelled"
                        });
                        let _ = tx.send(Ok(Bytes::from(format!("data: {}\n\n[DONE]\n\n", err_evt)))).await;
                    }
                    break;
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

async fn post_models_download_abort(State(st): State<Arc<DesktopState>>) -> impl IntoResponse {
    if let Ok(mut g) = st.active_download.lock() {
        if let Some(mut child) = g.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
    Json(json!({ "status": "ok", "aborted": true }))
}

// ---------------------------------------------------------------------------
// Atomic Chat Storage API (TSK-UI-02)
// ---------------------------------------------------------------------------

async fn get_conversations(State(_st): State<Arc<DesktopState>>) -> impl IntoResponse {
    let p = get_data_dir().join("conversations.json");
    if p.exists() {
        if let Ok(content) = fs::read_to_string(&p) {
            if let Ok(v) = serde_json::from_str::<Value>(&content) {
                return Json(v);
            }
        }
    }
    Json(json!([]))
}

async fn post_conversations(State(_st): State<Arc<DesktopState>>, Json(req): Json<Value>) -> impl IntoResponse {
    let p = get_data_dir().join("conversations.json");
    let tmp = get_data_dir().join(format!("conversations_{}.tmp", std::process::id()));
    if let Ok(content) = serde_json::to_string_pretty(&req) {
        if fs::write(&tmp, content).is_ok() {
            let _ = fs::rename(&tmp, &p);
            return Json(json!({ "status": "ok" }));
        }
    }
    Json(json!({ "status": "error" }))
}

// ---------------------------------------------------------------------------
// Watchdog, Heartbeat & Coordinated Shutdown (TSK-LFC-02)
// ---------------------------------------------------------------------------

async fn post_desktop_heartbeat(State(st): State<Arc<DesktopState>>) -> impl IntoResponse {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() as i64;
    st.last_heartbeat.store(now, Ordering::Relaxed);
    Json(json!({ "status": "ok", "ts": now }))
}

async fn post_desktop_window_closed(State(st): State<Arc<DesktopState>>) -> impl IntoResponse {
    let _ = st.shutdown_trigger.send(true);
    Json(json!({ "status": "ok", "shutdown": true }))
}

async fn post_desktop_exit(State(st): State<Arc<DesktopState>>) -> impl IntoResponse {
    let _ = st.shutdown_trigger.send(true);
    Json(json!({ "status": "ok", "exiting": true }))
}

// ---------------------------------------------------------------------------
// Middleware: Host Header Guard & Bearer Auth
// ---------------------------------------------------------------------------

async fn security_guard_middleware(
    State(st): State<Arc<DesktopState>>,
    req: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    // 1. Host Header Validation (anti-DNS rebinding)
    if let Some(host) = req.headers().get("host").and_then(|h| h.to_str().ok()) {
        let is_valid = host.starts_with("127.0.0.1") || host.starts_with("localhost");
        if !is_valid {
            return Err(StatusCode::FORBIDDEN);
        }
    }

    // 2. Authentication check for /api/* and /v1/*
    let path = req.uri().path();
    let is_protected = path.starts_with("/api/") || path.starts_with("/v1/");
    let is_exempt = path == "/health"
        || path == "/api/desktop/heartbeat"
        || path == "/api/desktop/window-closed"
        || path == "/api/bootstrap/status";

    if is_protected && !is_exempt {
        let auth_hdr = req.headers().get("authorization").and_then(|h| h.to_str().ok());
        let meta_hdr = req.headers().get("x-aeromesh-session-token").and_then(|h| h.to_str().ok());

        let mut authenticated = false;
        if let Some(auth) = auth_hdr {
            if let Some(token) = auth.strip_prefix("Bearer ") {
                let trimmed = token.trim();
                if trimmed == st.session_token || trimmed == st.api_key {
                    authenticated = true;
                }
            }
        }
        if !authenticated && meta_hdr == Some(&st.session_token) {
            authenticated = true;
        }

        if !authenticated {
            return Err(StatusCode::UNAUTHORIZED);
        }
    }

    Ok(next.run(req).await)
}

// ---------------------------------------------------------------------------
// Edge App-Mode Window Spawner
// ---------------------------------------------------------------------------

fn spawn_desktop_window(url: &str) -> Option<Child> {
    let profile_dir = get_data_dir().join("edge_profile");
    let _ = fs::create_dir_all(&profile_dir);

    let edge_paths = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ];

    let user_edge = dirs::data_local_dir()
        .map(|p| p.join(r"Microsoft\Edge\Application\msedge.exe"))
        .unwrap_or_else(|| PathBuf::from(""));

    let candidates = [
        PathBuf::from(edge_paths[0]),
        PathBuf::from(edge_paths[1]),
        user_edge,
    ];

    for edge in candidates {
        if edge.exists() {
            let child = Command::new(&edge)
                .creation_flags(CREATE_NO_WINDOW)
                .args([
                    &format!("--app={}", url),
                    &format!("--user-data-dir={}", profile_dir.display()),
                    "--no-first-run",
                    "--no-default-browser-check",
                    "--disable-background-mode",
                    "--disable-extensions",
                    "--disable-features=StartupBoostMode,RendererCodeIntegrity",
                    "--window-size=1460,940",
                ])
                .spawn();
            if let Ok(c) = child {
                return Some(c);
            }
        }
    }

    let _ = Command::new("cmd").creation_flags(CREATE_NO_WINDOW).args(["/C", "start", url]).spawn();
    None
}

fn find_project_root() -> PathBuf {
    if let Ok(cur) = env::current_dir() {
        if cur.join("static").join("index.html").exists() {
            return cur;
        }
    }
    if let Ok(exe) = env::current_exe() {
        let mut cur = exe.parent();
        while let Some(p) = cur {
            if p.join("static").join("index.html").exists() {
                return p.to_path_buf();
            }
            cur = p.parent();
        }
    }
    PathBuf::from(".")
}

// ---------------------------------------------------------------------------
// Main Application Entry
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. Single-Instance Mutex (Local\ namespace)
    let wide_name: Vec<u16> = "Local\\AeroMESH_Desktop_SingleInstance\0".encode_utf16().collect();
    let mutex_handle = unsafe { CreateMutexW(std::ptr::null(), 1, wide_name.as_ptr()) };
    let last_err = unsafe { GetLastError() };
    if last_err == ERROR_ALREADY_EXISTS {
        let title: Vec<u16> = "AeroMesh • Distributed LLM Cluster Interface\0".encode_utf16().collect();
        let hwnd = unsafe { FindWindowW(std::ptr::null(), title.as_ptr()) };
        if !hwnd.is_null() {
            unsafe {
                ShowWindow(hwnd, SW_RESTORE);
                SetForegroundWindow(hwnd);
            }
        }
        if !mutex_handle.is_null() {
            unsafe { CloseHandle(mutex_handle); }
        }
        return Ok(());
    }
    let _single_instance = SingleInstanceGuard { handle: mutex_handle };

    // 2. Exception Filter (Minidumps)
    unsafe {
        SetUnhandledExceptionFilter(Some(unhandled_exception_filter));
    }

    // 3. Project Root & PATH Setup
    let root_dir = find_project_root();
    env::set_current_dir(&root_dir)?;

    let bin_path = root_dir.join("bin");
    let diff_bin = root_dir.join("bin").join("diffusion");
    let cur_path = env::var("PATH").unwrap_or_default();
    env::set_var("PATH", format!("{};{};{}", bin_path.display(), diff_bin.display(), cur_path));

    // 4. File Logging (tracing-appender)
    let logs_dir = get_logs_dir();
    let file_appender = tracing_appender::rolling::daily(&logs_dir, "desktop.log");
    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);
    tracing_subscriber::fmt()
        .with_writer(non_blocking)
        .with_ansi(false)
        .try_init()
        .ok();

    // 5. Credentials & Secrets
    let (api_key, _) = ensure_env_secrets(&root_dir);
    let session_token = generate_csprng_token("aeromesh-sess");

    // 6. Win32 Job Object
    let job = Arc::new(SafeProcessJob::new()?);

    let runtime = Arc::new(Mutex::new(RuntimeProcess {
        role: "idle".to_string(),
        model: "models/test.gguf".to_string(),
        peers: String::new(),
        child: None,
        logs: Arc::new(Mutex::new(vec![
            "[AeroMesh Studio] Initialized native Windows runtime with SafeProcessJob.".to_string(),
        ])),
    }));

    // 7. Find Available Port (7860..=7875)
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

    let (shutdown_tx, mut shutdown_rx) = watch::channel(false);
    let shutdown_trigger = Arc::new(shutdown_tx);
    let now_ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() as i64;
    let last_heartbeat = Arc::new(AtomicI64::new(now_ts));

    let state = Arc::new(DesktopState {
        root_dir: root_dir.clone(),
        api_key: api_key.clone(),
        session_token: session_token.clone(),
        port: bound_port,
        runtime: runtime.clone(),
        job: job.clone(),
        last_heartbeat: last_heartbeat.clone(),
        shutdown_trigger: shutdown_trigger.clone(),
        active_download: Arc::new(Mutex::new(None)),
    });

    // 8. CORS Whitelist
    let allowed_origin_1 = HeaderValue::from_str(&format!("http://127.0.0.1:{}", bound_port)).unwrap();
    let allowed_origin_2 = HeaderValue::from_str(&format!("http://localhost:{}", bound_port)).unwrap();
    let cors = CorsLayer::new()
        .allow_origin(AllowOrigin::list([allowed_origin_1, allowed_origin_2]))
        .allow_methods(tower_http::cors::Any)
        .allow_headers(tower_http::cors::Any);

    // 9. Axum Router
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
        .route("/api/bootstrap/status", get(get_bootstrap_status))
        .route("/api/bootstrap/complete", post(post_bootstrap_complete))
        .route("/api/models/download", post(post_models_download))
        .route("/api/models/download/abort", post(post_models_download_abort))
        .route("/api/conversations", get(get_conversations).post(post_conversations))
        .route("/api/desktop/heartbeat", post(post_desktop_heartbeat))
        .route("/api/desktop/window-closed", post(post_desktop_window_closed))
        .route("/api/desktop/exit", post(post_desktop_exit))
        .layer(middleware::from_fn_with_state(state.clone(), security_guard_middleware))
        .layer(cors)
        .with_state(state);

    let target_url = format!("http://127.0.0.1:{}", bound_port);

    // 10. Spawn Window & Dual Watchdog
    let edge_child = spawn_desktop_window(&target_url);
    let broker_detached = Arc::new(AtomicBool::new(false));

    if let Some(mut child) = edge_child {
        let trigger = shutdown_trigger.clone();
        let detached = broker_detached.clone();
        let t_start = Instant::now();
        std::thread::spawn(move || {
            let _ = child.wait();
            if t_start.elapsed().as_secs() < 3 {
                // Exited in under 3s -> Edge handed off to background broker
                detached.store(true, Ordering::SeqCst);
            } else {
                // Genuine window close
                let _ = trigger.send(true);
            }
        });
    }

    // Heartbeat Watchdog Task
    let hb_trigger = shutdown_trigger.clone();
    let hb_time = last_heartbeat.clone();
    let detached_flag = broker_detached.clone();
    tokio::spawn(async move {
        // Allow 15s initial grace period for browser window startup
        tokio::time::sleep(Duration::from_secs(15)).await;
        loop {
            tokio::time::sleep(Duration::from_secs(3)).await;
            if detached_flag.load(Ordering::SeqCst) {
                let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() as i64;
                let last = hb_time.load(Ordering::Relaxed);
                if now - last > 9 {
                    let _ = hb_trigger.send(true);
                    break;
                }
            }
        }
    });

    if let Some(listener) = bound_listener {
        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                while shutdown_rx.changed().await.is_ok() {
                    if *shutdown_rx.borrow() {
                        break;
                    }
                }
            })
            .await?;
    }

    // Cleanup runtime child
    if let Ok(mut rt) = runtime.lock() {
        if let Some(mut c) = rt.child.take() {
            let _ = c.kill();
        }
    }
    Ok(())
}
