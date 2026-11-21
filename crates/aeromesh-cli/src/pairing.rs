//! One-Click Cluster Pairing & Encrypted Secret Sync (`aeromesh-pair-v1`).
//!
//! Allows two AeroMESH PCs over Wi-Fi LAN or Tailscale to pair with a
//! short-lived 6-digit PIN (or 1-line `AERO-...` invite code), cryptographically
//! verify the code via PBKDF2-HMAC-SHA256 (100,000 rounds) + mutual HMAC-SHA256,
//! exchange `AEROMESH_WORKER_SECRET` encrypted with SHA-256-CTR + Encrypt-then-MAC,
//! and persist mutual peer metadata (GPU, endpoints, installed GGUF models, key
//! fingerprint) in `<data_dir>/paired_peers.json`.

use std::collections::HashMap;
use std::env;
use std::net::{IpAddr, SocketAddr};
#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;
#[cfg(target_os = "windows")]
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::oneshot;

pub const DEFAULT_PAIR_PORT: u16 = 50053;
const PAIR_SESSION_TTL_SECS: u64 = 300;
const PAIR_MAX_FAILED_ATTEMPTS: u32 = 5;
const PAIR_PBKDF2_ITERS: u32 = 100_000;
const MAX_LINE_BYTES: usize = 16 * 1024;

#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x08000000;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PairedPeer {
    pub id: String,
    pub node_name: String,
    pub gpu: String,
    pub wifi_ip: String,
    pub tailscale_ip: String,
    pub worker_endpoint: String,
    pub models: Vec<String>,
    pub cluster_fingerprint: String,
    pub paired_at: u64,
    pub last_seen_at: u64,
}

#[derive(Clone)]
pub struct ActivePairSession {
    pub pin: String,
    pub port: u16,
    pub salt: [u8; 16],
    pub expires_at: u64,
    pub failed_attempts: Arc<AtomicU32>,
    pub paired_count: Arc<AtomicU32>,
    pub active: Arc<AtomicBool>,
    pub shutdown_tx: Arc<Mutex<Option<oneshot::Sender<()>>>>,
}

#[derive(Clone, Default)]
pub struct PairingState {
    pub session: Arc<Mutex<Option<ActivePairSession>>>,
}

fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn random_bytes<const N: usize>() -> [u8; N] {
    let mut buf = [0u8; N];
    if getrandom::getrandom(&mut buf).is_err() {
        let mut h = Sha256::new();
        h.update(now_unix_secs().to_le_bytes());
        h.update(std::process::id().to_le_bytes());
        let digest = h.finalize();
        for i in 0..N {
            buf[i] = digest[i % digest.len()];
        }
    }
    buf
}

fn generate_six_digit_pin() -> String {
    let raw = random_bytes::<4>();
    let n = u32::from_be_bytes(raw) % 1_000_000;
    format!("{:06}", n)
}

pub fn format_pin_display(pin: &str) -> String {
    let digits: String = pin.chars().filter(|c| c.is_ascii_digit()).collect();
    if digits.len() == 6 {
        format!("{}-{}", &digits[..3], &digits[3..])
    } else {
        digits
    }
}

/// Short non-reversible visual fingerprint of `AEROMESH_WORKER_SECRET` (e.g. `KEY-A4F1-90C2`).
pub fn secret_fingerprint(secret: &str) -> String {
    let trimmed = secret.trim();
    if trimmed.is_empty() {
        return "UNSET".to_string();
    }
    let mut h = Sha256::new();
    h.update(b"aeromesh-cluster-fingerprint-v1:");
    h.update(trimmed.as_bytes());
    let digest = h.finalize();
    let hex_str = hex::encode(&digest[..4]).to_ascii_uppercase();
    format!("KEY-{}-{}", &hex_str[..4], &hex_str[4..8])
}

fn current_worker_secret() -> String {
    env::var("AEROMESH_WORKER_SECRET").unwrap_or_default()
}

fn local_node_name() -> String {
    for var in ["COMPUTERNAME", "HOSTNAME"] {
        if let Ok(val) = env::var(var) {
            let trimmed = val.trim();
            if !trimmed.is_empty() {
                return trimmed.to_string();
            }
        }
    }
    "AeroMesh-Node".to_string()
}

fn local_wifi_ip() -> String {
    if let Ok(sock) = std::net::UdpSocket::bind("0.0.0.0:0") {
        if sock.connect("8.8.8.8:80").is_ok() {
            if let Ok(addr) = sock.local_addr() {
                return addr.ip().to_string();
            }
        }
    }
    "127.0.0.1".to_string()
}

fn local_tailscale_ip() -> String {
    #[cfg(target_os = "windows")]
    {
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
    }
    "Not Connected".to_string()
}

fn local_gpu_summary() -> String {
    #[cfg(target_os = "windows")]
    {
        let candidates = [
            "nvidia-smi",
            r"C:\Windows\System32\nvidia-smi.exe",
            r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
        ];
        for p in candidates {
            if let Ok(out) = Command::new(p)
                .creation_flags(CREATE_NO_WINDOW)
                .args([
                    "--query-gpu=name,memory.total",
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
    }
    "CPU / Integrated GPU".to_string()
}

fn local_llm_models() -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for dir in aeromesh_core::model_search_dirs() {
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for e in entries.flatten() {
                let name = e.file_name().to_string_lossy().to_string();
                let low = name.to_ascii_lowercase();
                if low.ends_with(".gguf")
                    && !name.contains("_stage")
                    && !name.starts_with(".aeromesh_")
                    && seen.insert(name.clone())
                {
                    out.push(name);
                }
            }
        }
    }
    out.sort();
    out
}

fn preferred_worker_endpoint(wifi_ip: &str, tailscale_ip: &str) -> String {
    if !tailscale_ip.is_empty()
        && tailscale_ip != "Not Connected"
        && tailscale_ip != "Local"
        && tailscale_ip.contains('.')
    {
        format!("{}:50052", tailscale_ip)
    } else {
        format!("{}:50052", wifi_ip)
    }
}

// ---------------------------------------------------------------------------
// Pure-Rust Protocol Cryptography (`aeromesh-pair-v1`)
// Identical bit-for-bit to Python's `app.py` implementation so Rust and Python
// nodes can pair seamlessly across Windows/Linux/macOS.
// ---------------------------------------------------------------------------

fn hmac_sha256(key: &[u8], data: &[u8]) -> [u8; 32] {
    let mut key_block = [0u8; 64];
    if key.len() > 64 {
        let hashed = Sha256::digest(key);
        key_block[..32].copy_from_slice(&hashed);
    } else {
        key_block[..key.len()].copy_from_slice(key);
    }

    let mut ipad = [0x36u8; 64];
    let mut opad = [0x5cu8; 64];
    for i in 0..64 {
        ipad[i] ^= key_block[i];
        opad[i] ^= key_block[i];
    }

    let mut inner = Sha256::new();
    inner.update(ipad);
    inner.update(data);
    let inner_hash = inner.finalize();

    let mut outer = Sha256::new();
    outer.update(opad);
    outer.update(inner_hash);
    let res = outer.finalize();

    let mut out = [0u8; 32];
    out.copy_from_slice(&res);
    out
}

fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (&x, &y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// PBKDF2-HMAC-SHA256 (RFC 2898) producing 32 bytes (single block).
fn pbkdf2_hmac_sha256_32(password: &[u8], salt: &[u8], iterations: u32) -> [u8; 32] {
    let mut salt_block = Vec::with_capacity(salt.len() + 4);
    salt_block.extend_from_slice(salt);
    salt_block.extend_from_slice(&1u32.to_be_bytes());

    let mut u = hmac_sha256(password, &salt_block);
    let mut out = u;
    for _ in 1..iterations {
        u = hmac_sha256(password, &u);
        for i in 0..32 {
            out[i] ^= u[i];
        }
    }
    out
}

struct SessionKeys {
    auth_key: [u8; 32],
    enc_key: [u8; 32],
    mac_key: [u8; 32],
}

fn derive_session_keys(pin_digits: &str, salt: &[u8], host_nonce: &[u8], client_nonce: &[u8]) -> SessionKeys {
    let mut kdf_salt = Vec::with_capacity(salt.len() + host_nonce.len() + client_nonce.len());
    kdf_salt.extend_from_slice(salt);
    kdf_salt.extend_from_slice(host_nonce);
    kdf_salt.extend_from_slice(client_nonce);

    let master = pbkdf2_hmac_sha256_32(pin_digits.as_bytes(), &kdf_salt, PAIR_PBKDF2_ITERS);

    let derive_sub = |label: &[u8]| -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(master);
        h.update(label);
        let d = h.finalize();
        let mut out = [0u8; 32];
        out.copy_from_slice(&d);
        out
    };

    SessionKeys {
        auth_key: derive_sub(b"aeromesh-pair-v1-auth"),
        enc_key: derive_sub(b"aeromesh-pair-v1-enc"),
        mac_key: derive_sub(b"aeromesh-pair-v1-mac"),
    }
}

fn sha256_ctr_xor(enc_key: &[u8; 32], iv: &[u8; 16], data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    let mut counter: u32 = 0;
    for chunk in data.chunks(32) {
        let mut h = Sha256::new();
        h.update(enc_key);
        h.update(iv);
        h.update(counter.to_be_bytes());
        let block = h.finalize();
        for (i, &b) in chunk.iter().enumerate() {
            out.push(b ^ block[i]);
        }
        counter = counter.wrapping_add(1);
    }
    out
}

fn encrypt_and_mac(enc_key: &[u8; 32], mac_key: &[u8; 32], plaintext: &[u8]) -> ([u8; 16], Vec<u8>, [u8; 32]) {
    let iv = random_bytes::<16>();
    let ciphertext = sha256_ctr_xor(enc_key, &iv, plaintext);
    let mut mac_input = Vec::with_capacity(16 + ciphertext.len());
    mac_input.extend_from_slice(&iv);
    mac_input.extend_from_slice(&ciphertext);
    let tag = hmac_sha256(mac_key, &mac_input);
    (iv, ciphertext, tag)
}

fn verify_and_decrypt(
    enc_key: &[u8; 32],
    mac_key: &[u8; 32],
    iv: &[u8],
    ciphertext: &[u8],
    tag: &[u8],
) -> Result<Vec<u8>, String> {
    if iv.len() != 16 || tag.len() != 32 {
        return Err("Invalid encrypted payload framing.".to_string());
    }
    let mut mac_input = Vec::with_capacity(16 + ciphertext.len());
    mac_input.extend_from_slice(iv);
    mac_input.extend_from_slice(ciphertext);
    let expected_tag = hmac_sha256(mac_key, &mac_input);
    if !ct_eq(&expected_tag, tag) {
        return Err("Cryptographic integrity check failed on synced payload.".to_string());
    }
    let mut iv_arr = [0u8; 16];
    iv_arr.copy_from_slice(iv);
    Ok(sha256_ctr_xor(enc_key, &iv_arr, ciphertext))
}

fn compute_client_proof(auth_key: &[u8; 32], client_nonce: &[u8], host_nonce: &[u8]) -> [u8; 32] {
    let mut msg = Vec::with_capacity(13 + client_nonce.len() + host_nonce.len());
    msg.extend_from_slice(b"client-proof:");
    msg.extend_from_slice(client_nonce);
    msg.extend_from_slice(host_nonce);
    hmac_sha256(auth_key, &msg)
}

fn compute_host_proof(auth_key: &[u8; 32], host_nonce: &[u8], client_nonce: &[u8]) -> [u8; 32] {
    let mut msg = Vec::with_capacity(11 + host_nonce.len() + client_nonce.len());
    msg.extend_from_slice(b"host-proof:");
    msg.extend_from_slice(host_nonce);
    msg.extend_from_slice(client_nonce);
    hmac_sha256(auth_key, &msg)
}

// ---------------------------------------------------------------------------
// Paired Peers Registry (`paired_peers.json`)
// ---------------------------------------------------------------------------

pub fn load_paired_peers() -> Vec<PairedPeer> {
    let path = aeromesh_core::paired_peers_file();
    let Ok(raw) = std::fs::read_to_string(&path) else {
        return Vec::new();
    };
    serde_json::from_str::<Vec<PairedPeer>>(&raw).unwrap_or_default()
}

pub fn save_paired_peers(peers: &[PairedPeer]) {
    let path = aeromesh_core::paired_peers_file();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(payload) = serde_json::to_string_pretty(peers) {
        let _ = std::fs::write(&path, payload);
    }
}

pub fn upsert_paired_peer(peer: PairedPeer) -> Vec<PairedPeer> {
    let mut peers = load_paired_peers();
    if let Some(existing) = peers.iter_mut().find(|p| {
        p.id == peer.id
            || (!peer.worker_endpoint.is_empty() && p.worker_endpoint == peer.worker_endpoint)
    }) {
        *existing = peer;
    } else {
        peers.insert(0, peer);
    }
    save_paired_peers(&peers);
    peers
}

pub fn remove_paired_peer(id_or_endpoint: &str) -> Vec<PairedPeer> {
    let target = id_or_endpoint.trim();
    let mut peers = load_paired_peers();
    peers.retain(|p| p.id != target && p.worker_endpoint != target);
    save_paired_peers(&peers);
    peers
}

// ---------------------------------------------------------------------------
// Invite Code & Target Validation
// ---------------------------------------------------------------------------

/// Parses either a compact invite string (`AERO-192.168.1.42:50053-842917`)
/// or explicit `(host, pin)` inputs.
fn parse_join_target(
    invite: &str,
    host_raw: &str,
    pin_raw: &str,
) -> Result<(String, u16, String), String> {
    let mut host_part = host_raw.trim().to_string();
    let mut pin_part = pin_raw.trim().to_string();

    let inv = invite.trim();
    if !inv.is_empty() {
        let stripped = inv
            .strip_prefix("AERO-")
            .or_else(|| inv.strip_prefix("aero-"))
            .unwrap_or(inv);
        if let Some((h, p)) = stripped.rsplit_once('-') {
            host_part = h.trim().to_string();
            pin_part = p.trim().to_string();
        } else {
            return Err("Invalid invite code format (expected AERO-<ip>:50053-<6-digit-pin>).".to_string());
        }
    }

    let pin_digits: String = pin_part.chars().filter(|c| c.is_ascii_digit()).collect();
    if pin_digits.len() != 6 {
        return Err("Pairing code must be exactly 6 digits (e.g. 482-915).".to_string());
    }

    let cleaned_host = host_part
        .trim_start_matches("http://")
        .trim_start_matches("https://")
        .trim_end_matches('/');
    if cleaned_host.is_empty() || cleaned_host.contains('/') || cleaned_host.contains(' ') {
        return Err("Enter a valid peer IP address or hostname.".to_string());
    }

    let (host, port) = if let Some((h, p_str)) = cleaned_host.rsplit_once(':') {
        if let Ok(port_num) = p_str.parse::<u16>() {
            (h.trim().to_string(), port_num)
        } else {
            return Err("Invalid pairing port number.".to_string());
        }
    } else {
        (cleaned_host.to_string(), DEFAULT_PAIR_PORT)
    };

    if port < 1024 {
        return Err("Pairing port must be between 1024 and 65535.".to_string());
    }

    // Block cloud metadata link-local addresses
    if let Ok(ip) = host.parse::<IpAddr>() {
        if let IpAddr::V4(v4) = ip {
            let oct = v4.octets();
            if oct[0] == 169 && oct[1] == 254 {
                return Err("Link-local metadata addresses are not permitted for pairing.".to_string());
            }
        }
    } else {
        let valid_hostname = !host.is_empty()
            && host.len() <= 253
            && host
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-'));
        if !valid_hostname {
            return Err("Invalid peer hostname.".to_string());
        }
    }

    Ok((host, port, pin_digits))
}

async fn read_bounded_json_line(
    reader: &mut BufReader<tokio::net::tcp::ReadHalf<'_>>,
) -> Result<Value, String> {
    let mut line = String::new();
    let n = tokio::time::timeout(Duration::from_secs(8), reader.read_line(&mut line))
        .await
        .map_err(|_| "Timed out waiting for peer message.".to_string())?
        .map_err(|e| format!("Socket read error: {}", e))?;
    if n == 0 {
        return Err("Peer closed connection.".to_string());
    }
    if line.len() > MAX_LINE_BYTES {
        return Err("Peer message exceeded maximum allowed length.".to_string());
    }
    serde_json::from_str::<Value>(line.trim()).map_err(|_| "Malformed JSON from peer.".to_string())
}

async fn write_json_line(
    writer: &mut tokio::net::tcp::WriteHalf<'_>,
    val: &Value,
) -> Result<(), String> {
    let mut raw = serde_json::to_string(val).map_err(|e| e.to_string())?;
    raw.push('\n');
    tokio::time::timeout(Duration::from_secs(8), writer.write_all(raw.as_bytes()))
        .await
        .map_err(|_| "Timed out writing to peer.".to_string())?
        .map_err(|e| format!("Socket write error: {}", e))?;
    let _ = writer.flush().await;
    Ok(())
}

// ---------------------------------------------------------------------------
// Host Pairing Session TCP Listener (`0.0.0.0:50053`)
// ---------------------------------------------------------------------------

async fn handle_pair_client(
    mut stream: TcpStream,
    peer_addr: SocketAddr,
    session: ActivePairSession,
) {
    let (rd, mut wr) = stream.split();
    let mut reader = BufReader::new(rd);

    let first_msg = match read_bounded_json_line(&mut reader).await {
        Ok(v) => v,
        Err(_) => return,
    };

    let op = first_msg.get("op").and_then(|v| v.as_str()).unwrap_or("");
    let wifi_ip = local_wifi_ip();
    let ts_ip = local_tailscale_ip();
    let node_name = local_node_name();
    let gpu = local_gpu_summary();
    let models = local_llm_models();
    let worker_endpoint = preferred_worker_endpoint(&wifi_ip, &ts_ip);
    let secret = current_worker_secret();
    let fp = secret_fingerprint(&secret);

    let now = now_unix_secs();
    if !session.active.load(Ordering::Relaxed) || now >= session.expires_at {
        let _ = write_json_line(&mut wr, &json!({"ok": false, "error": "Pairing session expired."})).await;
        return;
    }

    if op == "probe" {
        let _ = write_json_line(
            &mut wr,
            &json!({
                "ok": true,
                "protocol": "aeromesh-pair-v1",
                "node_name": node_name,
                "gpu": gpu,
                "wifi_ip": wifi_ip,
                "tailscale_ip": ts_ip,
                "pair_port": session.port,
                "expires_in": session.expires_at.saturating_sub(now),
                "cluster_fingerprint": fp
            }),
        )
        .await;
        return;
    }

    if op != "hello" {
        let _ = write_json_line(&mut wr, &json!({"ok": false, "error": "Expected hello or probe."})).await;
        return;
    }

    let client_nonce_hex = first_msg
        .get("client_nonce")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let Ok(client_nonce) = hex::decode(client_nonce_hex) else {
        let _ = write_json_line(&mut wr, &json!({"ok": false, "error": "Invalid client nonce."})).await;
        return;
    };
    if client_nonce.len() != 16 {
        let _ = write_json_line(&mut wr, &json!({"ok": false, "error": "Invalid client nonce length."})).await;
        return;
    }

    let host_nonce = random_bytes::<16>();
    let challenge = json!({
        "ok": true,
        "protocol": "aeromesh-pair-v1",
        "host_nonce": hex::encode(host_nonce),
        "salt": hex::encode(session.salt),
        "node_name": node_name,
        "gpu": gpu,
        "wifi_ip": wifi_ip,
        "tailscale_ip": ts_ip,
        "worker_endpoint": worker_endpoint,
        "models": models,
        "cluster_fingerprint": fp
    });
    if write_json_line(&mut wr, &challenge).await.is_err() {
        return;
    }

    let verify_msg = match read_bounded_json_line(&mut reader).await {
        Ok(v) => v,
        Err(_) => return,
    };
    if verify_msg.get("op").and_then(|v| v.as_str()) != Some("verify") {
        let _ = write_json_line(&mut wr, &json!({"ok": false, "error": "Expected verify step."})).await;
        return;
    }

    let proof_hex = verify_msg
        .get("proof")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let Ok(proof_bytes) = hex::decode(proof_hex) else {
        let _ = write_json_line(&mut wr, &json!({"ok": false, "error": "Malformed proof."})).await;
        return;
    };

    let pin_copy = session.pin.clone();
    let salt_copy = session.salt;
    let cn_copy = client_nonce.clone();
    let keys = tokio::task::spawn_blocking(move || {
        derive_session_keys(&pin_copy, &salt_copy, &host_nonce, &cn_copy)
    })
    .await
    .unwrap();

    let expected_proof = compute_client_proof(&keys.auth_key, &client_nonce, &host_nonce);
    if !ct_eq(&expected_proof, &proof_bytes) {
        let fails = session.failed_attempts.fetch_add(1, Ordering::Relaxed) + 1;
        if fails >= PAIR_MAX_FAILED_ATTEMPTS {
            session.active.store(false, Ordering::Relaxed);
            if let Ok(mut tx) = session.shutdown_tx.lock() {
                if let Some(s) = tx.take() {
                    let _ = s.send(());
                }
            }
            let _ = write_json_line(
                &mut wr,
                &json!({"ok": false, "error": "Too many invalid attempts. Pairing session locked and closed."}),
            )
            .await;
        } else {
            let remaining = PAIR_MAX_FAILED_ATTEMPTS.saturating_sub(fails);
            let _ = write_json_line(
                &mut wr,
                &json!({"ok": false, "error": format!("Invalid pairing code ({} attempts left).", remaining)}),
            )
            .await;
        }
        return;
    }

    // Proof verified! Encrypt the cluster secret with Encrypt-then-MAC.
    let secret_payload = json!({
        "worker_secret": secret,
        "cluster_fingerprint": fp
    })
    .to_string();
    let (iv, ciphertext, tag) = encrypt_and_mac(&keys.enc_key, &keys.mac_key, secret_payload.as_bytes());
    let host_proof = compute_host_proof(&keys.auth_key, &host_nonce, &client_nonce);

    // Record joining peer in local paired_peers.json
    let peer_node_name = first_msg
        .get("node_name")
        .and_then(|v| v.as_str())
        .unwrap_or("Remote-Node")
        .chars()
        .take(80)
        .collect::<String>();
    let peer_gpu = first_msg
        .get("gpu")
        .and_then(|v| v.as_str())
        .unwrap_or("GPU / CPU")
        .chars()
        .take(120)
        .collect::<String>();
    let peer_wifi = first_msg
        .get("wifi_ip")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty() && *s != "127.0.0.1")
        .map(|s| s.to_string())
        .unwrap_or_else(|| peer_addr.ip().to_string());
    let peer_ts = first_msg
        .get("tailscale_ip")
        .and_then(|v| v.as_str())
        .unwrap_or("Not Connected")
        .to_string();
    let peer_worker_ep = first_msg
        .get("worker_endpoint")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty() && !s.starts_with("127.0.0.1"))
        .map(|s| s.to_string())
        .unwrap_or_else(|| preferred_worker_endpoint(&peer_wifi, &peer_ts));
    let peer_models: Vec<String> = first_msg
        .get("models")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|x| x.as_str().map(|s| s.to_string()))
                .take(100)
                .collect()
        })
        .unwrap_or_default();

    let peer_id = hex::encode(&Sha256::digest(peer_worker_ep.as_bytes())[..6]);
    let ts_now = now_unix_secs();
    upsert_paired_peer(PairedPeer {
        id: peer_id,
        node_name: peer_node_name,
        gpu: peer_gpu,
        wifi_ip: peer_wifi,
        tailscale_ip: peer_ts,
        worker_endpoint: peer_worker_ep,
        models: peer_models,
        cluster_fingerprint: fp,
        paired_at: ts_now,
        last_seen_at: ts_now,
    });
    session.paired_count.fetch_add(1, Ordering::Relaxed);

    let _ = write_json_line(
        &mut wr,
        &json!({
            "ok": true,
            "host_proof": hex::encode(host_proof),
            "iv": hex::encode(iv),
            "ciphertext": hex::encode(ciphertext),
            "tag": hex::encode(tag)
        }),
    )
    .await;
}

// ---------------------------------------------------------------------------
// HTTP Endpoints (`/api/cluster/pair/*`)
// ---------------------------------------------------------------------------

pub fn build_pair_status_payload(state: &PairingState) -> Value {
    let secret = current_worker_secret();
    let fp = secret_fingerprint(&secret);
    let wifi_ip = local_wifi_ip();
    let ts_ip = local_tailscale_ip();
    let local_models = local_llm_models();
    let now = now_unix_secs();

    let mut active_session_json = Value::Null;
    if let Ok(mut guard) = state.session.lock() {
        if let Some(sess) = guard.as_ref() {
            let alive = sess.active.load(Ordering::Relaxed) && now < sess.expires_at;
            if alive {
                let primary_host = if !ts_ip.is_empty() && ts_ip != "Not Connected" && ts_ip.contains('.') {
                    &ts_ip
                } else {
                    &wifi_ip
                };
                let invite_lan = format!("AERO-{}:{}-{}", wifi_ip, sess.port, sess.pin);
                let invite_primary = format!("AERO-{}:{}-{}", primary_host, sess.port, sess.pin);
                active_session_json = json!({
                    "active": true,
                    "pin": sess.pin,
                    "pin_formatted": format_pin_display(&sess.pin),
                    "port": sess.port,
                    "expires_in": sess.expires_at.saturating_sub(now),
                    "failed_attempts": sess.failed_attempts.load(Ordering::Relaxed),
                    "max_attempts": PAIR_MAX_FAILED_ATTEMPTS,
                    "paired_count": sess.paired_count.load(Ordering::Relaxed),
                    "invite_code": invite_primary,
                    "invite_lan": invite_lan
                });
            } else {
                *guard = None;
            }
        }
    }

    let peers: Vec<Value> = load_paired_peers()
        .into_iter()
        .map(|p| {
            let secret_synced = p.cluster_fingerprint == fp && fp != "UNSET";
            let shared_models: Vec<String> = p
                .models
                .iter()
                .filter(|m| local_models.contains(*m))
                .cloned()
                .collect();
            json!({
                "id": p.id,
                "node_name": p.node_name,
                "gpu": p.gpu,
                "wifi_ip": p.wifi_ip,
                "tailscale_ip": p.tailscale_ip,
                "worker_endpoint": p.worker_endpoint,
                "models": p.models,
                "shared_models": shared_models,
                "cluster_fingerprint": p.cluster_fingerprint,
                "secret_synced": secret_synced,
                "paired_at": p.paired_at,
                "last_seen_at": p.last_seen_at
            })
        })
        .collect();

    json!({
        "node_name": local_node_name(),
        "gpu": local_gpu_summary(),
        "wifi_ip": wifi_ip,
        "tailscale_ip": ts_ip,
        "worker_endpoint": preferred_worker_endpoint(&wifi_ip, &ts_ip),
        "secret_configured": !secret.trim().is_empty(),
        "cluster_fingerprint": fp,
        "local_models": local_models,
        "session": active_session_json,
        "paired_peers": peers
    })
}

pub async fn get_pair_status(State(state): State<PairingState>) -> impl IntoResponse {
    (StatusCode::OK, Json(build_pair_status_payload(&state)))
}

pub async fn post_pair_start(State(state): State<PairingState>) -> impl IntoResponse {
    // Stop any previous active session cleanly
    if let Ok(mut guard) = state.session.lock() {
        if let Some(prev) = guard.take() {
            prev.active.store(false, Ordering::Relaxed);
            if let Ok(mut tx) = prev.shutdown_tx.lock() {
                if let Some(s) = tx.take() {
                    let _ = s.send(());
                }
            }
        }
    }

    // Bind ephemeral TCP pairing listener on 50053..=50058
    let mut listener_opt = None;
    let mut bound_port = DEFAULT_PAIR_PORT;
    for port in DEFAULT_PAIR_PORT..=(DEFAULT_PAIR_PORT + 5) {
        if let Ok(l) = TcpListener::bind(("0.0.0.0", port)).await {
            listener_opt = Some(l);
            bound_port = port;
            break;
        }
    }
    let Some(listener) = listener_opt else {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"detail": "Could not bind pairing port 50053 (check firewall or existing session)."})),
        );
    };

    let pin = generate_six_digit_pin();
    let salt = random_bytes::<16>();
    let expires_at = now_unix_secs() + PAIR_SESSION_TTL_SECS;
    let (shutdown_tx, mut shutdown_rx) = oneshot::channel::<()>();

    let session = ActivePairSession {
        pin,
        port: bound_port,
        salt,
        expires_at,
        failed_attempts: Arc::new(AtomicU32::new(0)),
        paired_count: Arc::new(AtomicU32::new(0)),
        active: Arc::new(AtomicBool::new(true)),
        shutdown_tx: Arc::new(Mutex::new(Some(shutdown_tx))),
    };

    if let Ok(mut guard) = state.session.lock() {
        *guard = Some(session.clone());
    }

    let state_clone = state.clone();
    let sess_loop = session.clone();
    tokio::spawn(async move {
        let sleep_ttl = tokio::time::sleep(Duration::from_secs(PAIR_SESSION_TTL_SECS));
        tokio::pin!(sleep_ttl);

        loop {
            tokio::select! {
                _ = &mut shutdown_rx => {
                    break;
                }
                _ = &mut sleep_ttl => {
                    break;
                }
                accept_res = listener.accept() => {
                    if !sess_loop.active.load(Ordering::Relaxed) {
                        break;
                    }
                    if let Ok((stream, addr)) = accept_res {
                        let s_ref = sess_loop.clone();
                        tokio::spawn(async move {
                            handle_pair_client(stream, addr, s_ref).await;
                        });
                    }
                }
            }
        }
        sess_loop.active.store(false, Ordering::Relaxed);
        if let Ok(mut guard) = state_clone.session.lock() {
            if let Some(cur) = guard.as_ref() {
                if cur.pin == sess_loop.pin {
                    *guard = None;
                }
            }
        }
    });

    (
        StatusCode::OK,
        Json(json!({"status": "started", "pair_status": build_pair_status_payload(&state)})),
    )
}

pub async fn post_pair_stop(State(state): State<PairingState>) -> impl IntoResponse {
    if let Ok(mut guard) = state.session.lock() {
        if let Some(prev) = guard.take() {
            prev.active.store(false, Ordering::Relaxed);
            if let Ok(mut tx) = prev.shutdown_tx.lock() {
                if let Some(s) = tx.take() {
                    let _ = s.send(());
                }
            }
        }
    }
    (
        StatusCode::OK,
        Json(json!({"status": "stopped", "pair_status": build_pair_status_payload(&state)})),
    )
}

#[derive(Deserialize)]
pub struct PairJoinReq {
    pub invite_code: Option<String>,
    pub host: Option<String>,
    pub pin: Option<String>,
}

pub async fn post_pair_join(
    State(state): State<PairingState>,
    Json(req): Json<PairJoinReq>,
) -> impl IntoResponse {
    let invite = req.invite_code.unwrap_or_default();
    let host_raw = req.host.unwrap_or_default();
    let pin_raw = req.pin.unwrap_or_default();

    let (host, port, pin_digits) = match parse_join_target(&invite, &host_raw, &pin_raw) {
        Ok(v) => v,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(json!({"detail": e}))),
    };

    let target = format!("{}:{}", host, port);
    let mut stream = match tokio::time::timeout(Duration::from_secs(6), TcpStream::connect(&target)).await {
        Ok(Ok(s)) => s,
        Ok(Err(e)) => {
            return (
                StatusCode::BAD_GATEWAY,
                Json(json!({"detail": format!("Could not connect to pairing host {}: {}", target, e)})),
            )
        }
        Err(_) => {
            return (
                StatusCode::GATEWAY_TIMEOUT,
                Json(json!({"detail": format!("Timed out connecting to pairing host {} (ensure 'Host Pairing Session' is open on that PC and port {} is allowed).", target, port)})),
            )
        }
    };

    let (rd, mut wr) = stream.split();
    let mut reader = BufReader::new(rd);

    let client_nonce = random_bytes::<16>();
    let my_wifi = local_wifi_ip();
    let my_ts = local_tailscale_ip();
    let hello = json!({
        "op": "hello",
        "protocol": "aeromesh-pair-v1",
        "client_nonce": hex::encode(client_nonce),
        "node_name": local_node_name(),
        "gpu": local_gpu_summary(),
        "wifi_ip": my_wifi,
        "tailscale_ip": my_ts,
        "worker_endpoint": preferred_worker_endpoint(&my_wifi, &my_ts),
        "models": local_llm_models()
    });
    if let Err(e) = write_json_line(&mut wr, &hello).await {
        return (StatusCode::BAD_GATEWAY, Json(json!({"detail": e})));
    }

    let challenge = match read_bounded_json_line(&mut reader).await {
        Ok(v) => v,
        Err(e) => return (StatusCode::BAD_GATEWAY, Json(json!({"detail": e}))),
    };
    if challenge.get("ok").and_then(|v| v.as_bool()) != Some(true) {
        let msg = challenge
            .get("error")
            .and_then(|v| v.as_str())
            .unwrap_or("Host rejected pairing hello.");
        return (StatusCode::BAD_REQUEST, Json(json!({"detail": msg})));
    }

    let host_nonce = match challenge
        .get("host_nonce")
        .and_then(|v| v.as_str())
        .and_then(|s| hex::decode(s).ok())
        .filter(|b| b.len() == 16)
    {
        Some(b) => b,
        None => {
            return (
                StatusCode::BAD_GATEWAY,
                Json(json!({"detail": "Invalid host nonce in pairing handshake."})),
            )
        }
    };
    let salt = match challenge
        .get("salt")
        .and_then(|v| v.as_str())
        .and_then(|s| hex::decode(s).ok())
        .filter(|b| b.len() == 16)
    {
        Some(b) => b,
        None => {
            return (
                StatusCode::BAD_GATEWAY,
                Json(json!({"detail": "Invalid salt in pairing handshake."})),
            )
        }
    };

    let hn_copy = host_nonce.clone();
    let salt_copy = salt.clone();
    let pin_copy = pin_digits.clone();
    let keys = tokio::task::spawn_blocking(move || {
        derive_session_keys(&pin_copy, &salt_copy, &hn_copy, &client_nonce)
    })
    .await
    .unwrap();

    let proof = compute_client_proof(&keys.auth_key, &client_nonce, &host_nonce);
    if let Err(e) = write_json_line(
        &mut wr,
        &json!({"op": "verify", "proof": hex::encode(proof)}),
    )
    .await
    {
        return (StatusCode::BAD_GATEWAY, Json(json!({"detail": e})));
    }

    let resp = match read_bounded_json_line(&mut reader).await {
        Ok(v) => v,
        Err(e) => return (StatusCode::BAD_GATEWAY, Json(json!({"detail": e}))),
    };
    if resp.get("ok").and_then(|v| v.as_bool()) != Some(true) {
        let msg = resp
            .get("error")
            .and_then(|v| v.as_str())
            .unwrap_or("Pairing code rejected by host.");
        return (StatusCode::UNAUTHORIZED, Json(json!({"detail": msg})));
    }

    // Verify mutual host proof before decrypting
    let host_proof_bytes = resp
        .get("host_proof")
        .and_then(|v| v.as_str())
        .and_then(|s| hex::decode(s).ok())
        .unwrap_or_default();
    let expected_host_proof = compute_host_proof(&keys.auth_key, &host_nonce, &client_nonce);
    if !ct_eq(&expected_host_proof, &host_proof_bytes) {
        return (
            StatusCode::BAD_GATEWAY,
            Json(json!({"detail": "Mutual host authentication failed. Aborted secret sync."})),
        );
    }

    let iv = resp
        .get("iv")
        .and_then(|v| v.as_str())
        .and_then(|s| hex::decode(s).ok())
        .unwrap_or_default();
    let ct = resp
        .get("ciphertext")
        .and_then(|v| v.as_str())
        .and_then(|s| hex::decode(s).ok())
        .unwrap_or_default();
    let tag = resp
        .get("tag")
        .and_then(|v| v.as_str())
        .and_then(|s| hex::decode(s).ok())
        .unwrap_or_default();

    let plaintext = match verify_and_decrypt(&keys.enc_key, &keys.mac_key, &iv, &ct, &tag) {
        Ok(p) => p,
        Err(e) => return (StatusCode::BAD_GATEWAY, Json(json!({"detail": e}))),
    };
    let Ok(payload_json) = serde_json::from_slice::<Value>(&plaintext) else {
        return (
            StatusCode::BAD_GATEWAY,
            Json(json!({"detail": "Decrypted pairing payload was malformed."})),
        );
    };
    let new_secret = payload_json
        .get("worker_secret")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim();
    if new_secret.len() < 16 {
        return (
            StatusCode::BAD_GATEWAY,
            Json(json!({"detail": "Host did not provide a valid cluster secret."})),
        );
    }

    if let Err(e) = aeromesh_core::upsert_env_var("AEROMESH_WORKER_SECRET", new_secret) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"detail": format!("Synced secret could not be saved to .env: {}", e)})),
        );
    }

    let fp = secret_fingerprint(new_secret);
    let host_name = challenge
        .get("node_name")
        .and_then(|v| v.as_str())
        .unwrap_or(&host)
        .to_string();
    let host_gpu = challenge
        .get("gpu")
        .and_then(|v| v.as_str())
        .unwrap_or("GPU / CPU")
        .to_string();
    let host_wifi = challenge
        .get("wifi_ip")
        .and_then(|v| v.as_str())
        .unwrap_or(&host)
        .to_string();
    let host_ts = challenge
        .get("tailscale_ip")
        .and_then(|v| v.as_str())
        .unwrap_or("Not Connected")
        .to_string();
    let host_worker_ep = challenge
        .get("worker_endpoint")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty() && !s.starts_with("127.0.0.1"))
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("{}:50052", host));
    let host_models: Vec<String> = challenge
        .get("models")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|x| x.as_str().map(|s| s.to_string()))
                .take(100)
                .collect()
        })
        .unwrap_or_default();

    let peer_id = hex::encode(&Sha256::digest(host_worker_ep.as_bytes())[..6]);
    let ts_now = now_unix_secs();
    upsert_paired_peer(PairedPeer {
        id: peer_id,
        node_name: host_name.clone(),
        gpu: host_gpu,
        wifi_ip: host_wifi,
        tailscale_ip: host_ts,
        worker_endpoint: host_worker_ep,
        models: host_models,
        cluster_fingerprint: fp.clone(),
        paired_at: ts_now,
        last_seen_at: ts_now,
    });

    (
        StatusCode::OK,
        Json(json!({
            "status": "paired",
            "paired_with": host_name,
            "cluster_fingerprint": fp,
            "pair_status": build_pair_status_payload(&state)
        })),
    )
}

pub async fn get_pair_discover() -> impl IntoResponse {
    let mut candidates: HashMap<String, String> = HashMap::new();
    let my_wifi = local_wifi_ip();
    let my_ts = local_tailscale_ip();

    #[cfg(target_os = "windows")]
    {
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
                                if let Some(ips) = peer.get("TailscaleIPs").and_then(|i| i.as_array()) {
                                    for ip in ips.iter().filter_map(|x| x.as_str()) {
                                        if ip.contains('.') && ip != my_ts {
                                            candidates.insert(ip.to_string(), "Tailscale".to_string());
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        if let Ok(out) = Command::new("arp")
            .creation_flags(CREATE_NO_WINDOW)
            .arg("-a")
            .output()
        {
            for line in String::from_utf8_lossy(&out.stdout).lines() {
                if line.contains("dynamic") {
                    if let Some(ip_tok) = line.split_whitespace().next() {
                        if ip_tok != my_wifi
                            && !ip_tok.ends_with(".255")
                            && !ip_tok.starts_with("224.")
                            && !ip_tok.starts_with("239.")
                        {
                            candidates.entry(ip_tok.to_string()).or_insert_with(|| "Wi-Fi LAN".to_string());
                        }
                    }
                }
            }
        }
    }
    let _ = (&my_wifi, &my_ts);

    let mut handles = Vec::new();
    for (ip, net_type) in candidates {
        handles.push(tokio::spawn(async move {
            let target = format!("{}:{}", ip, DEFAULT_PAIR_PORT);
            let Ok(Ok(mut stream)) =
                tokio::time::timeout(Duration::from_millis(450), TcpStream::connect(&target)).await
            else {
                return None;
            };
            let (rd, mut wr) = stream.split();
            let mut reader = BufReader::new(rd);
            if write_json_line(&mut wr, &json!({"op": "probe"})).await.is_err() {
                return None;
            }
            let info = read_bounded_json_line(&mut reader).await.ok()?;
            if info.get("ok").and_then(|v| v.as_bool()) != Some(true) {
                return None;
            }
            Some(json!({
                "host": ip,
                "port": DEFAULT_PAIR_PORT,
                "network": net_type,
                "node_name": info.get("node_name").and_then(|v| v.as_str()).unwrap_or("AeroMesh-Host"),
                "gpu": info.get("gpu").and_then(|v| v.as_str()).unwrap_or("GPU / CPU"),
                "expires_in": info.get("expires_in").and_then(|v| v.as_u64()).unwrap_or(0),
                "cluster_fingerprint": info.get("cluster_fingerprint").and_then(|v| v.as_str()).unwrap_or("")
            }))
        }));
    }

    let mut hosts = Vec::new();
    for h in handles {
        if let Ok(Some(item)) = h.await {
            hosts.push(item);
        }
    }

    (StatusCode::OK, Json(json!({"pairing_hosts": hosts})))
}

pub async fn post_pair_rotate_secret(State(state): State<PairingState>) -> impl IntoResponse {
    // Close any active pairing session first so old code cannot leak the new secret
    if let Ok(mut guard) = state.session.lock() {
        if let Some(prev) = guard.take() {
            prev.active.store(false, Ordering::Relaxed);
            if let Ok(mut tx) = prev.shutdown_tx.lock() {
                if let Some(s) = tx.take() {
                    let _ = s.send(());
                }
            }
        }
    }

    let new_secret = hex::encode(random_bytes::<32>());
    if let Err(e) = aeromesh_core::upsert_env_var("AEROMESH_WORKER_SECRET", &new_secret) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"detail": format!("Failed to rotate cluster secret: {}", e)})),
        );
    }

    (
        StatusCode::OK,
        Json(json!({
            "status": "rotated",
            "cluster_fingerprint": secret_fingerprint(&new_secret),
            "pair_status": build_pair_status_payload(&state)
        })),
    )
}

#[derive(Deserialize)]
pub struct UnpairReq {
    pub id: String,
}

pub async fn post_pair_remove(
    State(state): State<PairingState>,
    Json(req): Json<UnpairReq>,
) -> impl IntoResponse {
    remove_paired_peer(&req.id);
    (
        StatusCode::OK,
        Json(json!({"status": "removed", "pair_status": build_pair_status_payload(&state)})),
    )
}
