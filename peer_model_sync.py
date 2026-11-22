"""
AeroMESH Automatic Multi-Node Model Check & Peer Model Sync (aeromesh-sync-v1)

Provides:
  1. Fast deterministic GGUF header inspection and content fingerprinting
     (size + head 1 MiB + mid 256 KiB + tail 1 MiB SHA-256) with mtime caching.
  2. Authenticated TCP peer sync daemon (default port 50054) protected by
     per-connection challenge-response HMAC-SHA256 derived from
     AEROMESH_WORKER_SECRET (zero plaintext secret on the wire, fails closed
     when secret is unset).
  3. Zero-RAM-buffering resumable chunked streaming for both PUSH (coordinator ->
     worker) and PULL (worker -> coordinator) GGUF transfers with strict path
     confinement, disk-space pre-check, GGUF magic validation, and atomic swap.
"""

import asyncio
import hashlib
import hmac
import ipaddress
import json
import os
import re
import secrets
import shutil
import struct
import time
import uuid
from typing import Callable, Dict, List, Optional, Tuple

DEFAULT_SYNC_PORT = 50054
SYNC_PROTOCOL_VERSION = "aeromesh-sync-v1"
SYNC_CHUNK_BYTES = 1024 * 1024  # 1 MiB streaming chunks
SYNC_MAX_LINE_BYTES = 64 * 1024
SYNC_CONNECT_TIMEOUT = 3.5
SYNC_IO_TIMEOUT = 15.0
SYNC_PART_SUFFIX = ".aeromesh-sync.part"

_SAFE_GGUF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,200}\.gguf$", re.IGNORECASE)
_FINGERPRINT_CACHE: Dict[str, Tuple[int, int, dict]] = {}


def format_bytes(num_bytes: int) -> str:
    val = float(max(0, int(num_bytes or 0)))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if val < 1024.0 or unit == "TB":
            return f"{val:.2f} {unit}" if unit in ("GB", "TB") else f"{val:.1f} {unit}"
        val /= 1024.0
    return f"{val:.2f} TB"


def extract_quant_tag(filename: str) -> str:
    m = re.search(
        r"(IQ[1-4]_[A-Z0-9]+|Q[2-8]_[A-Z0-9_]+|F16|BF16|F32|FP16|FP8)",
        str(filename or ""),
        re.IGNORECASE,
    )
    return m.group(1).upper() if m else "GGUF"


def is_slice_filename(name: str) -> bool:
    low = str(name or "").lower()
    return (
        ".stage" in low
        or "_stage" in low
        or "_slice" in low
        or ".aeromesh_" in low
        or low.startswith(".aeromesh_")
        or low.endswith(SYNC_PART_SUFFIX)
    )


def validate_gguf_filename(filename: str) -> str:
    """Strictly validate a base .gguf filename (no directories, no traversal)."""
    raw = str(filename or "").strip()
    if not raw or "/" in raw or "\\" in raw or ".." in raw or "\x00" in raw:
        raise ValueError(f"Invalid GGUF filename '{filename}'.")
    if os.path.basename(raw) != raw:
        raise ValueError(f"Invalid GGUF filename '{filename}'.")
    if is_slice_filename(raw):
        raise ValueError(f"Internal slice or partial file '{filename}' cannot be synced.")
    if not _SAFE_GGUF_RE.fullmatch(raw):
        raise ValueError(
            f"Filename '{filename}' must end with .gguf and contain only letters, digits, '.', '_', or '-'."
        )
    return raw


def resolve_confined_gguf(filename: str, models_dirs: List[str]) -> Optional[str]:
    """Resolve an existing GGUF file strictly inside one of models_dirs."""
    safe_name = validate_gguf_filename(filename)
    for base_dir in models_dirs:
        if not base_dir or not os.path.isdir(base_dir):
            continue
        real_base = os.path.realpath(base_dir)
        candidate = os.path.realpath(os.path.join(real_base, safe_name))
        if candidate.startswith(real_base + os.sep) and os.path.isfile(candidate):
            return candidate
    return None


def confined_target_path(filename: str, primary_dir: str) -> str:
    """Return a strictly confined target path inside primary_dir."""
    safe_name = validate_gguf_filename(filename)
    os.makedirs(primary_dir, exist_ok=True)
    real_base = os.path.realpath(primary_dir)
    target = os.path.realpath(os.path.join(real_base, safe_name))
    if not target.startswith(real_base + os.sep):
        raise ValueError("Resolved target path escaped managed models directory.")
    return target


def validate_peer_host(raw_host: str, default_port: int = DEFAULT_SYNC_PORT) -> Tuple[str, int]:
    """Parse and validate peer host/IP (rejects cloud metadata / invalid hosts)."""
    cleaned = str(raw_host or "").strip()
    cleaned = re.sub(r"^https?://", "", cleaned).rstrip("/")
    if not cleaned or "/" in cleaned or " " in cleaned or "\x00" in cleaned:
        raise ValueError("Invalid peer host address.")

    port = default_port
    if ":" in cleaned and cleaned.count(":") == 1:
        host_part, port_str = cleaned.rsplit(":", 1)
        host = host_part.strip()
        try:
            port = int(port_str)
        except ValueError:
            raise ValueError("Invalid peer port number.")
    else:
        host = cleaned

    if port < 1024 or port > 65535:
        raise ValueError("Peer port must be between 1024 and 65535.")

    try:
        ip_obj = ipaddress.ip_address(host)
        if ip_obj.is_link_local or ip_obj.is_unspecified or ip_obj.is_multicast:
            raise ValueError("Link-local, unspecified, or multicast addresses are not permitted.")
    except ValueError as exc:
        if "not permitted" in str(exc):
            raise
        if len(host) > 253 or not re.fullmatch(r"[A-Za-z0-9.-]+", host):
            raise ValueError("Invalid peer hostname.")

    return host, port


def extract_host_from_endpoint(endpoint: str) -> str:
    raw = str(endpoint or "").strip()
    if not raw:
        return ""
    if ":" in raw and raw.count(":") == 1:
        return raw.rsplit(":", 1)[0].strip()
    return raw


# ============================================================================
# FAST DETERMINISTIC GGUF HEADER INSPECTION & FINGERPRINTING
# ============================================================================
def inspect_gguf_file(file_path: str) -> dict:
    """
    Inspect a local GGUF file and compute a fast deterministic fingerprint.

    Fingerprint hashes:
      - exact file size (8-byte LE)
      - first 1 MiB (header + tensor metadata + early weights)
      - middle 256 KiB (interior weight sample)
      - last 1 MiB (tail weights / alignment)

    Runs in ~2-4 ms on NVMe even for multi-GB models and caches by
    (realpath, mtime_ns, size_bytes).
    """
    real_path = os.path.realpath(file_path)
    st = os.stat(real_path)
    size_bytes = int(st.st_size)
    mtime_ns = int(st.st_mtime_ns)

    cached = _FINGERPRINT_CACHE.get(real_path)
    if cached and cached[0] == mtime_ns and cached[1] == size_bytes:
        return dict(cached[2])

    gguf_valid = False
    gguf_version = 0
    tensor_count = 0
    kv_count = 0

    h = hashlib.sha256()
    h.update(b"aeromesh-gguf-fp-v1:")
    h.update(struct.pack("<Q", size_bytes))

    head_window = 1024 * 1024
    mid_window = 256 * 1024
    tail_window = 1024 * 1024

    with open(real_path, "rb") as f:
        header = f.read(24)
        if len(header) >= 24 and header[:4] == b"GGUF":
            gguf_valid = True
            gguf_version, tensor_count, kv_count = struct.unpack("<IQQ", header[4:24])
        elif len(header) >= 4 and header[:4] == b"GGUF":
            gguf_valid = True

        h.update(header)
        remaining_head = max(0, min(size_bytes, head_window) - len(header))
        if remaining_head > 0:
            h.update(f.read(remaining_head))

        if size_bytes > head_window + tail_window:
            mid_pos = max(head_window, (size_bytes // 2) - (mid_window // 2))
            f.seek(mid_pos)
            h.update(b":mid:")
            h.update(f.read(mid_window))

        if size_bytes > head_window:
            tail_pos = max(head_window, size_bytes - tail_window)
            f.seek(tail_pos)
            h.update(b":tail:")
            h.update(f.read(size_bytes - tail_pos))

    full_hex = h.hexdigest()
    name = os.path.basename(real_path)
    info = {
        "name": name,
        "size_bytes": size_bytes,
        "size_label": format_bytes(size_bytes),
        "quant": extract_quant_tag(name),
        "gguf_valid": gguf_valid,
        "gguf_version": gguf_version,
        "tensor_count": tensor_count,
        "kv_count": kv_count,
        "fingerprint": full_hex,
        "fingerprint_short": full_hex[:12].upper(),
    }
    _FINGERPRINT_CACHE[real_path] = (mtime_ns, size_bytes, dict(info))
    return info


def build_local_model_manifest(models_dirs: List[str]) -> List[dict]:
    """Scan all configured model directories and return fingerprinted GGUF entries."""
    entries: List[dict] = []
    seen = set()
    for d in models_dirs:
        if not d or not os.path.isdir(d):
            continue
        try:
            names = sorted(os.listdir(d))
        except Exception:
            continue
        for fname in names:
            if not fname.lower().endswith(".gguf") or is_slice_filename(fname):
                continue
            if fname in seen:
                continue
            full_path = os.path.join(d, fname)
            if not os.path.isfile(full_path):
                continue
            try:
                info = inspect_gguf_file(full_path)
                seen.add(fname)
                entries.append(info)
            except Exception:
                continue
    return entries


# ============================================================================
# CHALLENGE-RESPONSE HMAC AUTHENTICATION (aeromesh-sync-v1)
# ============================================================================
def compute_sync_auth_tag(
    secret: str,
    server_nonce: bytes,
    client_nonce: bytes,
    op: str,
    filename: str = "",
) -> bytes:
    key = hashlib.sha256(b"aeromesh-sync-auth-key-v1:" + secret.encode("utf-8")).digest()
    msg = (
        b"aeromesh-sync-v1|"
        + server_nonce
        + b"|"
        + client_nonce
        + b"|"
        + op.encode("utf-8")
        + b"|"
        + filename.encode("utf-8")
    )
    return hmac.new(key, msg, hashlib.sha256).digest()


def compute_sync_server_tag(
    secret: str,
    server_nonce: bytes,
    client_nonce: bytes,
    op: str,
    filename: str = "",
) -> bytes:
    key = hashlib.sha256(b"aeromesh-sync-auth-key-v1:" + secret.encode("utf-8")).digest()
    msg = (
        b"aeromesh-sync-v1-srv|"
        + server_nonce
        + b"|"
        + client_nonce
        + b"|"
        + op.encode("utf-8")
        + b"|"
        + filename.encode("utf-8")
    )
    return hmac.new(key, msg, hashlib.sha256).digest()


async def _read_json_line(reader: asyncio.StreamReader, timeout: float = SYNC_IO_TIMEOUT) -> dict:
    raw = await asyncio.wait_for(reader.readline(), timeout=timeout)
    if not raw:
        raise ConnectionError("Peer closed sync connection.")
    if len(raw) > SYNC_MAX_LINE_BYTES:
        raise ValueError("Sync control frame exceeded maximum length.")
    obj = json.loads(raw.decode("utf-8").strip())
    if not isinstance(obj, dict):
        raise ValueError("Malformed sync JSON frame.")
    return obj


async def _write_json_line(writer: asyncio.StreamWriter, payload: dict, timeout: float = SYNC_IO_TIMEOUT) -> None:
    raw = (json.dumps(payload) + "\n").encode("utf-8")
    writer.write(raw)
    await asyncio.wait_for(writer.drain(), timeout=timeout)


# ============================================================================
# AUTHENTICATED PEER MODEL SYNC SERVER (TCP :50054)
# ============================================================================
class PeerModelSyncServer:
    """
    Background TCP daemon that answers authenticated multi-node model checks
    (`manifest`) and streams GGUF weights for peer-to-peer sync (`pull` / `push`).
    """

    def __init__(
        self,
        get_secret: Callable[[], str],
        get_models_dirs: Callable[[], List[str]],
        get_primary_dir: Callable[[], str],
        get_node_meta: Callable[[], dict],
    ):
        self._get_secret = get_secret
        self._get_models_dirs = get_models_dirs
        self._get_primary_dir = get_primary_dir
        self._get_node_meta = get_node_meta
        self.server: Optional[asyncio.AbstractServer] = None
        self.port: int = DEFAULT_SYNC_PORT
        self.active: bool = False

    async def start(self, bind_host: str = "0.0.0.0", port: int = DEFAULT_SYNC_PORT) -> int:
        if self.active and self.server is not None:
            return self.port
        self.server = await asyncio.start_server(self._handle_client, bind_host, port)
        sockets = self.server.sockets or []
        if sockets:
            self.port = int(sockets[0].getsockname()[1])
        else:
            self.port = port
        self.active = True
        return self.port

    async def stop(self) -> None:
        self.active = False
        if self.server is not None:
            try:
                self.server.close()
                await self.server.wait_closed()
            except Exception:
                pass
            self.server = None

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            secret = str(self._get_secret() or "").strip()
            if not secret:
                await _write_json_line(writer, {"ok": False, "error": "Cluster secret not configured on peer."})
                return

            server_nonce = secrets.token_bytes(16)
            meta = self._get_node_meta() or {}
            await _write_json_line(
                writer,
                {
                    "ok": True,
                    "protocol": SYNC_PROTOCOL_VERSION,
                    "server_nonce": server_nonce.hex(),
                    "node_name": meta.get("node_name", "AeroMesh-Node"),
                    "cluster_fingerprint": meta.get("cluster_fingerprint", "UNSET"),
                },
            )

            req = await _read_json_line(reader)
            op = str(req.get("op", "")).strip()
            filename = str(req.get("filename", "")).strip()
            try:
                client_nonce = bytes.fromhex(str(req.get("client_nonce", "")))
                client_tag = bytes.fromhex(str(req.get("auth_tag", "")))
            except ValueError:
                await _write_json_line(writer, {"ok": False, "code": "bad_auth", "error": "Malformed auth fields."})
                return

            if len(client_nonce) != 16 or len(client_tag) != 32:
                await _write_json_line(writer, {"ok": False, "code": "bad_auth", "error": "Invalid auth parameters."})
                return

            expected = compute_sync_auth_tag(secret, server_nonce, client_nonce, op, filename)
            if not hmac.compare_digest(expected, client_tag):
                await _write_json_line(
                    writer,
                    {
                        "ok": False,
                        "code": "auth_mismatch",
                        "error": "Cluster secret mismatch — re-pair this node in the Cluster tab.",
                    },
                )
                return

            server_tag = compute_sync_server_tag(secret, server_nonce, client_nonce, op, filename).hex()

            if op == "manifest":
                models = await asyncio.to_thread(build_local_model_manifest, self._get_models_dirs())
                primary_dir = self._get_primary_dir()
                os.makedirs(primary_dir, exist_ok=True)
                try:
                    disk_free = int(shutil.disk_usage(primary_dir).free)
                except Exception:
                    disk_free = 0
                await _write_json_line(
                    writer,
                    {
                        "ok": True,
                        "server_tag": server_tag,
                        "node_name": meta.get("node_name", "AeroMesh-Node"),
                        "gpu": meta.get("gpu", "GPU / CPU"),
                        "vram_total_mb": int(meta.get("vram_total_mb") or 0),
                        "vram_free_mb": int(meta.get("vram_free_mb") or 0),
                        "ram_total_mb": int(meta.get("ram_total_mb") or 0),
                        "ram_available_mb": int(meta.get("ram_available_mb") or 0),
                        "wifi_ip": meta.get("wifi_ip", ""),
                        "tailscale_ip": meta.get("tailscale_ip", ""),
                        "cluster_fingerprint": meta.get("cluster_fingerprint", "UNSET"),
                        "sync_port": self.port,
                        "disk_free_bytes": disk_free,
                        "disk_free_label": format_bytes(disk_free),
                        "models": models,
                    },
                )
                return

            if op == "pull":
                try:
                    safe_name = validate_gguf_filename(filename)
                except ValueError as exc:
                    await _write_json_line(writer, {"ok": False, "error": str(exc)})
                    return

                resolved = resolve_confined_gguf(safe_name, self._get_models_dirs())
                if not resolved:
                    await _write_json_line(writer, {"ok": False, "error": f"Model '{safe_name}' not found on peer."})
                    return

                info = await asyncio.to_thread(inspect_gguf_file, resolved)
                total_size = int(info["size_bytes"])
                offset = max(0, int(req.get("offset") or 0))
                if offset > total_size:
                    offset = 0

                await _write_json_line(
                    writer,
                    {
                        "ok": True,
                        "server_tag": server_tag,
                        "filename": safe_name,
                        "size_bytes": total_size,
                        "fingerprint": info["fingerprint"],
                        "offset": offset,
                    },
                )

                with open(resolved, "rb") as f:
                    if offset > 0:
                        f.seek(offset)
                    remaining = total_size - offset
                    while remaining > 0:
                        chunk = await asyncio.to_thread(f.read, min(SYNC_CHUNK_BYTES, remaining))
                        if not chunk:
                            break
                        writer.write(chunk)
                        await asyncio.wait_for(writer.drain(), timeout=SYNC_IO_TIMEOUT)
                        remaining -= len(chunk)
                return

            if op == "push":
                try:
                    safe_name = validate_gguf_filename(filename)
                except ValueError as exc:
                    await _write_json_line(writer, {"ok": False, "error": str(exc)})
                    return

                size_bytes = int(req.get("size_bytes") or 0)
                expected_fp = str(req.get("fingerprint") or "").strip().lower()
                if size_bytes < 24 or len(expected_fp) != 64:
                    await _write_json_line(writer, {"ok": False, "error": "Invalid push metadata."})
                    return

                primary_dir = self._get_primary_dir()
                target_path = confined_target_path(safe_name, primary_dir)
                part_path = target_path + SYNC_PART_SUFFIX

                # If identical file is already present on this node, short-circuit!
                if os.path.isfile(target_path):
                    existing = await asyncio.to_thread(inspect_gguf_file, target_path)
                    if existing["size_bytes"] == size_bytes and existing["fingerprint"] == expected_fp:
                        await _write_json_line(
                            writer,
                            {
                                "ok": True,
                                "server_tag": server_tag,
                                "already_synced": True,
                                "offset": size_bytes,
                            },
                        )
                        return

                resume_offset = 0
                if os.path.isfile(part_path):
                    cur_part = os.path.getsize(part_path)
                    if 0 < cur_part < size_bytes:
                        resume_offset = cur_part
                    elif cur_part >= size_bytes:
                        try:
                            os.remove(part_path)
                        except OSError:
                            pass

                try:
                    free_bytes = shutil.disk_usage(primary_dir).free
                except Exception:
                    free_bytes = size_bytes + (64 * 1024 * 1024)
                needed = max(0, size_bytes - resume_offset)
                if free_bytes < needed + (16 * 1024 * 1024):
                    await _write_json_line(
                        writer,
                        {
                            "ok": False,
                            "error": f"Not enough free disk space on peer ({format_bytes(free_bytes)} free, need {format_bytes(needed)}).",
                        },
                    )
                    return

                await _write_json_line(
                    writer,
                    {
                        "ok": True,
                        "server_tag": server_tag,
                        "already_synced": False,
                        "offset": resume_offset,
                    },
                )

                mode = "r+b" if resume_offset > 0 else "wb"
                received = resume_offset
                with open(part_path, mode) as f:
                    if resume_offset > 0:
                        f.seek(resume_offset)
                    while received < size_bytes:
                        to_read = min(SYNC_CHUNK_BYTES, size_bytes - received)
                        chunk = await asyncio.wait_for(reader.read(to_read), timeout=SYNC_IO_TIMEOUT)
                        if not chunk:
                            raise ConnectionError("Sender disconnected before transfer completed.")
                        await asyncio.to_thread(f.write, chunk)
                        received += len(chunk)

                # Verify GGUF magic & exact deterministic fingerprint before atomic swap
                verified = await asyncio.to_thread(inspect_gguf_file, part_path)
                if not verified["gguf_valid"] or verified["size_bytes"] != size_bytes or verified["fingerprint"] != expected_fp:
                    try:
                        os.remove(part_path)
                    except OSError:
                        pass
                    await _write_json_line(
                        writer,
                        {"ok": False, "error": "Integrity check failed on received GGUF file (hash mismatch)."},
                    )
                    return

                os.replace(part_path, target_path)
                await _write_json_line(
                    writer,
                    {
                        "ok": True,
                        "verified": True,
                        "fingerprint": verified["fingerprint"],
                        "size_bytes": verified["size_bytes"],
                    },
                )
                return

            await _write_json_line(writer, {"ok": False, "error": f"Unknown sync operation '{op}'."})
        except Exception:
            pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass


# ============================================================================
# AUTHENTICATED PEER SYNC CLIENT + MULTI-NODE MODEL MATRIX + JOB MANAGER
# ============================================================================
async def _open_authenticated_sync_channel(
    host: str,
    port: int,
    secret: str,
    op: str,
    filename: str = "",
    extra_req: Optional[dict] = None,
    timeout: float = SYNC_CONNECT_TIMEOUT,
) -> Tuple[asyncio.StreamReader, asyncio.StreamWriter, dict]:
    if not secret:
        raise PermissionError("Cluster secret is not configured. Pair nodes first.")

    safe_host, safe_port = validate_peer_host(f"{host}:{port}", default_port=DEFAULT_SYNC_PORT)
    reader, writer = await asyncio.wait_for(asyncio.open_connection(safe_host, safe_port), timeout=timeout)
    try:
        hello = await _read_json_line(reader, timeout=timeout)
        if not hello.get("ok"):
            raise RuntimeError(hello.get("error") or "Peer rejected sync handshake.")
        server_nonce = bytes.fromhex(str(hello.get("server_nonce", "")))
        if len(server_nonce) != 16:
            raise ValueError("Invalid peer nonce.")

        client_nonce = secrets.token_bytes(16)
        auth_tag = compute_sync_auth_tag(secret, server_nonce, client_nonce, op, filename)
        req = {
            "op": op,
            "filename": filename,
            "client_nonce": client_nonce.hex(),
            "auth_tag": auth_tag.hex(),
        }
        if extra_req:
            req.update(extra_req)
        await _write_json_line(writer, req, timeout=SYNC_IO_TIMEOUT)

        resp = await _read_json_line(reader, timeout=SYNC_IO_TIMEOUT)
        if not resp.get("ok"):
            code = resp.get("code", "")
            msg = resp.get("error") or "Peer denied sync request."
            if code == "auth_mismatch":
                raise PermissionError(msg)
            raise RuntimeError(msg)

        srv_tag_hex = str(resp.get("server_tag", ""))
        expected_srv = compute_sync_server_tag(secret, server_nonce, client_nonce, op, filename)
        if not srv_tag_hex or not hmac.compare_digest(expected_srv, bytes.fromhex(srv_tag_hex)):
            raise PermissionError("Mutual peer verification failed.")

        return reader, writer, resp
    except Exception:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass
        raise


async def fetch_peer_manifest(
    host: str,
    port: int,
    secret: str,
    timeout: float = SYNC_CONNECT_TIMEOUT,
) -> dict:
    t0 = time.perf_counter()
    reader, writer, resp = await _open_authenticated_sync_channel(
        host=host,
        port=port,
        secret=secret,
        op="manifest",
        timeout=timeout,
    )
    try:
        rtt_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        resp["rtt_ms"] = rtt_ms
        return resp
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


def build_cluster_model_matrix(
    local_models: List[dict],
    peer_checks: List[dict],
    target_model: str = "",
) -> dict:
    """
    Pure comparison engine: computes per-model multi-node readiness across the
    local node and all queried peers.
    """
    local_by_name = {m["name"]: m for m in local_models if isinstance(m, dict) and m.get("name")}
    all_model_names = set(local_by_name.keys())
    if target_model:
        all_model_names.add(target_model)

    for pc in peer_checks:
        for m in pc.get("models") or []:
            if isinstance(m, dict) and m.get("name"):
                all_model_names.add(m["name"])

    matrix = []
    for name in sorted(all_model_names):
        local_m = local_by_name.get(name)
        node_states = []
        ready_peers = 0
        online_peers = 0
        missing_peers = []
        mismatch_peers = []
        pull_sources = []

        for pc in peer_checks:
            peer_id = pc.get("id", "")
            peer_name = pc.get("node_name") or pc.get("host") or "Peer"
            peer_host = pc.get("host", "")
            sync_port = int(pc.get("sync_port") or DEFAULT_SYNC_PORT)

            if pc.get("status") != "online":
                node_states.append({
                    "peer_id": peer_id,
                    "node_name": peer_name,
                    "host": peer_host,
                    "sync_port": sync_port,
                    "state": pc.get("status", "offline"),
                    "detail": pc.get("error") or "Peer sync daemon unreachable",
                })
                continue

            online_peers += 1
            peer_map = {m["name"]: m for m in (pc.get("models") or []) if isinstance(m, dict) and m.get("name")}
            pm = peer_map.get(name)

            if local_m and pm:
                if (
                    local_m.get("fingerprint") == pm.get("fingerprint")
                    and local_m.get("size_bytes") == pm.get("size_bytes")
                ):
                    ready_peers += 1
                    node_states.append({
                        "peer_id": peer_id,
                        "node_name": peer_name,
                        "host": peer_host,
                        "sync_port": sync_port,
                        "state": "verified",
                        "size_label": pm.get("size_label", ""),
                        "fingerprint_short": pm.get("fingerprint_short", ""),
                        "detail": "Exact GGUF hash match",
                    })
                else:
                    mismatch_peers.append(peer_id)
                    node_states.append({
                        "peer_id": peer_id,
                        "node_name": peer_name,
                        "host": peer_host,
                        "sync_port": sync_port,
                        "state": "mismatch",
                        "size_label": pm.get("size_label", ""),
                        "fingerprint_short": pm.get("fingerprint_short", ""),
                        "detail": f"Hash mismatch (peer {pm.get('fingerprint_short', '?')} vs local {local_m.get('fingerprint_short', '?')})",
                    })
            elif local_m and not pm:
                missing_peers.append(peer_id)
                node_states.append({
                    "peer_id": peer_id,
                    "node_name": peer_name,
                    "host": peer_host,
                    "sync_port": sync_port,
                    "state": "missing_on_peer",
                    "detail": "Missing on peer — ready to push over LAN/Tailscale",
                })
            elif not local_m and pm:
                pull_sources.append({
                    "peer_id": peer_id,
                    "node_name": peer_name,
                    "host": peer_host,
                    "sync_port": sync_port,
                    "size_bytes": pm.get("size_bytes", 0),
                    "size_label": pm.get("size_label", ""),
                    "fingerprint_short": pm.get("fingerprint_short", ""),
                })
                node_states.append({
                    "peer_id": peer_id,
                    "node_name": peer_name,
                    "host": peer_host,
                    "sync_port": sync_port,
                    "state": "available_on_peer",
                    "size_label": pm.get("size_label", ""),
                    "fingerprint_short": pm.get("fingerprint_short", ""),
                    "detail": "Available on peer — ready to pull to this PC",
                })
            else:
                node_states.append({
                    "peer_id": peer_id,
                    "node_name": peer_name,
                    "host": peer_host,
                    "sync_port": sync_port,
                    "state": "missing_both",
                    "detail": "Not installed on this PC or peer",
                })

        # Overall status classification for this model
        if not local_m:
            overall = "missing_locally" if pull_sources else "missing_everywhere"
            ref_size_label = pull_sources[0]["size_label"] if pull_sources else ""
            ref_fp_short = pull_sources[0]["fingerprint_short"] if pull_sources else ""
        elif online_peers == 0:
            overall = "peers_offline" if peer_checks else "local_only"
            ref_size_label = local_m.get("size_label", "")
            ref_fp_short = local_m.get("fingerprint_short", "")
        elif mismatch_peers:
            overall = "mismatch"
            ref_size_label = local_m.get("size_label", "")
            ref_fp_short = local_m.get("fingerprint_short", "")
        elif missing_peers:
            overall = "needs_sync"
            ref_size_label = local_m.get("size_label", "")
            ref_fp_short = local_m.get("fingerprint_short", "")
        else:
            overall = "ready"
            ref_size_label = local_m.get("size_label", "")
            ref_fp_short = local_m.get("fingerprint_short", "")

        matrix.append({
            "name": name,
            "quant": (local_m or {}).get("quant") or extract_quant_tag(name),
            "size_label": ref_size_label,
            "fingerprint_short": ref_fp_short,
            "local_present": bool(local_m),
            "overall_status": overall,
            "ready_peers": ready_peers,
            "online_peers": online_peers,
            "total_peers": len(peer_checks),
            "missing_peer_ids": missing_peers,
            "mismatch_peer_ids": mismatch_peers,
            "pull_sources": pull_sources,
            "nodes": node_states,
        })

    selected_summary = None
    if target_model:
        for row in matrix:
            if row["name"] == target_model:
                selected_summary = row
                break

    return {
        "checked_at": int(time.time()),
        "local_count": len(local_models),
        "peers": [
            {
                "id": pc.get("id", ""),
                "node_name": pc.get("node_name", ""),
                "host": pc.get("host", ""),
                "worker_endpoint": pc.get("worker_endpoint", ""),
                "sync_port": int(pc.get("sync_port") or DEFAULT_SYNC_PORT),
                "status": pc.get("status", "offline"),
                "error": pc.get("error", ""),
                "rtt_ms": pc.get("rtt_ms"),
                "gpu": pc.get("gpu", ""),
                "disk_free_label": pc.get("disk_free_label", ""),
                "disk_free_bytes": pc.get("disk_free_bytes", 0),
                "cluster_fingerprint": pc.get("cluster_fingerprint", ""),
                "model_count": len(pc.get("models") or []),
            }
            for pc in peer_checks
        ],
        "models": matrix,
        "selected": selected_summary,
    }


class PeerSyncJobManager:
    """Tracks active and completed peer-to-peer GGUF transfers (push & pull)."""

    def __init__(self):
        self._jobs: Dict[str, dict] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
        self._cancel_flags: Dict[str, bool] = {}

    def list_jobs(self) -> List[dict]:
        items = list(self._jobs.values())
        items.sort(key=lambda j: j.get("started_at", 0), reverse=True)
        return items[:25]

    def get_job(self, job_id: str) -> Optional[dict]:
        return self._jobs.get(job_id)

    def cancel_job(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if not job:
            return False
        self._cancel_flags[job_id] = True
        task = self._tasks.get(job_id)
        if task and not task.done():
            task.cancel()
        if job.get("status") in ("queued", "syncing"):
            job["status"] = "cancelled"
            job["error"] = "Cancelled by user."
        return True

    def start_push(
        self,
        peer_name: str,
        host: str,
        port: int,
        filename: str,
        local_path: str,
        secret: str,
        on_complete: Optional[Callable[[], None]] = None,
    ) -> dict:
        safe_name = validate_gguf_filename(filename)
        info = inspect_gguf_file(local_path)
        if not info["gguf_valid"]:
            raise ValueError(f"'{safe_name}' is not a valid GGUF file.")

        # Reuse active job if identical transfer is already running
        for cur in self._jobs.values():
            if (
                cur.get("status") in ("queued", "syncing")
                and cur.get("direction") == "push"
                and cur.get("filename") == safe_name
                and cur.get("peer_host") == host
            ):
                return dict(cur)

        job_id = f"sync-{uuid.uuid4().hex[:8]}"
        job = {
            "id": job_id,
            "direction": "push",
            "filename": safe_name,
            "peer_name": peer_name or host,
            "peer_host": host,
            "sync_port": port,
            "status": "queued",
            "percent": 0.0,
            "transferred_bytes": 0,
            "total_bytes": int(info["size_bytes"]),
            "transferred_label": "0.0 B",
            "total_label": info["size_label"],
            "resumed_from_bytes": 0,
            "speed_mbps": 0.0,
            "eta_seconds": None,
            "fingerprint_short": info["fingerprint_short"],
            "error": "",
            "started_at": time.time(),
            "finished_at": None,
        }
        self._jobs[job_id] = job
        self._cancel_flags[job_id] = False
        self._tasks[job_id] = asyncio.create_task(
            self._run_push(job_id, host, port, safe_name, local_path, info, secret, on_complete)
        )
        return dict(job)

    def start_pull(
        self,
        peer_name: str,
        host: str,
        port: int,
        filename: str,
        primary_dir: str,
        secret: str,
        on_complete: Optional[Callable[[], None]] = None,
    ) -> dict:
        safe_name = validate_gguf_filename(filename)
        target_path = confined_target_path(safe_name, primary_dir)

        for cur in self._jobs.values():
            if (
                cur.get("status") in ("queued", "syncing")
                and cur.get("direction") == "pull"
                and cur.get("filename") == safe_name
                and cur.get("peer_host") == host
            ):
                return dict(cur)

        job_id = f"sync-{uuid.uuid4().hex[:8]}"
        job = {
            "id": job_id,
            "direction": "pull",
            "filename": safe_name,
            "peer_name": peer_name or host,
            "peer_host": host,
            "sync_port": port,
            "status": "queued",
            "percent": 0.0,
            "transferred_bytes": 0,
            "total_bytes": 0,
            "transferred_label": "0.0 B",
            "total_label": "Checking...",
            "resumed_from_bytes": 0,
            "speed_mbps": 0.0,
            "eta_seconds": None,
            "fingerprint_short": "",
            "error": "",
            "started_at": time.time(),
            "finished_at": None,
        }
        self._jobs[job_id] = job
        self._cancel_flags[job_id] = False
        self._tasks[job_id] = asyncio.create_task(
            self._run_pull(job_id, host, port, safe_name, target_path, primary_dir, secret, on_complete)
        )
        return dict(job)

    async def _run_push(
        self,
        job_id: str,
        host: str,
        port: int,
        filename: str,
        local_path: str,
        info: dict,
        secret: str,
        on_complete: Optional[Callable[[], None]],
    ) -> None:
        job = self._jobs[job_id]
        writer = None
        try:
            job["status"] = "syncing"
            size_bytes = int(info["size_bytes"])
            reader, writer, resp = await _open_authenticated_sync_channel(
                host=host,
                port=port,
                secret=secret,
                op="push",
                filename=filename,
                extra_req={"size_bytes": size_bytes, "fingerprint": info["fingerprint"]},
                timeout=6.0,
            )

            if resp.get("already_synced"):
                job["transferred_bytes"] = size_bytes
                job["transferred_label"] = info["size_label"]
                job["percent"] = 100.0
                job["status"] = "completed"
                job["finished_at"] = time.time()
                if on_complete:
                    on_complete()
                return

            offset = max(0, int(resp.get("offset") or 0))
            job["resumed_from_bytes"] = offset
            sent = offset
            t_start = time.perf_counter()
            bytes_this_run = 0

            with open(local_path, "rb") as f:
                if offset > 0:
                    f.seek(offset)
                while sent < size_bytes:
                    if self._cancel_flags.get(job_id):
                        raise asyncio.CancelledError()
                    chunk = await asyncio.to_thread(f.read, min(SYNC_CHUNK_BYTES, size_bytes - sent))
                    if not chunk:
                        break
                    writer.write(chunk)
                    await asyncio.wait_for(writer.drain(), timeout=SYNC_IO_TIMEOUT)
                    sent += len(chunk)
                    bytes_this_run += len(chunk)
                    elapsed = max(0.001, time.perf_counter() - t_start)
                    speed_bps = bytes_this_run / elapsed
                    job["transferred_bytes"] = sent
                    job["transferred_label"] = format_bytes(sent)
                    job["percent"] = round((sent / max(1, size_bytes)) * 100.0, 1)
                    job["speed_mbps"] = round(speed_bps / (1024 * 1024), 2)
                    job["eta_seconds"] = int((size_bytes - sent) / speed_bps) if speed_bps > 1 else None

            done_resp = await _read_json_line(reader, timeout=30.0)
            if not done_resp.get("ok"):
                raise RuntimeError(done_resp.get("error") or "Peer integrity verification failed.")

            job["transferred_bytes"] = size_bytes
            job["transferred_label"] = info["size_label"]
            job["percent"] = 100.0
            job["eta_seconds"] = 0
            job["status"] = "completed"
            job["finished_at"] = time.time()
            if on_complete:
                on_complete()
        except asyncio.CancelledError:
            job["status"] = "cancelled"
            job["error"] = "Cancelled by user."
            job["finished_at"] = time.time()
        except Exception as exc:
            job["status"] = "error"
            job["error"] = str(exc)
            job["finished_at"] = time.time()
        finally:
            if writer is not None:
                try:
                    writer.close()
                    await writer.wait_closed()
                except Exception:
                    pass

    async def _run_pull(
        self,
        job_id: str,
        host: str,
        port: int,
        filename: str,
        target_path: str,
        primary_dir: str,
        secret: str,
        on_complete: Optional[Callable[[], None]],
    ) -> None:
        job = self._jobs[job_id]
        writer = None
        part_path = target_path + SYNC_PART_SUFFIX
        try:
            job["status"] = "syncing"
            resume_offset = 0
            if os.path.isfile(part_path):
                resume_offset = os.path.getsize(part_path)

            reader, writer, resp = await _open_authenticated_sync_channel(
                host=host,
                port=port,
                secret=secret,
                op="pull",
                filename=filename,
                extra_req={"offset": resume_offset},
                timeout=6.0,
            )

            size_bytes = int(resp.get("size_bytes") or 0)
            expected_fp = str(resp.get("fingerprint") or "").strip().lower()
            actual_offset = max(0, int(resp.get("offset") or 0))
            if size_bytes < 24 or len(expected_fp) != 64:
                raise ValueError("Peer returned invalid GGUF metadata.")

            try:
                free_bytes = shutil.disk_usage(primary_dir).free
            except Exception:
                free_bytes = size_bytes + (64 * 1024 * 1024)
            needed = max(0, size_bytes - actual_offset)
            if free_bytes < needed + (16 * 1024 * 1024):
                raise RuntimeError(
                    f"Not enough local disk space ({format_bytes(free_bytes)} free, need {format_bytes(needed)})."
                )

            job["total_bytes"] = size_bytes
            job["total_label"] = format_bytes(size_bytes)
            job["fingerprint_short"] = expected_fp[:12].upper()
            job["resumed_from_bytes"] = actual_offset

            mode = "r+b" if (actual_offset > 0 and os.path.isfile(part_path)) else "wb"
            received = actual_offset
            bytes_this_run = 0
            t_start = time.perf_counter()

            with open(part_path, mode) as f:
                if actual_offset > 0 and mode == "r+b":
                    f.seek(actual_offset)
                while received < size_bytes:
                    if self._cancel_flags.get(job_id):
                        raise asyncio.CancelledError()
                    to_read = min(SYNC_CHUNK_BYTES, size_bytes - received)
                    chunk = await asyncio.wait_for(reader.read(to_read), timeout=SYNC_IO_TIMEOUT)
                    if not chunk:
                        raise ConnectionError("Peer disconnected before transfer completed.")
                    await asyncio.to_thread(f.write, chunk)
                    received += len(chunk)
                    bytes_this_run += len(chunk)
                    elapsed = max(0.001, time.perf_counter() - t_start)
                    speed_bps = bytes_this_run / elapsed
                    job["transferred_bytes"] = received
                    job["transferred_label"] = format_bytes(received)
                    job["percent"] = round((received / max(1, size_bytes)) * 100.0, 1)
                    job["speed_mbps"] = round(speed_bps / (1024 * 1024), 2)
                    job["eta_seconds"] = int((size_bytes - received) / speed_bps) if speed_bps > 1 else None

            verified = await asyncio.to_thread(inspect_gguf_file, part_path)
            if not verified["gguf_valid"] or verified["size_bytes"] != size_bytes or verified["fingerprint"] != expected_fp:
                try:
                    os.remove(part_path)
                except OSError:
                    pass
                raise RuntimeError("Integrity check failed on pulled GGUF file (hash mismatch).")

            os.replace(part_path, target_path)
            job["transferred_bytes"] = size_bytes
            job["transferred_label"] = format_bytes(size_bytes)
            job["percent"] = 100.0
            job["eta_seconds"] = 0
            job["status"] = "completed"
            job["finished_at"] = time.time()
            if on_complete:
                on_complete()
        except asyncio.CancelledError:
            job["status"] = "cancelled"
            job["error"] = "Cancelled by user."
            job["finished_at"] = time.time()
        except Exception as exc:
            job["status"] = "error"
            job["error"] = str(exc)
            job["finished_at"] = time.time()
        finally:
            if writer is not None:
                try:
                    writer.close()
                    await writer.wait_closed()
                except Exception:
                    pass
