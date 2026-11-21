import os
import sys
import re
import json
import glob
import hmac
import logging
import subprocess
import socket
import threading
import time
import asyncio
import base64
import hashlib
import ipaddress
import shutil
import uuid
from urllib.parse import urlparse, quote
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def get_install_dir() -> str:
    return os.environ.get("AEROMESH_INSTALL_DIR") or os.path.dirname(os.path.abspath(__file__))


def get_data_dir() -> str:
    explicit = os.environ.get("AEROMESH_DATA_DIR", "").strip()
    if explicit:
        return explicit
    install_dir = get_install_dir()
    if os.path.exists(os.path.join(install_dir, "Cargo.toml")) or os.path.exists(
        os.path.join(install_dir, ".aeromesh_portable")
    ):
        return install_dir
    if sys.platform == "win32":
        local_app = os.environ.get("LOCALAPPDATA")
        if local_app:
            return os.path.join(local_app, "AeroMesh")
    return install_dir


def get_custom_model_paths_file() -> str:
    return os.path.join(get_data_dir(), "model_paths.json")


def load_custom_model_dirs() -> list:
    path = get_custom_model_paths_file()
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return []
        result = []
        for item in data:
            if isinstance(item, str) and item.strip():
                norm = os.path.abspath(item.strip())
                if norm not in result:
                    result.append(norm)
        return result
    except Exception:
        return []


def save_custom_model_dirs(dirs: list) -> None:
    path = get_custom_model_paths_file()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dirs, f, indent=2)


def get_primary_models_dir() -> str:
    explicit = os.environ.get("AEROMESH_MODELS_DIR", "").strip()
    target = explicit if explicit else os.path.join(get_data_dir(), "models")
    os.makedirs(target, exist_ok=True)
    return target


def get_models_dirs() -> list:
    dirs = []
    explicit = os.environ.get("AEROMESH_MODELS_DIR", "").strip()
    if explicit:
        dirs.append(explicit)
    for d in (
        os.path.join(get_data_dir(), "models"),
        os.path.join(get_install_dir(), "models"),
    ):
        if d not in dirs:
            dirs.append(d)
    for custom in load_custom_model_dirs():
        if os.path.isdir(custom) and custom not in dirs:
            dirs.append(custom)
    return dirs


def get_cache_dir() -> str:
    d = os.path.join(get_data_dir(), "cache")
    os.makedirs(d, exist_ok=True)
    return d


# Lightweight .env loader (checks writable data dir first, then install dir)
def load_dotenv():
    candidates = [
        os.path.join(get_data_dir(), ".env"),
        os.path.join(get_install_dir(), ".env"),
    ]
    for env_file in candidates:
        if os.path.exists(env_file):
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k not in os.environ:
                        os.environ[k] = v
            break

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("aeromesh-gateway")

from fastapi import FastAPI, Request, Depends, HTTPException, Response
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask
import httpx
import uvicorn

# ---------------------------------------------------------------------------
# GATEWAY & BACKEND CONFIGURATION (FIX-16 / SEC-01)
# ---------------------------------------------------------------------------
# Architecture:
#   Browser -> FastAPI Gateway (7860) -> Axum Engine Server (8080)
#
# Environment variables:
#   AEROMESH_UI_HOST:    Gateway bind host (default: 127.0.0.1)
#   AEROMESH_UI_PORT:    Gateway bind port (default: 7860)
#   AEROMESH_BACKEND_URL: Axum coordinator backend URL (default: http://127.0.0.1:8080)
#   AEROMESH_API_KEY:    Bearer token forwarded to Axum backend
#   AEROMESH_GATEWAY_API_KEY: Optional bearer token required to access the gateway
# ---------------------------------------------------------------------------
AEROMESH_UI_HOST = os.getenv("AEROMESH_UI_HOST", os.getenv("HOST", "127.0.0.1"))
AEROMESH_UI_PORT = int(os.getenv("AEROMESH_UI_PORT", os.getenv("AEROMESH_PORT", os.getenv("PORT", "7860"))))
AEROMESH_BACKEND_URL = os.getenv("AEROMESH_BACKEND_URL", os.getenv("AEROMESH_ENDPOINT", "http://127.0.0.1:8080")).rstrip("/")

AEROMESH_API_KEY = os.getenv("AEROMESH_API_KEY", "")
GATEWAY_API_KEY = os.getenv("AEROMESH_GATEWAY_API_KEY", "")

# Requests larger than this are rejected rather than buffered. The gateway
# historically read whole bodies into memory with no cap, which is a trivial
# memory-exhaustion vector on an interface that may be reachable from a LAN.
MAX_REQUEST_BYTES = int(os.getenv("AEROMESH_MAX_REQUEST_BYTES", str(8 * 1024 * 1024)))

# Hosts treated as "this machine". `testclient` is the Starlette TestClient host.
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "testclient"})

if AEROMESH_UI_HOST not in ("127.0.0.1", "localhost", "::1"):
    logger.warning(
        "AeroMESH gateway is binding to '%s', which is not loopback. "
        "Requests from non-loopback clients require AEROMESH_GATEWAY_API_KEY. "
        "This is intended for trusted tailnets only.",
        AEROMESH_UI_HOST,
    )
if not AEROMESH_API_KEY:
    logger.warning(
        "AEROMESH_API_KEY is not set. The coordinator will reject every proxied "
        "request with 401. Run ./start.ps1 once to generate .env, or set it manually."
    )

# Hop-by-hop headers that must not be forwarded by a proxy (RFC 7230 section 6.1)
HOP_BY_HOP_HEADERS = frozenset({
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
})

def filter_headers(headers) -> list:
    """Strip hop-by-hop HTTP headers before proxying.

    Returns a list of (name, value) pairs rather than a dict: `dict` collapses
    duplicate header names, which silently drops all but one `Set-Cookie`.
    """
    return [
        (key, value)
        for key, value in headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS
    ]


class HeaderPairs(list):
    """A list of (name, value) header pairs that also behaves like a Mapping.

    Starlette's `Response` accepts anything with `.items()` / `.raw`, but a
    plain `dict` would collapse repeated headers. This keeps duplicates while
    satisfying the interface, so a response can carry multiple `Set-Cookie`
    values without losing any.
    """

    _MISSING = object()

    def items(self):
        return list(self)

    def keys(self):
        return [k for k, _ in self]

    def get(self, key, default=None):
        lowered = key.lower()
        for k, v in self:
            if k.lower() == lowered:
                return v
        return default

    def __getitem__(self, key):
        if isinstance(key, (int, slice)):
            return super().__getitem__(key)
        value = self.get(key, self._MISSING)
        if value is self._MISSING:
            raise KeyError(key)
        return value

    def __setitem__(self, key, value):
        if isinstance(key, (int, slice)):
            super().__setitem__(key, value)
            return
        lowered = key.lower()
        for idx, (k, _) in enumerate(self):
            if k.lower() == lowered:
                super().__setitem__(idx, (key, str(value)))
                return
        self.append((key, str(value)))

    def __contains__(self, key):
        return self.get(key, self._MISSING) is not self._MISSING

    @property
    def raw(self):
        return [(k.lower().encode("latin-1"), str(v).encode("latin-1")) for k, v in self]


def get_auth_headers() -> dict:
    """Bearer credential injected on every upstream call."""
    key = os.getenv("AEROMESH_API_KEY", "")
    return {"Authorization": f"Bearer {key}"} if key else {}


def build_upstream_headers(client_headers) -> list:
    """Builds the header list sent to the coordinator.

    Two rules, both load-bearing:

    1. Hop-by-hop headers (RFC 7230 section 6.1) are dropped.
    2. **Any caller-supplied `Authorization` is dropped.** The gateway owns the
       upstream credential. Retaining the caller's value produced two
       Authorization headers and the coordinator reads the first, i.e. the
       caller's. The gateway's own key is appended last so it always wins.

    Returns a list of pairs, preserving duplicates such as multiple Set-Cookie.
    """
    headers = [
        (name, value)
        for (name, value) in filter_headers(client_headers)
        if name.lower() != "authorization"
    ]
    headers.extend(get_auth_headers().items())
    return headers


def unauthorized(message: str) -> HTTPException:
    return HTTPException(
        status_code=401,
        detail={"error": {"message": message, "type": "authentication_error",
                          "param": None, "code": "invalid_api_key"}},
    )

async def verify_gateway_access(request: Request):
    """Authorise a request to a proxied route.

    Policy:
      * loopback clients are always allowed;
      * if AEROMESH_GATEWAY_API_KEY is configured, non-loopback clients must
        present it;
      * if it is *not* configured, non-loopback clients are refused outright.

    The previous implementation returned True unconditionally when the key was
    unset, so the loopback check was dead code and any host on the network had
    full API access by default.
    """
    client_host = request.client.host if request.client else "127.0.0.1"
    is_local = client_host in LOOPBACK_HOSTS

    if is_local:
        return True

    if not GATEWAY_API_KEY:
        logger.warning(
            "Refused non-loopback request from %s: AEROMESH_GATEWAY_API_KEY is not "
            "configured. Set it to allow remote access to the gateway.",
            client_host,
        )
        raise unauthorized(
            "Gateway access from non-loopback clients requires AEROMESH_GATEWAY_API_KEY "
            "to be configured, or a valid Bearer token."
        )

    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise unauthorized("Unauthorized gateway access")
    token = auth[len("Bearer "):].strip()
    if not hmac.compare_digest(token, GATEWAY_API_KEY):
        raise unauthorized("Invalid gateway API key")
    return True


# ---------------------------------------------------------------------------
# LIFESPAN & ASYNC HTTP CLIENT MANAGEMENT
# ---------------------------------------------------------------------------
backend_client: httpx.AsyncClient = None

def get_backend_client() -> httpx.AsyncClient:
    """Returns or lazily creates the shared async HTTP backend client."""
    global backend_client
    if backend_client is None or backend_client.is_closed:
        backend_client = httpx.AsyncClient(
            base_url=AEROMESH_BACKEND_URL,
            timeout=httpx.Timeout(
                connect=5.0,
                read=None,  # Do not drop long SSE streaming completions
                write=30.0,
                pool=10.0,
            ),
            follow_redirects=False,
        )
    return backend_client

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages shared HTTP client lifecycle across gateway execution."""
    client = get_backend_client()
    yield
    if client and not client.is_closed:
        await client.aclose()


from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="AeroMesh Intelligence Engine Gateway",
    version="1.0.0",
    lifespan=lifespan,
)

_ALLOWED_CORS_ORIGINS = [
    f"http://127.0.0.1:{AEROMESH_UI_PORT}",
    f"http://localhost:{AEROMESH_UI_PORT}",
]
for _extra_origin in os.getenv("AEROMESH_ALLOWED_ORIGIN", "").split(","):
    _extra_origin = _extra_origin.strip()
    if _extra_origin and _extra_origin not in _ALLOWED_CORS_ORIGINS:
        _ALLOWED_CORS_ORIGINS.append(_extra_origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# HEALTH & READINESS ENDPOINTS (FIX-16)
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    """
    Gateway Liveness Check.
    Returns 200 OK if the FastAPI gateway process is alive.
    """
    return {
        "gateway": "ok",
        "backend_url": AEROMESH_BACKEND_URL,
    }


@app.get("/ready")
async def ready():
    """
    End-to-End Readiness Probe.
    Checks whether the backend Axum coordinator server is reachable.
    """
    client = get_backend_client()
    try:
        resp = await client.get("/health", headers=get_auth_headers(), timeout=3.0)
        backend_ok = (resp.status_code == 200)
    except Exception:
        backend_ok = False

    if backend_ok:
        return {"gateway": "ok", "backend": "ok"}
    else:
        return JSONResponse(
            status_code=503,
            content={
                "gateway": "ok",
                "backend": "unreachable",
                "backend_url": AEROMESH_BACKEND_URL,
            },
        )


# ---------------------------------------------------------------------------
# CORE REUSABLE PROXY FUNCTION
# ---------------------------------------------------------------------------
PROXY_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]

async def read_body_limited(request: Request) -> bytes:
    """Reads the request body, rejecting anything over MAX_REQUEST_BYTES.

    Raises HTTPException(413) rather than buffering an unbounded body.
    """
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            if int(declared) > MAX_REQUEST_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail={"error": {"message": f"Request body exceeds {MAX_REQUEST_BYTES} bytes",
                                      "type": "invalid_request_error", "param": None,
                                      "code": "payload_too_large"}},
                )
        except ValueError:
            pass
    body = await request.body()
    if len(body) > MAX_REQUEST_BYTES:
        raise HTTPException(
            status_code=413,
            detail={"error": {"message": f"Request body exceeds {MAX_REQUEST_BYTES} bytes",
                              "type": "invalid_request_error", "param": None,
                              "code": "payload_too_large"}},
        )
    return body


async def proxy_request(request: Request, backend_path: str):
    """
    Forwards HTTP request to Axum backend, preserving stream semantics and headers.
    """
    client = get_backend_client()
    method = request.method
    headers = build_upstream_headers(dict(request.headers))

    params = request.query_params
    body = await read_body_limited(request)

    try:
        backend_request = client.build_request(
            method=method,
            url=backend_path,
            headers=headers,
            params=params,
            content=body if body else None,
        )
        backend_response = await client.send(backend_request, stream=True)
    except httpx.ConnectError:
        logger.warning("Backend unreachable at %s", AEROMESH_BACKEND_URL)
        return JSONResponse(
            status_code=502,
            content={
                "error": "backend_unreachable",
                "detail": "Cannot connect to the AeroMESH coordinator. Is it running?",
            },
        )
    except Exception:
        # Log the detail server-side; do not echo exception text to the client.
        logger.exception("Gateway proxy error")
        return JSONResponse(
            status_code=502,
            content={
                "error": "gateway_proxy_error",
                "detail": "The gateway could not complete the upstream request.",
            },
        )

    response_headers = HeaderPairs(filter_headers(dict(backend_response.headers)))
    content_type = backend_response.headers.get("content-type", "")

    if content_type.startswith("text/event-stream"):
        response_headers["Cache-Control"] = "no-cache"
        response_headers["X-Accel-Buffering"] = "no"

        async def event_stream():
            try:
                async for chunk in backend_response.aiter_raw():
                    yield chunk
            finally:
                await backend_response.aclose()

        return StreamingResponse(
            event_stream(),
            status_code=backend_response.status_code,
            headers=response_headers,
            media_type=content_type,
            background=BackgroundTask(backend_response.aclose),
        )

    content = await backend_response.aread()
    await backend_response.aclose()

    return Response(
        content=content,
        status_code=backend_response.status_code,
        headers=response_headers,
        media_type=content_type or None,
    )


# ---------------------------------------------------------------------------
# ROOT ROUTE - SERVE FULL-SPREAD CLAYMORPHIC SPA
# ---------------------------------------------------------------------------
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def serve_spa():
    """Serves the single-page dashboard."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    logger.error("static/index.html is missing; the dashboard cannot be served")
    return "<h1>AeroMesh UI error: static/index.html not found</h1>"


# ---------------------------------------------------------------------------
# SPECIALIZED API ENDPOINTS WITH GRACEFUL OFFLINE FALLBACKS
# ---------------------------------------------------------------------------
def _is_slice_file(name: str) -> bool:
    """True for internal partition files, which must never be offered as models.

    Covers both naming schemes the slicer can produce: `_stage1.gguf` /
    `_stage2.gguf` for a 2-stage split, and `_stage{N}_of_{M}.gguf` otherwise.
    """
    lowered = name.lower()
    if lowered.endswith("_stage1.gguf") or lowered.endswith("_stage2.gguf"):
        return True
    return bool(re.search(r"_stage\d+_of_\d+\.gguf$", lowered))


@app.get("/v1/models", dependencies=[Depends(verify_gateway_access)])
async def get_models():
    """
    Fetches active models from the running AeroMesh coordinator.
    If the coordinator is offline, discovers local GGUF models from models/.
    Internal partition slice files are excluded.
    """
    client = get_backend_client()
    try:
        resp = await client.get("/v1/models", headers=get_auth_headers(), timeout=2.5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass

    # Coordinator offline: fall back to a local disk scan. The response shape
    # matches API_REFERENCE.md (id/object/created/owned_by/is_active) so the UI
    # can still mark the active model. Only the basename is exposed: absolute
    # local paths are not the client's business.
    local_models = []
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    if os.path.exists(models_dir):
        for f in sorted(glob.glob(os.path.join(models_dir, "*.gguf"))):
            name = os.path.basename(f)
            if _is_slice_file(name):
                continue
            local_models.append({
                "id": name,
                "object": "model",
                "created": int(os.path.getmtime(f)),
                "owned_by": "local_disk",
                "is_active": len(local_models) == 0,  # first (alphabetical) is the assumed active one
            })

    return {
        "object": "list",
        "data": local_models
    }


@app.post("/api/model/switch", dependencies=[Depends(verify_gateway_access)])
@app.post("/v1/models/load", dependencies=[Depends(verify_gateway_access)])
async def switch_model(request: Request):
    """
    Proxies model switch requests to the AeroMesh coordinator.
    Uses a 180s timeout to allow large models (9GB+) to slice and memory-map.
    """
    client = get_backend_client()
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "Request body must be JSON",
                               "type": "invalid_request_error", "param": "model",
                               "code": "invalid_json"}},
        )
    if not isinstance(payload, dict) or not payload.get("model"):
        return JSONResponse(
            status_code=422,
            content={"error": {"message": "Field 'model' is required",
                               "type": "invalid_request_error", "param": "model",
                               "code": "missing_required_field"}},
        )
    try:
        resp = await client.post("/api/model/switch", json=payload, headers=get_auth_headers(), timeout=180.0)
        try:
            content = resp.json()
        except Exception:
            content = {"status": "ok" if resp.status_code == 200 else "error", "message": resp.text}
        return JSONResponse(status_code=resp.status_code, content=content)
    except httpx.ConnectError:
        logger.warning("Backend unreachable during model switch")
        return JSONResponse(
            status_code=502,
            content={"error": "backend_unreachable",
                     "detail": "Cannot connect to the AeroMESH coordinator. Is it running?"}
        )
    except httpx.TimeoutException:
        logger.warning("Model switch request timed out after 180s")
        return JSONResponse(
            status_code=504,
            content={"status": "error", "message": "Model switch timed out (large model slicing / GPU allocation in progress). Check server logs."}
        )
    except Exception:
        logger.exception("Error during model switch")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Model switch failed. Check the coordinator logs."}
        )


@app.get("/api/cluster/status", dependencies=[Depends(verify_gateway_access)])
async def cluster_status():
    """
    Queries the live AeroMesh Coordinator node dynamically.
    Returns real connected node topology, layer distribution, and transport status.
    Guarantees fallback payload with state: 'offline' on connection error (never a raw 5xx).
    """
    client = get_backend_client()
    try:
        resp = await client.get("/api/cluster/status", headers=get_auth_headers(), timeout=2.0)
        if resp.status_code == 200:
            data = resp.json()
            if "state" not in data:
                data["state"] = "online_idle" if data.get("connected", True) else "offline"
            return data
    except Exception:
        pass

    # Coordinator is offline - fallback payload with state: "offline"
    return {
        "state": "offline",
        "status": "offline",
        "connected": False,
        "coordinator_endpoint": AEROMESH_BACKEND_URL,
        "message": "AeroMesh Coordinator is offline. Start the coordinator with: cargo run --bin aeromesh -- serve --model <model.gguf> --layers <range> --peers <worker-ip>:50052"
    }


# ---------------------------------------------------------------------------
# UNIFIED NODE CONTROL & CLUSTER DISCOVERY HELPERS (FOR DESKTOP / KIMI STUDIO)
# ---------------------------------------------------------------------------
_CACHED_GPU = None
def get_gpu_summary() -> str:
    global _CACHED_GPU
    if _CACHED_GPU is not None:
        return _CACHED_GPU
    candidates = [
        "nvidia-smi",
        r"C:\Windows\System32\nvidia-smi.exe",
        r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"
    ]
    creationflags = 0x08000000 if sys.platform == "win32" else 0
    for c in candidates:
        try:
            res = subprocess.run(
                [c, "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=2.0,
                creationflags=creationflags
            )
            if res.returncode == 0 and res.stdout.strip():
                _CACHED_GPU = res.stdout.strip().splitlines()[0]; return _CACHED_GPU
        except Exception:
            pass
    return "CPU / Integrated GPU"

def get_local_wifi_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

_CACHED_TS_IP = None
def get_tailscale_ip() -> str:
    global _CACHED_TS_IP
    if _CACHED_TS_IP is not None:
        return _CACHED_TS_IP
    creationflags = 0x08000000 if sys.platform == "win32" else 0
    try:
        res = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=2.0, creationflags=creationflags)
        if res.returncode == 0 and res.stdout.strip():
            _CACHED_TS_IP = res.stdout.strip().splitlines()[0]
            return _CACHED_TS_IP
    except Exception:
        pass
    _CACHED_TS_IP = "Not Connected"
    return _CACHED_TS_IP

def _sanitize_model_name(raw_model: str) -> str:
    cleaned = str(raw_model or "").strip()
    if cleaned.startswith("models/") or cleaned.startswith("models\\"):
        cleaned = cleaned[7:]
    if (
        not cleaned
        or cleaned.startswith("-")
        or ".." in cleaned
        or "/" in cleaned
        or "\\" in cleaned
        or "\x00" in cleaned
        or ":" in cleaned
    ):
        raise ValueError(f"Invalid model filename: {raw_model!r}")
    return cleaned


def _resolve_confined_model_file(raw_model: str) -> str:
    filename = _sanitize_model_name(raw_model)
    for models_dir in get_models_dirs():
        if not os.path.isdir(models_dir):
            continue
        base_real = os.path.realpath(models_dir)
        candidate = os.path.realpath(os.path.join(models_dir, filename))
        if candidate.startswith(base_real + os.sep) and os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"Model {filename!r} not found in models directory")


def _validate_peers_csv(raw_peers: str) -> str:
    trimmed = str(raw_peers or "").strip()
    if not trimmed:
        return ""
    validated = []
    for item in trimmed.split(","):
        part = item.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid peer address {part!r}: expected IP:PORT")
        host, port_s = part.rsplit(":", 1)
        host = host.strip("[]")
        ipaddress.ip_address(host)
        port_num = int(port_s)
        if not (1 <= port_num <= 65535):
            raise ValueError(f"Invalid peer port in {part!r}")
        validated.append(f"{host}:{port_num}")
    return ",".join(validated)


class NodeRuntimeManager:
    def __init__(self):
        self.lock = threading.RLock()
        self.role = "idle"
        self.model = "test.gguf"
        self.peers = ""
        self.process = None
        self.logs = ["[AeroMesh Studio] Ready. Click a Role Card above to start Coordinator, Worker, or Full Local Mesh."]

    def push_log(self, line: str):
        with self.lock:
            self.logs.append(line)
            if len(self.logs) > 120:
                self.logs = self.logs[-120:]

    def is_running(self) -> bool:
        with self.lock:
            if self.process is None:
                return False
            poll = self.process.poll()
            if poll is not None:
                self.role = "idle"
                self.process = None
                return False
            return True

    def stop(self):
        with self.lock:
            if self.process is not None:
                pid = self.process.pid
                if sys.platform == "win32":
                    try:
                        subprocess.run(
                            ["taskkill", "/F", "/T", "/PID", str(pid)],
                            capture_output=True,
                            creationflags=0x08000000,
                            timeout=5,
                        )
                    except Exception:
                        pass
                try:
                    self.process.kill()
                    self.process.wait(timeout=3)
                except Exception:
                    pass
                self.process = None
            self.role = "idle"
            self.logs.append("[AeroMesh Studio] Node stopped by user.")

    def start(self, role: str, model: str, peers: str) -> dict:
        role_clean = str(role or "").strip().lower()
        if role_clean not in ("worker", "coordinator", "all"):
            return {"status": "error", "message": f"Invalid role: {role!r}"}
        try:
            model_path = _resolve_confined_model_file(model)
            safe_peers = _validate_peers_csv(peers)
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

        self.stop()
        install_dir = get_install_dir()
        data_dir = get_data_dir()
        os.makedirs(data_dir, exist_ok=True)
        engine_candidates = [
            os.path.join(install_dir, "aeromesh.exe"),
            os.path.join(install_dir, "bin", "aeromesh.exe"),
            os.path.join(install_dir, "target", "release", "aeromesh.exe"),
        ]
        engine_exe = next((c for c in engine_candidates if os.path.exists(c)), "aeromesh.exe")

        cmd = [engine_exe]
        if role_clean == "worker":
            cmd += ["start", "--role", "worker", "--port", "50052", "--model", model_path]
        elif role_clean == "coordinator":
            cmd += ["start", "--role", "coordinator", "--model", model_path, "--port", "8080"]
            if safe_peers:
                cmd += ["--peers", safe_peers]
        else:
            cmd += ["start", "--role", "all", "--model", model_path, "--port", "8080"]

        with self.lock:
            self.logs.clear()
            self.logs.append(f"[AeroMesh Studio] Started role '{role_clean.upper()}' with model '{os.path.basename(model_path)}'")
            try:
                creationflags = 0x08000000 if sys.platform == "win32" else 0
                proc = subprocess.Popen(
                    cmd,
                    cwd=data_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    creationflags=creationflags,
                )
                self.process = proc
                self.role = role_clean
                self.model = os.path.basename(model_path)
                self.peers = safe_peers

                def log_reader(p: subprocess.Popen):
                    for line in iter(p.stdout.readline, ""):
                        clean = line.rstrip()
                        if clean:
                            self.push_log(clean)
                    p.stdout.close()

                t = threading.Thread(target=log_reader, args=(proc,), daemon=True)
                t.start()
                return {"status": "ok", "role": role_clean, "model": self.model, "peers": safe_peers}
            except Exception as e:
                self.logs.append(f"[AeroMesh Studio] Failed to start aeromesh.exe: {e}")
                return {"status": "error", "message": str(e)}

node_manager = NodeRuntimeManager()


@app.get("/api/node/state", dependencies=[Depends(verify_gateway_access)])
async def node_state():
    gguf_models = []
    image_models = []
    seen_gguf = set()
    seen_img = set()
    for models_dir in get_models_dirs():
        if os.path.isdir(models_dir):
            for f in sorted(os.listdir(models_dir)):
                if f.endswith(".gguf") and not _is_slice_file(f) and not f.startswith(".aeromesh_"):
                    if f not in seen_gguf:
                        seen_gguf.add(f)
                        gguf_models.append(f)
                elif f.endswith(".safetensors"):
                    if f not in seen_img:
                        seen_img.add(f)
                        image_models.append(f)

    with node_manager.lock:
        role = node_manager.role
        model = node_manager.model
        peers = node_manager.peers
        running = node_manager.is_running()
        logs_copy = list(node_manager.logs)

    return {
        "role": role,
        "running": running,
        "model": model,
        "peers": peers,
        "gpu": get_gpu_summary(),
        "wifi_ip": get_local_wifi_ip(),
        "tailscale_ip": get_tailscale_ip(),
        "models_dir": get_models_dirs()[0],
        "models": gguf_models,
        "image_models": image_models,
        "logs": logs_copy,
    }


@app.post("/api/node/start", dependencies=[Depends(verify_gateway_access)])
async def node_start(request: Request):
    body = await _read_json_body(request)
    role = str(body.get("role", "all")).strip().lower()
    model = str(body.get("model", "")).strip()
    peers = str(body.get("peers", "")).strip()

    if role == "auto":
        discovered = await asyncio.to_thread(discover_mesh_workers_sync, 50052)
        if discovered:
            peers = ",".join(d["endpoint"] for d in discovered)
            role = "coordinator"
        else:
            role = "all"

    res = node_manager.start(role, model, peers)
    status_code = 200 if res.get("status") == "ok" else 400
    return JSONResponse(status_code=status_code, content=res)


@app.post("/api/node/stop", dependencies=[Depends(verify_gateway_access)])
async def node_stop():
    node_manager.stop()
    return {"status": "stopped", "role": "idle"}


async def _read_json_body(request: Request) -> dict:
    """Reads and validates a JSON object body."""
    raw = await read_body_limited(request)
    try:
        body = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail={"error": {"message": "Request body must be valid JSON",
                              "type": "invalid_request_error", "param": None,
                              "code": "invalid_json"}},
        )
    if not isinstance(body, dict):
        raise HTTPException(
            status_code=400,
            detail={"error": {"message": "Request body must be a JSON object",
                              "type": "invalid_request_error", "param": None,
                              "code": "invalid_json"}},
        )
    return body


def _sse_error_frame(message: str) -> str:
    """Builds an SSE data frame carrying a user-visible error message."""
    payload = json.dumps({"choices": [{"index": 0, "delta": {"content": message}}]})
    return f"data: {payload}\n\n"


@app.post("/v1/chat/completions", dependencies=[Depends(verify_gateway_access)])
@app.post("/api/chat", dependencies=[Depends(verify_gateway_access)])
async def chat_completions(request: Request):
    """OpenAI-compatible chat completions proxy.

    Streaming requests (`stream: true`, the default) are delegated to the SSE
    path. The body is parsed exactly once and passed down: the previous version
    parsed it here and then handed the same `Request` to `chat_stream`, which
    read it a second time.
    """
    body = await _read_json_body(request)

    if body.get("stream", True):
        return _stream_response(body)

    client = get_backend_client()
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json=body,
            headers=get_auth_headers(),
            timeout=httpx.Timeout(180.0, connect=10.0),
        )
        return JSONResponse(status_code=resp.status_code, content=resp.json())
    except httpx.ConnectError:
        logger.warning("Backend unreachable during chat completion")
        return JSONResponse(
            status_code=502,
            content={"error": "backend_unreachable",
                     "detail": "Cannot connect to the AeroMESH coordinator. Is it running?"}
        )
    except Exception:
        logger.exception("Chat completion proxy failed")
        return JSONResponse(
            status_code=500,
            content={"error": {"message": "Upstream request failed",
                               "type": "api_error", "param": None, "code": None}}
        )


@app.post("/api/chat/abort", dependencies=[Depends(verify_gateway_access)])
async def chat_abort():
    """Signals stream cancellation to the backend coordinator."""
    client = get_backend_client()
    try:
        resp = await client.post("/api/chat/abort", headers=get_auth_headers(), timeout=2.0)
        # Report the real outcome rather than always claiming success.
        return JSONResponse(
            status_code=200,
            content={"status": "aborted", "backend_status": resp.status_code},
        )
    except Exception as exc:
        logger.info("Abort signal could not be delivered: %s", exc)
        return JSONResponse(
            status_code=200,
            content={"status": "aborted",
                     "backend_status": None,
                     "detail": "Coordinator unreachable; the client-side stream was still cancelled."},
        )


@app.post("/api/chat/stream", dependencies=[Depends(verify_gateway_access)])
async def chat_stream_endpoint(request: Request):
    """SSE streaming endpoint used by the browser dashboard."""
    body = await _read_json_body(request)
    return _stream_response(body)


def _stream_response(body: dict) -> StreamingResponse:
    """Streams a chat completion as Server-Sent Events.

    Note on timeouts: `read=None` is deliberate. A finite read timeout would
    terminate long reasoning traces mid-generation. The upstream connection is
    closed when the client disconnects (StreamingResponse cancels the generator)
    or when the coordinator ends the stream.
    """
    client = get_backend_client()

    async def stream_generator():
        try:
            async with client.stream(
                "POST",
                "/v1/chat/completions",
                json=body,
                headers=get_auth_headers(),
                timeout=httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0),
            ) as response:
                if response.status_code != 200:
                    err_text = (await response.aread()).decode("utf-8", errors="ignore")
                    logger.warning("Backend returned HTTP %s: %s", response.status_code, err_text[:500])
                    yield _sse_error_frame(
                        f"\n\n❌ **AeroMesh backend error**: HTTP {response.status_code}\n\n"
                        f"```\n{err_text[:1000]}\n```"
                    )
                    return

                async for chunk in response.aiter_raw():
                    yield chunk
        except httpx.ConnectError:
            yield _sse_error_frame(
                "❌ **AeroMesh coordinator is offline**\n\n"
                f"Could not connect to `{AEROMESH_BACKEND_URL}`.\n\n"
                "Start the zero-weight cluster coordinator with:\n"
                "```powershell\n"
                "aeromesh serve --model models/<model>.gguf --layers 0..24 --peers <worker-ip>:50052 --port 8080\n"
                "```"
            )
        except Exception:
            logger.exception("SSE stream failed")
            yield _sse_error_frame(
                "\n\n❌ **Pipeline error**: the coordinator connection dropped. "
                "Check the coordinator logs for details."
            )

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# LOCAL STABLE DIFFUSION IMAGE GENERATION & PEER DISCOVERY
# ---------------------------------------------------------------------------
def _find_sd_binary_and_model():
    install_dir = get_install_dir()
    sd_candidates = [
        os.path.join(install_dir, "bin", "diffusion", "sd-cli.exe"),
        os.path.join(install_dir, "bin", "sd-cli.exe"),
        os.path.join(install_dir, "bin", "sd.exe"),
        os.path.join(install_dir, "bin", "stable-diffusion.exe"),
    ]
    sd_bin = next((p for p in sd_candidates if os.path.exists(p)), None)
    sd_model = None
    candidates = []
    for models_dir in get_models_dirs():
        if os.path.isdir(models_dir):
            candidates += (
                glob.glob(os.path.join(models_dir, "*.safetensors"))
                + glob.glob(os.path.join(models_dir, "diffusion", "*.gguf"))
                + glob.glob(os.path.join(models_dir, "diffusion", "*.safetensors"))
            )
    if candidates:
        preferred = [
            c for c in candidates
            if "realvis" in os.path.basename(c).lower() or "dreamshaper" in os.path.basename(c).lower()
        ]
        sd_model = sorted(preferred)[0] if preferred else sorted(candidates)[0]
    return sd_bin, sd_model


def _probe_worker_port(ip_str: str, port: int = 50052, timeout: float = 0.35):
    t0 = time.perf_counter()
    try:
        with socket.create_connection((ip_str, port), timeout=timeout):
            rtt = round((time.perf_counter() - t0) * 1000.0, 2)
            return {"endpoint": f"{ip_str}:{port}", "ip": ip_str, "port": port, "rtt_ms": rtt}
    except Exception:
        return None


def discover_mesh_workers_sync(port: int = 50052):
    """Scans Tailscale mesh peers and local Wi-Fi LAN ARP neighbors in parallel."""
    candidate_ips = {}
    creationflags = 0x08000000 if sys.platform == "win32" else 0
    try:
        res = subprocess.run(
            ["tailscale", "status", "--json"],
            capture_output=True,
            text=True,
            timeout=2.0,
            creationflags=creationflags,
        )
        if res.returncode == 0 and res.stdout:
            ts_data = json.loads(res.stdout)
            for peer in (ts_data.get("Peer") or {}).values():
                if not peer.get("Online"):
                    continue
                host = peer.get("HostName", "tailscale-peer")
                for ip in (peer.get("TailscaleIPs") or []):
                    if "." in ip:
                        candidate_ips[ip] = {"type": "Tailscale Direct WireGuard", "hostname": host}
    except Exception:
        pass

    local_ips = {"127.0.0.1", "0.0.0.0"}
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            local_ips.add(info[4][0])
    except Exception:
        pass

    try:
        arp = subprocess.run(
            ["arp", "-a"],
            capture_output=True,
            text=True,
            timeout=2.0,
            creationflags=creationflags,
        )
        if arp.returncode == 0:
            for m in re.finditer(r"(\d+\.\d+\.\d+\.\d+)\s+[0-9a-fA-F-]+\s+dynamic", arp.stdout):
                ip = m.group(1)
                if ip not in local_ips and not ip.endswith(".255") and not ip.startswith("224."):
                    candidate_ips.setdefault(ip, {"type": "Wi-Fi / LAN Direct TCP", "hostname": f"lan-{ip.split('.')[-1]}"})
    except Exception:
        pass

    discovered = []
    if candidate_ips:
        with ThreadPoolExecutor(max_workers=min(32, len(candidate_ips))) as ex:
            futures = {ex.submit(_probe_worker_port, ip, port): (ip, meta) for ip, meta in candidate_ips.items()}
            for fut, (ip, meta) in futures.items():
                hit = fut.result()
                if hit:
                    hit.update(meta)
                    discovered.append(hit)
    discovered.sort(key=lambda x: x["rtt_ms"])
    return discovered


@app.get("/api/cluster/discover", dependencies=[Depends(verify_gateway_access)])
async def cluster_discover():
    peers = await asyncio.to_thread(discover_mesh_workers_sync, 50052)
    sd_bin, sd_model = _find_sd_binary_and_model()
    return {
        "discovered_workers": peers,
        "count": len(peers),
        "diffusion_engine_ready": bool(sd_bin and sd_model),
        "diffusion_model": os.path.basename(sd_model) if sd_model else None,
    }


def _run_sd_cli_sync(
    sd_bin: str,
    sd_model: str,
    prompt: str,
    steps: int,
    width: int,
    height: int,
    out_png: str,
):
    creationflags = 0x08000000 if sys.platform == "win32" else 0
    cmd = [
        sd_bin,
        "-m", sd_model,
        "--type", "f16",
        "-p", prompt,
        "-n", "blurry, out of focus, low quality, deformed, watermark, text",
        "--cfg-scale", "7.0",
        "--sampling-method", "euler_a",
        "--steps", str(steps),
        "-W", str(width),
        "-H", str(height),
        "-o", out_png,
    ]
    logger.info("Launching local sd-cli (%dx%d, %d steps)", width, height, steps)
    res = subprocess.run(
        cmd,
        cwd=os.path.dirname(sd_bin),
        capture_output=True,
        text=True,
        timeout=180,
        creationflags=creationflags,
    )
    if res.returncode != 0:
        raise RuntimeError(((res.stderr or "") + "\n" + (res.stdout or "")).strip()[-300:] or f"sd-cli exited with {res.returncode}")


@app.post("/v1/images/generations", dependencies=[Depends(verify_gateway_access)])
@app.post("/api/image/stream", dependencies=[Depends(verify_gateway_access)])
async def generate_image_pipeline(request: Request):
    body = await _read_json_body(request)
    prompt = str(body.get("prompt", "")).strip()
    if not prompt:
        raise HTTPException(
            status_code=422,
            detail={"error": {"message": "Field 'prompt' is required",
                              "type": "invalid_request_error", "param": "prompt",
                              "code": "missing_required_field"}},
        )

    sd_bin, sd_model = _find_sd_binary_and_model()
    model_label = os.path.basename(sd_model) if sd_model else "Not Installed"
    steps = max(1, min(int(body.get("steps", 12)), 50))
    width = (max(256, min(int(body.get("width", 512)), 1024)) // 64) * 64
    height = (max(256, min(int(body.get("height", 512)), 1024)) // 64) * 64
    stream = bool(body.get("stream", True))

    async def dit_event_stream():
        t0 = time.perf_counter()
        client = get_backend_client()

        cluster_meta = {
            "connected": False,
            "transport": "Local GPU/CPU Stable Diffusion (sd-cli)",
            "rtt_ms": 0.0,
        }
        try:
            c_resp = await client.get("/api/cluster/status", headers=get_auth_headers(), timeout=1.5)
            if c_resp.status_code == 200:
                c_data = c_resp.json()
                cluster_meta.update(c_data)
                cluster_meta["connected"] = bool(c_data.get("connected", False))
        except Exception:
            pass

        mesh_active = bool(cluster_meta.get("connected"))
        transport_label = (
            f"Local sd-cli + LLM Prompt Conditioning ({cluster_meta.get('transport', 'Local')})"
            if mesh_active
            else "Local GPU/CPU Stable Diffusion (sd-cli)"
        )

        enhanced_prompt = prompt
        mesh_tokens_generated = 0
        if mesh_active:
            try:
                cond_resp = await client.post(
                    "/v1/chat/completions",
                    json={
                        "messages": [
                            {"role": "user", "content": f"Write a 12-word vivid visual lighting and lens description for: {prompt}"}
                        ],
                        "max_tokens": 24,
                        "temperature": 0.6,
                        "stream": False,
                    },
                    headers=get_auth_headers(),
                    timeout=10.0,
                )
                if cond_resp.status_code == 200:
                    ext = cond_resp.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                    ext = re.sub(r"<think>[\s\S]*?</think>", "", ext).strip()
                    if ext:
                        enhanced_prompt = f"{prompt}, {ext}"
                        mesh_tokens_generated = len(ext.split())
            except Exception as e:
                logger.warning("Prompt conditioning fallback: %s", e)

        if not (sd_bin and sd_model):
            final_evt = {
                "type": "dit_complete",
                "prompt": prompt,
                "conditioning": enhanced_prompt[:240],
                "mesh_active": mesh_active,
                "model_name": model_label,
                "steps": steps,
                "width": width,
                "height": height,
                "accum_wire_mb": 0.0,
                "weights_on_wire_mb": 0.0,
                "elapsed_s": round(time.perf_counter() - t0, 2),
                "b64_png": None,
                "sd_error": "Stable Diffusion engine (sd-cli.exe) or a .safetensors model is not installed. Select the Stable Diffusion component in the AeroMESH installer.",
                "transport": "Local sd-cli (Not Installed)",
            }
            yield f"data: {json.dumps(final_evt)}\n\n"
            yield "data: [DONE]\n\n"
            return

        yield f"data: {json.dumps({'type': 'dit_step', 'step': 1, 'total_steps': steps, 'mesh_active': mesh_active, 'stage1_blocks': 'Local UNet + CLIP Conditioning', 'stage2_blocks': f'Local FP16 VAE ({model_label})', 'step_wire_mb': 0.0, 'accum_wire_mb': 0.0, 'elapsed_s': round(time.perf_counter() - t0, 2), 'transport': transport_label, 'conditioning': enhanced_prompt[:180], 'model_name': model_label})}\n\n"

        out_png = os.path.join(get_cache_dir(), f".aeromesh_sd_{os.getpid()}_{int(time.time()*1000)}.png")
        b64_png = None
        sd_error = None
        try:
            await asyncio.to_thread(_run_sd_cli_sync, sd_bin, sd_model, enhanced_prompt, steps, width, height, out_png)
            if os.path.exists(out_png):
                with open(out_png, "rb") as f:
                    b64_png = base64.b64encode(f.read()).decode("ascii")
                os.remove(out_png)
        except Exception as e:
            sd_error = str(e)[:200]
            logger.warning("sd-cli error: %s", e)

        seed_hex = hashlib.sha256(enhanced_prompt.encode("utf-8")).hexdigest()[:12]
        final_evt = {
            "type": "dit_complete",
            "prompt": prompt,
            "conditioning": enhanced_prompt[:240],
            "mesh_active": mesh_active,
            "mesh_tokens_generated": mesh_tokens_generated,
            "model_name": model_label,
            "steps": steps,
            "width": width,
            "height": height,
            "accum_wire_mb": 0.0,
            "weights_on_wire_mb": 0.0,
            "elapsed_s": round(time.perf_counter() - t0, 2),
            "seed": seed_hex,
            "b64_png": b64_png,
            "sd_error": sd_error,
            "transport": transport_label,
        }
        yield f"data: {json.dumps(final_evt)}\n\n"
        yield "data: [DONE]\n\n"

    if stream:
        return StreamingResponse(
            dit_event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    frames = []
    async for chunk in dit_event_stream():
        if chunk.startswith("data: ") and not chunk.startswith("data: [DONE]"):
            frames.append(json.loads(chunk[6:].strip()))
    last = frames[-1] if frames else {}
    return JSONResponse(status_code=200, content={"created": int(time.time()), "data": [{"b64_json": last.get("b64_png")}]})


# ---------------------------------------------------------------------------
# MODEL HUB & INSTALLER ROUTES (LM Studio / Ollama-style Library & HF Hub)
# ---------------------------------------------------------------------------
ALLOWED_DOWNLOAD_HOSTS = frozenset({
    "huggingface.co",
    "www.huggingface.co",
    "hf.co",
    "cdn-lfs.huggingface.co",
    "cdn-lfs-us-1.huggingface.co",
    "cdn-lfs-eu-1.huggingface.co",
    "cas-bridge.xethub.hf.co",
})

CURATED_MODEL_CATALOG = [
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
        "description": "Ultra-fast starter model for quick testing, laptops, and CPU fallback.",
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
        "description": "Balanced everyday assistant with high token speed on single GPUs and Wi-Fi meshes.",
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
        "description": "Strong instruction following and coding quality that fits comfortably in 6-8 GB VRAM.",
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
        "description": "Step-by-step chain-of-thought reasoning with native <think> foldout support.",
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
        "description": "Built for 2-node AeroMESH layer pipelining across two 6-8 GB gaming laptops.",
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
        "description": "High-accuracy local software engineering model designed for distributed VRAM pooling.",
    },
]

_DOWNLOAD_LOCK = threading.Lock()
_DOWNLOAD_TASKS: dict = {}
_DOWNLOAD_CANCEL: dict = {}


def _format_bytes(num_bytes: int) -> str:
    if num_bytes >= 1024 ** 3:
        return f"{num_bytes / (1024 ** 3):.2f} GB"
    if num_bytes >= 1024 ** 2:
        return f"{num_bytes / (1024 ** 2):.1f} MB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.0f} KB"
    return f"{num_bytes} B"


def _extract_quant_tag(filename: str) -> str:
    m = re.search(r'(?i)(I?Q\d_[A-Z0-9_]+|Q\d_0|F16|BF16|F32|FP16)', filename)
    return m.group(1).upper() if m else ("FP16" if filename.lower().endswith(".safetensors") else "GGUF")


def _verify_gguf_magic_file(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"GGUF"
    except Exception:
        return False


def _validate_hf_download_url(raw_url: str) -> str:
    url = str(raw_url or "").strip()
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError("Only HTTPS download URLs are permitted.")
    host = (parsed.hostname or "").lower()
    if host not in ALLOWED_DOWNLOAD_HOSTS and not host.endswith(".hf.co") and not host.endswith(".huggingface.co"):
        raise ValueError(f"Download host '{host}' is not on the Hugging Face allowlist.")
    if not parsed.path or ".." in parsed.path:
        raise ValueError("Invalid download URL path.")
    return url


def _validate_hf_repo_id(repo_id: str) -> str:
    cleaned = str(repo_id or "").strip().strip("/")
    if not re.match(r'^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$', cleaned):
        raise ValueError("Invalid Hugging Face repository ID (expected 'owner/repo').")
    return cleaned


def _list_model_library() -> dict:
    primary_dir = get_primary_models_dir()
    primary_real = os.path.realpath(primary_dir)
    custom_dirs = load_custom_model_dirs()
    custom_norm = {os.path.realpath(d).lower() for d in custom_dirs if os.path.isdir(d)}

    directories = []
    models = []
    seen_real_files = set()

    for idx, d in enumerate(get_models_dirs()):
        exists = os.path.isdir(d)
        real_d = os.path.realpath(d) if exists else d
        count = 0
        if exists:
            try:
                for entry in os.scandir(d):
                    if not entry.is_file():
                        continue
                    fname = entry.name
                    lower = fname.lower()
                    if lower.endswith(".gguf"):
                        if "_stage" in fname or fname.startswith(".aeromesh_"):
                            continue
                        kind = "llm"
                    elif lower.endswith(".safetensors"):
                        kind = "diffusion"
                    else:
                        continue
                    real_file = os.path.realpath(entry.path)
                    if real_file in seen_real_files:
                        continue
                    seen_real_files.add(real_file)
                    st = entry.stat()
                    count += 1
                    models.append({
                        "name": fname,
                        "path": entry.path,
                        "directory": d,
                        "size_bytes": st.st_size,
                        "size_label": _format_bytes(st.st_size),
                        "kind": kind,
                        "quant": _extract_quant_tag(fname),
                        "managed": real_file.startswith(primary_real + os.sep),
                        "modified_ts": int(st.st_mtime),
                    })
            except OSError:
                pass

        if real_d.lower() == primary_real.lower() and idx == 0:
            dir_kind = "primary"
        elif real_d.lower() in custom_norm:
            dir_kind = "custom"
        else:
            dir_kind = "default"
        directories.append({
            "path": d,
            "kind": dir_kind,
            "exists": exists,
            "model_count": count,
        })

    models.sort(key=lambda m: (0 if m["kind"] == "llm" else 1, m["name"].lower()))
    with _DOWNLOAD_LOCK:
        downloads = list(_DOWNLOAD_TASKS.values())

    disk_free_bytes = 0
    try:
        disk_free_bytes = shutil.disk_usage(primary_dir).free
    except Exception:
        pass

    return {
        "primary_dir": primary_dir,
        "disk_free_bytes": disk_free_bytes,
        "disk_free_label": _format_bytes(disk_free_bytes),
        "directories": directories,
        "models": models,
        "catalog": CURATED_MODEL_CATALOG,
        "downloads": downloads,
    }


def _browse_windows_dialog(mode: str) -> Optional[str]:
    if sys.platform != "win32":
        return None
    if mode == "folder":
        ps_script = (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$d = New-Object System.Windows.Forms.FolderBrowserDialog; "
            "$d.Description = 'Select a folder containing GGUF or Safetensors models'; "
            "$d.UseDescriptionForTitle = $true; "
            "if ($d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { Write-Output $d.SelectedPath }"
        )
    else:
        ps_script = (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$d = New-Object System.Windows.Forms.OpenFileDialog; "
            "$d.Title = 'Select a GGUF or Safetensors Model File'; "
            "$d.Filter = 'AI Models (*.gguf;*.safetensors)|*.gguf;*.safetensors|All Files (*.*)|*.*'; "
            "if ($d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { Write-Output $d.FileName }"
        )
    flags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
    res = subprocess.run(
        ["powershell", "-NoProfile", "-STA", "-Command", ps_script],
        capture_output=True,
        text=True,
        timeout=120,
        creationflags=flags,
    )
    selected = (res.stdout or "").strip()
    return selected or None


async def _run_model_download_task(download_id: str, url: str, filename: str, repo: str, hf_token: str = ""):
    primary_dir = get_primary_models_dir()
    dest_path = os.path.join(primary_dir, filename)
    part_path = dest_path + ".part"
    headers = {"User-Agent": "AeroMESH-Studio/0.3"}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    t_start = time.perf_counter()
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=httpx.Timeout(connect=15.0, read=60.0, write=30.0, pool=15.0)) as client:
            async with client.stream("GET", url, headers=headers) as resp:
                # Re-validate final redirected host against allowlist
                final_host = (resp.url.host or "").lower()
                if resp.url.scheme != "https" or (
                    final_host not in ALLOWED_DOWNLOAD_HOSTS
                    and not final_host.endswith(".hf.co")
                    and not final_host.endswith(".huggingface.co")
                ):
                    raise ValueError(f"Blocked redirect to non-allowlisted host '{final_host}'.")

                if resp.status_code != 200:
                    raise RuntimeError(f"Hugging Face returned HTTP {resp.status_code}")

                total_bytes = int(resp.headers.get("content-length") or 0)
                downloaded = 0
                magic_checked = False
                is_gguf = filename.lower().endswith(".gguf")

                with _DOWNLOAD_LOCK:
                    _DOWNLOAD_TASKS[download_id]["total_bytes"] = total_bytes
                    _DOWNLOAD_TASKS[download_id]["status"] = "downloading"

                with open(part_path, "wb") as out_f:
                    async for chunk in resp.aiter_bytes(chunk_size=1024 * 512):
                        if _DOWNLOAD_CANCEL.get(download_id):
                            raise asyncio.CancelledError("Cancelled by user")
                        if not chunk:
                            continue
                        if not magic_checked and is_gguf:
                            if len(chunk) >= 4 and chunk[:4] != b"GGUF":
                                raise ValueError("Downloaded payload is not a valid GGUF file (missing GGUF magic header).")
                            magic_checked = True
                        out_f.write(chunk)
                        downloaded += len(chunk)
                        elapsed = max(0.05, time.perf_counter() - t_start)
                        speed_mbps = round((downloaded / (1024 * 1024)) / elapsed, 2)
                        pct = round((downloaded / total_bytes) * 100.0, 1) if total_bytes > 0 else 0.0
                        eta = int((total_bytes - downloaded) / (downloaded / elapsed)) if (total_bytes > downloaded > 0) else 0
                        with _DOWNLOAD_LOCK:
                            task = _DOWNLOAD_TASKS[download_id]
                            task["downloaded_bytes"] = downloaded
                            task["percent"] = pct
                            task["speed_mbps"] = speed_mbps
                            task["eta_seconds"] = eta

        if filename.lower().endswith(".gguf") and not _verify_gguf_magic_file(part_path):
            raise ValueError("Downloaded file failed GGUF header verification.")

        os.replace(part_path, dest_path)
        with _DOWNLOAD_LOCK:
            task = _DOWNLOAD_TASKS[download_id]
            task["status"] = "completed"
            task["percent"] = 100.0
            task["downloaded_bytes"] = os.path.getsize(dest_path)
            task["total_bytes"] = task["downloaded_bytes"]
            task["path"] = dest_path
            task["eta_seconds"] = 0
    except asyncio.CancelledError:
        try:
            if os.path.exists(part_path):
                os.remove(part_path)
        except OSError:
            pass
        with _DOWNLOAD_LOCK:
            _DOWNLOAD_TASKS[download_id]["status"] = "cancelled"
            _DOWNLOAD_TASKS[download_id]["error"] = "Cancelled by user"
    except Exception as exc:
        try:
            if os.path.exists(part_path):
                os.remove(part_path)
        except OSError:
            pass
        with _DOWNLOAD_LOCK:
            _DOWNLOAD_TASKS[download_id]["status"] = "error"
            _DOWNLOAD_TASKS[download_id]["error"] = str(exc)[:240]
    finally:
        _DOWNLOAD_CANCEL.pop(download_id, None)


@app.get("/api/models/library", dependencies=[Depends(verify_gateway_access)])
async def get_models_library():
    return JSONResponse(status_code=200, content=_list_model_library())


@app.post("/api/models/paths", dependencies=[Depends(verify_gateway_access)])
async def manage_model_paths(request: Request):
    body = await request.json()
    action = str(body.get("action", "add")).strip().lower()
    raw_path = str(body.get("path", "")).strip().strip('"')
    if not raw_path:
        raise HTTPException(status_code=400, detail="Directory path is required.")

    abs_path = os.path.abspath(os.path.expanduser(raw_path))
    dirs = load_custom_model_dirs()
    if action == "add":
        if not os.path.isdir(abs_path):
            raise HTTPException(status_code=400, detail=f"Folder does not exist: {abs_path}")
        if not any(os.path.realpath(d).lower() == os.path.realpath(abs_path).lower() for d in dirs):
            dirs.append(abs_path)
            save_custom_model_dirs(dirs)
    elif action == "remove":
        target_low = os.path.realpath(abs_path).lower() if os.path.exists(abs_path) else abs_path.lower()
        dirs = [d for d in dirs if os.path.realpath(d).lower() != target_low and d.lower() != abs_path.lower()]
        save_custom_model_dirs(dirs)
    else:
        raise HTTPException(status_code=400, detail="Unsupported action (expected 'add' or 'remove').")

    return JSONResponse(status_code=200, content={"status": "ok", "library": _list_model_library()})


@app.post("/api/models/import-path", dependencies=[Depends(verify_gateway_access)])
async def import_model_by_path(request: Request):
    body = await request.json()
    raw_path = str(body.get("path", "")).strip().strip('"')
    copy_to_library = bool(body.get("copy_to_library", False))
    if not raw_path:
        raise HTTPException(status_code=400, detail="Model file path is required.")

    abs_path = os.path.abspath(os.path.expanduser(raw_path))
    if os.path.isdir(abs_path):
        # User pasted a directory into the path bar — register it as a custom model folder
        dirs = load_custom_model_dirs()
        if not any(os.path.realpath(d).lower() == os.path.realpath(abs_path).lower() for d in dirs):
            dirs.append(abs_path)
            save_custom_model_dirs(dirs)
        return JSONResponse(status_code=200, content={"status": "ok", "imported_type": "directory", "path": abs_path, "library": _list_model_library()})

    if not os.path.isfile(abs_path):
        raise HTTPException(status_code=404, detail=f"File not found: {abs_path}")

    fname = os.path.basename(abs_path)
    lower = fname.lower()
    if not (lower.endswith(".gguf") or lower.endswith(".safetensors")):
        raise HTTPException(status_code=400, detail="Only .gguf and .safetensors model files are supported.")
    if lower.endswith(".gguf") and not _verify_gguf_magic_file(abs_path):
        raise HTTPException(status_code=400, detail="Selected file is not a valid GGUF model (missing GGUF magic header).")

    if copy_to_library:
        safe_name = _sanitize_model_name(fname)
        dest = os.path.join(get_primary_models_dir(), safe_name)
        part = dest + ".part"
        await asyncio.to_thread(shutil.copy2, abs_path, part)
        os.replace(part, dest)
        final_path = dest
    else:
        parent_dir = os.path.dirname(abs_path)
        dirs = load_custom_model_dirs()
        if not any(os.path.realpath(d).lower() == os.path.realpath(parent_dir).lower() for d in dirs):
            dirs.append(parent_dir)
            save_custom_model_dirs(dirs)
        final_path = abs_path

    return JSONResponse(
        status_code=200,
        content={"status": "ok", "imported_type": "file", "model": fname, "path": final_path, "library": _list_model_library()},
    )


@app.post("/api/models/browse", dependencies=[Depends(verify_gateway_access)])
async def browse_local_model_path(request: Request):
    body = await request.json()
    mode = str(body.get("mode", "file")).strip().lower()
    if sys.platform != "win32":
        return JSONResponse(status_code=200, content={"supported": False, "path": None, "message": "Native folder/file picker is available on Windows desktop; type or paste the path directly."})
    selected = await asyncio.to_thread(_browse_windows_dialog, mode)
    return JSONResponse(status_code=200, content={"supported": True, "path": selected})


@app.post("/api/models/upload", dependencies=[Depends(verify_gateway_access)])
async def upload_dropped_model(request: Request):
    raw_name = request.query_params.get("filename") or request.headers.get("x-model-filename") or ""
    try:
        filename = _sanitize_model_name(raw_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    lower = filename.lower()
    if not (lower.endswith(".gguf") or lower.endswith(".safetensors")):
        raise HTTPException(status_code=400, detail="Only .gguf and .safetensors model files can be uploaded.")

    primary_dir = get_primary_models_dir()
    dest = os.path.join(primary_dir, filename)
    part = dest + ".part"
    written = 0
    header_buf = bytearray()
    try:
        with open(part, "wb") as out_f:
            async for chunk in request.stream():
                if not chunk:
                    continue
                if len(header_buf) < 4:
                    header_buf.extend(chunk[: 4 - len(header_buf)])
                    if len(header_buf) == 4 and lower.endswith(".gguf") and bytes(header_buf) != b"GGUF":
                        raise HTTPException(status_code=400, detail="Dropped file is not a valid GGUF model (missing GGUF header).")
                out_f.write(chunk)
                written += len(chunk)
        if written == 0:
            raise HTTPException(status_code=400, detail="Uploaded file was empty.")
        os.replace(part, dest)
    except Exception:
        try:
            if os.path.exists(part):
                os.remove(part)
        except OSError:
            pass
        raise

    return JSONResponse(
        status_code=200,
        content={"status": "ok", "model": filename, "size_bytes": written, "path": dest, "library": _list_model_library()},
    )


@app.post("/api/models/delete", dependencies=[Depends(verify_gateway_access)])
async def delete_managed_model(request: Request):
    body = await request.json()
    raw_name = str(body.get("name", "")).strip()
    try:
        filename = _sanitize_model_name(raw_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    primary_dir = get_primary_models_dir()
    primary_real = os.path.realpath(primary_dir)
    candidate = os.path.realpath(os.path.join(primary_dir, filename))
    if not candidate.startswith(primary_real + os.sep) or not os.path.isfile(candidate):
        raise HTTPException(
            status_code=404,
            detail="Only model files stored in AeroMESH's managed models directory can be deleted from the UI. To unlink an external folder, remove it from Model Folders.",
        )
    os.remove(candidate)
    return JSONResponse(status_code=200, content={"status": "ok", "deleted": filename, "library": _list_model_library()})


@app.get("/api/models/hf/search", dependencies=[Depends(verify_gateway_access)])
async def search_hf_gguf_models(q: str = "", limit: int = 12):
    query = str(q or "").strip()
    lim = max(1, min(int(limit), 25))
    params = {
        "filter": "gguf",
        "sort": "downloads",
        "direction": "-1",
        "limit": str(lim),
    }
    if query:
        params["search"] = query
    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            resp = await client.get(
                "https://huggingface.co/api/models",
                params=params,
                headers={"User-Agent": "AeroMESH-Studio/0.3"},
            )
            if resp.status_code != 200:
                raise HTTPException(status_code=502, detail=f"Hugging Face search returned HTTP {resp.status_code}")
            items = resp.json()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not reach Hugging Face Hub: {exc}")

    results = []
    for item in items:
        repo_id = item.get("id") or item.get("modelId") or ""
        if not repo_id:
            continue
        results.append({
            "repo": repo_id,
            "author": item.get("author") or repo_id.split("/")[0],
            "downloads": item.get("downloads", 0),
            "likes": item.get("likes", 0),
            "gated": bool(item.get("gated", False)),
            "last_modified": item.get("lastModified", ""),
        })
    return JSONResponse(status_code=200, content={"results": results})


@app.get("/api/models/hf/files", dependencies=[Depends(verify_gateway_access)])
async def list_hf_repo_gguf_files(repo: str):
    try:
        repo_id = _validate_hf_repo_id(repo)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            resp = await client.get(
                f"https://huggingface.co/api/models/{repo_id}/tree/main",
                headers={"User-Agent": "AeroMESH-Studio/0.3"},
            )
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=f"Hugging Face returned HTTP {resp.status_code} for {repo_id}")
            entries = resp.json()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not inspect repository {repo_id}: {exc}")

    files = []
    for entry in entries:
        path = str(entry.get("path") or "")
        if not path.lower().endswith(".gguf"):
            continue
        size = int(entry.get("size") or (entry.get("lfs") or {}).get("size") or 0)
        fname = os.path.basename(path)
        est_vram_gb = round((size / (1024 ** 3)) * 1.18 + 0.5, 1) if size > 0 else 0.0
        files.append({
            "filename": fname,
            "path": path,
            "size_bytes": size,
            "size_label": _format_bytes(size) if size > 0 else "Unknown",
            "quant": _extract_quant_tag(fname),
            "est_vram_gb": est_vram_gb,
            "download_url": f"https://huggingface.co/{repo_id}/resolve/main/{quote(path, safe='/')}",
            "recommended": "Q4_K_M" in fname.upper(),
        })
    files.sort(key=lambda f: (0 if f["recommended"] else 1, f["size_bytes"]))
    return JSONResponse(status_code=200, content={"repo": repo_id, "files": files})


@app.post("/api/models/download", dependencies=[Depends(verify_gateway_access)])
async def start_model_download(request: Request):
    body = await request.json()
    raw_url = str(body.get("url", "")).strip()
    repo = str(body.get("repo", "")).strip()
    raw_filename = str(body.get("filename", "")).strip()
    hf_token = str(body.get("hf_token", "")).strip()

    if not raw_url and repo and raw_filename:
        repo_id = _validate_hf_repo_id(repo)
        safe_fname = _sanitize_model_name(os.path.basename(raw_filename))
        raw_url = f"https://huggingface.co/{repo_id}/resolve/main/{quote(raw_filename, safe='/')}"
    else:
        if not raw_filename and raw_url:
            raw_filename = os.path.basename(urlparse(raw_url).path)

    try:
        url = _validate_hf_download_url(raw_url)
        filename = _sanitize_model_name(os.path.basename(raw_filename))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if not (filename.lower().endswith(".gguf") or filename.lower().endswith(".safetensors")):
        raise HTTPException(status_code=400, detail="Only .gguf and .safetensors files can be downloaded.")

    download_id = uuid.uuid4().hex[:12]
    task_info = {
        "id": download_id,
        "filename": filename,
        "repo": repo or urlparse(url).path.strip("/"),
        "status": "starting",
        "downloaded_bytes": 0,
        "total_bytes": 0,
        "percent": 0.0,
        "speed_mbps": 0.0,
        "eta_seconds": 0,
        "error": None,
    }
    with _DOWNLOAD_LOCK:
        _DOWNLOAD_TASKS[download_id] = task_info
        _DOWNLOAD_CANCEL[download_id] = False

    asyncio.create_task(_run_model_download_task(download_id, url, filename, repo, hf_token))
    return JSONResponse(status_code=202, content={"status": "started", "download": task_info})


@app.get("/api/models/downloads", dependencies=[Depends(verify_gateway_access)])
async def list_model_downloads():
    with _DOWNLOAD_LOCK:
        items = list(_DOWNLOAD_TASKS.values())
    return JSONResponse(status_code=200, content={"downloads": items})


@app.post("/api/models/download/cancel", dependencies=[Depends(verify_gateway_access)])
async def cancel_model_download(request: Request):
    body = await request.json()
    download_id = str(body.get("id", "")).strip()
    with _DOWNLOAD_LOCK:
        if download_id not in _DOWNLOAD_TASKS:
            raise HTTPException(status_code=404, detail="Download task not found.")
        _DOWNLOAD_CANCEL[download_id] = True
    return JSONResponse(status_code=200, content={"status": "cancelling", "id": download_id})


# ---------------------------------------------------------------------------
# CATCH-ALL PROXY ROUTES FOR ARBITRARY /v1/* AND /api/* TRAFFIC
# ---------------------------------------------------------------------------
@app.api_route("/api/{path:path}", methods=PROXY_METHODS, dependencies=[Depends(verify_gateway_access)])
async def proxy_api_catchall(path: str, request: Request):
    """Catch-all proxy for /api/* routes."""
    return await proxy_request(request, f"/api/{path}")


@app.api_route("/v1/{path:path}", methods=PROXY_METHODS, dependencies=[Depends(verify_gateway_access)])
async def proxy_v1_catchall(path: str, request: Request):
    """Catch-all proxy for /v1/* routes."""
    return await proxy_request(request, f"/v1/{path}")


# ---------------------------------------------------------------------------
# MOUNT STATIC ASSETS
# ---------------------------------------------------------------------------
# Mounted after API routes to guarantee static files do not shadow endpoints
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------------------------------------------------------------------------
# MAIN ENTRYPOINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n========================================================")
    print("   [+] AEROMESH LIQUID GLASS CHAT GATEWAY                 ")
    print("========================================================")
    print(f"  Web Interface URL:      http://{AEROMESH_UI_HOST}:{AEROMESH_UI_PORT}")
    print(f"  Target Coordinator API: {AEROMESH_BACKEND_URL}")
    print(f"  Theme:                  Liquid Glass & Fluid Topological Mesh")
    print(f"  Dynamic Backend:        ENABLED")
    print("========================================================\n")
    uvicorn.run(app, host=AEROMESH_UI_HOST, port=AEROMESH_UI_PORT)
