import os
import sys
import re
import json
import glob
import hmac
import logging
from contextlib import asynccontextmanager
from typing import Optional

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Lightweight .env loader
def load_dotenv():
    env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
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
        value = self.get(key, self._MISSING)
        if value is self._MISSING:
            raise KeyError(key)
        return value

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


app = FastAPI(
    title="AeroMesh Intelligence Engine Gateway",
    version="1.0.0",
    lifespan=lifespan,
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
    """Serves the single-page dashboard.

    The file is read once and cached: it was previously re-read from disk on
    every request.
    """
    global _index_html_cache
    if _index_html_cache is None:
        index_path = os.path.join(STATIC_DIR, "index.html")
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                _index_html_cache = f.read()
        else:
            logger.error("static/index.html is missing; the dashboard cannot be served")
            _index_html_cache = "<h1>AeroMesh UI error: static/index.html not found</h1>"
    return _index_html_cache


_index_html_cache = None


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
