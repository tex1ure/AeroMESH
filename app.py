import os
import sys
import json
import glob
import hmac
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

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

# Backwards-compatibility aliases
AEROMESH_ENDPOINT = AEROMESH_BACKEND_URL
HOST = AEROMESH_UI_HOST
PORT = AEROMESH_UI_PORT

AEROMESH_API_KEY = os.getenv("AEROMESH_API_KEY", "")
GATEWAY_API_KEY = os.getenv("AEROMESH_GATEWAY_API_KEY", "")

if AEROMESH_UI_HOST == "0.0.0.0":
    logger.warning(
        "AeroMESH gateway is binding to all network interfaces (0.0.0.0). "
        "This is intended for local development and trusted tailnets only."
    )

# Hop-by-hop headers that must not be forwarded by a proxy
HOP_BY_HOP_HEADERS = {
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
}

def filter_headers(headers: dict) -> dict:
    """Strip hop-by-hop HTTP headers before proxying."""
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS
    }

def get_auth_headers() -> dict:
    """Inject Bearer token for backend coordinator authentication."""
    key = os.getenv("AEROMESH_API_KEY", "")
    return {"Authorization": f"Bearer {key}"} if key else {}

async def verify_gateway_access(request: Request):
    """FastAPI dependency returning 401 envelope on proxy routes if exposed beyond localhost."""
    client_host = request.client.host if request.client else "127.0.0.1"
    is_local = client_host in ("127.0.0.1", "localhost", "::1", "testclient")
    if not GATEWAY_API_KEY and is_local:
        return True
    if not GATEWAY_API_KEY:
        return True

    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail={"error": {"message": "Unauthorized gateway access", "type": "authentication_error", "param": None, "code": "invalid_api_key"}}
        )
    token = auth.replace("Bearer ", "").strip()
    if not hmac.compare_digest(token, GATEWAY_API_KEY):
        raise HTTPException(
            status_code=401,
            detail={"error": {"message": "Invalid gateway API key", "type": "authentication_error", "param": None, "code": "invalid_api_key"}}
        )
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

async def proxy_request(request: Request, backend_path: str):
    """
    Forwards HTTP request to Axum backend, preserving stream semantics and headers.
    """
    client = get_backend_client()
    method = request.method
    headers = filter_headers(dict(request.headers))
    auth_headers = get_auth_headers()
    headers.update(auth_headers)
    params = request.query_params
    body = await request.body()

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
        return JSONResponse(
            status_code=502,
            content={
                "error": "backend_unreachable",
                "detail": f"Cannot connect to AeroMESH backend at {AEROMESH_BACKEND_URL}",
            },
        )
    except Exception as exc:
        logger.exception("Gateway proxy error")
        return JSONResponse(
            status_code=502,
            content={
                "error": "gateway_proxy_error",
                "detail": str(exc),
            },
        )

    response_headers = filter_headers(dict(backend_response.headers))
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
    """Serves the main claymorphic chat dashboard single-page application."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>AeroMesh UI Error: static/index.html not found</h1>"


# ---------------------------------------------------------------------------
# SPECIALIZED API ENDPOINTS WITH GRACEFUL OFFLINE FALLBACKS
# ---------------------------------------------------------------------------
@app.get("/v1/models", dependencies=[Depends(verify_gateway_access)])
async def get_models():
    """
    Fetches active models from the running AeroMesh coordinator.
    If coordinator is offline, dynamically discovers local GGUF models from models/ directory.
    """
    client = get_backend_client()
    try:
        resp = await client.get("/v1/models", headers=get_auth_headers(), timeout=1.5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass

    # Discover local GGUF files dynamically if backend is offline
    local_models = []
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    if os.path.exists(models_dir):
        for f in glob.glob(os.path.join(models_dir, "*.gguf")):
            local_models.append({
                "id": os.path.basename(f),
                "object": "model",
                "owned_by": "local_disk",
                "path": f
            })

    return {
        "object": "list",
        "data": local_models
    }


@app.post("/api/model/switch", dependencies=[Depends(verify_gateway_access)])
@app.post("/v1/models/load", dependencies=[Depends(verify_gateway_access)])
async def switch_model(payload: dict):
    """Proxies model switch requests directly to the AeroMesh coordinator."""
    client = get_backend_client()
    try:
        resp = await client.post("/api/model/switch", json=payload, headers=get_auth_headers(), timeout=30.0)
        return JSONResponse(status_code=resp.status_code, content=resp.json())
    except httpx.ConnectError:
        return JSONResponse(
            status_code=502,
            content={"error": "backend_unreachable", "detail": f"Cannot connect to AeroMESH backend at {AEROMESH_BACKEND_URL}"}
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


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


@app.post("/v1/chat/completions", dependencies=[Depends(verify_gateway_access)])
@app.post("/api/chat", dependencies=[Depends(verify_gateway_access)])
async def chat_completions(request: Request):
    """OpenAI-compatible chat completions proxy, automatically delegating streaming requests."""
    body = await request.json()
    if body.get("stream", True):
        return await chat_stream(request)

    client = get_backend_client()
    try:
        resp = await client.post("/v1/chat/completions", json=body, headers=get_auth_headers(), timeout=httpx.Timeout(180.0, connect=10.0))
        return JSONResponse(status_code=resp.status_code, content=resp.json())
    except httpx.ConnectError:
        return JSONResponse(
            status_code=502,
            content={"error": "backend_unreachable", "detail": f"Cannot connect to AeroMESH backend at {AEROMESH_BACKEND_URL}"}
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/chat/abort", dependencies=[Depends(verify_gateway_access)])
async def chat_abort():
    """Signals stream cancellation to the backend coordinator."""
    client = get_backend_client()
    try:
        await client.post("/api/chat/abort", headers=get_auth_headers(), timeout=2.0)
    except Exception:
        pass
    return {"status": "aborted"}


@app.post("/api/chat/stream", dependencies=[Depends(verify_gateway_access)])
async def chat_stream(request: Request):
    """
    Server-Sent Events (SSE) streaming endpoint.
    Directly streams token generation from the active AeroMesh Coordinator backend.
    """
    body = await request.json()
    client = get_backend_client()

    async def stream_generator():
        try:
            async with client.stream("POST", "/v1/chat/completions", json=body, headers=get_auth_headers(), timeout=httpx.Timeout(180.0, connect=10.0)) as response:
                if response.status_code != 200:
                    err_text = await response.aread()
                    err_msg = json.dumps({
                        "choices": [{
                            "delta": {
                                "content": f"\n\n❌ **AeroMesh Backend Error**: HTTP {response.status_code} - {err_text.decode('utf-8', errors='ignore')}"
                            }
                        }]
                    })
                    yield f"data: {err_msg}\n\n"
                    return

                async for chunk in response.aiter_text():
                    yield chunk
        except httpx.ConnectError:
            err_msg = json.dumps({
                "choices": [{
                    "delta": {
                        "content": (
                            "❌ **AeroMesh Coordinator is Offline**\n\n"
                            f"Could not connect to `{AEROMESH_BACKEND_URL}`.\n\n"
                            "To start the zero-weight cluster coordinator, run:\n"
                            "```powershell\n"
                            "cargo run --bin aeromesh -- serve --model models/<model>.gguf --layers 0..24 --peers <worker-ip>:50052 --port 8080\n"
                            "```"
                        )
                    }
                }]
            })
            yield f"data: {err_msg}\n\n"
        except Exception as e:
            err_msg = json.dumps({
                "choices": [{
                    "delta": {
                        "content": f"\n\n❌ **Pipeline Network Error**: {str(e)}"
                    }
                }]
            })
            yield f"data: {err_msg}\n\n"

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(stream_generator(), media_type="text/event-stream", headers=headers)


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
    print("   [+] AEROMESH CLAYMORPHIC CHAT GATEWAY                 ")
    print("========================================================")
    print(f"  Web Interface URL:      http://{AEROMESH_UI_HOST}:{AEROMESH_UI_PORT}")
    print(f"  Target Coordinator API: {AEROMESH_BACKEND_URL}")
    print(f"  Theme:                  Warm Grey & Smoked Orange Claymorphism")
    print(f"  Dynamic Backend:        ENABLED")
    print("========================================================\n")
    uvicorn.run(app, host=AEROMESH_UI_HOST, port=AEROMESH_UI_PORT)
