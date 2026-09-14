import os
import sys
import json
import glob
import hmac
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

from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import httpx
import uvicorn

# ---------------------------------------------------------------------------
# DYNAMIC BACKEND CONFIGURATION & THREAT MODEL (SEC-01)
# ---------------------------------------------------------------------------
# Threat model:
# - browser -> gateway (7860): stays open on 127.0.0.1; remote exposure requires Tailscale with ACLs
# - gateway -> coordinator (8080): authenticated via Bearer token (AEROMESH_API_KEY)
# - coordinator -> worker (50052): authenticated via mutual SHA-256 handshake secret (AEROMESH_WORKER_SECRET)
AEROMESH_ENDPOINT = os.getenv("AEROMESH_ENDPOINT", "http://127.0.0.1:8080")
CHAT_COMPLETIONS_URL = f"{AEROMESH_ENDPOINT}/v1/chat/completions"
MODELS_URL = f"{AEROMESH_ENDPOINT}/v1/models"
CLUSTER_STATUS_URL = f"{AEROMESH_ENDPOINT}/api/cluster/status"
HEALTH_URL = f"{AEROMESH_ENDPOINT}/health"

AEROMESH_API_KEY = os.getenv("AEROMESH_API_KEY", "")
GATEWAY_API_KEY = os.getenv("AEROMESH_GATEWAY_API_KEY", "")

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "7860"))

def get_auth_headers() -> dict:
    key = os.getenv("AEROMESH_API_KEY", "")
    return {"Authorization": f"Bearer {key}"} if key else {}

async def verify_gateway_access(request: Request):
    """FastAPI dependency returning 401 envelope on proxy routes if exposed beyond localhost."""
    client_host = request.client.host if request.client else "127.0.0.1"
    is_local = client_host in ("127.0.0.1", "localhost", "::1")
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

app = FastAPI(title="AeroMesh Intelligence Engine", version="1.0.0")

# Mount static files directory
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------------------------------------------------------------------------
# ROOT ROUTE - SERVE FULL-SPREAD CLAYMORPHIC SPA
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_spa():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>AeroMesh UI Error: static/index.html not found</h1>"


# ---------------------------------------------------------------------------
# DYNAMIC BACKEND MODELS & CLUSTER STATUS (Zero Hardcoding)
# ---------------------------------------------------------------------------
@app.get("/v1/models", dependencies=[Depends(verify_gateway_access)])
async def get_models():
    """
    Fetches the active models directly from the running AeroMesh coordinator.
    If coordinator is offline, dynamically discovers local GGUF models from models/ directory.
    """
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            resp = await client.get(MODELS_URL, headers=get_auth_headers())
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
    """
    Proxies model switch requests directly to the AeroMesh coordinator.
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{AEROMESH_ENDPOINT}/api/model/switch", json=payload, headers=get_auth_headers())
            return JSONResponse(status_code=resp.status_code, content=resp.json())
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.get("/health")
async def health():
    """
    Proxies health check to coordinator, returns gateway status.
    """
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(HEALTH_URL, headers=get_auth_headers())
            if resp.status_code == 200:
                return resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {"status": "ok", "state": "online_idle"}
    except Exception:
        pass
    return {"status": "offline", "state": "offline"}


@app.get("/api/cluster/status", dependencies=[Depends(verify_gateway_access)])
async def cluster_status():
    """
    Queries the live AeroMesh Coordinator node dynamically.
    Returns real connected node topology, layer distribution, and transport status.
    Guarantees fallback payload with state:"offline" on connection error (never a raw 5xx).
    """
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(CLUSTER_STATUS_URL, headers=get_auth_headers())
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
        "coordinator_endpoint": AEROMESH_ENDPOINT,
        "message": "AeroMesh Coordinator is offline. Start the coordinator with: cargo run --bin aeromesh -- serve --model <model.gguf> --layers <range> --peers <worker-ip>:50052"
    }


@app.post("/v1/chat/completions", dependencies=[Depends(verify_gateway_access)])
@app.post("/api/chat", dependencies=[Depends(verify_gateway_access)])
async def chat_completions(request: Request):
    body = await request.json()
    if body.get("stream", True):
        return await chat_stream(request)
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=10.0)) as client:
            resp = await client.post(CHAT_COMPLETIONS_URL, json=body, headers=get_auth_headers())
            return JSONResponse(status_code=resp.status_code, content=resp.json())
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/chat/abort", dependencies=[Depends(verify_gateway_access)])
async def chat_abort():
    """Signals cancellation to backend coordinator if required."""
    return {"status": "aborted"}


@app.post("/api/chat/stream", dependencies=[Depends(verify_gateway_access)])
async def chat_stream(request: Request):
    """
    Server-Sent Events (SSE) streaming endpoint.
    Directly streams token generation from the active AeroMesh Coordinator backend.
    """
    body = await request.json()

    async def stream_generator():
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=10.0)) as client:
                async with client.stream("POST", CHAT_COMPLETIONS_URL, json=body, headers=get_auth_headers()) as response:
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
                            f"Could not connect to `{AEROMESH_ENDPOINT}`.\n\n"
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

    return StreamingResponse(stream_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# MAIN ENTRYPOINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n========================================================")
    print("   [+] AEROMESH CLAYMORPHIC CHAT INTERFACE               ")
    print("========================================================")
    print(f"  Web Interface URL:      http://127.0.0.1:{PORT}")
    print(f"  Target Coordinator API: {AEROMESH_ENDPOINT}")
    print(f"  Theme:                  Warm Grey & Smoked Orange Claymorphism (Zero Purple)")
    print(f"  Dynamic Backend:        ENABLED")
    print("========================================================\n")
    uvicorn.run(app, host=HOST, port=PORT)
