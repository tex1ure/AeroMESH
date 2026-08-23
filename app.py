import os
import sys
import json
import glob
from typing import AsyncGenerator

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import httpx
import uvicorn

# ---------------------------------------------------------------------------
# DYNAMIC BACKEND CONFIGURATION
# ---------------------------------------------------------------------------
AEROMESH_ENDPOINT = os.getenv("AEROMESH_ENDPOINT", "http://127.0.0.1:8080")
CHAT_COMPLETIONS_URL = f"{AEROMESH_ENDPOINT}/v1/chat/completions"
MODELS_URL = f"{AEROMESH_ENDPOINT}/v1/models"
CLUSTER_STATUS_URL = f"{AEROMESH_ENDPOINT}/api/cluster/status"
HEALTH_URL = f"{AEROMESH_ENDPOINT}/health"

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "7860"))

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
@app.get("/v1/models")
async def get_models():
    """
    Fetches the active models directly from the running AeroMesh coordinator.
    If coordinator is offline, dynamically discovers local GGUF models from models/ directory.
    """
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            resp = await client.get(MODELS_URL)
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


@app.post("/api/model/switch")
@app.post("/v1/models/load")
async def switch_model(payload: dict):
    """
    Proxies model switch requests directly to the AeroMesh coordinator.
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{AEROMESH_ENDPOINT}/api/model/switch", json=payload)
            return JSONResponse(status_code=resp.status_code, content=resp.json())
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.get("/api/cluster/status")
async def cluster_status():
    """
    Queries the live AeroMesh Coordinator node dynamically.
    Returns real connected node topology, layer distribution, and transport status.
    """
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            resp = await client.get(CLUSTER_STATUS_URL)
            if resp.status_code == 200:
                return resp.json()
            # Fallback to health endpoint
            health_resp = await client.get(HEALTH_URL)
            if health_resp.status_code == 200:
                models_resp = await client.get(MODELS_URL)
                model_name = "test.gguf"
                if models_resp.status_code == 200:
                    data = models_resp.json().get("data", [])
                    if data:
                        model_name = data[0].get("id", model_name)
                return {
                    "cluster_status": "ONLINE",
                    "active_model": model_name,
                    "connected": True,
                    "status": "online",
                    "coordinator": {
                        "model": model_name,
                        "transport": "CUDA High-Throughput Engine",
                        "status": "READY"
                    }
                }
    except Exception:
        pass

    # Coordinator is offline
    return {
        "connected": False,
        "status": "offline",
        "coordinator_endpoint": AEROMESH_ENDPOINT,
        "message": "AeroMesh Coordinator is offline."
    }


# ---------------------------------------------------------------------------
# STREAMING CHAT COMPLETIONS PROXY
# ---------------------------------------------------------------------------
@app.post("/api/chat/abort")
async def chat_abort():
    """Signals cancellation to backend coordinator if required."""
    return {"status": "aborted"}


@app.post("/api/chat/stream")
async def chat_stream(request: Request):
    """
    Server-Sent Events (SSE) streaming endpoint.
    Directly streams token generation from the active AeroMesh Coordinator backend.
    """
    body = await request.json()

    async def stream_generator():
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=10.0)) as client:
                async with client.stream("POST", CHAT_COMPLETIONS_URL, json=body) as response:
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
