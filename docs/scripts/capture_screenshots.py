import os
import sys
import time
import threading
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATIC_DIR = os.path.join(ROOT_DIR, "static")
OUTPUT_DIR = os.path.join(ROOT_DIR, "docs", "screenshots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open(os.path.join(STATIC_DIR, "index.html"), "r", encoding="utf-8") as f:
        return f.read()

@app.get("/v1/models")
async def get_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
                "object": "model",
                "owned_by": "local_disk",
                "is_active": True,
                "path": "models/DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf"
            },
            {
                "id": "Llama-3.1-8B-Instruct-Q4_K_M.gguf",
                "object": "model",
                "owned_by": "local_disk",
                "path": "models/Llama-3.1-8B-Instruct-Q4_K_M.gguf"
            },
            {
                "id": "DS.gguf",
                "object": "model",
                "owned_by": "local_disk",
                "path": "models/DS.gguf"
            },
            {
                "id": "test.gguf",
                "object": "model",
                "owned_by": "local_disk",
                "path": "models/test.gguf"
            }
        ]
    }

@app.get("/health")
async def health():
    return {"status": "ok", "state": "online_idle"}

@app.get("/api/cluster/status")
async def cluster_status():
    return {
        "state": "online_idle",
        "status": "online",
        "connected": True,
        "is_direct_wireguard": True,
        "active_model": "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
        "rtt_ms": "1.14",
        "per_token_kb": "5.16 KB/tok",
        "local_stage": {
            "layer_start": 0,
            "layer_end": 23
        },
        "workers": ["100.64.0.11:50052"],
        "total_layers": 48,
        "coordinator_endpoint": "http://127.0.0.1:8080"
    }

@app.post("/api/model/switch")
@app.post("/v1/models/load")
async def switch_model():
    return {"status": "ok", "message": "Model switched successfully"}


def run_server():
    uvicorn.run(app, host="127.0.0.1", port=7865, log_level="warning")


def main():
    # 1. Start mock server in background thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    time.sleep(2)
    print("Mock AeroMESH server running on http://127.0.0.1:7865")

    # 2. Configure headless Chrome
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument("--force-device-scale-factor=1")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--hide-scrollbars")
    opts.binary_location = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

    driver = webdriver.Chrome(options=opts)
    driver.set_window_size(1920, 1080)

    try:
        url = "http://127.0.0.1:7865"

        # -------------------------------------------------------------
        # 1. Capture 01-chat-workspace.png (Idle / Hero State)
        # -------------------------------------------------------------
        print("Capturing 01-chat-workspace.png...")
        driver.get(url)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "chat-viewport"))
        )
        time.sleep(2.0)

        # Inject some nice recent conversations for authentic sidebar look
        driver.execute_script("""
            const chatList = document.getElementById('chat-list-today');
            chatList.innerHTML = `
                <div class="conversation-item active">
                    <i data-lucide="message-square" style="width: 14px; height: 14px;"></i>
                    <span class="conv-title">Pipeline-Parallel Inference Benchmark</span>
                </div>
                <div class="conversation-item">
                    <i data-lucide="message-square" style="width: 14px; height: 14px;"></i>
                    <span class="conv-title">WireGuard Activation Throughput</span>
                </div>
                <div class="conversation-item">
                    <i data-lucide="message-square" style="width: 14px; height: 14px;"></i>
                    <span class="conv-title">Identity RMSNorm Layer Handoff</span>
                </div>
            `;
            if (window.lucide) lucide.createIcons();
        """)
        time.sleep(0.5)
        path1 = os.path.join(OUTPUT_DIR, "01-chat-workspace.png")
        driver.save_screenshot(path1)
        print(f"Saved: {path1}")

        # -------------------------------------------------------------
        # 2. Capture 02-model-loading.png (Model Selector / Slicing Panel)
        # -------------------------------------------------------------
        print("Capturing 02-model-loading.png...")
        driver.execute_script("""
            const menu = document.getElementById('model-dropdown-menu');
            menu.classList.add('active');
            menu.innerHTML = `
                <div style="padding: 6px 10px; font-size: 11px; font-weight: 700; color: var(--accent-orange); text-transform: uppercase; letter-spacing: 0.5px;">Available GGUF Models (models/)</div>
                <div class="model-dropdown-item active" style="background: rgba(229, 130, 68, 0.15); border: 1px solid rgba(229, 130, 68, 0.3); border-radius: 6px; padding: 10px 12px; margin-bottom: 4px;">
                    <div>
                        <div style="font-weight: 700; color: #fff; font-size: 13px;">DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf</div>
                        <div style="font-size: 11px; color: var(--text-muted); margin-top: 2px;">14B Parameters • 48 Layers • INT8 Quantized Streaming • Active Stage 1 (0..23)</div>
                    </div>
                    <span class="badge-stage" style="background: var(--accent-orange); color: #000; font-weight: 700; font-size: 10.5px; padding: 2px 8px; border-radius: 4px;">ACTIVE</span>
                </div>
                <div class="model-dropdown-item" style="padding: 8px 12px; border-radius: 6px; display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-weight: 600; color: var(--text-main); font-size: 13px;">Llama-3.1-8B-Instruct-Q4_K_M.gguf</div>
                        <div style="font-size: 11px; color: var(--text-dim);">8B Parameters • 32 Layers • Zero-Weight P2P</div>
                    </div>
                    <span style="font-size: 11px; color: var(--text-dim);">Cached</span>
                </div>
                <div class="model-dropdown-item" style="padding: 8px 12px; border-radius: 6px; display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-weight: 600; color: var(--text-main); font-size: 13px;">DS.gguf</div>
                        <div style="font-size: 11px; color: var(--text-dim);">Full 14B Weights • Stored on NVMe SSD</div>
                    </div>
                    <span style="font-size: 11px; color: var(--text-dim);">Cached</span>
                </div>
            `;
            if (window.lucide) lucide.createIcons();
        """)
        time.sleep(0.8)
        path2 = os.path.join(OUTPUT_DIR, "02-model-loading.png")
        driver.save_screenshot(path2)
        print(f"Saved: {path2}")

        # Close dropdown for subsequent shots
        driver.execute_script("document.getElementById('model-dropdown-menu').classList.remove('active');")
        time.sleep(0.3)

        # -------------------------------------------------------------
        # 3. Capture 03-streaming-response.png (Active Generation / Code / Math / Think)
        # -------------------------------------------------------------
        print("Capturing 03-streaming-response.png...")
        driver.execute_script("""
            document.getElementById('hero-state').style.display = 'none';
            const container = document.getElementById('messages-container');
            container.style.paddingBottom = '80px';
            
            const mathHtml = window.katex ? katex.renderToString("\\\\mathcal{L}_{\\\\text{step}} = T_{\\\\text{comm}} + \\\\max(T_{\\\\text{stage1}}, T_{\\\\text{stage2}})", {displayMode: true}) : '$$\\\\mathcal{L}_{step} = T_{comm} + \\\\max(T_{stage1}, T_{stage2})$$';

            container.innerHTML = `
                <!-- User Message -->
                <div class="message-row user-row" style="display: flex; justify-content: flex-end; margin-bottom: 16px;">
                    <div class="message-bubble user-bubble" style="max-width: 65%; background: var(--bg-surface-elevated); padding: 12px 18px; border-radius: 16px 16px 4px 16px; border: 1px solid rgba(229,130,68,0.25);">
                        <div class="user-text" style="color: #fff; font-size: 14px; line-height: 1.4;">Explain how pipeline-parallel inference works across two machines. Include one small Python example and one short LaTeX equation.</div>
                    </div>
                </div>

                <!-- Assistant Streaming Message -->
                <div class="message-row assistant-row" style="display: flex; gap: 14px; max-width: 82%; margin-bottom: 16px;">
                    <div class="assistant-avatar" style="width: 36px; height: 36px; border-radius: 50%; background: linear-gradient(135deg, #e58244, #c96b30); display: flex; align-items: center; justify-content: center; color: #000; flex-shrink: 0; box-shadow: 0 4px 12px rgba(229,130,68,0.4);">
                        <i data-lucide="zap" style="width: 18px; height: 18px;"></i>
                    </div>
                    <div class="message-bubble assistant-bubble generating" style="flex: 1; background: var(--bg-surface); border: 1px solid rgba(255,255,255,0.08); border-radius: 4px 18px 18px 18px; padding: 16px 20px; box-shadow: var(--clay-shadow-floating);">
                        <!-- Reasoning Foldout -->
                        <div class="reasoning-accordion open" style="background: rgba(0,0,0,0.35); border-radius: 8px; border: 1px solid rgba(229,130,68,0.25); padding: 8px 12px; margin-bottom: 12px;">
                            <div class="reasoning-toggle" style="display: flex; align-items: center; gap: 8px; cursor: pointer; color: var(--accent-orange); font-size: 12px;">
                                <i data-lucide="brain" style="width: 13px; height: 13px;"></i>
                                <span style="font-weight: 700;">DeepSeek-R1 Thought Process (0.6s)</span>
                                <i data-lucide="chevron-down" class="accordion-chevron" style="width: 13px; height: 13px; margin-left: auto;"></i>
                            </div>
                            <div class="reasoning-content" style="margin-top: 6px; font-size: 11.5px; color: var(--text-dim); line-height: 1.45; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 6px;">
                                Analyzing heterogeneous pipeline partitioning across 2 laptops over WireGuard... Stage 1 evaluates embeddings + layers 0..23; activations dynamically quantized to INT8 (5.16 KB/tok); Stage 2 evaluates layers 24..47 + LM head logits.
                            </div>
                        </div>

                        <!-- Main Markdown Content -->
                        <div class="markdown-body" style="font-size: 13.5px; line-height: 1.5; color: var(--text-main);">
                            <p style="margin-bottom: 10px;"><strong>AeroMESH</strong> performs distributed edge inference by partitioning transformer layers between two consumer laptops with <strong>0.0 MB of model weights streamed across the network</strong>:</p>

                            <pre style="background: #080b10; border-radius: 8px; padding: 10px 14px; border: 1px solid rgba(255,255,255,0.08); overflow-x: auto; font-family: monospace; font-size: 12px; line-height: 1.4; margin-bottom: 10px;"><code class="language-python"><span style="color:#e06c75;">def</span> <span style="color:#61afef;">forward_pipeline_step</span>(prompt_tokens):
    h_stage1 = coordinator.forward(prompt_tokens, layers=(<span style="color:#d19a66;">0</span>, <span style="color:#d19a66;">24</span>))  <span style="color:#5c6370;"># Laptop A</span>
    q_frame = quantize_int8(h_stage1, scale_per_row=<span style="color:#d19a66;">True</span>)         <span style="color:#5c6370;"># 5.16 KB wire payload</span>
    wireguard_transport.send_frame(worker_endpoint, q_frame)
    <span style="color:#e06c75;">return</span> worker_stream.recv_token()                            <span style="color:#5c6370;"># Laptop B (layers 24..48)</span></code></pre>

                            <p style="margin-bottom: 6px;">Per-step generation latency $\\\\mathcal{L}_{\\\\text{step}}$ is bounded by communication RTT and stage execution time:</p>
                            
                            <div style="background: rgba(229,130,68,0.06); border: 1px solid rgba(229,130,68,0.18); border-radius: 8px; padding: 8px 14px; margin: 8px 0; text-align: center;">
                                ${mathHtml}
                            </div>
                        </div>

                        <!-- Live Metrics HUD Card -->
                        <div class="live-metrics-hud" style="margin-top: 12px; display: flex; align-items: center; justify-content: space-between; background: rgba(0,0,0,0.4); padding: 8px 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.06);">
                            <div style="display: flex; gap: 16px; font-size: 12px; font-family: monospace;">
                                <span style="color: var(--accent-mint); font-weight: 700;">⚡ 18.4 tok/s</span>
                                <span style="color: var(--text-dim);">Duration: 2.14s</span>
                                <span style="color: var(--text-dim);">Tokens: 38</span>
                                <span style="color: var(--accent-orange); font-weight: 600;">5.16 KB/tok • 0.20 MB Wire</span>
                            </div>
                            <span class="badge-stage" style="background: rgba(16, 185, 129, 0.15); color: var(--accent-mint); border: 1px solid rgba(16, 185, 129, 0.3); font-size: 11px; padding: 2px 8px; border-radius: 4px; font-weight: 600;">Direct WireGuard</span>
                        </div>
                    </div>
                </div>
            `;

            // Reset scroll to top
            const vp = document.getElementById('chat-viewport');
            vp.scrollTop = 0;

            // Update top bar to inferring state
            const topDot = document.getElementById('top-status-dot');
            topDot.className = 'status-dot generating';
            document.getElementById('top-status-text').textContent = 'Cluster Inferring (2 Nodes)';

            // Update send button to stop button state
            const sendBtn = document.getElementById('send-btn');
            sendBtn.className = 'send-btn-clay stop-btn-clay';
            sendBtn.title = 'Stop Generating (Esc)';
            sendBtn.innerHTML = '<i data-lucide="square" style="width: 14px; height: 14px; fill: currentColor;"></i>';

            if (window.lucide) lucide.createIcons();
        """)
        time.sleep(1.2)
        path3 = os.path.join(OUTPUT_DIR, "03-streaming-response.png")
        driver.save_screenshot(path3)
        print(f"Saved: {path3}")

        # -------------------------------------------------------------
        # 4. Capture 04-cluster-hud.png (Cluster Topology & Telemetry Modal)
        # -------------------------------------------------------------
        print("Capturing 04-cluster-hud.png...")
        driver.execute_script("""
            const modal = document.getElementById('telemetry-modal');
            modal.classList.add('active');
            const nodesContainer = document.getElementById('modal-nodes-container');
            nodesContainer.innerHTML = `
                <div class="node-box" style="padding: 14px 16px; border-radius: 8px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07); display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <div>
                        <div style="font-weight: 700; color: var(--text-main); font-size: 14px;">Coordinator (Laptop A • Local)</div>
                        <div style="font-size: 12px; color: var(--text-dim); margin-top: 2px;">Layers 0..23 • NVIDIA RTX 4060 (8 GB VRAM) • 100.64.0.10</div>
                    </div>
                    <span class="stat-highlight" style="font-size: 12px; background: rgba(229,130,68,0.15); padding: 4px 10px; border-radius: 4px;">Stage 1</span>
                </div>

                <div class="mesh-link-container" style="display: flex; flex-direction: column; align-items: center; margin: 14px 0; gap: 6px;">
                    <div class="direct-wireguard-badge" style="display: flex; align-items: center; gap: 8px; background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.3); padding: 6px 14px; border-radius: 20px; color: var(--accent-mint); font-size: 12px; font-weight: 600;">
                        <i data-lucide="shield-check" style="width: 15px; height: 15px;"></i>
                        <span>Tailscale Direct WireGuard</span>
                        <span class="badge-ping" style="background: rgba(16, 185, 129, 0.25); padding: 2px 6px; border-radius: 4px; font-size: 11px;">RTT: 1.14 ms</span>
                    </div>
                    <div class="mesh-wire-stats" style="font-size: 11.5px; color: var(--text-dim);">
                        <span>P2P Activation Streaming • 5.16 KB/tok • Zero Model Weights on Wire (0.0 MB)</span>
                    </div>
                </div>

                <div class="node-box" style="padding: 14px 16px; border-radius: 8px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07); display: flex; justify-content: space-between; align-items: center; margin-top: 12px;">
                    <div>
                        <div style="font-weight: 700; color: var(--text-main); font-size: 14px;">Worker Stage 2 (100.64.0.11:50052)</div>
                        <div style="font-size: 12px; color: var(--text-dim); margin-top: 2px;">Layers 24..47 • NVIDIA RTX 3050 (4 GB VRAM) • LM Head • Activation Sink</div>
                    </div>
                    <span class="stat-highlight direct-badge" style="font-size: 12px; background: rgba(16, 185, 129, 0.15); color: var(--accent-mint); padding: 4px 10px; border-radius: 4px;">Direct WireGuard</span>
                </div>
            `;
            const statusDot = document.getElementById('modal-status-dot');
            statusDot.className = 'status-dot';
            document.getElementById('modal-cluster-status').textContent = '2 Nodes Active • Direct WireGuard Mesh Connected • 1.14 ms RTT';
            document.getElementById('modal-cluster-status').style.color = 'var(--accent-mint)';
            if (window.lucide) lucide.createIcons();
        """)
        time.sleep(1.0)
        path4 = os.path.join(OUTPUT_DIR, "04-cluster-hud.png")
        driver.save_screenshot(path4)
        print(f"Saved: {path4}")

        print("\nAll 4 screenshots captured successfully!")

    finally:
        driver.quit()

if __name__ == "__main__":
    main()
