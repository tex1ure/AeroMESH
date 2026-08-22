import json
import os
import sys
import time
import requests
import gradio as gr

# 1. AEROMESH ZERO-WEIGHT PIPELINE CONFIGURATION
AEROMESH_ENDPOINT = os.getenv("AEROMESH_ENDPOINT", "http://127.0.0.1:8080")
CHAT_COMPLETIONS_URL = f"{AEROMESH_ENDPOINT}/v1/chat/completions"
HEALTH_URL = f"{AEROMESH_ENDPOINT}/health"

print("========================================================")
print("   🚀 AEROMESH ZERO-WEIGHT PIPELINE WEB UI              ")
print("========================================================")
print(f"  Target Coordinator API: {AEROMESH_ENDPOINT}")
print(f"  Wire Weight Transfer:   0.0 MB (Zero-Weight Pipeline)")
print("========================================================\n")

# Check if AeroMesh Pipeline Coordinator is online
def check_aeromesh_status():
    try:
        resp = requests.get(HEALTH_URL, timeout=1.5)
        if resp.status_code == 200:
            return True, "✅ AeroMesh Zero-Weight Pipeline Coordinator Connected"
        return False, f"⚠️ Coordinator returned HTTP {resp.status_code}"
    except Exception:
        return False, "❌ AeroMesh Coordinator offline. Start via: cargo run --bin aeromesh -- serve --model models/<model>.gguf --layers 0..24 --peers <peer-ip>:50052"

# 2. STREAMING CHAT LOGIC (Zero-Weight Pipeline over HTTP SSE)
def chat_function(message, history):
    """
    Streams tokens from AeroMesh Zero-Weight Pipeline Coordinator.
    Zero model weights transferred across network; only activations on backend.
    """
    messages = []
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})

    payload = {
        "messages": messages,
        "max_tokens": 128,
        "stream": True,
        "session_id": int(time.time() * 1000),
    }

    try:
        with requests.post(CHAT_COMPLETIONS_URL, json=payload, stream=True, timeout=30.0) as response:
            if response.status_code != 200:
                yield f"❌ Error from AeroMesh Coordinator: HTTP {response.status_code} - {response.text}"
                return

            accumulated_response = ""
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8').strip()
                    if decoded_line.startswith("data: "):
                        data_str = decoded_line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            data_json = json.loads(data_str)
                            delta = data_json.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                accumulated_response += content
                                yield accumulated_response
                        except json.JSONDecodeError:
                            continue

            if not accumulated_response:
                yield "*(No output received from pipeline stage)*"

    except requests.exceptions.ConnectionError:
        yield (
            "❌ Could not connect to AeroMesh Pipeline Coordinator at http://127.0.0.1:8080.\n\n"
            "Please ensure the AeroMesh coordinator is running in another terminal:\n"
            "```powershell\n"
            "cargo run --bin aeromesh -- serve --model models/<model>.gguf --layers 0..24 --peers <worker-ip>:50052 --port 8080\n"
            "```"
        )
    except Exception as e:
        yield f"❌ Pipeline communication error: {str(e)}"

# 3. CREATE GRADIO INTERFACE
is_online, status_msg = check_aeromesh_status()
print(f"Status: {status_msg}")

demo = gr.ChatInterface(
    fn=chat_function,
    title="⚡ AeroMesh Distributed LLM Engine (0.0 MB Wire Transfer)",
    description=(
        "**Zero-Weight P2P Pipeline Execution**: Laptop A (Layers 0..24) ⇄ Laptop B (Layers 25..48). "
        "Activations streamed over Direct Tailscale WireGuard. No model weights transferred."
    ),
    examples=[
        "Explain distributed GPU clustering in one sentence.",
        "How does zero-weight activation streaming work?",
        "What are the benefits of memory-mapped layer slicing?"
    ]
)

# 4. LAUNCH WEB UI
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
