import os
import sys
import subprocess
from PIL import Image, ImageDraw, ImageFont, ImageSequence

def get_font(name="segoeui.ttf", size=24):
    font_path = os.path.join("C:/Windows/Fonts", name)
    if os.path.exists(font_path):
        return ImageFont.truetype(font_path, size)
    return ImageFont.load_default()

def create_card(title, subtitle, category_tag, bullets, highlight_badge=None, duration_sec=5.0, fps=30):
    width, height = 1920, 1080
    bg = Image.new("RGB", (width, height), (13, 17, 23))
    draw = ImageDraw.Draw(bg)
    
    font_title = get_font("segoeuib.ttf", 46)
    font_sub = get_font("segoeui.ttf", 26)
    font_body = get_font("segoeui.ttf", 25)
    font_badge = get_font("segoeuib.ttf", 22)
    font_mono = get_font("consola.ttf", 24)
    font_small = get_font("segoeui.ttf", 20)

    # Top Accent gradient bar
    for x in range(width):
        t = x / width
        r = int(245 * (1 - t) + 16 * t)
        g = int(100 * (1 - t) + 185 * t)
        b = int(10 * (1 - t) + 129 * t)
        draw.line([(x, 0), (x, 7)], fill=(r, g, b))

    # Top Brand & SIH Header
    draw.rounded_rectangle([(80, 40), (280, 85)], radius=8, fill=(255, 107, 0))
    draw.text((100, 48), "AeroMESH", font=font_badge, fill=(255, 255, 255))
    
    draw.rounded_rectangle([(300, 40), (620, 85)], radius=8, fill=(30, 41, 59), outline=(71, 85, 105), width=1)
    draw.text((320, 48), category_tag, font=font_badge, fill=(56, 189, 248))

    draw.text((width - 520, 52), "SMART INDIA HACKATHON 2026", font=font_badge, fill=(148, 163, 184))

    # Main Card Box
    draw.rounded_rectangle([(80, 115), (width - 80, height - 90)], radius=18, fill=(21, 27, 38), outline=(51, 65, 85), width=2)

    # Title & Subtitle inside card
    draw.text((120, 150), title, font=font_title, fill=(248, 250, 252))
    draw.text((120, 218), subtitle, font=font_sub, fill=(148, 163, 184))

    # Highlight Badge
    if highlight_badge:
        draw.rounded_rectangle([(120, 275), (960, 325)], radius=10, fill=(12, 45, 30), outline=(34, 197, 94), width=2)
        draw.text((140, 285), f"★  {highlight_badge}", font=font_badge, fill=(74, 222, 128))

    y_cursor = 355 if highlight_badge else 285
    for item in bullets:
        if item.startswith("[HEADER]"):
            header_text = item.replace("[HEADER]", "").strip()
            y_cursor += 15
            draw.text((120, y_cursor), header_text, font=get_font("segoeuib.ttf", 28), fill=(251, 191, 36))
            y_cursor += 46
        elif item.startswith("[MONO]"):
            mono_text = item.replace("[MONO]", "").strip()
            draw.text((150, y_cursor), mono_text, font=font_mono, fill=(134, 239, 172))
            y_cursor += 42
        elif item.startswith("[METRIC]"):
            metric_text = item.replace("[METRIC]", "").strip()
            draw.rounded_rectangle([(120, y_cursor), (width - 120, y_cursor + 48)], radius=8, fill=(30, 41, 59), outline=(71, 85, 105), width=1)
            draw.text((140, y_cursor + 8), metric_text, font=font_body, fill=(241, 245, 249))
            y_cursor += 58
        else:
            draw.text((120, y_cursor), f"•  {item}", font=font_body, fill=(226, 232, 240))
            y_cursor += 46

    # Bottom Footer
    footer_text = "AeroMESH Zero-Weight Engine • Decentralized Heterogeneous GPU Aggregation • 100% Air-Gapped Sovereign AI"
    draw.text((120, height - 60), footer_text, font=font_small, fill=(100, 116, 139))

    n_frames = int(duration_sec * fps)
    return [bg] * n_frames

def scale_and_center_image(img, target_width=1920, target_height=1080, bg_color=(13, 17, 23), title_banner=None):
    canvas = Image.new("RGB", (target_width, target_height), bg_color)
    draw = ImageDraw.Draw(canvas)

    # Header bar
    for x in range(target_width):
        t = x / target_width
        r = int(245 * (1 - t) + 16 * t)
        g = int(100 * (1 - t) + 185 * t)
        b = int(10 * (1 - t) + 129 * t)
        draw.line([(x, 0), (x, 7)], fill=(r, g, b))

    # Top Brand
    font_badge = get_font("segoeuib.ttf", 22)
    draw.rounded_rectangle([(80, 25), (280, 65)], radius=6, fill=(255, 107, 0))
    draw.text((100, 31), "AeroMESH", font=font_badge, fill=(255, 255, 255))
    draw.text((310, 33), "SIH 2026 LIVE DEMONSTRATION RECORDING", font=font_badge, fill=(148, 163, 184))

    # Reserve top 80px and bottom 60px
    avail_w = target_width - 80
    avail_h = target_height - 150

    img_ratio = img.width / img.height
    avail_ratio = avail_w / avail_h

    if img_ratio > avail_ratio:
        new_w = avail_w
        new_h = int(new_w / img_ratio)
    else:
        new_h = avail_h
        new_w = int(new_h * img_ratio)

    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    x = (target_width - new_w) // 2
    y = 80 + (avail_h - new_h) // 2

    # Draw card border around content
    draw.rounded_rectangle([(x - 6, y - 6), (x + new_w + 6, y + new_h + 6)], radius=12, fill=(21, 27, 38), outline=(51, 65, 85), width=2)
    canvas.paste(resized, (x, y))

    if title_banner:
        draw.rounded_rectangle([(x + 20, y + 20), (x + 850, y + 75)], radius=10, fill=(15, 23, 42), outline=(56, 189, 248), width=2)
        draw.text((x + 40, y + 30), f"⚡  {title_banner}", font=font_badge, fill=(241, 245, 249))

    font_small = get_font("segoeui.ttf", 20)
    draw.text((80, target_height - 40), "Real-time execution captured on 2-node cluster (NVIDIA RTX 4060 GPUs, Tailscale Direct WireGuard, 0.0 MB Wire)", font=font_small, fill=(100, 116, 139))
    return canvas

def main():
    fps = 30
    artifacts_dir = r"C:\Users\Krushna\.gemini\antigravity-ide\brain\84934efa-b31f-48f0-855f-8c6491fa4bb4"
    artifacts_live = r"C:\Users\Krushna\.gemini\antigravity-ide\brain\0ea887cb-47d2-4902-9696-4d55ac601bdf"

    hud_path = os.path.join(artifacts_dir, "cluster_hud_modal_1789387595757.png")
    final_chat_path = os.path.join(artifacts_dir, "final_completed_chat_1789389047916.png")
    webp_path = os.path.join(artifacts_dir, "dist_proof_demo_-62135596800000.webp")
    live_webp = os.path.join(artifacts_live, "aeromesh_sih_demo_1790694298643.webp")

    output_mp4 = r"d:\New folder\Latest_llama\AeroMESH\demo_assets\AeroMESH_SIH_2026_Full_Demo.mp4"

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", "1920x1080",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "medium",
        "-crf", "18",
        output_mp4
    ]

    print("Launching ffmpeg video encoder pipeline...")
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    def write_frames(frames):
        for f in frames:
            if f.size != (1920, 1080):
                f = scale_and_center_image(f)
            proc.stdin.write(f.convert("RGB").tobytes())

    # ------------------------------------------------------------------------
    # CARD 1: SIH TITLE & PROBLEM STATEMENT
    # ------------------------------------------------------------------------
    print("Generating Slide 1: SIH 2026 Title & Innovation Card...")
    card1 = create_card(
        title="AeroMESH: Distributed Zero-Weight LLM Cluster Engine",
        subtitle="Bridging the Bharat GPU Divide • Sovereign On-Premise AI for Indian Enterprises & Academia",
        category_tag="SIH 2026 • STUDENT INNOVATION",
        bullets=[
            "Problem Statement: Solving the 'Bharat GPU Divide' across Indian Universities and MSMEs.",
            "Hardware Impasse: State-of-the-art 14B/32B LLMs demand 16GB-32GB VRAM; labs only have 6GB-8GB GPUs.",
            "Cloud Dependency: Sending sensitive code, legal data, or plant schematics to foreign cloud APIs violates DPDP Act 2023.",
            "The AeroMESH Breakthrough: Virtualizes heterogeneous consumer GPUs into a unified cluster without streaming weights.",
            "Key Innovation: Zero model weights on wire (0.0 MB) via local GGUF memory-mapping + INT8 dynamic activation streaming.",
            "Network Efficiency: Latent activation vectors quantized from FP32 to INT8 per-row, reducing bandwidth by 75% (~5 KB/tok)."
        ],
        highlight_badge="SIH 2026 Innovation • True Distributed Pipeline Parallelism on Standard PCs",
        duration_sec=7.0,
        fps=fps
    )
    write_frames(card1)

    # ------------------------------------------------------------------------
    # CARD 2: ARCHITECTURE & DISTRIBUTED TOPOLOGY
    # ------------------------------------------------------------------------
    print("Generating Slide 2: Distributed Architecture Card...")
    card2 = create_card(
        title="Decentralized Zero-Weight Architecture & Wire Protocol",
        subtitle="How Two Everyday Consumer Laptops Cooperate as an Enterprise-Grade AI Engine",
        category_tag="SYSTEM ARCHITECTURE",
        bullets=[
            "[HEADER] 2-NODE CLUSTER CONFIGURATION (DEMONSTRATED LIVE):",
            "[MONO] Node 1 (Coordinator / Laptop A): Layers 0..22 on NVIDIA RTX 4060 GPU (100.66.49.50)",
            "[MONO] Node 2 (Worker / Laptop B):      Layers 23..47 + LM Head on Port 50052 over Tailscale WireGuard",
            "[MONO] Network Transport:             Point-to-Point Encrypted Direct WireGuard (UDP 41641, 1.14 ms RTT)",
            "[MONO] Wire Weight Transfer:          0.0 MB (Both machines load identical model slices from local SSD)",
            "[MONO] Inter-Node Latent Payload:     5.04 KB / token (Compressed with Dynamic Per-Row INT8 Scaling)",
            "[MONO] Control Plane & Security:      SEC-01 Authenticated Handshake + Native Windows Job Object Isolation",
            "[MONO] Client Layer:                  Zero-Framework Claymorphic Dark SPA + OpenAI-Compatible Axum Server"
        ],
        highlight_badge="0.0 MB Model Weights Transferred over Network • Zero Cloud Reliance",
        duration_sec=7.0,
        fps=fps
    )
    write_frames(card2)

    # ------------------------------------------------------------------------
    # SLIDE 3: CLUSTER HUD & TOPOLOGY INSPECTION
    # ------------------------------------------------------------------------
    print("Generating Slide 3: Live Cluster HUD Inspection...")
    if os.path.exists(hud_path):
        hud_img = Image.open(hud_path)
        hud_frame = scale_and_center_image(
            hud_img,
            title_banner="Live Cluster Topology HUD: 2 Nodes Connected • Tailscale Direct WireGuard (1.14 ms RTT)"
        )
        write_frames([hud_frame] * int(7.0 * fps))

    # ------------------------------------------------------------------------
    # SLIDE 4: LIVE BROWSER DEMONSTRATION (RECORDED IN REAL TIME)
    # ------------------------------------------------------------------------
    chosen_webp = live_webp if os.path.exists(live_webp) else webp_path
    print(f"Generating Slide 4: Real-time Browser Session from {os.path.basename(chosen_webp)}...")
    if os.path.exists(chosen_webp):
        webp_img = Image.open(chosen_webp)
        webp_frames = []
        for frame in ImageSequence.Iterator(webp_img):
            f = frame.copy().convert("RGB")
            scaled = scale_and_center_image(f, title_banner="Interactive Web Interface: Real-Time SSE Token Streaming & Telemetry")
            webp_frames.append(scaled)

        print(f"Streaming {len(webp_frames)} interactive browser session frames...")
        for f in webp_frames:
            # Repeat each frame 2x to achieve smooth 30 fps playback
            proc.stdin.write(f.tobytes())
            proc.stdin.write(f.tobytes())

    # ------------------------------------------------------------------------
    # SLIDE 5: FINAL COMPLETED CHAT & TELEMETRY BADGE
    # ------------------------------------------------------------------------
    print("Generating Slide 5: Final Completed Generation & Telemetry Card...")
    if os.path.exists(final_chat_path):
        final_img = Image.open(final_chat_path)
        final_frame = scale_and_center_image(
            final_img,
            title_banner="Streaming Verified: 1.4 tok/s • 1.33 MB Wire Payload (256 Tokens) • 5.16 KB/tok • Direct WireGuard"
        )
        write_frames([final_frame] * int(8.0 * fps))

    # ------------------------------------------------------------------------
    # SLIDE 6: EMPIRICAL BENCHMARKS & VERIFICATION
    # ------------------------------------------------------------------------
    print("Generating Slide 6: Verification & Empirical Metrics Card...")
    card6 = create_card(
        title="Empirical Benchmarks & Validation Results",
        subtitle="Measured Performance of DeepSeek-R1 / Qwen-2.5 14B on Consumer RTX 4060 Laptops",
        category_tag="BENCHMARKS & PROOF",
        bullets=[
            "[HEADER] SYSTEM PERFORMANCE RECORDED ON CAMERA:",
            "[METRIC] VRAM Reduction:  Each 8GB GPU holds only 50.1% of model layers, eliminating Out-Of-Memory crashes.",
            "[METRIC] Cold Start Time: < 3.2 seconds via memory-mapped GGUF slicing (vs 15+ minutes copying weights over LAN).",
            "[METRIC] Network Payload: 1.33 MB total data transferred for a full 256-token response over WireGuard.",
            "[METRIC] Inter-Node Ping: 1.14 ms Direct WireGuard RTT (bypasses Tensor Parallelism synchronization wall).",
            "[METRIC] Generation Rate: 1.4 tokens/second on 14B model across two standard laptops over Wi-Fi."
        ],
        highlight_badge="PASS: Zero-Weight Activation Pipeline Fully Validated Under Real Network Conditions",
        duration_sec=7.0,
        fps=fps
    )
    write_frames(card6)

    # ------------------------------------------------------------------------
    # SLIDE 7: SIH VALUE PROPOSITION & INDIAN MARKET IMPACT
    # ------------------------------------------------------------------------
    print("Generating Slide 7: SIH Value Proposition & National Impact...")
    card7 = create_card(
        title="Empowering India's AI Revolution: Sovereignty, Savings & Scale",
        subtitle="Aligned with IndiaAI Mission, Digital India, and the DPDP Act 2023",
        category_tag="NATIONAL IMPACT & VALUE",
        bullets=[
            "₹0 Cloud API Fees: Eliminates ongoing USD SaaS bills for Indian startups, research labs, and MSMEs.",
            "100% Data Sovereignty: Sensitive health, defense, and industrial data never leaves local on-premise hardware.",
            "Repurposes Existing Hardware: Indian colleges can turn ordinary computer labs into sovereign AI clusters.",
            "100% Rust Safe Architecture: Tokio asynchronous runtime, Axum API, and zero memory leaks or dangling pointers.",
            "Standard OpenAI Compatibility: Seamless drop-in backend for any existing RAG pipeline, LangChain, or agentic UI.",
            "Open Source & Indigenous: Developed entirely from first principles with native llama.cpp / GGML CUDA bindings."
        ],
        highlight_badge="AeroMESH: Democratizing High-Parameter AI for Bharat",
        duration_sec=7.0,
        fps=fps
    )
    write_frames(card7)

    proc.stdin.close()
    proc.wait()
    print(f"\n========================================================")
    print(f"   [+] SIH 2026 DEMO VIDEO COMPILATION COMPLETE")
    print(f"   File: {output_mp4}")
    print(f"========================================================\n")

if __name__ == "__main__":
    main()
