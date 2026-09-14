import os
import sys
import subprocess
from PIL import Image, ImageDraw, ImageFont, ImageSequence

def create_card(title, subtitle, bullets, highlight_badge=None, duration_sec=5.0, fps=30):
    width, height = 1920, 1080
    bg = Image.new("RGB", (width, height), (15, 17, 23))
    draw = ImageDraw.Draw(bg)
    
    font_title = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 48)
    font_sub = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 28)
    font_body = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 26)
    font_badge = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 22)
    font_mono = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 24)

    # Gradient header bar
    for x in range(width):
        t = x / width
        r = int(255 * (1 - t) + 16 * t)
        g = int(107 * (1 - t) + 185 * t)
        b = int(0 * (1 - t) + 129 * t)
        draw.line([(x, 0), (x, 8)], fill=(r, g, b))

    # Top Brand
    draw.text((80, 50), "AeroMESH", font=font_title, fill=(255, 107, 0))
    draw.text((340, 68), "• DISTRIBUTED PIPELINE INFERENCE DEMO", font=font_badge, fill=(160, 170, 190))

    # Title & Subtitle
    draw.text((80, 140), title, font=font_title, fill=(240, 245, 255))
    draw.text((80, 210), subtitle, font=font_sub, fill=(120, 190, 255))

    # Card background box
    draw.rounded_rectangle([(80, 280), (width - 80, height - 120)], radius=18, fill=(22, 26, 36), outline=(45, 55, 75), width=2)

    # Highlight Badge
    if highlight_badge:
        draw.rounded_rectangle([(120, 315), (760, 365)], radius=10, fill=(10, 45, 30), outline=(16, 185, 129), width=2)
        draw.text((140, 325), f"🛡️  {highlight_badge}", font=font_badge, fill=(52, 211, 153))

    y_cursor = 390 if highlight_badge else 320
    for bullet in bullets:
        if bullet.startswith("[MONO]"):
            text = bullet.replace("[MONO]", "").strip()
            draw.text((140, y_cursor), text, font=font_mono, fill=(130, 220, 160))
            y_cursor += 42
        elif bullet.startswith("[HEADER]"):
            text = bullet.replace("[HEADER]", "").strip()
            y_cursor += 15
            draw.text((120, y_cursor), text, font=font_title, fill=(255, 180, 50))
            y_cursor += 50
        else:
            draw.text((120, y_cursor), f"•  {bullet}", font=font_body, fill=(220, 225, 235))
            y_cursor += 48

    # Footer
    draw.text((80, height - 80), "AeroMESH Zero-Weight Pipeline • Consumer Hardware Mesh Parallelism", font=font_body, fill=(100, 115, 135))
    
    n_frames = int(duration_sec * fps)
    return [bg] * n_frames

def scale_and_center_image(img, target_width=1920, target_height=1080, bg_color=(15, 17, 23)):
    canvas = Image.new("RGB", (target_width, target_height), bg_color)
    img_ratio = img.width / img.height
    target_ratio = target_width / target_height

    if img_ratio > target_ratio:
        new_w = target_width
        new_h = int(new_w / img_ratio)
    else:
        new_h = target_height
        new_w = int(new_h * img_ratio)

    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    x = (target_width - new_w) // 2
    y = (target_height - new_h) // 2
    canvas.paste(resized, (x, y))
    return canvas

def main():
    fps = 30
    artifacts_dir = r"C:\Users\Krushna\.gemini\antigravity-ide\brain\84934efa-b31f-48f0-855f-8c6491fa4bb4"
    hud_path = os.path.join(artifacts_dir, "cluster_hud_modal_1789387595757.png")
    final_chat_path = os.path.join(artifacts_dir, "final_completed_chat_1789389047916.png")
    webp_path = os.path.join(artifacts_dir, "dist_proof_demo_-62135596800000.webp")
    output_mp4 = r"d:\New folder\Latest_llama\AeroMESH\demo_assets\demo_distributed_2node_tailscale_direct.mp4"

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

    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    def write_frames(frames):
        for f in frames:
            if f.size != (1920, 1080):
                f = scale_and_center_image(f)
            proc.stdin.write(f.convert("RGB").tobytes())

    print("Generating Card 1: Title...")
    card1 = create_card(
        title="2-Node Pipeline Parallelism over Tailscale Direct WireGuard",
        subtitle="Zero Model Weights on Wire • INT8 Per-Row Quantization • True Mesh Execution",
        bullets=[
            "Demonstrating pipeline parallel split of DeepSeek-R1-Distill-14B (8.37 GB, 48 Layers)",
            "Coordinator Node (Local Stage 1): Layers 0..22 on NVIDIA RTX 4060 GPU",
            "Worker Node (Peer Stage 2): Layers 23..47 on 100.66.49.50:50052 with CUDA offload",
            "Transport: Point-to-Point Tailscale Direct WireGuard (UDP 41641, RTT 1.14 ms, Non-DERP)",
            "Activation Compression: Dynamic INT8 Per-Row (5.04 KB/tok vs 20.0 KB FP32 - 75% savings)",
            "Zero Model Weights Transferred: Both nodes memory-map weights locally via GGUF slicer"
        ],
        highlight_badge="Tailscale Direct WireGuard Active • Peer RTT: 1.14 ms",
        duration_sec=6.0,
        fps=fps
    )
    write_frames(card1)

    print("Generating Card 2: Cluster HUD Inspection...")
    hud_img = Image.open(hud_path)
    hud_frame = scale_and_center_image(hud_img)
    # Overlay an explanation banner on top of HUD
    draw = ImageDraw.Draw(hud_frame)
    font_badge = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 24)
    draw.rounded_rectangle([(80, 30), (880, 85)], radius=12, fill=(15, 23, 42, 220), outline=(16, 185, 129), width=2)
    draw.text((105, 42), "✓ 2 Nodes Connected • Tailscale Direct WireGuard (1.14 ms RTT)", font=font_badge, fill=(52, 211, 153))
    write_frames([hud_frame] * int(8.0 * fps))

    print("Generating Card 3: Browser Session Video from Recorded WebP...")
    webp_img = Image.open(webp_path)
    webp_frames = []
    for frame in ImageSequence.Iterator(webp_img):
        f = frame.copy().convert("RGB")
        webp_frames.append(scale_and_center_image(f))

    # Play webp frames at 15 fps repeated to match 30 fps (smooth playback)
    print(f"Adding {len(webp_frames)} webp frames...")
    for f in webp_frames:
        proc.stdin.write(f.tobytes())
        proc.stdin.write(f.tobytes())

    print("Generating Card 4: Final Completed Generation & Telemetry Card...")
    final_img = Image.open(final_chat_path)
    final_frame = scale_and_center_image(final_img)
    draw_final = ImageDraw.Draw(final_frame)
    # Highlight badge box on telemetry
    draw_final.rounded_rectangle([(80, 30), (960, 95)], radius=12, fill=(15, 23, 42), outline=(255, 107, 0), width=2)
    draw_final.text((105, 48), "⚡ 1.4 tok/s  |  🌐 1.33 MB Wire  |  📦 5.16 KB/tok  |  🛡️ Direct WireGuard", font=font_badge, fill=(255, 180, 50))
    write_frames([final_frame] * int(10.0 * fps))

    print("Generating Card 5: Verdict & Benchmarks Summary...")
    card5 = create_card(
        title="DIST-01 Verification Passed • AeroMESH Distributed Proof",
        subtitle="Benchmark Metrics Recorded from Live 2-Node Tailscale Cluster Run",
        bullets=[
            "[HEADER] MEASURED METRICS OVER TAILSCALE DIRECT WIREGUARD:",
            "[MONO] Model Topology:       DeepSeek-R1-Distill-14B / Qwen-2.5-14B (8.37 GB, 48 Layers)",
            "[MONO] Pipeline Partition:   Coordinator: L0..22 (23 layers) | Worker: L23..47 (25 layers)",
            "[MONO] Network Transport:    Tailscale Direct WireGuard (UDP 41641, Peer Ping: 1.14 ms)",
            "[MONO] Weights on Wire:      0.0 MB (Zero-Weight Architecture, local mmap execution)",
            "[MONO] Per-Token Payload:    5.04 KB - 5.16 KB / token (INT8 Per-Row Quantized)",
            "[MONO] Wire Compression:     75% reduction vs raw FP32 (20.0 KB -> 5.04 KB)",
            "[MONO] Total Wire Payload:   1.33 MB transferred over wire for 256 tokens",
            "[MONO] Prefill Latency:      968 ms (52 prompt tokens over WireGuard)",
            "[MONO] Generation Speed:     1.4 tok/s continuous streaming inference across mesh"
        ],
        highlight_badge="PASS: DIST-01 Verified on Camera & In Benchmarks",
        duration_sec=8.0,
        fps=fps
    )
    write_frames(card5)

    proc.stdin.close()
    proc.wait()
    print(f"Video created successfully at: {output_mp4}")

if __name__ == "__main__":
    main()
