"""
AeroMESH Hardware-Aware VRAM Fit Meter & Auto Layer-Split Planner.

Pure Python stdlib module that:
  1. Detects local GPU VRAM (via nvidia-smi) and System RAM (Windows/Linux/macOS)
     and parses remote peer GPU/RAM telemetry.
  2. Reads binary GGUF v2/v3 KV metadata headers in <2ms to extract exact
     layer count, hidden dim, attention/KV head counts, and context length.
  3. Estimates exact per-layer weight size, embeddings, LM head, KV-cache
     footprint (GQA-aware), and CUDA runtime scratch for both local GGUF files
     and pre-download Hugging Face / catalog models.
  4. Computes proportional heterogeneous multi-node layer splits across any
     combination of GPUs + System RAM (e.g., 8 GB RTX 4060 + 6 GB RTX 3060 +
     16 GB RAM) and classifies fit status (`fits_vram`, `tight_vram`,
     `ram_offload`, `wont_fit`) with actionable quantization/context advice.
"""

import ctypes
import math
import os
import re
import struct
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

# GGUF KV value types (ggml / llama.cpp specification)
_GGUF_TYPE_UINT8 = 0
_GGUF_TYPE_INT8 = 1
_GGUF_TYPE_UINT16 = 2
_GGUF_TYPE_INT16 = 3
_GGUF_TYPE_UINT32 = 4
_GGUF_TYPE_INT32 = 5
_GGUF_TYPE_FLOAT32 = 6
_GGUF_TYPE_BOOL = 7
_GGUF_TYPE_STRING = 8
_GGUF_TYPE_ARRAY = 9
_GGUF_TYPE_UINT64 = 10
_GGUF_TYPE_INT64 = 11
_GGUF_TYPE_FLOAT64 = 12

_GGUF_SCALAR_SIZES = {
    _GGUF_TYPE_UINT8: 1,
    _GGUF_TYPE_INT8: 1,
    _GGUF_TYPE_UINT16: 2,
    _GGUF_TYPE_INT16: 2,
    _GGUF_TYPE_UINT32: 4,
    _GGUF_TYPE_INT32: 4,
    _GGUF_TYPE_FLOAT32: 4,
    _GGUF_TYPE_BOOL: 1,
    _GGUF_TYPE_UINT64: 8,
    _GGUF_TYPE_INT64: 8,
    _GGUF_TYPE_FLOAT64: 8,
}

# Average bits-per-weight (including quantization block scales & norms)
QUANT_BPW = {
    "IQ1_S": 1.65,
    "IQ1_M": 1.85,
    "IQ2_XXS": 2.15,
    "IQ2_XS": 2.40,
    "IQ2_S": 2.60,
    "Q2_K": 3.05,
    "Q2_K_S": 2.85,
    "IQ3_XXS": 3.15,
    "IQ3_XS": 3.40,
    "IQ3_S": 3.55,
    "Q3_K_S": 3.55,
    "Q3_K_M": 3.90,
    "Q3_K_L": 4.25,
    "IQ4_XS": 4.35,
    "IQ4_NL": 4.60,
    "Q4_0": 4.55,
    "Q4_K_S": 4.60,
    "Q4_K_M": 4.85,
    "Q5_0": 5.55,
    "Q5_K_S": 5.55,
    "Q5_K_M": 5.75,
    "Q6_K": 6.60,
    "Q8_0": 8.55,
    "FP8": 8.50,
    "F16": 16.0,
    "BF16": 16.0,
    "FP16": 16.0,
    "F32": 32.0,
}

# Representative modern transformer architecture profiles (GQA-aware)
# (params_b, layers, hidden_dim, n_heads, n_kv_heads, head_dim, vocab_size)
_ARCH_PROFILES = [
    (0.5, 24, 896, 14, 2, 64, 151936),
    (1.5, 28, 1536, 12, 2, 128, 151936),
    (3.0, 28, 3072, 24, 8, 128, 128256),
    (7.0, 28, 3584, 28, 4, 128, 152064),
    (8.0, 32, 4096, 32, 8, 128, 128256),
    (9.0, 42, 3584, 16, 8, 256, 256000),
    (12.0, 40, 5120, 32, 8, 128, 131072),
    (14.0, 48, 5120, 40, 8, 128, 152064),
    (22.0, 56, 6144, 48, 8, 128, 131072),
    (24.0, 40, 5120, 32, 8, 128, 131072),
    (27.0, 46, 4608, 32, 16, 128, 256000),
    (32.0, 64, 5120, 40, 8, 128, 152064),
    (70.0, 80, 8192, 64, 8, 128, 128256),
    (72.0, 80, 8192, 64, 8, 128, 152064),
]

_GGUF_META_CACHE: Dict[str, Tuple[int, int, dict]] = {}
_CACHED_LOCAL_HW: Optional[dict] = None


def mb_to_gb_label(mb: float) -> str:
    gb = max(0.0, float(mb or 0.0)) / 1024.0
    if gb >= 10.0:
        return f"{gb:.1f} GB"
    return f"{gb:.2f} GB"


def extract_quant_from_name(name: str) -> str:
    m = re.search(
        r"(IQ[1-4]_[A-Z0-9]+|Q[2-8]_[A-Z0-9_]+|Q[458]_0|F16|BF16|F32|FP16|FP8)",
        str(name or ""),
        re.IGNORECASE,
    )
    return m.group(1).upper() if m else "Q4_K_M"


def extract_params_b_from_name(name: str, fallback_size_bytes: int = 0, quant: str = "Q4_K_M") -> float:
    text = str(name or "")
    # Match patterns like 1.5B, 7B, 14B, 32B, 70B, 8x7B
    moe = re.search(r"(\d+)\s*[xX]\s*(\d+(?:\.\d+)?)\s*[bB](?:\b|[^a-zA-Z0-9])", text)
    if moe:
        return round(float(moe.group(1)) * float(moe.group(2)) * 0.85, 2)
    m = re.search(r"(?:^|[^0-9.])(\d+(?:\.\d+)?)\s*[bB](?:\b|[^a-zA-Z0-9])", text)
    if m:
        val = float(m.group(1))
        if 0.1 <= val <= 1000.0:
            return val
    # Infer from file size and quant BPW if filename doesn't state parameter count
    if fallback_size_bytes and fallback_size_bytes > 50 * 1024 * 1024:
        bpw = QUANT_BPW.get(quant.upper(), 4.85)
        est_b = (fallback_size_bytes * 8.0) / (bpw * 1e9)
        return round(max(0.5, est_b), 2)
    return 7.0


def parse_size_hint_bytes(size_hint: str) -> int:
    text = str(size_hint or "").strip()
    m = re.search(r"(\d+(?:\.\d+)?)\s*(TB|GB|GiB|MB|MiB)", text, re.IGNORECASE)
    if not m:
        return 0
    val = float(m.group(1))
    unit = m.group(2).upper()
    if unit == "TB":
        return int(val * (1024 ** 4))
    if unit in ("GB", "GIB"):
        return int(val * (1024 ** 3))
    return int(val * (1024 ** 2))


def parse_gpu_vram_mb(gpu_summary: str) -> Tuple[str, int, int]:
    """
    Parse a GPU summary string such as:
      "NVIDIA GeForce RTX 3060 Laptop GPU, 6144 MiB, 512 MiB"
      "NVIDIA GeForce RTX 4060, 8188 MiB"
      "RTX 4090 (24 GB)"
    Returns (gpu_name, total_vram_mb, used_vram_mb).
    """
    raw = str(gpu_summary or "").strip()
    if not raw or "cpu / integrated" in raw.lower() or raw.lower() == "cpu":
        return (raw or "CPU / Integrated GPU", 0, 0)

    parts = [p.strip() for p in raw.split(",") if p.strip()]
    name = parts[0] if parts else raw
    total_mb = 0
    used_mb = 0

    def _parse_mem_token(tok: str) -> int:
        m = re.search(r"(\d+(?:\.\d+)?)\s*(GiB|GB|MiB|MB)", tok, re.IGNORECASE)
        if not m:
            return 0
        v = float(m.group(1))
        u = m.group(2).upper()
        if u in ("GB", "GIB"):
            return int(round(v * 1024))
        return int(round(v))

    if len(parts) >= 2:
        total_mb = _parse_mem_token(parts[1])
    if len(parts) >= 3:
        used_mb = _parse_mem_token(parts[2])
    if total_mb == 0:
        total_mb = _parse_mem_token(raw)

    # Heuristic fallback if GPU model is named without explicit MiB
    if total_mb == 0:
        low = raw.lower()
        for pat, mb in (
            ("rtx 5090", 32768),
            ("rtx 4090", 24576),
            ("rtx 3090", 24576),
            ("rtx 4080", 16384),
            ("rtx 4070 ti super", 16384),
            ("rtx 4070", 12288),
            ("rtx 3080 ti", 12288),
            ("rtx 3080", 10240),
            ("rtx 4060 ti", 8192),
            ("rtx 4060", 8192),
            ("rtx 3070", 8192),
            ("rtx 3060 laptop", 6144),
            ("rtx 3060", 12288),
            ("rtx 3050", 4096),
            ("rtx 2060", 6144),
            ("gtx 1660", 6144),
        ):
            if pat in low:
                total_mb = mb
                break

    return (name, max(0, total_mb), max(0, min(used_mb, total_mb)))


def detect_system_ram_mb() -> Tuple[int, int]:
    """Return (total_ram_mb, available_ram_mb) using OS-native stdlib APIs."""
    # Windows
    if sys.platform == "win32":
        try:
            class _MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = _MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                total_mb = int(stat.ullTotalPhys // (1024 * 1024))
                avail_mb = int(stat.ullAvailPhys // (1024 * 1024))
                return (total_mb, avail_mb)
        except Exception:
            pass

    # Linux /proc/meminfo
    if os.path.isfile("/proc/meminfo"):
        try:
            total_kb = 0
            avail_kb = 0
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        total_kb = int(line.split()[1])
                    elif line.startswith("MemAvailable:"):
                        avail_kb = int(line.split()[1])
            if total_kb > 0:
                if avail_kb <= 0:
                    avail_kb = int(total_kb * 0.65)
                return (total_kb // 1024, avail_kb // 1024)
        except Exception:
            pass

    # macOS sysctl
    if sys.platform == "darwin":
        try:
            out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=1.5)
            if out.returncode == 0 and out.stdout.strip().isdigit():
                total_mb = int(out.stdout.strip()) // (1024 * 1024)
                return (total_mb, int(total_mb * 0.65))
        except Exception:
            pass

    return (16384, 10240)


def detect_local_hardware(force_refresh: bool = False) -> dict:
    """Detect local GPU VRAM + System RAM (cached lightly unless force_refresh=True)."""
    global _CACHED_LOCAL_HW
    if _CACHED_LOCAL_HW is not None and not force_refresh:
        return dict(_CACHED_LOCAL_HW)

    gpu_name = "CPU / Integrated GPU"
    vram_total_mb = 0
    vram_free_mb = 0
    vram_used_mb = 0
    gpu_count = 0

    candidates = [
        "nvidia-smi",
        r"C:\Windows\System32\nvidia-smi.exe",
        r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
    ]
    creationflags = 0x08000000 if sys.platform == "win32" else 0
    for cmd in candidates:
        try:
            res = subprocess.run(
                [cmd, "--query-gpu=name,memory.total,memory.free,memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=2.0,
                creationflags=creationflags,
            )
            if res.returncode == 0 and res.stdout.strip():
                lines = [ln.strip() for ln in res.stdout.strip().splitlines() if ln.strip()]
                if lines:
                    gpu_count = len(lines)
                    parts = [p.strip() for p in lines[0].split(",")]
                    gpu_name = parts[0]
                    if len(parts) >= 2:
                        vram_total_mb = sum(
                            int(float(ln.split(",")[1].strip())) for ln in lines if len(ln.split(",")) >= 2
                        )
                    if len(parts) >= 3:
                        vram_free_mb = sum(
                            int(float(ln.split(",")[2].strip())) for ln in lines if len(ln.split(",")) >= 3
                        )
                    if len(parts) >= 4:
                        vram_used_mb = sum(
                            int(float(ln.split(",")[3].strip())) for ln in lines if len(ln.split(",")) >= 4
                        )
                    break
        except Exception:
            continue

    ram_total_mb, ram_avail_mb = detect_system_ram_mb()
    if vram_total_mb > 0 and vram_free_mb <= 0:
        vram_free_mb = max(0, vram_total_mb - vram_used_mb)

    hw = {
        "gpu_name": gpu_name,
        "gpu_count": gpu_count,
        "vram_total_mb": vram_total_mb,
        "vram_free_mb": vram_free_mb,
        "vram_used_mb": vram_used_mb,
        "vram_total_label": mb_to_gb_label(vram_total_mb) if vram_total_mb > 0 else "0 GB (CPU)",
        "ram_total_mb": ram_total_mb,
        "ram_available_mb": ram_avail_mb,
        "ram_total_label": mb_to_gb_label(ram_total_mb),
    }
    _CACHED_LOCAL_HW = dict(hw)
    return hw


# ============================================================================
# FAST BINARY GGUF v2/v3 KV METADATA PARSER
# ============================================================================
def _read_gguf_string(f) -> str:
    raw_len = f.read(8)
    if len(raw_len) < 8:
        raise EOFError("Truncated GGUF string length")
    str_len = struct.unpack("<Q", raw_len)[0]
    if str_len > 16 * 1024 * 1024:
        raise ValueError("GGUF string length exceeds safety bound")
    if str_len > 2048:
        # Skip huge strings (e.g. chat templates) without allocating
        f.seek(str_len, os.SEEK_CUR)
        return ""
    data = f.read(str_len)
    if len(data) < str_len:
        raise EOFError("Truncated GGUF string data")
    return data.decode("utf-8", errors="replace")


def _skip_or_read_gguf_value(f, vtype: int, want_value: bool):
    if vtype == _GGUF_TYPE_STRING:
        if want_value:
            return _read_gguf_string(f)
        raw_len = f.read(8)
        if len(raw_len) < 8:
            raise EOFError("Truncated GGUF string")
        str_len = struct.unpack("<Q", raw_len)[0]
        if str_len > 16 * 1024 * 1024:
            raise ValueError("GGUF string length exceeds safety bound")
        f.seek(str_len, os.SEEK_CUR)
        return None

    if vtype == _GGUF_TYPE_ARRAY:
        hdr = f.read(12)
        if len(hdr) < 12:
            raise EOFError("Truncated GGUF array header")
        elem_type, count = struct.unpack("<IQ", hdr)
        if count > 2_000_000:
            raise ValueError("GGUF array count exceeds safety bound")
        if elem_type in _GGUF_SCALAR_SIZES:
            f.seek(_GGUF_SCALAR_SIZES[elem_type] * count, os.SEEK_CUR)
        elif elem_type == _GGUF_TYPE_STRING:
            for _ in range(count):
                raw_len = f.read(8)
                if len(raw_len) < 8:
                    raise EOFError("Truncated GGUF string array")
                str_len = struct.unpack("<Q", raw_len)[0]
                if str_len > 16 * 1024 * 1024:
                    raise ValueError("GGUF array string length exceeds safety bound")
                f.seek(str_len, os.SEEK_CUR)
        else:
            for _ in range(count):
                _skip_or_read_gguf_value(f, elem_type, False)
        return count if want_value else None

    size = _GGUF_SCALAR_SIZES.get(vtype)
    if size is None:
        raise ValueError(f"Unknown GGUF KV type {vtype}")
    if not want_value:
        f.seek(size, os.SEEK_CUR)
        return None
    raw = f.read(size)
    if len(raw) < size:
        raise EOFError("Truncated GGUF scalar")
    fmt_map = {
        _GGUF_TYPE_UINT8: "<B",
        _GGUF_TYPE_INT8: "<b",
        _GGUF_TYPE_UINT16: "<H",
        _GGUF_TYPE_INT16: "<h",
        _GGUF_TYPE_UINT32: "<I",
        _GGUF_TYPE_INT32: "<i",
        _GGUF_TYPE_FLOAT32: "<f",
        _GGUF_TYPE_BOOL: "<?",
        _GGUF_TYPE_UINT64: "<Q",
        _GGUF_TYPE_INT64: "<q",
        _GGUF_TYPE_FLOAT64: "<d",
    }
    return struct.unpack(fmt_map[vtype], raw)[0]


def read_gguf_architecture(file_path: str) -> Optional[dict]:
    """
    Parse GGUF v2/v3 KV metadata header to extract exact layer count, hidden dim,
    attention/KV head counts, vocab size, and max context length.
    Returns None if file is not a valid GGUF v2/v3 file.
    """
    try:
        real_path = os.path.realpath(file_path)
        st = os.stat(real_path)
        cached = _GGUF_META_CACHE.get(real_path)
        if cached and cached[0] == st.st_mtime_ns and cached[1] == st.st_size:
            return dict(cached[2])

        wanted_suffixes = (
            ".block_count",
            ".embedding_length",
            ".attention.head_count",
            ".attention.head_count_kv",
            ".attention.key_length",
            ".attention.value_length",
            ".context_length",
            ".vocab_size",
        )
        wanted_exact = {
            "general.architecture",
            "general.name",
            "general.file_type",
            "tokenizer.ggml.tokens",
        }

        parsed = {}
        with open(real_path, "rb") as f:
            hdr = f.read(24)
            if len(hdr) < 24 or hdr[:4] != b"GGUF":
                return None
            version, tensor_count, kv_count = struct.unpack("<IQQ", hdr[4:24])
            if version not in (2, 3) or kv_count > 10000:
                return None

            for _ in range(kv_count):
                key = _read_gguf_string(f)
                vtype_raw = f.read(4)
                if len(vtype_raw) < 4:
                    break
                vtype = struct.unpack("<I", vtype_raw)[0]
                want = (key in wanted_exact) or key.endswith(wanted_suffixes)
                val = _skip_or_read_gguf_value(f, vtype, want)
                if want and val is not None:
                    parsed[key] = val
                # Stop early once we have all core architecture fields
                arch = parsed.get("general.architecture")
                if (
                    arch
                    and f"{arch}.block_count" in parsed
                    and f"{arch}.embedding_length" in parsed
                    and f"{arch}.attention.head_count" in parsed
                    and f"{arch}.attention.head_count_kv" in parsed
                    and (f"{arch}.vocab_size" in parsed or "tokenizer.ggml.tokens" in parsed)
                ):
                    break

        arch = str(parsed.get("general.architecture") or "").strip()
        if not arch:
            return None
        layers = int(parsed.get(f"{arch}.block_count") or 0)
        hidden_dim = int(parsed.get(f"{arch}.embedding_length") or 0)
        n_heads = int(parsed.get(f"{arch}.attention.head_count") or 0)
        n_kv_heads = int(parsed.get(f"{arch}.attention.head_count_kv") or n_heads or 0)
        key_len = int(parsed.get(f"{arch}.attention.key_length") or 0)
        if key_len <= 0 and hidden_dim > 0 and n_heads > 0:
            key_len = max(64, hidden_dim // n_heads)
        vocab_size = int(parsed.get(f"{arch}.vocab_size") or parsed.get("tokenizer.ggml.tokens") or 128256)
        max_ctx = int(parsed.get(f"{arch}.context_length") or 32768)

        if layers <= 0 or hidden_dim <= 0:
            return None

        meta = {
            "source": "gguf_header",
            "architecture": arch,
            "gguf_version": version,
            "tensor_count": tensor_count,
            "layers": layers,
            "hidden_dim": hidden_dim,
            "n_heads": max(1, n_heads),
            "n_kv_heads": max(1, n_kv_heads),
            "head_dim": max(64, key_len or 128),
            "vocab_size": max(32000, vocab_size),
            "max_context": max(2048, max_ctx),
            "size_bytes": int(st.st_size),
        }
        _GGUF_META_CACHE[real_path] = (st.st_mtime_ns, st.st_size, dict(meta))
        return meta
    except Exception:
        return None


# ============================================================================
# MODEL SPEC & HETEROGENEOUS AUTO LAYER-SPLIT PLANNER
# ============================================================================
def _closest_arch_profile(params_b: float) -> Tuple[int, int, int, int, int, int]:
    best = min(_ARCH_PROFILES, key=lambda p: abs(math.log(max(0.2, p[0])) - math.log(max(0.2, params_b))))
    return best[1], best[2], best[3], best[4], best[5], best[6]


def build_model_spec(
    name: str,
    size_bytes: int = 0,
    size_hint: str = "",
    params_hint: str = "",
    quant_hint: str = "",
    file_path: Optional[str] = None,
) -> dict:
    """Build a normalized architecture & weight breakdown for any local or remote GGUF."""
    gguf_meta = read_gguf_architecture(file_path) if (file_path and os.path.isfile(file_path)) else None

    actual_bytes = int(size_bytes or 0)
    if actual_bytes <= 0 and gguf_meta:
        actual_bytes = int(gguf_meta.get("size_bytes") or 0)
    if actual_bytes <= 0 and file_path and os.path.isfile(file_path):
        try:
            actual_bytes = int(os.path.getsize(file_path))
        except OSError:
            actual_bytes = 0
    if actual_bytes <= 0 and size_hint:
        actual_bytes = parse_size_hint_bytes(size_hint)

    quant = str(quant_hint or "").strip().upper() or extract_quant_from_name(name)
    bpw = QUANT_BPW.get(quant, 4.85)

    params_b = 0.0
    if params_hint:
        params_b = extract_params_b_from_name(params_hint)
    if params_b <= 0:
        params_b = extract_params_b_from_name(name, fallback_size_bytes=actual_bytes, quant=quant)

    if actual_bytes <= 0:
        # Estimate GGUF file size from parameter count & quantization BPW
        actual_bytes = int((params_b * 1e9 * bpw) / 8.0 * 1.03)

    if gguf_meta:
        layers = int(gguf_meta["layers"])
        hidden_dim = int(gguf_meta["hidden_dim"])
        n_heads = int(gguf_meta["n_heads"])
        n_kv_heads = int(gguf_meta["n_kv_heads"])
        head_dim = int(gguf_meta["head_dim"])
        vocab_size = int(gguf_meta["vocab_size"])
        max_context = int(gguf_meta["max_context"])
        arch_source = "gguf_header"
        arch_name = gguf_meta.get("architecture", "llama")
    else:
        layers, hidden_dim, n_heads, n_kv_heads, head_dim, vocab_size = _closest_arch_profile(params_b)
        max_context = 32768
        arch_source = "estimated_profile"
        arch_name = "transformer"

    total_weight_mb = actual_bytes / (1024.0 * 1024.0)

    # Separate token embeddings (Stage 0) and LM output head (Terminal Stage) from transformer blocks.
    # In modern GGUFs, embeddings & lm_head are quantized (typically Q4/Q6/Q8), ~vocab * hidden * bpw / 8.
    emb_bpw = min(max(bpw, 4.5), 8.5)
    raw_emb_mb = (vocab_size * hidden_dim * emb_bpw) / (8.0 * 1024.0 * 1024.0)
    # Clamp embeddings + lm_head to realistic fraction of total file size (max 22% combined)
    embed_mb = round(min(raw_emb_mb, total_weight_mb * 0.11), 1)
    lm_head_mb = round(min(raw_emb_mb, total_weight_mb * 0.11), 1)
    blocks_total_mb = max(total_weight_mb * 0.72, total_weight_mb - embed_mb - lm_head_mb)
    layer_weight_mb = round(blocks_total_mb / max(1, layers), 2)

    # AeroMESH Outlier-Preserving INT8 activation wire payload per token:
    # 44B header + 48B Top-8 FP32 outliers + hidden_dim * 1B INT8
    per_token_kb = round((44 + 48 + hidden_dim) / 1024.0, 2)

    return {
        "name": name,
        "quant": quant,
        "params_b": round(params_b, 2),
        "params_label": f"{params_b:g}B",
        "size_bytes": actual_bytes,
        "size_gb": round(actual_bytes / (1024.0 ** 3), 2),
        "size_label": mb_to_gb_label(total_weight_mb),
        "architecture": arch_name,
        "arch_source": arch_source,
        "layers": layers,
        "hidden_dim": hidden_dim,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "vocab_size": vocab_size,
        "max_context": max_context,
        "total_weight_mb": round(total_weight_mb, 1),
        "embed_mb": embed_mb,
        "lm_head_mb": lm_head_mb,
        "layer_weight_mb": layer_weight_mb,
        "per_token_kb": per_token_kb,
    }


def kv_cache_mb_per_layer(model_spec: dict, context_length: int, kv_bytes: float = 2.0) -> float:
    """
    Exact per-layer KV cache footprint in MiB:
      2 (K + V) * n_kv_heads * head_dim * context_length * kv_bytes / 1024^2
    Default kv_bytes=2.0 (FP16 K/V cache as used by llama.cpp default).
    """
    n_kv = max(1, int(model_spec.get("n_kv_heads") or 8))
    h_dim = max(64, int(model_spec.get("head_dim") or 128))
    ctx = max(512, int(context_length or 4096))
    return (2.0 * n_kv * h_dim * ctx * kv_bytes) / (1024.0 * 1024.0)


def normalize_cluster_nodes(raw_nodes: List[dict]) -> List[dict]:
    """Normalize local coordinator + worker nodes into a clean planner input list."""
    normalized = []
    for idx, n in enumerate(raw_nodes or []):
        if not isinstance(n, dict):
            continue
        gpu_str = str(n.get("gpu_name") or n.get("gpu") or "").strip()
        parsed_name, parsed_vram, parsed_used = parse_gpu_vram_mb(gpu_str)
        vram_total_mb = int(n.get("vram_total_mb") if n.get("vram_total_mb") is not None else parsed_vram)
        vram_used_mb = int(n.get("vram_used_mb") if n.get("vram_used_mb") is not None else parsed_used)
        vram_free_mb = int(
            n.get("vram_free_mb")
            if n.get("vram_free_mb") is not None
            else max(0, vram_total_mb - vram_used_mb)
        )
        ram_total_mb = int(n.get("ram_total_mb") or 16384)
        ram_avail_mb = int(
            n.get("ram_available_mb")
            if n.get("ram_available_mb") is not None
            else int(ram_total_mb * 0.72)
        )
        role = str(n.get("role") or ("coordinator" if idx == 0 else "worker")).strip().lower()
        normalized.append({
            "id": str(n.get("id") or f"node-{idx}"),
            "name": str(n.get("name") or n.get("node_name") or ("This PC (Coordinator)" if idx == 0 else f"Worker-{idx}")),
            "role": role,
            "gpu_name": parsed_name,
            "vram_total_mb": max(0, vram_total_mb),
            "vram_free_mb": max(0, min(vram_free_mb, vram_total_mb)) if vram_total_mb > 0 else 0,
            "ram_total_mb": max(2048, ram_total_mb),
            "ram_available_mb": max(1024, min(ram_avail_mb, ram_total_mb)),
            "online": bool(n.get("online", True)),
            "endpoint": str(n.get("endpoint") or n.get("worker_endpoint") or ""),
        })
    if not normalized:
        hw = detect_local_hardware()
        normalized.append({
            "id": "local",
            "name": "This PC (Coordinator)",
            "role": "coordinator",
            "gpu_name": hw["gpu_name"],
            "vram_total_mb": hw["vram_total_mb"],
            "vram_free_mb": hw["vram_free_mb"],
            "ram_total_mb": hw["ram_total_mb"],
            "ram_available_mb": hw["ram_available_mb"],
            "online": True,
            "endpoint": "127.0.0.1:8080",
        })
    return normalized


def _recommend_alternative_quants(
    model_spec: dict,
    total_usable_vram_mb: float,
    total_kv_mb: float,
    overhead_all_nodes_mb: float,
) -> List[dict]:
    """Recommend quantizations that would fit 100% in pooled VRAM when current quant is tight/spilling."""
    params_b = float(model_spec.get("params_b") or 7.0)
    cur_quant = str(model_spec.get("quant") or "Q4_K_M").upper()
    available_for_weights = max(0.0, total_usable_vram_mb - total_kv_mb - overhead_all_nodes_mb)
    candidates = ("Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_K_S", "IQ4_XS", "Q3_K_M", "Q2_K")
    recs = []
    for q in candidates:
        if q == cur_quant:
            continue
        bpw = QUANT_BPW[q]
        est_weight_mb = ((params_b * 1e9 * bpw) / 8.0 * 1.03) / (1024.0 * 1024.0)
        if est_weight_mb <= available_for_weights * 0.97:
            recs.append({
                "quant": q,
                "est_size_label": mb_to_gb_label(est_weight_mb),
                "est_total_vram_label": mb_to_gb_label(est_weight_mb + total_kv_mb + overhead_all_nodes_mb),
                "fits_vram": True,
            })
    # Return top 3 highest-quality quants that fit in VRAM
    return recs[:3]


def plan_layer_split(
    model_spec: dict,
    nodes: List[dict],
    context_length: int = 4096,
    kv_bytes: float = 2.0,
) -> dict:
    """
    Compute the optimal heterogeneous layer split across all active nodes.

    Accounts for:
      - Stage 0 (Coordinator) token embedding table (`embed_mb`)
      - Terminal Stage LM head (`lm_head_mb`) — stripped from non-terminal slices
      - Per-layer weights (`layer_weight_mb`) + GQA KV cache (`kv_per_layer_mb`)
      - Per-GPU CUDA context & activation scratch reserve (~420 MiB + 6% OS/display reserve)
      - System RAM spill when a model exceeds pooled GPU VRAM (e.g., 32B Q4_K_M on 8GB + 6GB + 16GB RAM)
    """
    ctx = max(512, min(131072, int(context_length or 4096)))
    norm_nodes = normalize_cluster_nodes(nodes)
    active_nodes = [n for n in norm_nodes if n.get("online", True)] or [norm_nodes[0]]

    total_layers = max(1, int(model_spec.get("layers") or 32))
    layer_wt_mb = float(model_spec.get("layer_weight_mb") or 120.0)
    kv_per_layer = kv_cache_mb_per_layer(model_spec, ctx, kv_bytes=kv_bytes)
    per_layer_total_mb = layer_wt_mb + kv_per_layer
    embed_mb = float(model_spec.get("embed_mb") or 200.0)
    lm_head_mb = float(model_spec.get("lm_head_mb") or 200.0)

    n_count = len(active_nodes)
    node_budgets = []
    for idx, node in enumerate(active_nodes):
        vram_tot = float(node["vram_total_mb"])
        ram_tot = float(node["ram_total_mb"])
        ram_avail = float(node["ram_available_mb"])

        # Usable VRAM leaves ~6% + 380 MiB for Windows WDDM/CUDA context/compute graph scratch
        if vram_tot > 0:
            cuda_scratch_mb = round(min(700.0, 340.0 + vram_tot * 0.045), 1)
            os_reserve_mb = round(min(640.0, vram_tot * 0.05), 1)
            usable_vram_mb = max(0.0, vram_tot - cuda_scratch_mb - os_reserve_mb)
        else:
            cuda_scratch_mb = 0.0
            os_reserve_mb = 0.0
            usable_vram_mb = 0.0

        # Safe usable System RAM keeps at least 3.0 GB free for OS + desktop UI
        usable_ram_mb = max(0.0, min(ram_avail, ram_tot - 3072.0))

        fixed_stage_mb = 0.0
        if idx == 0:
            fixed_stage_mb += embed_mb
        if idx == n_count - 1:
            fixed_stage_mb += lm_head_mb

        vram_for_layers_mb = max(0.0, usable_vram_mb - fixed_stage_mb)
        max_vram_layers = int(vram_for_layers_mb // per_layer_total_mb) if per_layer_total_mb > 0 else 0

        # Combined VRAM + RAM capacity for fallback spill
        combined_for_layers_mb = max(0.0, usable_vram_mb + usable_ram_mb - fixed_stage_mb)
        max_total_layers = int(combined_for_layers_mb // per_layer_total_mb) if per_layer_total_mb > 0 else 0

        node_budgets.append({
            "node": node,
            "stage_index": idx,
            "is_first": idx == 0,
            "is_last": idx == n_count - 1,
            "vram_total_mb": vram_tot,
            "usable_vram_mb": usable_vram_mb,
            "cuda_scratch_mb": cuda_scratch_mb,
            "os_reserve_mb": os_reserve_mb,
            "usable_ram_mb": usable_ram_mb,
            "fixed_stage_mb": fixed_stage_mb,
            "vram_for_layers_mb": vram_for_layers_mb,
            "max_vram_layers": max_vram_layers,
            "max_total_layers": max_total_layers,
        })

    total_vram_mb = sum(b["vram_total_mb"] for b in node_budgets)
    total_usable_vram_mb = sum(b["usable_vram_mb"] for b in node_budgets)
    total_usable_ram_mb = sum(b["usable_ram_mb"] for b in node_budgets)
    total_ram_mb = sum(b["node"]["ram_total_mb"] for b in node_budgets)
    total_kv_mb = round(kv_per_layer * total_layers, 1)
    total_scratch_mb = round(sum(b["cuda_scratch_mb"] for b in node_budgets), 1)
    total_model_mem_mb = round(
        embed_mb + lm_head_mb + (per_layer_total_mb * total_layers) + total_scratch_mb, 1
    )

    # 1. Allocate layers across nodes
    # Priority: proportion by VRAM capacity first; if total VRAM layers < total_layers,
    # fill every GPU's VRAM capacity and distribute remaining layers into System RAM proportionally.
    total_vram_layer_cap = sum(b["max_vram_layers"] for b in node_budgets)
    allocations = [0] * n_count

    if total_vram_layer_cap >= total_layers and total_vram_mb > 0:
        # Pure GPU VRAM fit! Allocate proportionally to each node's vram_for_layers_mb
        weights = [max(0.0, b["vram_for_layers_mb"]) for b in node_budgets]
        w_sum = sum(weights) or 1.0
        raw_shares = [(w / w_sum) * total_layers for w in weights]
        allocations = [min(node_budgets[i]["max_vram_layers"], int(math.floor(s))) for i, s in enumerate(raw_shares)]

        # Ensure multi-node pipeline uses every online GPU with >=1 layer capacity when beneficial
        if n_count > 1:
            for i in range(n_count):
                if allocations[i] == 0 and node_budgets[i]["max_vram_layers"] >= 1:
                    allocations[i] = 1

        # Adjust rounding difference cleanly while respecting per-node VRAM ceiling
        diff = total_layers - sum(allocations)
        while diff > 0:
            # Pick node with highest fractional remainder or largest remaining VRAM headroom
            best_i = None
            best_score = -1.0
            for i in range(n_count):
                if allocations[i] < node_budgets[i]["max_vram_layers"]:
                    headroom = node_budgets[i]["vram_for_layers_mb"] - allocations[i] * per_layer_total_mb
                    if headroom > best_score:
                        best_score = headroom
                        best_i = i
            if best_i is None:
                break
            allocations[best_i] += 1
            diff -= 1
        while diff < 0:
            # Trim from node with largest allocation (>1 if possible)
            best_i = max(range(n_count), key=lambda i: allocations[i])
            if allocations[best_i] <= 0:
                break
            allocations[best_i] -= 1
            diff += 1
    else:
        # Hybrid GPU + System RAM offload (or CPU-only)
        # First assign each GPU its max VRAM layers, then distribute leftover layers across RAM
        allocations = [min(total_layers, b["max_vram_layers"]) for b in node_budgets]
        # Guard if sum exceeded total_layers (edge case)
        while sum(allocations) > total_layers:
            idx_max = max(range(n_count), key=lambda i: allocations[i])
            allocations[idx_max] -= 1

        remaining = total_layers - sum(allocations)
        if remaining > 0:
            ram_weights = [max(1.0, b["usable_ram_mb"]) for b in node_budgets]
            rw_sum = sum(ram_weights)
            for i in range(n_count):
                extra = int(math.floor((ram_weights[i] / rw_sum) * remaining))
                allocations[i] += extra
            leftover = total_layers - sum(allocations)
            order = sorted(range(n_count), key=lambda i: node_budgets[i]["usable_ram_mb"], reverse=True)
            for idx_k in range(leftover):
                allocations[order[idx_k % n_count]] += 1

    # 2. Build detailed per-node stage report
    stages = []
    cursor = 0
    total_gpu_layers = 0
    total_ram_layers = 0
    max_vram_pct = 0.0
    total_ram_spill_mb = 0.0
    ram_overflow_mb = 0.0

    for i, b in enumerate(node_budgets):
        cnt = allocations[i]
        layer_start = cursor if cnt > 0 else None
        layer_end = (cursor + cnt - 1) if cnt > 0 else None
        cursor += cnt

        gpu_layers = min(cnt, b["max_vram_layers"]) if b["vram_total_mb"] > 0 else 0
        ram_layers = max(0, cnt - gpu_layers)
        total_gpu_layers += gpu_layers
        total_ram_layers += ram_layers

        weights_on_node_mb = round(b["fixed_stage_mb"] + cnt * layer_wt_mb, 1)
        kv_on_node_mb = round(cnt * kv_per_layer, 1)

        if b["vram_total_mb"] > 0:
            vram_used_by_stage = b["cuda_scratch_mb"] + b["fixed_stage_mb"] + gpu_layers * per_layer_total_mb
            # If even fixed_stage_mb + scratch exceeds usable VRAM, push overflow into RAM
            if vram_used_by_stage > b["usable_vram_mb"] + b["cuda_scratch_mb"]:
                spill_extra = vram_used_by_stage - (b["usable_vram_mb"] + b["cuda_scratch_mb"])
                vram_used_by_stage = b["usable_vram_mb"] + b["cuda_scratch_mb"]
            else:
                spill_extra = 0.0
            ram_used_by_stage = round(ram_layers * per_layer_total_mb + spill_extra, 1)
            vram_pct = round((vram_used_by_stage / max(1.0, b["vram_total_mb"])) * 100.0, 1)
        else:
            vram_used_by_stage = 0.0
            vram_pct = 0.0
            ram_used_by_stage = round(b["fixed_stage_mb"] + cnt * per_layer_total_mb, 1)

        total_ram_spill_mb += ram_used_by_stage
        if ram_used_by_stage > b["usable_ram_mb"]:
            ram_overflow_mb += (ram_used_by_stage - b["usable_ram_mb"])

        if b["vram_total_mb"] > 0:
            max_vram_pct = max(max_vram_pct, vram_pct)

        range_str = f"{layer_start}..={layer_end}" if (layer_start is not None and layer_end is not None) else ""
        ram_pct = round((ram_used_by_stage / max(1.0, b["node"]["ram_total_mb"])) * 100.0, 1)

        stages.append({
            "node_id": b["node"]["id"],
            "node_name": b["node"]["name"],
            "role": b["node"]["role"],
            "stage_index": i,
            "gpu_name": b["node"]["gpu_name"],
            "endpoint": b["node"]["endpoint"],
            "layer_start": layer_start,
            "layer_end": layer_end,
            "layer_count": cnt,
            "layer_range": range_str,
            "gpu_layers": gpu_layers,
            "ram_layers": ram_layers,
            "share_percent": round((cnt / max(1, total_layers)) * 100.0, 1),
            "vram_total_mb": round(b["vram_total_mb"], 1),
            "vram_used_mb": round(vram_used_by_stage, 1),
            "vram_percent": min(100.0, vram_pct),
            "vram_used_label": mb_to_gb_label(vram_used_by_stage),
            "vram_total_label": mb_to_gb_label(b["vram_total_mb"]),
            "ram_total_mb": round(b["node"]["ram_total_mb"], 1),
            "ram_used_mb": ram_used_by_stage,
            "ram_percent": min(100.0, ram_pct),
            "ram_used_label": mb_to_gb_label(ram_used_by_stage),
            "ram_total_label": mb_to_gb_label(b["node"]["ram_total_mb"]),
            "weights_mb": weights_on_node_mb,
            "kv_cache_mb": kv_on_node_mb,
            "includes_embeddings": b["is_first"],
            "includes_lm_head": b["is_last"],
            "cli_flag": f"--layers {range_str}" if range_str else "",
        })

    # 3. Determine overall Fit Tier & Recommendations
    total_required_with_reserves = total_model_mem_mb
    total_cluster_capacity_mb = total_usable_vram_mb + total_scratch_mb + total_usable_ram_mb

    if total_ram_layers == 0 and total_vram_mb > 0 and ram_overflow_mb <= 0:
        if max_vram_pct <= 88.0:
            fit_status = "fits_vram"
            fit_label = "Full VRAM Fit" if n_count == 1 else f"{n_count}-Node VRAM Fit"
            speed_tier = "Optimal (100% GPU VRAM)"
            summary = (
                f"All {total_layers} layers + {ctx:,} ctx KV cache fit in pooled GPU VRAM "
                f"({mb_to_gb_label(total_model_mem_mb)} / {mb_to_gb_label(total_vram_mb)} total VRAM, peak {max_vram_pct:.0f}%)."
            )
        else:
            fit_status = "tight_vram"
            fit_label = "Tight VRAM Fit"
            speed_tier = "Fast (Near VRAM Ceiling)"
            summary = (
                f"Fits across {n_count} GPU(s) at {max_vram_pct:.0f}% peak VRAM. "
                f"Keep context ≤ {ctx:,} or close background GPU apps to avoid WDDM paging."
            )
    elif ram_overflow_mb <= 0 and total_required_with_reserves <= total_cluster_capacity_mb:
        fit_status = "ram_offload"
        fit_label = "GPU + RAM Spill" if total_gpu_layers > 0 else "CPU RAM Only"
        gpu_pct_layers = round((total_gpu_layers / max(1, total_layers)) * 100.0)
        speed_tier = f"Hybrid ({gpu_pct_layers}% GPU / {100 - gpu_pct_layers}% RAM — slower tok/s)"
        summary = (
            f"Exceeds pooled GPU VRAM ({mb_to_gb_label(total_vram_mb)}). "
            f"{total_gpu_layers}/{total_layers} layers fit in VRAM; {total_ram_layers} layers ({mb_to_gb_label(total_ram_spill_mb)}) spill into System RAM."
        )
    else:
        fit_status = "wont_fit"
        fit_label = "Exceeds Memory"
        shortfall_mb = max(ram_overflow_mb, total_required_with_reserves - total_cluster_capacity_mb)
        speed_tier = "Will OOM / Swap Heavily"
        summary = (
            f"Needs {mb_to_gb_label(total_model_mem_mb)} at {ctx:,} ctx, exceeding safe VRAM + RAM capacity "
            f"by ~{mb_to_gb_label(shortfall_mb)}. Pick a smaller quantization or add another GPU node."
        )

    recommendations = []
    alt_quants = []
    if fit_status in ("tight_vram", "ram_offload", "wont_fit"):
        alt_quants = _recommend_alternative_quants(
            model_spec,
            total_usable_vram_mb=total_usable_vram_mb,
            total_kv_mb=total_kv_mb,
            overhead_all_nodes_mb=embed_mb + lm_head_mb,
        )
        if alt_quants:
            best_q = alt_quants[0]
            recommendations.append(
                f"Use {best_q['quant']} ({best_q['est_size_label']}) to run 100% inside your {mb_to_gb_label(total_vram_mb)} VRAM pool."
            )
        elif fit_status in ("ram_offload", "wont_fit") and QUANT_BPW.get(model_spec.get("quant", "Q4_K_M"), 4.85) > 3.6:
            q3_mb = (model_spec["params_b"] * 1e9 * QUANT_BPW["Q3_K_M"]) / (8.0 * 1024.0 * 1024.0) * 1.03
            recommendations.append(
                f"Use Q3_K_M (~{mb_to_gb_label(q3_mb)}) to cut System RAM spill and speed up token generation."
            )
        if ctx > 4096 and total_kv_mb > 700:
            recommendations.append(
                f"Lowering context from {ctx:,} to 4,096 saves ~{mb_to_gb_label(total_kv_mb * (1.0 - 4096.0 / ctx))} of KV-cache VRAM."
            )
        if fit_status in ("ram_offload", "wont_fit"):
            extra_vram_needed = max(1024.0, (total_model_mem_mb - total_vram_mb) * 1.12)
            lead = "Or pair" if recommendations else "Pair"
            recommendations.append(
                f"{lead} a worker with +{mb_to_gb_label(extra_vram_needed)} VRAM to keep all {total_layers} layers on GPU."
            )

    coordinator_stage = stages[0] if stages else None
    worker_stages = stages[1:] if len(stages) > 1 else []

    return {
        "model": model_spec,
        "context_length": ctx,
        "fit_status": fit_status,
        "fit_label": fit_label,
        "speed_tier": speed_tier,
        "summary": summary,
        "recommendations": recommendations,
        "recommended_quants": alt_quants,
        "cluster_totals": {
            "node_count": n_count,
            "vram_total_mb": round(total_vram_mb, 1),
            "vram_total_label": mb_to_gb_label(total_vram_mb),
            "ram_total_mb": round(total_ram_mb, 1),
            "ram_total_label": mb_to_gb_label(total_ram_mb),
            "model_total_mb": total_model_mem_mb,
            "model_total_label": mb_to_gb_label(total_model_mem_mb),
            "kv_cache_mb": total_kv_mb,
            "kv_cache_label": mb_to_gb_label(total_kv_mb),
            "gpu_layers": total_gpu_layers,
            "ram_layers": total_ram_layers,
            "total_layers": total_layers,
            "peak_vram_percent": min(100.0, round(max_vram_pct, 1)),
            "ram_spill_mb": round(total_ram_spill_mb, 1),
            "ram_spill_label": mb_to_gb_label(total_ram_spill_mb),
            "per_token_kb": model_spec["per_token_kb"],
        },
        "coordinator_layers": coordinator_stage["layer_range"] if coordinator_stage else "",
        "worker_layers": [
            {"node_name": w["node_name"], "endpoint": w["endpoint"], "layer_range": w["layer_range"]}
            for w in worker_stages
        ],
        "stages": stages,
    }


def validate_layer_range_arg(raw_layers: str) -> str:
    """Strictly validate CLI --layers argument (e.g. '0..26' or '0..=26')."""
    cleaned = str(raw_layers or "").strip()
    if not cleaned:
        return ""
    m = re.fullmatch(r"(\d{1,4})\.\.(=?)(\d{1,4})", cleaned)
    if not m:
        raise ValueError(f"Invalid layer range '{raw_layers}'. Expected format like '0..26'.")
    start, end = int(m.group(1)), int(m.group(3))
    if start > end:
        raise ValueError(f"Layer range start ({start}) cannot exceed end ({end}).")
    return cleaned
