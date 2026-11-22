import os
import struct
import tempfile
import unittest

from vram_planner import (
    build_model_spec,
    parse_gpu_vram_mb,
    plan_layer_split,
    read_gguf_architecture,
    validate_layer_range_arg,
)


def _pack_gguf_string(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _write_synthetic_gguf_with_kv(path: str, arch: str = "qwen2", layers: int = 48, hidden: int = 5120, heads: int = 40, kv_heads: int = 8, ctx: int = 32768):
    # GGUF v3 header with 6 KV pairs
    kvs = [
        ("general.architecture", 8, arch),       # STRING
        (f"{arch}.block_count", 4, layers),      # UINT32
        (f"{arch}.embedding_length", 4, hidden), # UINT32
        (f"{arch}.attention.head_count", 4, heads),
        (f"{arch}.attention.head_count_kv", 4, kv_heads),
        (f"{arch}.context_length", 4, ctx),
    ]
    with open(path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<IQQ", 3, 579, len(kvs)))
        for k, vtype, val in kvs:
            f.write(_pack_gguf_string(k))
            f.write(struct.pack("<I", vtype))
            if vtype == 8:
                f.write(_pack_gguf_string(val))
            else:
                f.write(struct.pack("<I", val))
        # Pad with minimal bytes
        f.write(b"\x00" * 4096)


class TestVramPlanner(unittest.TestCase):
    def test_parse_gpu_vram_strings(self):
        name, total, used = parse_gpu_vram_mb("NVIDIA GeForce RTX 3060 Laptop GPU, 6144 MiB, 420 MiB")
        self.assertIn("RTX 3060", name)
        self.assertEqual(total, 6144)
        self.assertEqual(used, 420)

        name2, total2, _ = parse_gpu_vram_mb("NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB")
        self.assertIn("RTX 4060", name2)
        self.assertEqual(total2, 8188)

        _, cpu_vram, _ = parse_gpu_vram_mb("CPU / Integrated GPU")
        self.assertEqual(cpu_vram, 0)

    def test_read_gguf_v3_kv_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            fp = os.path.join(tmp, "Qwen2.5-14B-Instruct-Q4_K_M.gguf")
            _write_synthetic_gguf_with_kv(fp, arch="qwen2", layers=48, hidden=5120, heads=40, kv_heads=8, ctx=32768)
            meta = read_gguf_architecture(fp)
            self.assertIsNotNone(meta)
            self.assertEqual(meta["source"], "gguf_header")
            self.assertEqual(meta["architecture"], "qwen2")
            self.assertEqual(meta["layers"], 48)
            self.assertEqual(meta["hidden_dim"], 5120)
            self.assertEqual(meta["n_heads"], 40)
            self.assertEqual(meta["n_kv_heads"], 8)
            self.assertEqual(meta["head_dim"], 128)

    def test_14b_q4km_fits_across_rtx4060_8gb_and_rtx3060_6gb(self):
        # Exact user scenario: 14B Q4_K_M (~9.0 GB) across 8 GB RTX 4060 + 6 GB RTX 3060 + 16 GB RAM
        nodes = [
            {
                "id": "coord",
                "name": "Laptop-A (RTX 4060)",
                "role": "coordinator",
                "gpu_name": "NVIDIA GeForce RTX 4060 Laptop GPU",
                "vram_total_mb": 8192,
                "ram_total_mb": 16384,
                "ram_available_mb": 12288,
            },
            {
                "id": "worker1",
                "name": "Laptop-B (RTX 3060)",
                "role": "worker",
                "gpu_name": "NVIDIA GeForce RTX 3060 Laptop GPU",
                "vram_total_mb": 6144,
                "ram_total_mb": 16384,
                "ram_available_mb": 12288,
            },
        ]
        spec = build_model_spec(
            name="DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
            size_bytes=int(8.99 * (1024 ** 3)),
            params_hint="14B",
            quant_hint="Q4_K_M",
        )
        plan = plan_layer_split(spec, nodes, context_length=4096)

        self.assertIn(plan["fit_status"], ("fits_vram", "tight_vram"))
        self.assertEqual(plan["cluster_totals"]["total_layers"], 48)
        self.assertEqual(plan["cluster_totals"]["gpu_layers"], 48)
        self.assertEqual(plan["cluster_totals"]["ram_layers"], 0)
        # 8 GB node should receive strictly more layers than 6 GB node
        s0, s1 = plan["stages"][0], plan["stages"][1]
        self.assertGreater(s0["layer_count"], s1["layer_count"])
        self.assertEqual(s0["layer_start"], 0)
        self.assertEqual(s0["layer_end"] + 1, s1["layer_start"])
        self.assertEqual(s1["layer_end"], 47)
        self.assertEqual(validate_layer_range_arg(plan["coordinator_layers"]), plan["coordinator_layers"])

    def test_32b_q4km_spills_to_ram_across_8gb_6gb_16gb_and_suggests_quants(self):
        # Exact user scenario: 32B Q4_K_M (~19.8 GB) across 8 GB RTX 4060 + 6 GB RTX 3060 + 16 GB RAM
        nodes = [
            {
                "id": "coord",
                "name": "Laptop-A (RTX 4060)",
                "role": "coordinator",
                "gpu_name": "NVIDIA GeForce RTX 4060",
                "vram_total_mb": 8192,
                "ram_total_mb": 16384,
                "ram_available_mb": 12288,
            },
            {
                "id": "worker1",
                "name": "Laptop-B (RTX 3060)",
                "role": "worker",
                "gpu_name": "NVIDIA GeForce RTX 3060",
                "vram_total_mb": 6144,
                "ram_total_mb": 16384,
                "ram_available_mb": 12288,
            },
        ]
        spec = build_model_spec(
            name="Qwen2.5-32B-Instruct-Q4_K_M.gguf",
            size_bytes=int(19.85 * (1024 ** 3)),
            params_hint="32B",
            quant_hint="Q4_K_M",
        )
        plan = plan_layer_split(spec, nodes, context_length=4096)

        self.assertEqual(plan["fit_status"], "ram_offload")
        self.assertGreater(plan["cluster_totals"]["gpu_layers"], 25)
        self.assertGreater(plan["cluster_totals"]["ram_layers"], 10)
        self.assertEqual(
            plan["cluster_totals"]["gpu_layers"] + plan["cluster_totals"]["ram_layers"],
            64,
        )
        self.assertTrue(len(plan["recommendations"]) >= 1)

    def test_70b_wont_fit_small_single_laptop(self):
        nodes = [{
            "id": "coord",
            "name": "Single-Laptop",
            "role": "coordinator",
            "gpu_name": "RTX 3060",
            "vram_total_mb": 6144,
            "ram_total_mb": 16384,
            "ram_available_mb": 11000,
        }]
        spec = build_model_spec(
            name="Llama-3.3-70B-Instruct-Q4_K_M.gguf",
            size_bytes=int(42.5 * (1024 ** 3)),
            params_hint="70B",
            quant_hint="Q4_K_M",
        )
        plan = plan_layer_split(spec, nodes, context_length=8192)
        self.assertEqual(plan["fit_status"], "wont_fit")

    def test_validate_layer_range_arg(self):
        self.assertEqual(validate_layer_range_arg("0..26"), "0..26")
        self.assertEqual(validate_layer_range_arg("0..=26"), "0..=26")
        self.assertEqual(validate_layer_range_arg(""), "")
        for bad in ("26..10", "0..26;rm -rf /", "-1..10", "abc"):
            with self.assertRaises(ValueError):
                validate_layer_range_arg(bad)


if __name__ == "__main__":
    unittest.main()
