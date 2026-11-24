"""
Regression & edge-case tests for issues uncovered during the AeroMESH bug hunt.
Pure Python stdlib (unittest) so it runs in any environment without optional deps.
"""

import ast
import json
import os
import re
import struct
import tempfile
import unittest

import openai_compat as oc
import peer_model_sync as pms
import vram_planner as vp

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _pack_gguf_string(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


class TestVramPlannerRegressions(unittest.TestCase):
    def test_inclusive_layer_ranges_cover_every_layer_without_gaps(self):
        # 2-node heterogeneous cluster (8 GB RTX 4060 + 6 GB RTX 3060)
        nodes_2 = [
            {
                "id": "coord",
                "name": "RTX-4060",
                "role": "coordinator",
                "gpu_name": "NVIDIA GeForce RTX 4060",
                "vram_total_mb": 8192,
                "ram_total_mb": 16384,
                "ram_available_mb": 12288,
            },
            {
                "id": "worker1",
                "name": "RTX-3060",
                "role": "worker",
                "gpu_name": "NVIDIA GeForce RTX 3060",
                "vram_total_mb": 6144,
                "ram_total_mb": 16384,
                "ram_available_mb": 12288,
            },
        ]
        spec = vp.build_model_spec(
            name="Qwen2.5-14B-Instruct-Q4_K_M.gguf",
            size_bytes=int(8.99 * (1024 ** 3)),
            params_hint="14B",
            quant_hint="Q4_K_M",
        )
        plan = vp.plan_layer_split(spec, nodes_2, context_length=4096)

        # Rust parse_layer_range treats a..b as exclusive and a..=b as inclusive.
        # Every stage must emit inclusive ..= syntax so no boundary layer is dropped.
        covered = set()
        for stage in plan["stages"]:
            r = stage["layer_range"]
            self.assertIn("..=", r, f"Stage {stage['node_name']} must use inclusive '..=' syntax, got {r!r}")
            self.assertEqual(vp.validate_layer_range_arg(r), r)
            m = re.fullmatch(r"(\d+)\.\.=(\d+)", r)
            self.assertIsNotNone(m)
            lo, hi = int(m.group(1)), int(m.group(2))
            self.assertEqual(lo, stage["layer_start"])
            self.assertEqual(hi, stage["layer_end"])
            for layer_idx in range(lo, hi + 1):
                self.assertNotIn(layer_idx, covered, f"Duplicate layer {layer_idx}")
                covered.add(layer_idx)

        self.assertEqual(covered, set(range(48)))
        self.assertIn("..=", plan["coordinator_layers"])
        self.assertTrue(len(plan["worker_layers"]) >= 1)
        for worker_info in plan["worker_layers"]:
            self.assertIn("..=", worker_info["layer_range"])

    def test_inclusive_layer_ranges_three_node_cluster(self):
        nodes_3 = [
            {"id": "n0", "name": "Coord-12G", "role": "coordinator", "gpu_name": "RTX 4070", "vram_total_mb": 12288, "ram_total_mb": 32768, "ram_available_mb": 24000},
            {"id": "n1", "name": "Work-8G", "role": "worker", "gpu_name": "RTX 4060", "vram_total_mb": 8192, "ram_total_mb": 16384, "ram_available_mb": 12000},
            {"id": "n2", "name": "Work-6G", "role": "worker", "gpu_name": "RTX 3060", "vram_total_mb": 6144, "ram_total_mb": 16384, "ram_available_mb": 12000},
        ]
        spec = vp.build_model_spec(
            name="Qwen2.5-32B-Instruct-Q4_K_M.gguf",
            size_bytes=int(19.85 * (1024 ** 3)),
            params_hint="32B",
            quant_hint="Q4_K_M",
        )
        plan = vp.plan_layer_split(spec, nodes_3, context_length=8192)
        covered = set()
        for stage in plan["stages"]:
            m = re.fullmatch(r"(\d+)\.\.=(\d+)", stage["layer_range"])
            self.assertIsNotNone(m, f"Expected inclusive range, got {stage['layer_range']!r}")
            for i in range(int(m.group(1)), int(m.group(2)) + 1):
                covered.add(i)
        self.assertEqual(covered, set(range(64)))

    def test_malformed_gguf_huge_string_and_array_lengths_do_not_overflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            # 1) Direct STRING KV with absurd 2^62 byte length
            bad_str_path = os.path.join(tmp, "bad_str.gguf")
            with open(bad_str_path, "wb") as f:
                f.write(b"GGUF")
                f.write(struct.pack("<IQQ", 3, 10, 1))
                f.write(_pack_gguf_string("general.description"))
                f.write(struct.pack("<I", 8))  # GGUF_TYPE_STRING
                f.write(struct.pack("<Q", (1 << 62)))  # Malicious length
            self.assertIsNone(vp.read_gguf_architecture(bad_str_path))

            # 2) ARRAY of STRINGs with an element claiming 2^60 byte length
            bad_arr_path = os.path.join(tmp, "bad_arr.gguf")
            with open(bad_arr_path, "wb") as f:
                f.write(b"GGUF")
                f.write(struct.pack("<IQQ", 3, 10, 1))
                f.write(_pack_gguf_string("tokenizer.ggml.tokens"))
                f.write(struct.pack("<I", 9))  # GGUF_TYPE_ARRAY
                f.write(struct.pack("<IQ", 8, 1))  # elem_type=STRING, count=1
                f.write(struct.pack("<Q", (1 << 60)))  # Malicious element length
            self.assertIsNone(vp.read_gguf_architecture(bad_arr_path))


class TestOpenAICompatRegressions(unittest.TestCase):
    def test_load_and_save_server_config_honors_bind_host_without_lan_exposed(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "server_config.json")
            # Write config containing bind_host="0.0.0.0" without explicit lan_exposed
            with open(p, "w", encoding="utf-8") as f:
                json.dump({"n_ctx": 8192, "ngl": 24, "bind_host": "0.0.0.0"}, f)

            loaded = oc.load_server_config_file(p)
            self.assertEqual(loaded["n_ctx"], 8192)
            self.assertEqual(loaded["ngl"], 24)
            self.assertTrue(loaded["lan_exposed"])
            self.assertEqual(loaded["bind_host"], "0.0.0.0")

            # save_server_config_file with bind_host="0.0.0.0" and no lan_exposed
            saved = oc.save_server_config_file(p, {"n_ctx": 16384, "ngl": -1, "bind_host": "0.0.0.0"})
            self.assertTrue(saved["lan_exposed"])
            self.assertEqual(saved["bind_host"], "0.0.0.0")

            # Explicit bind_host="127.0.0.1" overrides lan_exposed=True safely
            saved_local = oc.save_server_config_file(p, {"n_ctx": 4096, "ngl": -1, "lan_exposed": True, "bind_host": "127.0.0.1"})
            self.assertFalse(saved_local["lan_exposed"])
            self.assertEqual(saved_local["bind_host"], "127.0.0.1")

    def test_tool_calling_edge_cases(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Weather lookup",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Time lookup",
                    "parameters": {"type": "object", "properties": {"tz": {"type": "string"}}},
                },
            },
        ]

        # 1) tool_choice == "none" disables tools completely
        _, meta_none = oc.prepare_openai_request({
            "model": "test.gguf",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": tools,
            "tool_choice": "none",
        })
        self.assertFalse(meta_none["tools_enabled"])
        resp_none = oc.transform_completion_response(
            {"choices": [{"index": 0, "message": {"role": "assistant", "content": '<tool_call>{"name":"get_weather","arguments":{}}</tool_call>'}, "finish_reason": "stop"}]},
            meta_none,
        )
        self.assertEqual(resp_none["choices"][0]["finish_reason"], "stop")
        self.assertNotIn("tool_calls", resp_none["choices"][0]["message"])

        # 2) Parallel tool calls + disallowed hallucinated tool filtered + malformed JSON skipped
        _, meta_auto = oc.prepare_openai_request({
            "model": "test.gguf",
            "messages": [{"role": "user", "content": "Weather & time"}],
            "tools": tools,
            "tool_choice": "auto",
        })
        mixed_output = (
            '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>\n'
            '<tool_call>{"name": "rm_rf_evil", "arguments": {}}</tool_call>\n'
            '<tool_call>{not valid json</tool_call>\n'
            '<tool_call>{"name": "get_time", "arguments": {"tz": "UTC"}}</tool_call>'
        )
        resp_mixed = oc.transform_completion_response(
            {"choices": [{"index": 0, "message": {"role": "assistant", "content": mixed_output}, "finish_reason": "stop"}]},
            meta_auto,
        )
        tcs = resp_mixed["choices"][0]["message"]["tool_calls"]
        self.assertEqual([t["function"]["name"] for t in tcs], ["get_weather", "get_time"])
        self.assertEqual(resp_mixed["choices"][0]["finish_reason"], "tool_calls")

        # 3) Forced function tool_choice and required directive
        up_req, _ = oc.prepare_openai_request({
            "model": "test.gguf",
            "messages": [{"role": "user", "content": "Do it"}],
            "tools": tools,
            "tool_choice": "required",
        })
        self.assertIn("MUST invoke at least one", up_req["messages"][0]["content"])

        up_forced, _ = oc.prepare_openai_request({
            "model": "test.gguf",
            "messages": [{"role": "user", "content": "Time please"}],
            "tools": tools,
            "tool_choice": {"type": "function", "function": {"name": "get_time"}},
        })
        self.assertIn('You MUST call the function "get_time"', up_forced["messages"][0]["content"])

        # 4) json_schema mode unwraps markdown fences cleanly
        _, meta_schema = oc.prepare_openai_request({
            "model": "test.gguf",
            "messages": [{"role": "user", "content": "Return status"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "status_obj",
                    "schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}},
                },
            },
        })
        self.assertEqual(meta_schema["json_mode"], "json_schema")
        resp_json = oc.transform_completion_response(
            {"choices": [{"index": 0, "message": {"role": "assistant", "content": "```json\n{\"ok\": true}\n```"}, "finish_reason": "stop"}]},
            meta_schema,
        )
        self.assertEqual(json.loads(resp_json["choices"][0]["message"]["content"]), {"ok": True})


class TestPeerModelSyncIPv6Regressions(unittest.TestCase):
    def test_ipv6_hosts_and_bracketed_endpoints(self):
        # Valid IPv6 loopback and ULA (e.g. Tailscale / local IPv6)
        self.assertEqual(pms.validate_peer_host("[::1]:50054"), ("::1", 50054))
        self.assertEqual(pms.validate_peer_host("::1"), ("::1", 50054))
        self.assertEqual(pms.validate_peer_host("[fd7a:115c:a1e0::1]:50055"), ("fd7a:115c:a1e0::1", 50055))
        self.assertEqual(pms.validate_peer_host("fd7a:115c:a1e0::1"), ("fd7a:115c:a1e0::1", 50054))

        self.assertEqual(pms.extract_host_from_endpoint("[fd7a:115c:a1e0::1]:50052"), "fd7a:115c:a1e0::1")
        self.assertEqual(pms.extract_host_from_endpoint("http://[::1]:50052"), "::1")
        self.assertEqual(pms.extract_host_from_endpoint("100.64.0.29:50052"), "100.64.0.29")

        # Forbidden IPv6 link-local, unspecified, multicast, and low port
        for bad in (
            "[fe80::1]:50054",
            "fe80::1",
            "[::]:50054",
            "::",
            "[ff02::1]:50054",
            "ff02::1",
            "[::1]:80",
            "[::1]:abc",
        ):
            with self.assertRaises(ValueError, msg=f"Should reject {bad!r}"):
                pms.validate_peer_host(bad)


class TestStaticCodebaseInvariants(unittest.TestCase):
    def _read(self, rel_path: str) -> str:
        with open(os.path.join(REPO_ROOT, rel_path), "r", encoding="utf-8") as f:
            return f.read()

    def test_rust_pipeline_does_not_hardcode_4096_ctx_and_honors_ngl(self):
        src = self._read("crates/aeromesh-engine/src/pipeline.rs")
        self.assertIsNone(
            re.search(r"LlamaPipelineInstance::open\([^;]*,\s*4096\s*\)", src),
            "pipeline.rs must pass 0 (not 4096) to LlamaPipelineInstance::open so AEROMESH_N_CTX is honored",
        )
        self.assertIn("resolve_runtime_ngl", src)
        self.assertIn("AEROMESH_NGL", src)

        cli_src = self._read("crates/aeromesh-cli/src/main.rs")
        self.assertIn('std::env::set_var("AEROMESH_NGL"', cli_src)

    def test_rust_server_test_compiles_struct_and_supports_tool_and_json_fixes(self):
        src = self._read("crates/aeromesh-engine/src/server.rs")
        # Verify test_handle_chat_abort_endpoint initializes last_rtt_ms
        idx = src.find("fn test_handle_chat_abort_endpoint")
        self.assertGreater(idx, 0)
        self.assertIn("last_rtt_ms: None", src[idx:])
        self.assertIn("strip_json_code_fence", src)
        self.assertIn("allowed_tool_names", src)

    def test_app_py_uses_bounded_json_reader_and_no_raw_request_json(self):
        src = self._read("app.py")
        # Ensure app.py parses cleanly with Python AST
        ast.parse(src)
        # Only _read_json_body itself should read raw bytes; no route should call await request.json() directly
        self.assertFalse("await request.json()" in src, "Raw await request.json() should not appear in app.py")
        self.assertIn("AEROMESH_UI_PORT", src)
        self.assertIn("ip_obj.is_link_local", src)

    def test_frontend_js_has_no_blocking_alert_or_confirm(self):
        for rel in (
            "static/js/app.js",
            "static/js/dev_server.js",
            "static/js/model_hub.js",
            "static/js/cluster_pair.js",
            "static/js/liquid_mesh.js",
        ):
            src = self._read(rel)
            self.assertIsNone(
                re.search(r"\b(?:window\.)?(?:alert|confirm)\s*\(", src),
                f"Blocking alert()/confirm() found in {rel}",
            )


if __name__ == "__main__":
    unittest.main()
