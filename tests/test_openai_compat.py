import json
import os
import tempfile
import unittest

import openai_compat as oc


class TestOpenAICompat(unittest.TestCase):
    def test_runtime_config_validation_and_persistence(self):
        self.assertEqual(oc.DEFAULT_N_CTX, 4096)
        self.assertEqual(oc.DEFAULT_NGL, -1)

        cfg = oc.validate_runtime_config(n_ctx=16384, ngl=28, lan_exposed=True)
        self.assertEqual(cfg["n_ctx"], 16384)
        self.assertEqual(cfg["ngl"], 28)
        self.assertTrue(cfg["lan_exposed"])
        self.assertEqual(cfg["bind_host"], "0.0.0.0")

        cfg_all = oc.validate_runtime_config(n_ctx="8192", ngl="all", lan_exposed=False)
        self.assertEqual(cfg_all["n_ctx"], 8192)
        self.assertEqual(cfg_all["ngl"], -1)
        self.assertEqual(cfg_all["bind_host"], "127.0.0.1")

        cfg_bind_lan = oc.validate_runtime_config(n_ctx=4096, ngl=-1, bind_host="0.0.0.0")
        self.assertTrue(cfg_bind_lan["lan_exposed"])
        self.assertEqual(cfg_bind_lan["bind_host"], "0.0.0.0")

        cfg_bind_local = oc.validate_runtime_config(n_ctx=4096, ngl=-1, lan_exposed=True, bind_host="127.0.0.1")
        self.assertFalse(cfg_bind_local["lan_exposed"])
        self.assertEqual(cfg_bind_local["bind_host"], "127.0.0.1")

        with self.assertRaises(ValueError):
            oc.validate_runtime_config(n_ctx=128)
        with self.assertRaises(ValueError):
            oc.validate_runtime_config(n_ctx=999999)
        with self.assertRaises(ValueError):
            oc.validate_runtime_config(ngl=-5)
        with self.assertRaises(ValueError):
            oc.validate_runtime_config(bind_host="1.2.3.4")

        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "server_config.json")
            oc.save_server_config_file(p, cfg)
            loaded = oc.load_server_config_file(p)
            self.assertEqual(loaded, cfg)

    def test_mask_and_generate_api_key(self):
        key = oc.generate_api_key()
        self.assertTrue(key.startswith("sk-aeromesh-"))
        self.assertGreaterEqual(len(key), 40)
        masked = oc.mask_api_key(key)
        self.assertNotIn(key[12:-4], masked)
        self.assertTrue(masked.startswith("sk-aerom"))
        self.assertTrue(masked.endswith(key[-4:]))

    def test_prepare_and_extract_tool_calling_unary_and_sse(self):
        req_body = {
            "model": "Qwen2.5-14B-Instruct-Q4_K_M.gguf",
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "What is the weather in Tokyo?"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Fetch live weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
            "tool_choice": "auto",
            "stream": False,
        }

        upstream, meta = oc.prepare_openai_request(req_body)
        self.assertTrue(meta["tools_enabled"])
        self.assertEqual(meta["allowed_tool_names"], ["get_weather"])
        self.assertIn("get_weather", upstream["messages"][0]["content"])
        self.assertEqual(upstream["reasoning"], False)

        # Model returns an XML-wrapped tool call
        raw_resp = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1726400000,
            "model": "Qwen2.5-14B-Instruct-Q4_K_M.gguf",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Tokyo"}}\n</tool_call>',
                    },
                    "finish_reason": "stop",
                }
            ],
        }
        transformed = oc.transform_completion_response(raw_resp, meta)
        choice = transformed["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertIsNone(choice["message"]["content"])
        self.assertEqual(len(choice["message"]["tool_calls"]), 1)
        tc = choice["message"]["tool_calls"][0]
        self.assertEqual(tc["function"]["name"], "get_weather")
        self.assertEqual(json.loads(tc["function"]["arguments"]), {"city": "Tokyo"})

        # Streaming SSE translation for tool calls
        frames = oc.build_tool_call_sse_frames(
            '<tool_call>{"name": "get_weather", "arguments": {"city": "Tokyo"}}</tool_call>',
            meta,
            completion_id="chatcmpl-sse-1",
            model="Qwen2.5-14B-Instruct-Q4_K_M.gguf",
        )
        self.assertEqual(frames[-1], "data: [DONE]\n\n")
        parsed_chunks = [
            json.loads(f[len("data: ") :].strip())
            for f in frames[:-1]
        ]
        tool_deltas = [
            c["choices"][0]["delta"]["tool_calls"]
            for c in parsed_chunks
            if "tool_calls" in c["choices"][0]["delta"]
        ]
        self.assertEqual(len(tool_deltas), 1)
        self.assertEqual(tool_deltas[0][0]["function"]["name"], "get_weather")
        self.assertEqual(parsed_chunks[-1]["choices"][0]["finish_reason"], "tool_calls")

    def test_multi_turn_tool_messages_and_json_mode(self):
        body = {
            "model": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Summarize Tokyo weather as JSON."}]},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city":"Tokyo"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": '{"temp_c": 19}'},
            ],
        }
        upstream, meta = oc.prepare_openai_request(body)
        self.assertEqual(meta["json_mode"], "json_object")
        self.assertEqual(upstream["messages"][0]["role"], "system")
        self.assertIn("JSON Output Mode", upstream["messages"][0]["content"])
        # Ensure assistant null content & tool result were normalized to non-empty strings
        self.assertTrue(all(m["content"].strip() for m in upstream["messages"]))

        # Unwrap fenced JSON with thinking block
        fenced = "<think>planning</think>\n```json\n{\"city\": \"Tokyo\", \"temp_c\": 19}\n```"
        cleaned = oc.extract_json_mode_content(fenced)
        self.assertEqual(json.loads(cleaned), {"city": "Tokyo", "temp_c": 19})

    def test_snippet_generator_covers_all_ides(self):
        snips = oc.generate_integration_snippets(
            "http://127.0.0.1:8080/v1",
            "sk-aeromesh-demo",
            "Qwen2.5-14B-Instruct-Q4_K_M.gguf",
        )
        for key in ("curl", "python", "tools_json", "continue", "cline", "cursor"):
            self.assertIn(key, snips)
            self.assertIn("http://127.0.0.1:8080/v1", snips[key])
            self.assertIn("Qwen2.5-14B-Instruct-Q4_K_M.gguf", snips[key])


if __name__ == "__main__":
    unittest.main()
