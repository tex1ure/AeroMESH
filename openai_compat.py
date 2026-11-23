"""
AeroMESH OpenAI Compatibility, Tool Calling, JSON Mode & Server Config Helpers
===============================================================================
Pure-stdlib module (zero third-party runtime dependencies) providing:
- Strict validation & persistence of local API server / inference runtime settings
  (n_ctx, -ngl GPU offload layers, LAN exposure bind host).
- OpenAI-compatible Tool Calling (`tools`, `tool_choice`, `role: "tool"`) and
  JSON Mode (`response_format`: `json_object` / `json_schema`) request preparation,
  response extraction, and SSE streaming chunk translation for local GGUF models.
- Ready-to-copy IDE & SDK integration snippet generators (cURL, OpenAI Python SDK,
  Tool Calling + JSON Mode, Continue.dev, Cline / Roo Code, Cursor / Open WebUI).
"""

from __future__ import annotations

import copy
import json
import os
import re
import secrets
import time
from typing import Any, Dict, List, Optional, Tuple


ALLOWED_N_CTX_PRESETS = (2048, 4096, 8192, 16384, 32768)
DEFAULT_N_CTX = 4096
DEFAULT_NGL = -1
MIN_N_CTX = 512
MAX_N_CTX = 131072
MIN_NGL = -1
MAX_NGL = 999
ALLOWED_BIND_HOSTS = ("127.0.0.1", "0.0.0.0")


# ============================================================================
# RUNTIME & SERVER CONFIG VALIDATION
# ============================================================================
def validate_n_ctx(raw: Any, default: int = DEFAULT_N_CTX) -> int:
    if raw is None or raw == "":
        return default
    try:
        val = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"n_ctx must be an integer between {MIN_N_CTX} and {MAX_N_CTX}.") from exc
    if val < MIN_N_CTX or val > MAX_N_CTX:
        raise ValueError(f"n_ctx must be between {MIN_N_CTX} and {MAX_N_CTX}, got {val}.")
    return val


def validate_ngl(raw: Any, default: int = DEFAULT_NGL) -> int:
    if raw is None or raw == "":
        return default
    if isinstance(raw, str) and raw.strip().lower() in ("auto", "all", "-1"):
        return -1
    try:
        val = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"ngl must be -1 (all GPU layers) or 0..{MAX_NGL}.") from exc
    if val < MIN_NGL or val > MAX_NGL:
        raise ValueError(f"ngl must be between {MIN_NGL} and {MAX_NGL}, got {val}.")
    return val


def validate_runtime_config(
    n_ctx: Any = DEFAULT_N_CTX,
    ngl: Any = DEFAULT_NGL,
    lan_exposed: Any = False,
    bind_host: Any = None,
) -> Dict[str, Any]:
    ctx = validate_n_ctx(n_ctx)
    gpu_layers = validate_ngl(ngl)
    if bind_host is not None and str(bind_host).strip() != "":
        clean_bind = str(bind_host).strip()
        if clean_bind not in ALLOWED_BIND_HOSTS:
            raise ValueError(
                f"bind_host must be one of {', '.join(ALLOWED_BIND_HOSTS)}, got {clean_bind!r}."
            )
        exposed = clean_bind == "0.0.0.0"
        resolved_bind = clean_bind
    else:
        exposed = bool(lan_exposed)
        resolved_bind = "0.0.0.0" if exposed else "127.0.0.1"
    return {
        "n_ctx": ctx,
        "ngl": gpu_layers,
        "lan_exposed": exposed,
        "bind_host": resolved_bind,
    }


def mask_api_key(key: str) -> str:
    clean = str(key or "").strip()
    if not clean:
        return "Not configured"
    if len(clean) <= 10:
        return clean[:2] + "•" * max(4, len(clean) - 2)
    return f"{clean[:8]}••••••••••••{clean[-4:]}"


def generate_api_key() -> str:
    return f"sk-aeromesh-{secrets.token_hex(24)}"


def load_server_config_file(path: str) -> Dict[str, Any]:
    defaults = validate_runtime_config(4096, -1, False)
    if not path or not os.path.isfile(path):
        return defaults
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            return defaults
        return validate_runtime_config(
            n_ctx=raw.get("n_ctx", 4096),
            ngl=raw.get("ngl", -1),
            lan_exposed=raw.get("lan_exposed", False),
        )
    except Exception:
        return defaults


def save_server_config_file(path: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    validated = validate_runtime_config(
        n_ctx=cfg.get("n_ctx", 4096),
        ngl=cfg.get("ngl", -1),
        lan_exposed=cfg.get("lan_exposed", False),
    )
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(validated, f, indent=2)
    os.replace(tmp, path)
    return validated


# ============================================================================
# OPENAI TOOL-CALLING & JSON-MODE COMPATIBILITY LAYER
# ============================================================================
_TOOL_CALL_TAG_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*\n?(.*?)\n?```", re.DOTALL)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _normalize_content(content: Any) -> str:
    """Convert OpenAI string or multimodal content-part arrays to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
        return "\n".join(p for p in parts if p)
    return str(content)


def _format_tools_system_block(tools: List[Dict[str, Any]], tool_choice: Any = None) -> str:
    """Build a compact, model-friendly Hermes/Qwen tool-calling system directive."""
    compact_tools = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        if t.get("type") == "function" and isinstance(t.get("function"), dict):
            fn = t["function"]
            compact_tools.append({
                "type": "function",
                "function": {
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
                },
            })
        elif "name" in t:
            compact_tools.append({
                "type": "function",
                "function": {
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters", {"type": "object", "properties": {}}),
                },
            })

    if not compact_tools:
        return ""

    choice_instruction = (
        "When calling a tool, respond ONLY with one or more XML blocks in this exact format:\n"
        "<tool_call>\n"
        "{\"name\": \"<function_name>\", \"arguments\": {<json_arguments>}}\n"
        "</tool_call>\n"
        "Do not fabricate tool outputs. If no tool is needed, reply normally."
    )
    if tool_choice == "required":
        choice_instruction = (
            "You MUST invoke at least one of the available tools using this exact XML format:\n"
            "<tool_call>\n"
            "{\"name\": \"<function_name>\", \"arguments\": {<json_arguments>}}\n"
            "</tool_call>"
        )
    elif isinstance(tool_choice, dict):
        forced_name = (
            tool_choice.get("function", {}).get("name")
            if isinstance(tool_choice.get("function"), dict)
            else tool_choice.get("name")
        )
        if forced_name:
            choice_instruction = (
                f"You MUST call the function \"{forced_name}\" using this exact XML format:\n"
                "<tool_call>\n"
                f"{{\"name\": \"{forced_name}\", \"arguments\": {{<json_arguments>}}}}\n"
                "</tool_call>"
            )

    tools_json = json.dumps(compact_tools, ensure_ascii=False, separators=(",", ":"))
    return (
        "# Available Tools\n"
        f"{choice_instruction}\n"
        f"Tools schema:\n{tools_json}"
    )


def _format_json_mode_block(response_format: Dict[str, Any]) -> str:
    rf_type = str(response_format.get("type") or "").strip().lower()
    if rf_type == "json_schema":
        js = response_format.get("json_schema") or {}
        schema_name = js.get("name") or "response"
        schema_obj = js.get("schema") or {}
        schema_str = json.dumps(schema_obj, ensure_ascii=False, separators=(",", ":"))
        return (
            "# Structured JSON Output Mode\n"
            f"Respond ONLY with a single valid JSON object matching schema \"{schema_name}\". "
            "Do not include markdown code fences, commentary, or extra text.\n"
            f"JSON Schema: {schema_str}"
        )
    if rf_type == "json_object":
        return (
            "# JSON Output Mode\n"
            "Respond ONLY with a single valid JSON object. "
            "Do not wrap it in markdown fences or include any text before or after the JSON."
        )
    return ""


def prepare_openai_request(body: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Normalizes an incoming OpenAI /v1/chat/completions payload for the AeroMESH engine.

    Handles:
    - `tools` + `tool_choice` (injects structured tool schema into system prompt)
    - `response_format` (`json_object` / `json_schema`)
    - Multi-turn `role: "tool"` responses and assistant messages with `tool_calls` + `content: null`
    - Multimodal content arrays (`[{"type": "text", "text": "..."}]`)

    Returns `(upstream_body, meta)` where `meta` guides response & SSE post-processing.
    """
    upstream = copy.deepcopy(body)
    raw_messages = upstream.get("messages")
    if not isinstance(raw_messages, list):
        raw_messages = []

    raw_tools = upstream.get("tools")
    tool_choice = upstream.get("tool_choice")
    tools_enabled = (
        isinstance(raw_tools, list)
        and len(raw_tools) > 0
        and tool_choice != "none"
    )

    allowed_tool_names: List[str] = []
    if tools_enabled:
        for t in raw_tools:
            if isinstance(t, dict):
                fn = t.get("function") if isinstance(t.get("function"), dict) else t
                name = str(fn.get("name") or "").strip()
                if name:
                    allowed_tool_names.append(name)

    response_format = upstream.get("response_format")
    json_mode_type = ""
    if isinstance(response_format, dict):
        rf_t = str(response_format.get("type") or "").strip().lower()
        if rf_t in ("json_object", "json_schema"):
            json_mode_type = rf_t

    directives: List[str] = []
    if tools_enabled:
        tb = _format_tools_system_block(raw_tools, tool_choice)
        if tb:
            directives.append(tb)
    if json_mode_type:
        jb = _format_json_mode_block(response_format)
        if jb:
            directives.append(jb)

    # Normalize messages so the Rust engine always receives non-empty string contents
    normalized_messages: List[Dict[str, str]] = []
    for msg in raw_messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "user").strip().lower()
        content_text = _normalize_content(msg.get("content"))

        if role == "assistant" and not content_text.strip() and isinstance(msg.get("tool_calls"), list):
            # Re-serialize prior assistant tool calls so follow-up turns preserve context
            rendered_calls = []
            for tc in msg["tool_calls"]:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                fn_name = fn.get("name") or ""
                fn_args = fn.get("arguments") or "{}"
                if isinstance(fn_args, str):
                    try:
                        parsed_args = json.loads(fn_args)
                    except Exception:
                        parsed_args = fn_args
                else:
                    parsed_args = fn_args
                rendered_calls.append(
                    f"<tool_call>\n{json.dumps({'name': fn_name, 'arguments': parsed_args}, ensure_ascii=False)}\n</tool_call>"
                )
            content_text = "\n".join(rendered_calls)
        elif role == "tool":
            tool_name = str(msg.get("name") or "").strip()
            call_id = str(msg.get("tool_call_id") or "").strip()
            header = f"Tool Result ({tool_name or call_id or 'function'})"
            content_text = f"<tool_response>\n{header}:\n{content_text}\n</tool_response>"
            # Map to user role for broad GGUF chat-template compatibility (DeepSeek/Llama3/ChatML)
            role = "user"

        if not content_text.strip():
            continue
        normalized_messages.append({"role": role, "content": content_text})

    if directives:
        directive_block = "\n\n".join(directives)
        if normalized_messages and normalized_messages[0]["role"] == "system":
            normalized_messages[0]["content"] = (
                normalized_messages[0]["content"].rstrip() + "\n\n" + directive_block
            )
        else:
            normalized_messages.insert(0, {"role": "system", "content": directive_block})

    upstream["messages"] = normalized_messages

    # When JSON mode or tool calling is requested, disable forced <think> prefix unless explicitly requested
    if (tools_enabled or json_mode_type) and "reasoning" not in upstream:
        upstream["reasoning"] = False

    meta = {
        "tools_enabled": tools_enabled,
        "allowed_tool_names": allowed_tool_names,
        "json_mode": json_mode_type,
        "model": str(upstream.get("model") or "aeromesh-local"),
    }
    return upstream, meta


def _parse_single_tool_object(obj: Any, allowed_names: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return None
    name = str(obj.get("name") or "").strip()
    if not name:
        return None
    if allowed_names and name not in allowed_names:
        return None

    raw_args = obj.get("arguments", obj.get("parameters", {}))
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
            args_str = json.dumps(parsed, ensure_ascii=False)
        except Exception:
            args_str = raw_args
    else:
        args_str = json.dumps(raw_args if raw_args is not None else {}, ensure_ascii=False)

    return {
        "id": f"call_{secrets.token_hex(12)}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": args_str,
        },
    }


def extract_tool_calls(
    text: str,
    allowed_tool_names: Optional[List[str]] = None,
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """
    Extract OpenAI-format `tool_calls` from model output.
    Supports:
    1. `<tool_call>{...}</tool_call>` XML-wrapped blocks (Qwen / Hermes / Llama3)
    2. Markdown fenced JSON `{"name": "...", "arguments": {...}}` or list of calls
    3. Bare JSON object/array matching an allowed tool schema
    """
    if not text or not text.strip():
        return text, []

    found_calls: List[Dict[str, Any]] = []
    remaining = text

    # 1. Explicit <tool_call>...</tool_call> blocks
    matches = list(_TOOL_CALL_TAG_RE.finditer(text))
    if matches:
        for m in matches:
            inner = m.group(1).strip()
            try:
                parsed = json.loads(inner)
            except Exception:
                continue
            items = parsed if isinstance(parsed, list) else [parsed]
            for item in items:
                tc = _parse_single_tool_object(item, allowed_tool_names)
                if tc:
                    found_calls.append(tc)
        if found_calls:
            remaining = _TOOL_CALL_TAG_RE.sub("", text).strip()
            return (remaining or None), found_calls

    # 2. Fallback: fenced JSON or bare JSON (ignoring <think> blocks)
    candidate = _THINK_BLOCK_RE.sub("", text).strip()
    fence_m = _CODE_FENCE_RE.search(candidate)
    json_candidate = fence_m.group(1).strip() if fence_m else candidate

    if json_candidate.startswith("{") or json_candidate.startswith("["):
        try:
            parsed = json.loads(json_candidate)
            items = parsed if isinstance(parsed, list) else [parsed]
            for item in items:
                tc = _parse_single_tool_object(item, allowed_tool_names)
                if tc:
                    found_calls.append(tc)
            if found_calls:
                return None, found_calls
        except Exception:
            pass

    return text, []


def extract_json_mode_content(text: str) -> str:
    """Cleanly unwrap fenced JSON or prose around a JSON object for response_format=json_object."""
    if not text:
        return "{}"
    cleaned = _THINK_BLOCK_RE.sub("", text).strip()
    fence_m = _CODE_FENCE_RE.search(cleaned)
    if fence_m:
        cleaned = fence_m.group(1).strip()
    try:
        parsed = json.loads(cleaned)
        return json.dumps(parsed, ensure_ascii=False)
    except Exception:
        pass

    # Try locating the outermost JSON object or array
    start_obj = cleaned.find("{")
    end_obj = cleaned.rfind("}")
    if start_obj != -1 and end_obj > start_obj:
        sub = cleaned[start_obj : end_obj + 1]
        try:
            parsed = json.loads(sub)
            return json.dumps(parsed, ensure_ascii=False)
        except Exception:
            pass

    start_arr = cleaned.find("[")
    end_arr = cleaned.rfind("]")
    if start_arr != -1 and end_arr > start_arr:
        sub = cleaned[start_arr : end_arr + 1]
        try:
            parsed = json.loads(sub)
            return json.dumps(parsed, ensure_ascii=False)
        except Exception:
            pass

    return cleaned


def transform_completion_response(resp_json: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Post-process a unary OpenAI chat.completion payload for tool_calls or JSON mode."""
    if not isinstance(resp_json, dict):
        return resp_json
    choices = resp_json.get("choices")
    if not isinstance(choices, list) or not choices:
        return resp_json

    out = copy.deepcopy(resp_json)
    first = out["choices"][0]
    msg = first.get("message") if isinstance(first, dict) else None
    if not isinstance(msg, dict):
        return out

    raw_content = str(msg.get("content") or "")
    if meta.get("tools_enabled"):
        rem_content, calls = extract_tool_calls(raw_content, meta.get("allowed_tool_names"))
        if calls:
            msg["content"] = rem_content
            msg["tool_calls"] = calls
            first["finish_reason"] = "tool_calls"
            return out

    if meta.get("json_mode"):
        msg["content"] = extract_json_mode_content(raw_content)

    return out


def build_tool_call_sse_frames(
    full_text: str,
    meta: Dict[str, Any],
    completion_id: str = "",
    model: str = "",
) -> List[str]:
    """
    Convert a complete buffered generation into spec-compliant OpenAI SSE frames
    for streaming `tools` or `json_object` requests.
    """
    cid = completion_id or f"chatcmpl-{int(time.time() * 1000)}"
    created = int(time.time())
    mdl = model or meta.get("model") or "aeromesh-local"
    frames: List[str] = []

    def _emit(delta: Dict[str, Any], finish_reason: Optional[str] = None):
        chunk = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": mdl,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }],
        }
        frames.append(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n")

    if meta.get("tools_enabled"):
        rem, calls = extract_tool_calls(full_text, meta.get("allowed_tool_names"))
        if calls:
            _emit({"role": "assistant", "content": rem})
            for idx, tc in enumerate(calls):
                _emit({
                    "tool_calls": [{
                        "index": idx,
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        },
                    }]
                })
            _emit({}, finish_reason="tool_calls")
            frames.append("data: [DONE]\n\n")
            return frames

    final_content = extract_json_mode_content(full_text) if meta.get("json_mode") else full_text
    _emit({"role": "assistant", "content": final_content})
    _emit({}, finish_reason="stop")
    frames.append("data: [DONE]\n\n")
    return frames


# ============================================================================
# IDE & SDK INTEGRATION SNIPPET GENERATOR
# ============================================================================
def generate_integration_snippets(
    base_url: str,
    api_key_placeholder: str,
    model_name: str,
) -> Dict[str, str]:
    url = str(base_url or "http://127.0.0.1:8080/v1").rstrip("/")
    key = str(api_key_placeholder or "YOUR_AEROMESH_API_KEY")
    model = str(model_name or "Qwen2.5-14B-Instruct-Q4_K_M.gguf")

    curl_snip = (
        f"curl {url}/chat/completions \\\n"
        f"  -H \"Content-Type: application/json\" \\\n"
        f"  -H \"Authorization: Bearer {key}\" \\\n"
        f"  -d '{{\n"
        f"    \"model\": \"{model}\",\n"
        f"    \"temperature\": 0.7,\n"
        f"    \"top_p\": 0.9,\n"
        f"    \"messages\": [\n"
        f"      {{\"role\": \"system\", \"content\": \"You are a concise coding assistant.\"}},\n"
        f"      {{\"role\": \"user\", \"content\": \"Explain zero-weight activation streaming in 3 bullets.\"}}\n"
        f"    ]\n"
        f"  }}'"
    )

    python_snip = (
        "from openai import OpenAI\n\n"
        "client = OpenAI(\n"
        f"    base_url=\"{url}\",\n"
        f"    api_key=\"{key}\",\n"
        ")\n\n"
        "stream = client.chat.completions.create(\n"
        f"    model=\"{model}\",\n"
        "    temperature=0.7,\n"
        "    top_p=0.9,\n"
        "    messages=[\n"
        "        {\"role\": \"system\", \"content\": \"You are a helpful local assistant.\"},\n"
        "        {\"role\": \"user\", \"content\": \"Write a fast Python binary search.\"}\n"
        "    ],\n"
        "    stream=True,\n"
        ")\n\n"
        "for chunk in stream:\n"
        "    print(chunk.choices[0].delta.content or \"\", end=\"\", flush=True)\n"
    )

    tools_json_snip = (
        "from openai import OpenAI\n\n"
        f"client = OpenAI(base_url=\"{url}\", api_key=\"{key}\")\n\n"
        "# 1. Function / Tool Calling\n"
        "resp = client.chat.completions.create(\n"
        f"    model=\"{model}\",\n"
        "    messages=[{\"role\": \"user\", \"content\": \"Check GPU VRAM on node-2\"}],\n"
        "    tools=[{\n"
        "        \"type\": \"function\",\n"
        "        \"function\": {\n"
        "            \"name\": \"get_node_vram\",\n"
        "            \"description\": \"Inspect free VRAM on a cluster node\",\n"
        "            \"parameters\": {\n"
        "                \"type\": \"object\",\n"
        "                \"properties\": {\"node\": {\"type\": \"string\"}},\n"
        "                \"required\": [\"node\"]\n"
        "            }\n"
        "        }\n"
        "    }],\n"
        "    stream=False,\n"
        ")\n"
        "print(resp.choices[0].message.tool_calls)\n\n"
        "# 2. Strict JSON Mode\n"
        "json_resp = client.chat.completions.create(\n"
        f"    model=\"{model}\",\n"
        "    response_format={\"type\": \"json_object\"},\n"
        "    messages=[{\"role\": \"user\", \"content\": \"Return JSON with keys: language, year, creator for Rust.\"}],\n"
        "    stream=False,\n"
        ")\n"
        "print(json_resp.choices[0].message.content)\n"
    )

    continue_snip = (
        "# ~/.continue/config.yaml (Continue.dev for VS Code / JetBrains)\n"
        "name: AeroMESH Local Cluster\n"
        "version: 1.0.0\n"
        "schema: v1\n"
        "models:\n"
        f"  - name: AeroMESH ({model})\n"
        "    provider: openai\n"
        f"    model: {model}\n"
        f"    apiBase: {url}\n"
        f"    apiKey: {key}\n"
        "    roles:\n"
        "      - chat\n"
        "      - edit\n"
        "      - apply\n"
        "    capabilities:\n"
        "      - tool_use\n"
    )

    cline_snip = (
        "// VS Code — Cline / Roo Code Provider Settings\n"
        "{\n"
        "  \"apiProvider\": \"openai\",\n"
        f"  \"openAiBaseUrl\": \"{url}\",\n"
        f"  \"openAiApiKey\": \"{key}\",\n"
        f"  \"openAiModelId\": \"{model}\",\n"
        "  \"openAiModelInfo\": {\n"
        "    \"supportsPromptCache\": false,\n"
        "    \"supportsImages\": false,\n"
        "    \"contextWindow\": 8192\n"
        "  }\n"
        "}"
    )

    cursor_webui_snip = (
        "# Cursor IDE (Settings -> Models -> OpenAI API Key -> Override OpenAI Base URL)\n"
        f"Base URL:  {url}\n"
        f"API Key:   {key}\n"
        f"Model ID:  {model}\n\n"
        "# Open WebUI (Admin Settings -> Connections -> OpenAI API)\n"
        f"URL:       {url}\n"
        f"Key:       {key}\n"
    )

    return {
        "curl": curl_snip,
        "python": python_snip,
        "tools_json": tools_json_snip,
        "continue": continue_snip,
        "cline": cline_snip,
        "cursor": cursor_webui_snip,
    }
