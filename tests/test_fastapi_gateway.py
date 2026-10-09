"""Integration tests for the FastAPI gateway (app.py).

These exercise the gateway in-process against a *mocked* backend client so they
do not depend on a coordinator actually listening on 127.0.0.1:8080.
"""

import asyncio
import json
import os

import pytest
from fastapi.testclient import TestClient

import app as gateway_app


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
@pytest.fixture
def client():
    """A test client for the gateway.

    `base_url` uses the host `testclient`, which `verify_gateway_access` treats
    as loopback -- exactly like a real browser on 127.0.0.1.
    """
    return TestClient(gateway_app.app, base_url="http://testclient")


@pytest.fixture
def remote_client():
    """A client whose source address is *not* loopback."""
    return TestClient(gateway_app.app, base_url="http://192.0.2.10")


class MockResponse:
    def __init__(self, status_code=200, payload=None, headers=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.headers = headers or {"content-type": "application/json"}
        self.text = text
        self.closed = False

    def json(self):
        return self._payload

    async def aread(self):
        return json.dumps(self._payload).encode("utf-8")

    async def aclose(self):
        self.closed = True

    async def aiter_raw(self):
        yield json.dumps(self._payload).encode("utf-8")

    async def aiter_text(self):
        yield json.dumps(self._payload)


class MockBackendClient:
    """Minimal stand-in for httpx.AsyncClient covering the calls app.py makes."""

    def __init__(self, get_response=None, post_response=None):
        self._get = get_response or MockResponse(200, {"status": "ok"})
        self._post = post_response or MockResponse(200, {"status": "ok"})
        self.sent_headers = []

    async def get(self, url, headers=None, timeout=None):
        self.sent_headers.append(headers)
        return self._get

    async def post(self, url, json=None, headers=None, timeout=None):
        self.sent_headers.append(headers)
        return self._post

    def build_request(self, **kwargs):
        return kwargs

    async def send(self, request, stream=False):
        return self._get

    def stream(self, *args, **kwargs):
        raise AssertionError("stream() is not used by these tests")


def _patch_backend(monkeypatch, **kwargs):
    mock = MockBackendClient(**kwargs)
    monkeypatch.setattr(gateway_app, "get_backend_client", lambda: mock)
    return mock


# ---------------------------------------------------------------------------
# Liveness / readiness
# ---------------------------------------------------------------------------
def test_gateway_health(client):
    """Gateway liveness returns 200 with the configured backend URL."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["gateway"] == "ok"
    assert "backend_url" in data


def test_gateway_ready_backend_offline(client):
    """Readiness reports 503 when the coordinator is unreachable."""
    response = client.get("/ready")
    assert response.status_code == 503
    data = response.json()
    assert data["gateway"] == "ok"
    assert data["backend"] == "unreachable"
    assert "backend_url" in data


def test_gateway_ready_backend_online(client, monkeypatch):
    """Readiness reports 200 when the coordinator answers."""
    _patch_backend(monkeypatch)
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["gateway"] == "ok"
    assert data["backend"] == "ok"


# ---------------------------------------------------------------------------
# SPA hosting
# ---------------------------------------------------------------------------
def test_serve_spa_root(client):
    """GET / serves the dashboard HTML."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    # Assert on markup that only the real file contains, so a missing
    # index.html (which returns an error stub) cannot pass this test.
    assert 'id="chat-viewport"' in response.text
    assert "<script" in response.text


def test_static_assets_are_served(client):
    """The stylesheet mounts under /static."""
    response = client.get("/static/css/liquid_glass.css")
    assert response.status_code == 200
    assert "text/css" in response.headers.get("content-type", "")


# ---------------------------------------------------------------------------
# Graceful offline fallbacks
# ---------------------------------------------------------------------------
def test_models_fallback_when_backend_offline(client):
    """Offline /v1/models falls back to a local scan with the documented schema."""
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data.get("object") == "list"
    assert isinstance(data["data"], list)

    for entry in data["data"]:
        # Shape promised by API_REFERENCE.md.
        assert set(entry) >= {"id", "object", "created", "owned_by", "is_active"}
        assert entry["object"] == "model"
        # Absolute local paths must not be disclosed.
        assert os.sep not in entry["id"]
        # Internal partition files must never be offered as loadable models.
        assert "_stage1.gguf" not in entry["id"]
        assert "_stage2.gguf" not in entry["id"]


def test_cluster_status_fallback_when_backend_offline(client):
    """Offline /api/cluster/status returns a structured payload, never a 5xx."""
    response = client.get("/api/cluster/status")
    assert response.status_code == 200
    data = response.json()
    assert data.get("state") == "offline"
    assert data.get("connected") is False
    assert "coordinator_endpoint" in data


def test_slice_file_detection():
    """Both partition naming schemes are recognised as internal."""
    is_slice = gateway_app._is_slice_file
    assert is_slice("DS_stage1.gguf")
    assert is_slice("DS_stage2.gguf")
    assert is_slice("DS_stage1_of_3.gguf")
    assert is_slice("DS_stage2_of_3.gguf")
    assert not is_slice("DS.gguf")
    assert not is_slice("qwen2.5-0.5b-instruct.gguf")
    # A model that merely mentions "stage" is not a slice.
    assert not is_slice("stagehand-model.gguf")


# ---------------------------------------------------------------------------
# Chat abort
# ---------------------------------------------------------------------------
def test_chat_abort_endpoint(client):
    """Abort reports the outcome and the upstream status."""
    response = client.post("/api/chat/abort")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "aborted"
    assert "backend_status" in data


# ---------------------------------------------------------------------------
# Header handling
# ---------------------------------------------------------------------------
def test_hop_by_hop_header_filtering():
    """Hop-by-hop headers are dropped; end-to-end headers survive."""
    headers = {
        "Connection": "keep-alive",
        "Upgrade": "websocket",
        "Content-Length": "1234",
        "X-Custom-Header": "AeroMesh-Cluster",
        "Content-Type": "application/json",
    }
    filtered = dict(gateway_app.filter_headers(headers))
    assert "Connection" not in filtered
    assert "connection" not in filtered
    assert "Upgrade" not in filtered
    assert "Content-Length" not in filtered
    assert filtered["X-Custom-Header"] == "AeroMesh-Cluster"
    assert filtered["Content-Type"] == "application/json"


def test_filter_headers_preserves_duplicates():
    """Repeated response headers must not collapse (e.g. multiple Set-Cookie)."""
    class MultiHeaders:
        def items(self):
            return [("Set-Cookie", "a=1"), ("Set-Cookie", "b=2"), ("Content-Type", "text/plain")]

    filtered = gateway_app.filter_headers(MultiHeaders())
    cookies = [v for k, v in filtered if k == "Set-Cookie"]
    assert cookies == ["a=1", "b=2"]


# ---------------------------------------------------------------------------
# Security: gateway access control
# ---------------------------------------------------------------------------
def _make_request(client_host, headers=None, body=b""):
    """Builds a bare Starlette Request with a chosen client address.

    `TestClient` always reports the host as `testclient`, so a transport-level
    test cannot exercise the non-loopback branch. Calling the dependency
    directly with a synthetic scope is the reliable way to cover it.

    A `receive` channel is provided so `await request.body()` works too, which
    lets the same helper drive `proxy_request`.
    """
    from starlette.requests import Request

    raw_headers = [
        (k.lower().encode("latin-1"), str(v).encode("latin-1"))
        for k, v in (headers.items() if hasattr(headers, "items") else (headers or []))
    ]
    if body and not any(k == b"content-length" for k, _ in raw_headers):
        raw_headers.append((b"content-length", str(len(body)).encode("latin-1")))

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/api/cluster/status",
        "headers": raw_headers,
        "query_string": b"",
        "client": (client_host, 51234),
        "server": ("127.0.0.1", 7860),
        "scheme": "http",
    }

    messages = [{"type": "http.request", "body": body, "more_body": False}]

    async def receive():
        if messages:
            return messages.pop(0)
        return {"type": "http.disconnect"}

    return Request(scope, receive)


@pytest.mark.parametrize("configured_key", ["", "s3cret"])
def test_non_loopback_refused_without_gateway_key(monkeypatch, configured_key):
    """A non-loopback client is refused when no gateway key is configured.

    Regression test for the guard that previously returned True unconditionally,
    which left the loopback check dead and the API open to the whole network.
    """
    monkeypatch.setattr(gateway_app, "GATEWAY_API_KEY", configured_key)
    with pytest.raises(gateway_app.HTTPException) as exc:
        asyncio.run(gateway_app.verify_gateway_access(_make_request("192.0.2.10")))
    assert exc.value.status_code == 401
    assert exc.value.detail["error"]["code"] == "invalid_api_key"


def test_non_loopback_allowed_with_valid_gateway_key(monkeypatch):
    """A non-loopback client presenting the configured key is allowed."""
    monkeypatch.setattr(gateway_app, "GATEWAY_API_KEY", "s3cret")

    ok = asyncio.run(
        gateway_app.verify_gateway_access(
            _make_request("192.0.2.10", {"Authorization": "Bearer s3cret"})
        )
    )
    assert ok is True

    with pytest.raises(gateway_app.HTTPException):
        asyncio.run(
            gateway_app.verify_gateway_access(
                _make_request("192.0.2.10", {"Authorization": "Bearer wrong"})
            )
        )


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "testclient"])
def test_loopback_allowed_without_gateway_key(monkeypatch, host):
    """Loopback clients never need the gateway key."""
    monkeypatch.setattr(gateway_app, "GATEWAY_API_KEY", "")
    assert asyncio.run(gateway_app.verify_gateway_access(_make_request(host))) is True


def test_caller_authorization_is_replaced_not_duplicated(monkeypatch):
    """A caller-supplied Authorization must never reach the coordinator.

    Regression test: the gateway previously merged headers by name, so the
    caller's Authorization survived alongside the injected one, and the
    coordinator (which reads the first) honoured the caller's value.
    """
    # get_auth_headers() reads the environment on each call.
    monkeypatch.setenv("AEROMESH_API_KEY", "real-key")

    headers = gateway_app.build_upstream_headers({
        "Authorization": "Bearer attacker-supplied",
        "authorization": "Bearer also-attacker",
        "Content-Type": "application/json",
    })

    authorization_values = [v for k, v in headers if k.lower() == "authorization"]
    assert authorization_values == ["Bearer real-key"], (
        f"expected exactly the injected key, got {authorization_values}"
    )
    # Ordinary headers still pass through.
    assert ("Content-Type", "application/json") in headers


def test_proxy_request_uses_injected_key(monkeypatch):
    """End-to-end through proxy_request: only the gateway's key goes upstream."""
    captured = {}

    class CapturingClient(MockBackendClient):
        def build_request(self, **kwargs):
            captured["headers"] = kwargs.get("headers")
            return kwargs

        async def send(self, request, stream=False):
            return MockResponse(200, {"ok": True})

    monkeypatch.setattr(gateway_app, "get_backend_client", lambda: CapturingClient())
    monkeypatch.setenv("AEROMESH_API_KEY", "real-key")

    request = _make_request("127.0.0.1", {"Authorization": "Bearer attacker"})
    asyncio.run(gateway_app.proxy_request(request, "/v1/models"))

    assert "headers" in captured
    values = [v for k, v in captured["headers"] if k.lower() == "authorization"]
    assert values == ["Bearer real-key"]


# ---------------------------------------------------------------------------
# Request validation / limits
# ---------------------------------------------------------------------------
def test_oversized_body_is_rejected(client, monkeypatch):
    """Bodies over the configured cap are refused with 413, not buffered."""
    monkeypatch.setattr(gateway_app, "MAX_REQUEST_BYTES", 64)
    response = client.post("/api/chat/stream", json={"messages": [{"role": "user", "content": "x" * 500}]})
    assert response.status_code == 413
    assert response.json()["detail"]["error"]["code"] == "payload_too_large"


def test_invalid_json_returns_400(client):
    """A malformed JSON body yields the documented 400 envelope."""
    response = client.post(
        "/api/chat/stream",
        content=b"{not json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 400
    assert response.json()["detail"]["error"]["code"] == "invalid_json"


def test_model_switch_requires_model_field(client):
    """Model switch validates its payload instead of proxying anything."""
    response = client.post("/api/model/switch", json={})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "missing_required_field"


def test_unreachable_backend_returns_502(client):
    """An unmapped proxy route returns a structured 502 when the backend is down."""
    response = client.get("/api/nonexistent-route")
    assert response.status_code == 502
    assert response.json()["error"] == "backend_unreachable"


def test_backend_error_detail_is_not_leaked(client, monkeypatch):
    """Upstream exception text must not be echoed to the client."""
    class ExplodingClient(MockBackendClient):
        def build_request(self, **kwargs):
            raise RuntimeError("secret internal path C:\\Users\\someone\\models")

    monkeypatch.setattr(gateway_app, "get_backend_client", lambda: ExplodingClient())
    response = client.get("/api/some-route")
    assert response.status_code == 502
    body = json.dumps(response.json())
    assert "secret internal path" not in body


# ---------------------------------------------------------------------------
# Front-end asset contracts
# ---------------------------------------------------------------------------
def test_multichat_concurrency_assets():
    """The dashboard still ships the multi-chat concurrency UI."""
    root = os.path.join(os.path.dirname(__file__), "..")
    app_js_path = os.path.join(root, "static", "js", "app.js")
    css_path = os.path.join(root, "static", "css", "liquid_glass.css")

    with open(app_js_path, "r", encoding="utf-8") as f:
        js_content = f.read()
    assert "MAX_CONCURRENT_CHATS = 2" in js_content
    assert "activeStreams" in js_content
    assert "showConcurrencyWarning" in js_content
    assert "chat-running-badge" in js_content

    with open(css_path, "r", encoding="utf-8") as f:
        css_content = f.read()
    assert ".chat-running-badge" in css_content
    assert ".concurrency-toast" in css_content


def test_no_unsanitized_markdown_sink():
    """Regression guard: rendered markdown must pass through the sanitizer.

    A raw `innerHTML = marked.parse(...)` is an XSS sink for model output.
    """
    root = os.path.join(os.path.dirname(__file__), "..")
    with open(os.path.join(root, "static", "js", "app.js"), "r", encoding="utf-8") as f:
        js = f.read()

    assert "DOMPurify" in js, "app.js must sanitize rendered markdown"
    assert "innerHTML = marked.parse(" not in js, "unsanitized marked.parse into innerHTML"
