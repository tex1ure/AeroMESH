import pytest
from fastapi.testclient import TestClient
import app as gateway_app


@pytest.fixture
def client():
    """Returns a test client for the FastAPI gateway application."""
    return TestClient(gateway_app.app, base_url="http://testclient")


def test_gateway_health(client):
    """Verify gateway liveness check returns 200 OK with backend URL."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["gateway"] == "ok"
    assert "backend_url" in data


def test_gateway_ready_backend_offline(client):
    """Verify gateway readiness probe returns 503 when Axum coordinator is unreachable."""
    response = client.get("/ready")
    assert response.status_code == 503
    data = response.json()
    assert data["gateway"] == "ok"
    assert data["backend"] == "unreachable"
    assert "backend_url" in data


def test_gateway_ready_backend_online(client, monkeypatch):
    """Verify gateway readiness probe returns 200 when Axum coordinator responds."""
    class MockBackendClient:
        async def get(self, url, headers=None, timeout=None):
            class MockResponse:
                status_code = 200
                headers = {"content-type": "application/json"}
                def json(self):
                    return {"status": "ok"}
            return MockResponse()

    monkeypatch.setattr(gateway_app, "get_backend_client", lambda: MockBackendClient())

    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["gateway"] == "ok"
    assert data["backend"] == "ok"


def test_serve_spa_root(client):
    """Verify root GET / serves static SPA HTML."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "AeroMesh" in response.text or "AeroMESH" in response.text


def test_models_fallback_when_backend_offline(client):
    """Verify /v1/models returns dynamic local file list if backend is offline."""
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data.get("object") == "list"
    assert "data" in data
    assert isinstance(data["data"], list)


def test_cluster_status_fallback_when_backend_offline(client):
    """Verify /api/cluster/status returns graceful offline state, never a 5xx."""
    response = client.get("/api/cluster/status")
    assert response.status_code == 200
    data = response.json()
    assert data.get("state") == "offline"
    assert data.get("connected") is False
    assert "coordinator_endpoint" in data


def test_chat_abort_endpoint(client):
    """Verify /api/chat/abort signals stream cancellation cleanly."""
    response = client.post("/api/chat/abort")
    assert response.status_code == 200
    assert response.json() == {"status": "aborted"}


def test_hop_by_hop_header_filtering():
    """Verify hop-by-hop headers are removed and legitimate headers preserved."""
    headers = {
        "Connection": "keep-alive",
        "Upgrade": "websocket",
        "Content-Length": "1234",
        "X-Custom-Header": "AeroMesh-Cluster",
        "Content-Type": "application/json",
    }
    filtered = gateway_app.filter_headers(headers)
    assert "Connection" not in filtered
    assert "connection" not in filtered
    assert "Upgrade" not in filtered
    assert "Content-Length" not in filtered
    assert filtered.get("X-Custom-Header") == "AeroMesh-Cluster"
    assert filtered.get("Content-Type") == "application/json"


def test_unreachable_backend_returns_502(client):
    """Verify unhandled API proxy routes return 502 with structured error when backend is offline."""
    response = client.get("/api/nonexistent-route")
    assert response.status_code == 502
    data = response.json()
    assert data.get("error") == "backend_unreachable"
