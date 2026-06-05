"""Host-header guard (DNS-rebinding mitigation, design doc §4.7)."""

from fastapi.testclient import TestClient

from hearth import config as cfgmod
from hearth.app import create_app
from tests.test_routes import FakeBackend


def _app(**kw):
    return create_app(cfgmod.default_config(), backends_override={"ollama": FakeBackend()}, **kw)


def test_localhost_host_allowed():
    client = TestClient(_app(), base_url="http://localhost:11435")
    assert client.get("/admin/health").status_code in (200, 503)  # reaches handler


def test_foreign_host_rejected():
    client = TestClient(_app())
    r = client.get("/admin/health", headers={"host": "evil.example.com"})
    assert r.status_code == 403


def test_unsafe_bind_requires_key_passes_with_token():
    client = TestClient(_app(allow_any_host=True, api_key="secret"))
    # Foreign host now allowed, but bearer required.
    assert client.get("/admin/health", headers={"host": "evil.example.com"}).status_code == 401
    r = client.get("/admin/health", headers={"host": "evil.example.com", "authorization": "Bearer secret"})
    assert r.status_code in (200, 503)
