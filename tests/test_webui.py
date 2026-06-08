"""The chat UI route — GET /app serves the bundled, self-contained HTML, and /
redirects to it. The UI is served by the gateway so it is same-origin with the
API and inherits the localhost-only security posture."""

from __future__ import annotations

from fastapi.testclient import TestClient

from hearth import config as cfgmod
from hearth.app import create_app


def _client(**kw):
    return TestClient(create_app(cfgmod.default_config(), allow_any_host=True, **kw))


def test_app_serves_chat_ui():
    r = _client().get("/app")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "<title>Hearth</title>" in r.text
    # Same-origin wiring + self-contained (no CDN / external assets), since the
    # engine is loopback-only with zero outbound.
    assert "/v1/chat/completions" in r.text
    assert 'src="http' not in r.text  # no external scripts/images
    assert "<link" not in r.text      # no external stylesheets — all CSS is inline


def test_root_redirects_to_app():
    r = _client().get("/", follow_redirects=False)
    assert r.status_code in (307, 308)
    assert r.headers["location"] == "/app"


def test_app_respects_localhost_only_host_check():
    # Default (loopback) security: /app serves to a localhost Host and rejects
    # anything else — same posture as the API routes (DNS-rebinding defense).
    app = create_app(cfgmod.default_config())  # allow_any_host=False
    assert TestClient(app, base_url="http://localhost").get("/app").status_code == 200
    assert TestClient(app, base_url="http://evil.example.com").get("/app").status_code == 403
