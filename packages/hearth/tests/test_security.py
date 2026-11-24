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


def test_passed_through_responses_carry_one_hearth_server_header():
    from hearth import __version__

    client = TestClient(_app(), base_url="http://localhost:11435")
    r = client.get("/admin/health")
    assert r.headers.get_list("server") == [f"hearth/{__version__}"]


def test_rejections_are_json_errors():
    r = TestClient(_app()).get("/admin/health", headers={"host": "evil.example.com"})
    assert r.json()["error"]["type"] == "forbidden"


# ── cross-site requests ───────────────────────────────────────────────────────

def _post(origin=None, host="127.0.0.1:11435", path="/admin/models/pull", **kw):
    headers = {"host": host, "content-type": "text/plain"}  # what a "simple" request may send
    if origin is not None:
        headers["origin"] = origin
    return TestClient(_app(**kw)).post(path, content=b"{}", headers=headers)


def test_another_site_cannot_post():
    for origin in ("https://evil.example", "http://evil.example:11435", "null", "",
                   "http://127.0.0.1:3000",        # another local page is another site
                   "http://localhost:11435",       # not the name this request used
                   "http://127.0.0.1:11435.evil.example", "http://user@127.0.0.1:11435",
                   "http://[", "file://"):
        r = _post(origin)
        assert r.status_code == 403, (origin, r.text)
        assert "cross-origin" in r.json()["error"]["message"]


def test_our_own_page_and_plain_clients_can_post():
    # 400 = reached the handler, which wants a model name.
    for host in ("127.0.0.1:11435", "localhost:11435", "[::1]:11435"):
        assert _post(None, host).status_code == 400
        assert _post(f"http://{host}", host).status_code == 400


def test_cross_origin_is_refused_with_a_key_too():
    r = _post("https://evil.example", "box.lan", allow_any_host=True, api_key="k")
    assert r.status_code == 403
    assert "cross-origin" in r.json()["error"]["message"]


def test_a_second_origin_header_does_not_slip_past():
    r = TestClient(_app()).post("/admin/models/pull", content=b"{}", headers=[
        ("host", "127.0.0.1:11435"), ("origin", "http://127.0.0.1:11435"),
        ("origin", "https://evil.example")])
    assert r.status_code == 403


def test_another_sites_image_or_frame_is_refused_but_a_followed_link_is_not():
    c = TestClient(_app(), base_url="http://127.0.0.1:11435")
    get = lambda path, **h: c.get(path, headers={k.replace("_", "-"): v for k, v in h.items()}).status_code  # noqa: E731
    for site in ("cross-site", "same-site"):
        assert get("/admin/health", sec_fetch_site=site, sec_fetch_mode="no-cors", sec_fetch_dest="image") == 403
        assert get("/app", sec_fetch_site=site, sec_fetch_mode="navigate", sec_fetch_dest="iframe") == 403
        assert get("/app", sec_fetch_site=site, sec_fetch_mode="navigate", sec_fetch_dest="document") == 200
    for site in ("same-origin", "none"):  # our page; a typed address
        assert get("/admin/health", sec_fetch_site=site, sec_fetch_mode="cors", sec_fetch_dest="empty") in (200, 503)
