"""hearth's own SearXNG: installed with every release, used when none is set."""

from __future__ import annotations

import asyncio

import pytest

from hearth import install, searxng, websearch
from hearth.config import SearchConfig
from hearth.provision import Problem

@pytest.fixture(autouse=True)
def _unix(monkeypatch):
    """add_searxng skips Windows; every test but test_not_on_windows is of the rest."""
    monkeypatch.setattr(install, "_WIN", False)


FILES = {"searxng-wheel": {"name": "searxng.whl", "sha256": "ab"}, "searxng-requirements": {"name": "r.txt", "sha256": "cd"}}


def test_installed_when_the_release_has_it(monkeypatch):
    built = []
    monkeypatch.setattr(install, "build_env", lambda rel, name, say, replacing: built.append(name))
    rel = install.Release({"schema": 1, "version": "1", "files": FILES}, "https://x/")
    assert install.add_searxng(rel, lambda t: None, lambda: None) and built == ["searxng"]


def test_an_older_release_without_it_is_fine(monkeypatch):
    monkeypatch.setattr(install, "build_env", lambda *a: pytest.fail("built"))
    rel = install.Release({"schema": 1, "version": "1", "files": {}}, "https://x/")
    assert install.add_searxng(rel, lambda t: None, lambda: None) is False


def test_a_failed_install_does_not_fail_hearths(monkeypatch):
    def boom(*a):
        raise Problem("Installing what searxng needs failed:\nno wheel for lxml")
    monkeypatch.setattr(install, "build_env", boom)
    said = []
    rel = install.Release({"schema": 1, "version": "1", "files": FILES}, "https://x/")
    assert install.add_searxng(rel, said.append, lambda: None) is False
    assert "SearXNG could not be installed" in said[0] and "\n" not in said[0]


def test_settings_are_loopback_json_and_keep_their_secret(tmp_path):
    first = searxng.write_settings(tmp_path, 8890).read_text()
    assert 'bind_address: "127.0.0.1"' in first and "port: 8890" in first and "- json" in first
    secret = first.split('secret_key: "')[1][:64]
    again = searxng.write_settings(tmp_path, 8891).read_text()
    assert secret in again and "port: 8891" in again


def test_no_environment_means_not_installed(monkeypatch, tmp_path):
    monkeypatch.setattr(searxng.paths, "data_dir", lambda: tmp_path)
    assert searxng.python() is None
    e = searxng.env()
    (e / "bin").mkdir(parents=True)
    (e / "bin" / "python").write_text("")
    assert searxng.python() is None  # not .complete
    (e / ".complete").write_text("")
    if not searxng.sys.platform == "win32":
        assert searxng.python() == e / "bin" / "python"


def _chain(monkeypatch, own: str, has_env: bool = True):
    """Search with no providers but SearXNG, recording which URL it was asked."""
    asked = []

    async def fake(client, query, count, base):
        asked.append(base)
        return [websearch.Result("t", "https://example.com/", "s", "", "searxng")]

    async def up():
        return own

    monkeypatch.setattr(websearch, "_searxng", fake)
    monkeypatch.setattr(searxng, "python", lambda: "py" if has_env else None)
    monkeypatch.setattr(searxng, "ensure_up", up)
    websearch._cache.clear()
    return asked


def test_search_uses_hearths_own_when_none_is_set(monkeypatch):
    asked = _chain(monkeypatch, "http://127.0.0.1:8890")
    res = asyncio.run(websearch.search("q", SearchConfig(provider="searxng")))
    assert asked == ["http://127.0.0.1:8890"] and res[0].source == "searxng"


def test_a_set_url_wins(monkeypatch):
    asked = _chain(monkeypatch, "http://127.0.0.1:8890")
    asyncio.run(websearch.search("q", SearchConfig(provider="searxng", searxng_url="http://mine:1")))
    assert asked == ["http://mine:1"]


@pytest.mark.parametrize("own,has_env,said", [("", True, "did not start"), ("", False, "not installed")])
def test_without_it_the_chain_says_why(monkeypatch, own, has_env, said):
    _chain(monkeypatch, own, has_env)
    with pytest.raises(websearch.SearchError, match=said):
        asyncio.run(websearch.search("q", SearchConfig(provider="searxng")))


def test_results_without_a_host_are_dropped():
    import httpx

    body = {"results": [{"url": "http:///go/v1/abc", "title": "brave redirect"},
                        {"url": "https://kernel.org/", "title": "kernel"}]}

    async def go():
        transport = httpx.MockTransport(lambda r: httpx.Response(200, json=body))
        async with httpx.AsyncClient(transport=transport) as c:
            return await websearch._searxng(c, "q", 5, "http://s")

    assert [r.url for r in asyncio.run(go())] == ["https://kernel.org/"]


def test_a_start_that_failed_is_not_tried_on_every_search(monkeypatch, tmp_path):
    starts = []

    class Dead:
        pid, returncode = 1, 1
        def poll(self):
            return 1

    monkeypatch.setattr(searxng, "python", lambda: tmp_path / "python")
    monkeypatch.setattr(searxng, "folder", lambda: tmp_path)
    monkeypatch.setattr(searxng.paths, "home", lambda: tmp_path)
    monkeypatch.setattr(searxng.subprocess, "Popen", lambda *a, **k: starts.append(a) or Dead())
    monkeypatch.setattr(searxng, "_proc", None)
    monkeypatch.setattr(searxng, "_failed_at", -searxng._RETRY)
    monkeypatch.setattr(searxng, "_lock", asyncio.Lock())

    async def twice():
        return await searxng.ensure_up(), await searxng.ensure_up()

    assert asyncio.run(twice()) == ("", "") and len(starts) == 1


def test_not_on_windows(monkeypatch):
    monkeypatch.setattr(install, "_WIN", True)
    monkeypatch.setattr(install, "build_env", lambda *a: pytest.fail("built"))
    rel = install.Release({"schema": 1, "version": "1", "files": FILES}, "https://x/")
    assert install.add_searxng(rel, lambda t: None, lambda: None) is False
