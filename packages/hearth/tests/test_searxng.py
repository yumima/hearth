"""The installer's SearXNG offer, without a container runtime."""

from __future__ import annotations

import subprocess

import pytest

from hearth import config as cfgmod
from hearth import install, searxng

REL = install.Release({"schema": 1, "version": "1", "searxng": {"image": "img@sha256:ab"}}, "https://x/")


@pytest.fixture
def cfg(monkeypatch, tmp_path):
    c = cfgmod.Config(path=tmp_path / "config.yaml")
    saved = []
    monkeypatch.setattr(install.cfgmod, "load", lambda create=False: c)
    monkeypatch.setattr(install.cfgmod, "save", lambda x: saved.append(x.search.searxng_url))
    monkeypatch.setattr(install, "root", lambda: tmp_path)
    c.saved = saved
    return c


def _setup_ok(monkeypatch):
    calls = []
    monkeypatch.setattr(searxng, "runtime", lambda: "/usr/bin/podman")
    monkeypatch.setattr(searxng, "setup", lambda image, folder, say: calls.append(image) or "http://127.0.0.1:8888")
    return calls


def test_fresh_install_asks_and_saves_the_url(monkeypatch, cfg):
    calls = _setup_ok(monkeypatch)
    monkeypatch.setattr(install, "ask", lambda q: True)
    install.offer_searxng(REL, "ask", True, lambda t: None)
    assert calls == ["img@sha256:ab"] and cfg.saved == ["http://127.0.0.1:8888"]


@pytest.mark.parametrize("want,fresh,url,answer", [
    ("no", True, "", True),          # declined by flag
    ("ask", False, "", True),        # an update: not asked again
    ("ask", True, "http://s", True), # already has one
    ("ask", True, "", False),        # said no
    ("ask", True, "", None),         # nobody at the terminal
])
def test_not_set_up(monkeypatch, cfg, want, fresh, url, answer):
    calls = _setup_ok(monkeypatch)
    monkeypatch.setattr(install, "ask", lambda q: answer)
    cfg.search.searxng_url = url
    install.offer_searxng(REL, want, fresh, lambda t: None)
    assert calls == [] and cfg.saved == []


def test_flag_sets_it_up_on_an_update(monkeypatch, cfg):
    calls = _setup_ok(monkeypatch)
    monkeypatch.setattr(install, "ask", lambda q: pytest.fail("asked"))
    install.offer_searxng(REL, "yes", False, lambda t: None)
    assert calls and cfg.saved


def test_no_runtime_is_said_not_asked(monkeypatch, cfg):
    def none():
        raise searxng.Unavailable("no")
    monkeypatch.setattr(searxng, "runtime", none)
    monkeypatch.setattr(install, "ask", lambda q: pytest.fail("asked"))
    said = []
    install.offer_searxng(REL, "ask", True, said.append)
    assert "--searxng" in said[0] and cfg.saved == []


def test_a_failed_setup_does_not_fail_the_install(monkeypatch, cfg):
    monkeypatch.setattr(searxng, "runtime", lambda: "/usr/bin/docker")
    def boom(*a):
        raise searxng.Unavailable("pull failed")
    monkeypatch.setattr(searxng, "setup", boom)
    said = []
    install.offer_searxng(REL, "yes", True, said.append)
    assert "pull failed" in said[-1] and cfg.saved == []


def test_settings_enable_json_and_keep_their_secret(tmp_path):
    p = searxng.write_settings(tmp_path)
    first = p.read_text()
    assert "- json" in first and "limiter: false" in first
    assert searxng.write_settings(tmp_path).read_text() == first


def test_someone_elses_container_of_that_name_is_left_alone(monkeypatch):
    monkeypatch.setattr(searxng.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="\n", stderr=""))
    with pytest.raises(searxng.Unavailable):
        searxng._ours("docker")
    monkeypatch.setattr(searxng, "runtime", lambda: "docker")
    assert searxng.remove() is False


def test_runs_on_loopback_with_the_pinned_image(monkeypatch, tmp_path):
    runs = []
    def run(cmd, **k):
        runs.append(cmd)
        rc = 1 if cmd[1] == "inspect" else 0  # no container yet
        return subprocess.CompletedProcess(cmd, rc, stdout="", stderr="")
    monkeypatch.setattr(searxng, "runtime", lambda: "podman")
    monkeypatch.setattr(searxng.subprocess, "run", run)
    monkeypatch.setattr(searxng, "_free_port", lambda: 8890)
    monkeypatch.setattr(searxng, "answers", lambda url: True)
    assert searxng.setup("img@sha256:ab", tmp_path / "s", lambda t: None) == "http://127.0.0.1:8890"
    started = next(c for c in runs if c[1] == "run")
    assert "127.0.0.1:8890:8080" in started and started[-1] == "img@sha256:ab" and searxng.LABEL in started
