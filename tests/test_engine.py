"""What tests/test_install.py stands in for: the real Ollama of the release, from
its own servers, and a small model — downloaded, checked, assigned, asked, removed.

Run with HEARTH_E2E_OLLAMA=1 (CI does): 2 to 4 GB of downloads.
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest
from test_install import ROOT, World, _until

MODEL = "qwen2.5:0.5b"

pytestmark = [pytest.mark.skipif(not os.environ.get("HEARTH_E2E_OLLAMA"), reason="HEARTH_E2E_OLLAMA is not set"),
              pytest.mark.timeout(2400)]


class Real(World):
    def _pins(self) -> Path:
        return ROOT / "installer" / "pins.json"


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    w = Real(tmp_path_factory.mktemp("r"))
    yield w
    w.close()


def _gateway(world, path: str, body: dict) -> dict:
    r = httpx.post(f"http://127.0.0.1:{world.port}{path}", json=body, timeout=600)
    assert r.status_code == 200, r.text
    return r.json()


def test_01_installed_with_the_ollama_of_the_release(world):
    world.release("0.2.0")
    world.install("0.2.0", "--no-open", "--no-client", "--no-autostart")
    assert world.version() == "0.2.0"
    v = httpx.get(f"http://127.0.0.1:{world.ollama_port}/api/version", timeout=10).json()["version"]
    assert f'"{v}"' in (ROOT / "installer" / "pins.json").read_text()


def test_02_a_model_is_downloaded_checked_and_assigned(world):
    # (It is small: what it cannot do is not what is tested.)
    r = world.hearth("model", "install", MODEL, "--anyway", ok=None)
    print(r.stdout[-1500:], r.stderr[-1500:])
    assert r.returncode == 0
    assert (world.data / "models" / "blobs").is_dir()
    assert MODEL in world.hearth("model", "list").stdout


def test_03_it_answers_by_role(world):
    out = _gateway(world, "/v1/chat/completions", {
        "model": "primary_chat", "messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 20})
    assert out["choices"][0]["message"]["content"].strip()
    assert len(_gateway(world, "/v1/embeddings", {"model": "embedding", "input": "hello"})["data"][0]["embedding"]) > 100


def test_04_stopping_stops_its_ollama(world):
    world.hearth("stop")
    assert _until(lambda: world.processes() == []), world.processes()
    world.hearth("start")
    assert world.version() == "0.2.0" and MODEL in world.hearth("model", "list").stdout


def test_05_removing_leaves_nothing(world):
    world.hearth("uninstall", "--yes")
    assert _until(lambda: not world.data.exists(), 90), list(world.data.iterdir())
    assert _until(lambda: world.processes() == []), world.processes()
