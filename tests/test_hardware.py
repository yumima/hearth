from hearth import hardware
from hearth.hardware import Gpu, Hardware


def test_probe_returns_shape():
    hw = hardware.probe()
    assert hw.cpu_cores is None or hw.cpu_cores > 0
    assert isinstance(hw.gpus, list)


def test_as_dict_has_inference_target():
    d = hardware.as_dict()
    assert "inference_target" in d
    assert "gpus" in d and "ram_total_mib" in d


# ── recommend_roles(): the VRAM-fit model picker ────────────────────────────
# Pins the calibration so a tweak can't silently re-introduce the bug where a
# model that overflows VRAM (30B on a 12 GB GPU) gets picked and runs on CPU.

def _hw(vram_mib=None, ram_mib=32000) -> Hardware:
    gpus = [Gpu("nvidia", "test-gpu", vram_mib, "550", "cuda")] if vram_mib else []
    return Hardware(cpu_cores=8, ram_total_mib=ram_mib, ram_available_mib=ram_mib, gpus=gpus)


def _rec(monkeypatch, vram_mib=None, ram_mib=32000):
    monkeypatch.setattr(hardware, "probe", lambda: _hw(vram_mib, ram_mib))
    return hardware.recommend_roles()


def test_12gb_gpu_picks_14b_not_30b(monkeypatch):
    # The headline regression: 30B overflows a 12 GB GPU -> must NOT be picked.
    assert _rec(monkeypatch, vram_mib=12227)["primary_chat"] == "qwen3:14b"


def test_24gb_gpu_picks_30b_moe(monkeypatch):
    assert _rec(monkeypatch, vram_mib=24576)["primary_chat"] == "qwen3:30b-a3b"


def test_32gb_gpu_picks_32b(monkeypatch):
    assert _rec(monkeypatch, vram_mib=32768)["primary_chat"] == "qwen3:32b"


def test_8gb_gpu_picks_8b(monkeypatch):
    assert _rec(monkeypatch, vram_mib=8192)["primary_chat"] == "qwen3:8b"


def test_thresholds_carry_kv_headroom(monkeypatch):
    # A GPU sized to a model's bare weights must drop a tier (room for KV cache).
    assert _rec(monkeypatch, vram_mib=20000)["primary_chat"] == "qwen3:14b"   # not 30b (~20GB)
    assert _rec(monkeypatch, vram_mib=22000)["primary_chat"] == "qwen3:14b"   # not 32b (~22GB)


def test_cpu_only_sized_by_ram(monkeypatch):
    assert _rec(monkeypatch, vram_mib=None, ram_mib=32000)["primary_chat"] == "qwen3:8b"
    assert _rec(monkeypatch, vram_mib=None, ram_mib=16000)["primary_chat"] == "qwen3:4b"
    assert _rec(monkeypatch, vram_mib=None, ram_mib=8000)["primary_chat"] == "qwen3:1.7b"


def test_embedding_and_fast_roles_stable(monkeypatch):
    r = _rec(monkeypatch, vram_mib=12227)
    assert r["embedding"] == "nomic-embed-text"
    assert r["fast_chat"] == "qwen3:8b" and r["coding"] == "qwen3:8b"
