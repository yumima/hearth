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


def test_12gb_gpu_picks_the_build_that_fits(monkeypatch):
    # The headline regression: the full 27B spills on 12 GB (about 3 tok/s).
    for vram in (11000, 12227, 16384, 19999):
        r = _rec(monkeypatch, vram_mib=vram)
        assert r["primary_chat"] == r["fast_chat"] == r["coding"] == hardware.QWEN38_2BIT


def test_20gb_and_up_picks_the_full_model(monkeypatch):
    for vram in (20000, 24576, 32768):
        assert _rec(monkeypatch, vram_mib=vram)["primary_chat"] == "qwen3.8:27b"


def test_below_11gb_still_fits_vram(monkeypatch):
    assert _rec(monkeypatch, vram_mib=10999)["primary_chat"] == "qwen3:8b"


def test_8gb_gpu_picks_8b(monkeypatch):
    assert _rec(monkeypatch, vram_mib=8192)["primary_chat"] == "qwen3:8b"


def test_cpu_only_sized_by_ram(monkeypatch):
    assert _rec(monkeypatch, vram_mib=None, ram_mib=32000)["primary_chat"] == "qwen3:8b"
    assert _rec(monkeypatch, vram_mib=None, ram_mib=16000)["primary_chat"] == "qwen3:4b"
    assert _rec(monkeypatch, vram_mib=None, ram_mib=8000)["primary_chat"] == "qwen3:1.7b"


def test_embedding_and_fast_roles_stable(monkeypatch):
    r = _rec(monkeypatch, vram_mib=12227)
    assert r["embedding"] == "nomic-embed-text"
    assert r["fast_chat"] == hardware.QWEN38_2BIT and r["coding"] == hardware.QWEN38_2BIT


# ── recommend_contexts(): the largest context that stays 100% on the GPU ────

def test_2bit_context_on_12gb_is_the_measured_fit():
    # 40960 measured fully on-GPU on this card; 48k and 64k spilled to the CPU.
    assert hardware.recommend_contexts({"primary_chat": hardware.QWEN38_2BIT}, 12227) == {"primary_chat": 40960}


def test_qwen38_fits_a_large_card_within_its_tested_ceiling():
    assert hardware.recommend_contexts({"primary_chat": "qwen3.8:27b"}, 32768) == {"primary_chat": 65536}


def test_qwen38_context_is_sized_to_a_mid_card():
    # 20 GB holds the weights, so size the context to stay on the GPU.
    assert hardware.recommend_contexts({"primary_chat": "qwen3.8:27b"}, 20000) == {"primary_chat": 24576}


def test_model_too_big_for_the_card_is_left_to_the_daemon():
    assert hardware.recommend_contexts({"primary_chat": "qwen3.8:27b"}, 12227) == {}


def test_context_ceiling_is_per_model():
    # Opening a 64k step for Qwen 3.8 must not push Qwen3 past its 40960 limit.
    assert hardware.recommend_contexts({"primary_chat": "qwen3:0.6b"}, 32768) == {"primary_chat": 40960}
