"""Which models each computer is offered. Pins the calibration, so a change
cannot bring back a model that overflows the GPU's memory."""

from __future__ import annotations

import json
import re

import pytest

from hearth import catalog
from hearth.hardware import Gpu, Hardware, _apple

BIT2, FULL = "hf.co/unsloth/Qwen3.8-27B-GGUF:UD-IQ2_S", "qwen3.8:27b"


def _hw(vram=None, ram=32000, disk=500_000, vendor="nvidia", **kw):
    gpus = [Gpu(vendor, "test", vram, vram, None, "cuda" if vendor == "nvidia" else "rocm")] if vram else []
    return Hardware("linux", "x86_64", "cpu", 8, kw.pop("avx2", True), ram, ram, disk, gpus, **kw)


def _mac(gb):
    return Hardware("macos", "arm64", "Apple M4", 10, None, gb * 1024, gb * 700, 500_000, _apple(gb * 1024, "Apple M4"))


def _pick(hw):
    return next((o.model for o in catalog.offers(hw) if o.recommended), None)


# ── what is recommended ───────────────────────────────────────────────────────

@pytest.mark.parametrize("vram,model", [
    (24564, FULL), (32768, FULL), (48000, FULL),
    (16384, BIT2), (12282, BIT2), (12227, BIT2), (11264, BIT2),
    (10240, "qwen3.5:9b"), (8188, "qwen3.5:9b"),
    (6144, "qwen3.5:4b"), (4608, "qwen3.5:4b"), (4096, "qwen3.5:4b"), (3072, "qwen3.5:2b"),
])
def test_a_gpu_gets_the_best_model_that_stays_on_it(vram, model):
    for vendor in ("nvidia", "amd"):
        hw = _hw(vram, vendor=vendor)
        assert _pick(hw) == model
        o = catalog.offers(hw)[0]
        assert o.model == model and o.fit == "gpu" and o.context >= catalog.LEAST_CONTEXT


@pytest.mark.parametrize("gb,model", [(64, FULL), (48, FULL), (36, BIT2), (24, BIT2), (16, "qwen3.5:9b")])
def test_a_mac_is_sized_by_its_share_of_the_memory(gb, model):
    assert _pick(_mac(gb)) == model


def test_the_27b_is_never_recommended_where_it_spills():
    """About 3 tokens a second on 12 GB, 27 for its 2-bit build."""
    for vram in range(4000, 19000, 500):
        assert _pick(_hw(vram)) != FULL
    spills = {o.model: o for o in catalog.offers(_hw(12227))}[FULL]
    assert spills.fit == "spills" and spills.context == 0 and "very slowly" in spills.note


def test_two_small_gpus_are_not_one_large_one():
    hw = _hw(12288)
    hw.gpus = hw.gpus * 2
    assert _pick(hw) == BIT2


def test_without_a_gpu_the_processor_takes_a_small_one_or_none():
    assert _pick(_hw(ram=64000)) == _pick(_hw(ram=16000)) == "qwen3.5:4b"
    for hw in (_hw(ram=8000), _hw(ram=14000), _hw(ram=64000, avx2=False)):
        assert _pick(hw) is None and catalog.verdict(hw)[0] == "no"
        assert all(o.fit == "no" for o in catalog.offers(hw))
    assert _pick(_hw(ram=32000, avx2=None)) == "qwen3.5:4b"  # not known is not lacking


def test_a_gpu_too_small_for_any_model_is_not_one():
    assert catalog.verdict(_hw(2048, ram=32000))[0] == "slow"


def test_the_verdict_and_the_offers_agree():
    for vram in range(1024, 26000, 256):
        for ram in (8000, 32000):
            hw = _hw(vram, ram=ram)
            on_gpu = any(o.fit == "gpu" for o in catalog.offers(hw))
            assert on_gpu == (catalog.verdict(hw)[0] in ("recommended", "works"))
            best = catalog.offers(hw)[0]
            assert best.recommended == (catalog.verdict(hw)[0] != "no") and (not on_gpu or best.fit == "gpu")


def test_what_is_here_needs_no_disk():
    hw = _hw(12227, disk=9000)
    assert _pick(hw) != BIT2
    assert catalog.recommend_roles(hw, [BIT2])["primary_chat"] == BIT2


def test_verdicts():
    level = lambda hw: catalog.verdict(hw)[0]  # noqa: E731
    assert [level(_hw(v)) for v in (24564, 11264, 8188, 4608)] == ["recommended", "recommended", "works", "works"]
    assert [level(_mac(g)) for g in (64, 24, 16, 8)] == ["recommended", "recommended", "works", "works"]
    assert level(Hardware()) == "no"  # nothing could be found out


def test_a_full_disk():
    hw = _hw(24564, disk=15_000)  # room for the 9.4 GB build and not for the 17.7 GB one
    assert _pick(hw) == BIT2
    full = {o.model: o for o in catalog.offers(hw)}[FULL]
    assert not full.disk_ok and "disk space" in full.note
    assert _pick(_hw(24564, disk=1000)) is None
    assert _pick(_hw(24564, disk=None)) == FULL  # not known is not full


# ── contexts ──────────────────────────────────────────────────────────────────

def test_2bit_context_on_12gb_is_the_measured_fit():
    # 40960 measured fully on-GPU on this card; 48k and 64k spilled to the CPU.
    assert catalog.context_in(catalog.find(BIT2), 12227) == 40960


def test_contexts_follow_the_card_and_each_models_ceiling():
    full = catalog.find(FULL)
    assert catalog.context_in(full, 32768) == 65536
    assert catalog.context_in(full, 20000) == 24576
    assert catalog.context_in(full, 12227) == 0
    assert catalog.context_in(catalog.find("qwen3:8b"), 32768) == 40960  # its own limit
    for m in catalog.chat_models():
        steps = [catalog.context_in(m, v) for v in range(1000, 50000, 250)]
        assert steps == sorted(steps)  # more memory never means less
        assert max(steps) <= m.max_context


def test_roles():
    hw = _hw(12227)
    roles = catalog.recommend_roles(hw)
    # one model for every chat role, so a hop between them does not swap models
    assert {roles[r] for r in ("primary_chat", "fast_chat", "coding", "vision")} == {BIT2}
    assert roles["embedding"] == "nomic-embed-text"
    assert catalog.recommend_roles(_hw(ram=8000)) == {}


# ── the catalog itself ────────────────────────────────────────────────────────

def test_every_entry_can_be_installed_and_checked():
    models = catalog.chat_models()
    assert len({m.model for m in models}) == len(models) and [m.quality for m in models] == sorted(
        (m.quality for m in models), reverse=True)
    for m in models:
        assert re.fullmatch(r"[0-9a-f]{64}", m.digest)
        assert m.license == "Apache-2.0"  # free to use, for anything
        assert 1e9 < m.download < 50e9 and 0 < m.weights_mib * 1.048576e6 <= m.download * 1.06
        assert catalog.digest_of(m.model) == m.digest
    e = catalog.embedding()
    assert re.fullmatch(r"[0-9a-f]{64}", e["digest"]) and catalog.digest_of(e["model"] + ":latest") == e["digest"]
    assert catalog.digest_of("somebody/elses:1b") == ""


def test_the_plan_is_plain_data_and_complete():
    for hw in (_hw(12227), _hw(), _mac(16), _mac(4), Hardware()):
        p = json.loads(json.dumps(catalog.plan(hw)))
        assert p["verdict"] in ("recommended", "works", "slow", "no") and p["summary"]
        assert len(p["offers"]) == len(catalog.chat_models())
        assert sum(o["recommended"] for o in p["offers"]) == (0 if p["verdict"] == "no" else 1)
        for o in p["offers"]:
            assert o["note"] and (o["context"] > 0) == (o["fit"] in ("gpu", "cpu"))
        assert p["embedding"]["model"] == "nomic-embed-text"
