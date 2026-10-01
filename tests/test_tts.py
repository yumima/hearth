"""PiperPool: the warm per-voice worker and its one-shot fallback, driven by a
fake ``piper`` script that speaks the same stdin/--output_dir protocol."""

from __future__ import annotations

import asyncio
import stat
from pathlib import Path

from hearth import tts

FAKE_PIPER = r"""#!/usr/bin/env python3
import os, sys, itertools
args = sys.argv[1:]
mode = os.environ.get("FAKE_PIPER_MODE", "ok")
if "--output_dir" in args:
    if mode == "worker-crash":
        sys.exit(3)
    d = args[args.index("--output_dir") + 1]
    with open(os.path.join(os.path.dirname(d), "spawns"), "a") as f:
        f.write("x")
    for n in itertools.count():
        line = sys.stdin.readline()
        if not line:
            break
        if "wedge" in line:
            import time; time.sleep(30)
        p = os.path.join(d, f"{n}.wav")
        open(p, "wb").write(b"RIFF" + line.strip().encode())
        print(p, flush=True)
else:
    out = args[args.index("--output_file") + 1]
    open(out, "wb").write(b"ONESHOT" + sys.stdin.read().strip().encode())
"""


def _fake(tmp_path: Path) -> str:
    p = tmp_path / "piper"
    p.write_text(FAKE_PIPER)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return str(p)


def test_worker_reuses_one_process(tmp_path, monkeypatch):
    monkeypatch.setattr(tts.tempfile, "tempdir", str(tmp_path))
    piper, voice = _fake(tmp_path), tmp_path / "v.onnx"

    async def go():
        pool = tts.PiperPool()
        try:
            a = await pool.synth(piper, voice, "ciao\ncome stai")
            b = await pool.synth(piper, voice, "grazie")
            c, d = await asyncio.gather(pool.synth(piper, voice, "uno"),
                                        pool.synth(piper, voice, "due"))
            return a, b, c, d
        finally:
            await pool.aclose()

    a, b, c, d = asyncio.run(go())
    assert a == b"RIFFciao come stai"  # newline folded: one utterance, one file
    assert b == b"RIFFgrazie"
    assert (c, d) == (b"RIFFuno", b"RIFFdue")  # concurrent requests stay paired
    assert (tmp_path / "spawns").read_text() == "x"  # a single warm process


def test_worker_failure_falls_back_to_oneshot(tmp_path, monkeypatch):
    monkeypatch.setattr(tts.tempfile, "tempdir", str(tmp_path))
    monkeypatch.setenv("FAKE_PIPER_MODE", "worker-crash")
    piper, voice = _fake(tmp_path), tmp_path / "v.onnx"

    async def go():
        pool = tts.PiperPool()
        try:
            return await pool.synth(piper, voice, "hello")
        finally:
            await pool.aclose()

    assert asyncio.run(go()) == b"ONESHOThello"


def test_idle_worker_is_reaped(tmp_path, monkeypatch):
    monkeypatch.setattr(tts.tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(tts, "IDLE_SECS", 0.05)
    piper, voice = _fake(tmp_path), tmp_path / "v.onnx"

    async def go():
        pool = tts.PiperPool()
        await pool.synth(piper, voice, "hi")
        w = next(iter(pool._workers.values()))
        await asyncio.sleep(0.2)
        reaped = not pool._workers and not w.alive()
        await pool.aclose()
        return reaped

    assert asyncio.run(go())


def test_cancelled_request_does_not_desync_the_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(tts.tempfile, "tempdir", str(tmp_path))
    piper, voice = _fake(tmp_path), tmp_path / "v.onnx"

    async def go():
        pool = tts.PiperPool()
        try:
            await pool.synth(piper, voice, "warm")
            t = asyncio.ensure_future(pool.synth(piper, voice, "interrupted"))
            await asyncio.sleep(0)   # let it write its line, then cancel the read
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass
            return await pool.synth(piper, voice, "next")
        finally:
            await pool.aclose()

    assert asyncio.run(go()) == b"RIFFnext"
    # ...and the interrupted request didn't cost a cold restart
    assert (tmp_path / "spawns").read_text() == "x"
    assert not [f for f in tmp_path.rglob("*.wav")]   # its output was collected, not leaked


def test_wedged_orphan_is_killed_after_grace(tmp_path, monkeypatch):
    monkeypatch.setattr(tts.tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(tts, "ORPHAN_GRACE", 0.1)
    piper, voice = _fake(tmp_path), tmp_path / "v.onnx"

    async def go():
        pool = tts.PiperPool()
        try:
            await pool.synth(piper, voice, "warm")
            t = asyncio.ensure_future(pool.synth(piper, voice, "wedge"))
            await asyncio.sleep(0.05)
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass
            return await asyncio.wait_for(pool.synth(piper, voice, "next"), 5)
        finally:
            await pool.aclose()

    assert asyncio.run(go()) == b"RIFFnext"   # not stuck behind the wedged sentence
