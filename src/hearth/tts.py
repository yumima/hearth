"""Low-latency Piper TTS: one warm ``piper`` process per voice.

Spawning ``piper`` per request reloads the ONNX voice every time — ~0.3 s of a
~0.5 s sentence. A voice conversation speaks sentence by sentence, so that load
cost lands on every sentence. Instead, keep one process per voice in
``--output_dir`` mode: it reads one utterance per stdin line, writes a WAV per
line, and prints that WAV's path on stdout. Requests to the same voice are
serialized on a lock (one line in, one path out).

Workers that sit idle are reaped so a rarely-used voice doesn't hold ~100 MB of
RAM. Any worker failure (crash, timeout, garbled output) kills that worker and
falls back to the one-shot path, so a wedged process can never take TTS down.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import time
from pathlib import Path

IDLE_SECS = float(os.environ.get("HEARTH_TTS_IDLE_SECS", "300"))
SYNTH_TIMEOUT = 60.0


def prosody_args() -> list[str]:
    """Prosody knobs — Piper's bare defaults sound clipped/robotic. Slightly
    longer phonemes + a touch more width-noise + a real inter-sentence pause
    give a calmer, more human cadence. Tunable via env without a rebuild."""
    return [
        "--length_scale", os.environ.get("HEARTH_TTS_LENGTH_SCALE", "1.08"),
        "--noise_w", os.environ.get("HEARTH_TTS_NOISE_W", "0.9"),
        "--sentence_silence", os.environ.get("HEARTH_TTS_SENTENCE_SILENCE", "0.35"),
    ]


class TTSError(Exception):
    pass


async def synth_oneshot(piper: str, voice: Path, text: str) -> bytes:
    """Classic path: one process per request. Used as the fallback."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    out_path = tmp.name
    tmp.close()
    try:
        proc = await asyncio.create_subprocess_exec(
            piper, "--model", str(voice), "--output_file", out_path, *prosody_args(),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, err = await proc.communicate(text.encode("utf-8"))
        if proc.returncode != 0 or not os.path.getsize(out_path):
            raise TTSError(f"piper failed: {err.decode('utf-8', 'ignore')[:200]}")
        return Path(out_path).read_bytes()
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass


class _Worker:
    def __init__(self, piper: str, voice: Path):
        self.piper = piper
        self.voice = voice
        self.lock = asyncio.Lock()
        self.proc: asyncio.subprocess.Process | None = None
        self.outdir: str | None = None
        self.last_used = time.monotonic()

    async def _start(self) -> None:
        self.outdir = tempfile.mkdtemp(prefix="hearth-tts-")
        self.proc = await asyncio.create_subprocess_exec(
            self.piper, "--model", str(self.voice), "--output_dir", self.outdir,
            *prosody_args(),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )

    def alive(self) -> bool:
        return self.proc is not None and self.proc.returncode is None

    async def synth(self, text: str) -> bytes:
        async with self.lock:
            self.last_used = time.monotonic()
            if not self.alive():
                await self.close_unlocked()
                await self._start()
            # One utterance per line: piper treats a newline as "next utterance".
            line = " ".join(text.split()) + "\n"
            try:
                self.proc.stdin.write(line.encode("utf-8"))
                await self.proc.stdin.drain()
                raw = await asyncio.wait_for(self.proc.stdout.readline(), SYNTH_TIMEOUT)
            except (asyncio.TimeoutError, OSError, BrokenPipeError) as e:
                await self.close_unlocked()
                raise TTSError(f"piper worker failed: {e!r}") from e
            except BaseException:
                # Cancelled mid-utterance (the client hung up, e.g. a barge-in):
                # piper will still print this line's path, which the NEXT request
                # would read as its own. Drop the worker so nothing desyncs.
                await asyncio.shield(self.close_unlocked())
                raise
            path = raw.decode("utf-8", "ignore").strip()
            # Only ever read a file inside our own output dir.
            if not path or os.path.dirname(os.path.abspath(path)) != os.path.abspath(self.outdir):
                await self.close_unlocked()
                raise TTSError(f"piper worker returned unexpected output {path[:120]!r}")
            try:
                data = Path(path).read_bytes()
            finally:
                try:
                    os.unlink(path)
                except OSError:
                    pass
            self.last_used = time.monotonic()
            if not data:
                raise TTSError("piper worker produced an empty file")
            return data

    async def close_unlocked(self) -> None:
        p, self.proc = self.proc, None
        if p is not None and p.returncode is None:
            try:
                p.stdin.close()
            except Exception:
                pass
            try:
                await asyncio.wait_for(p.wait(), 2.0)
            except asyncio.TimeoutError:
                p.kill()
                await p.wait()
        if self.outdir:
            shutil.rmtree(self.outdir, ignore_errors=True)
            self.outdir = None

    async def close(self) -> None:
        async with self.lock:
            await self.close_unlocked()


class PiperPool:
    def __init__(self) -> None:
        self._workers: dict[tuple[str, str], _Worker] = {}
        self._reaper: asyncio.Task | None = None

    async def synth(self, piper: str, voice: Path, text: str) -> bytes:
        key = (piper, str(voice))
        w = self._workers.get(key)
        if w is None:
            w = self._workers[key] = _Worker(piper, voice)
        self._ensure_reaper()
        try:
            return await w.synth(text)
        except TTSError:
            # Worker is closed; it restarts on the next request. Serve this one
            # the slow-but-reliable way.
            return await synth_oneshot(piper, voice, text)

    def _ensure_reaper(self) -> None:
        if self._reaper is None or self._reaper.done():
            self._reaper = asyncio.get_running_loop().create_task(self._reap())

    async def _reap(self) -> None:
        while self._workers:
            await asyncio.sleep(min(30.0, IDLE_SECS))
            now = time.monotonic()
            for key, w in list(self._workers.items()):
                if not w.lock.locked() and now - w.last_used > IDLE_SECS:
                    self._workers.pop(key, None)
                    await w.close()

    async def aclose(self) -> None:
        if self._reaper is not None:
            self._reaper.cancel()
            self._reaper = None
        workers, self._workers = list(self._workers.values()), {}
        for w in workers:
            await w.close()


pool = PiperPool()
