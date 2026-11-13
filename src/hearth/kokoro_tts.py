"""Kokoro-82M TTS: natural-sounding voices for /v1/audio/speech.

Piper is fast but sounds synthetic; Kokoro (82M params, Apache-2.0, ONNX) is
close to human read-aloud quality and still runs ~4x faster than real time on
a laptop CPU. It is optional: the ``kokoro-onnx`` package plus two model files
under ``~/.local/share/finterm/kokoro`` (``hearth voice --voice kokoro``
provisions both). When either is missing, callers fall back to Piper.

One model instance serves every voice (a voice is a 256-float style vector,
not a separate model), loaded on first use and unloaded after IDLE_SECS so an
unused engine doesn't hold ~350 MB of RAM. Synthesis runs in a worker thread,
serialized on a lock: onnxruntime already uses every core for one utterance.
"""

from __future__ import annotations

import asyncio
import io
import os
import re
import threading
import time
import wave
from pathlib import Path

from hearth.speech_text import normalize

MODEL_FILE = "kokoro-v1.0.onnx"
VOICES_FILE = "voices-v1.0.bin"
DEFAULT_VOICE = os.environ.get("HEARTH_KOKORO_VOICE", "af_heart")
IDLE_SECS = float(os.environ.get("HEARTH_KOKORO_IDLE_SECS", "600"))

# Curated read-aloud voices, best first (Kokoro's own quality grades: heart A,
# bella A-, the rest B/C). Other voices in the pack still work by id; these are
# the ones offered by name. Prefix: a = American English, b = British English;
# f/m = female/male.
CURATED: list[tuple[str, str]] = [
    ("af_heart", "Heart — warm, natural (US)"),
    ("af_bella", "Bella — bright, expressive (US)"),
    ("af_nicole", "Nicole — soft, close (US)"),
    ("af_sarah", "Sarah — calm (US)"),
    ("af_sky", "Sky — light (US)"),
    ("bf_emma", "Emma — gentle (UK)"),
    ("bf_isabella", "Isabella — clear (UK)"),
    ("am_michael", "Michael — male (US)"),
    ("bm_george", "George — male (UK)"),
]


class KokoroError(Exception):
    pass


_SENTENCE_END = re.compile(r"(?<=[.!?;:])\s+")

# Pause after a piece, by how it ends — Kokoro is synthesised a piece at a
# time and the pieces used to be butted together with no gap at all, which is
# what made sentences run into each other. Tuned to a newsreader's pace.
PAUSE_SENTENCE_S = 0.32
PAUSE_CLAUSE_S = 0.18  # ':' / ';'
SILENCE_THRESHOLD = 0.004  # |sample| below this at a piece's edges is trimmed


def split_sentences(text: str, min_len: int = 160) -> list[str]:
    """Sentence-sized pieces (short ones merged up to `min_len`, so the model
    phrases two short sentences as one passage). Synthesis runs one piece at a
    time so a cancelled request stops within a piece instead of holding the
    engine for the whole text."""
    out: list[str] = []
    for s in _SENTENCE_END.split(" ".join(text.split())):
        if out and len(out[-1]) < min_len:
            out[-1] = f"{out[-1]} {s}"
        elif s:
            out.append(s)
    return out


def model_dir() -> Path:
    override = os.environ.get("HEARTH_KOKORO_DIR")
    if override:
        return Path(override)
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "finterm" / "kokoro"


def installed() -> bool:
    """Package importable AND both model files present (cheap; no model load)."""
    d = model_dir()
    if not ((d / MODEL_FILE).is_file() and (d / VOICES_FILE).is_file()):
        return False
    try:
        import importlib.util

        return importlib.util.find_spec("kokoro_onnx") is not None
    except (ImportError, ValueError):
        return False


def lang_for(voice: str) -> str:
    return "en-gb" if voice.startswith("b") else "en-us"


def _trim(samples):
    """Drop the near-silent head/tail Kokoro leaves on a piece (uneven, so the
    gaps between pieces were uneven), keeping 25 ms so no onset is clipped."""
    import numpy as np

    loud = np.flatnonzero(np.abs(samples) > SILENCE_THRESHOLD)
    if not len(loud):
        return samples
    pad = 600  # 25 ms at 24 kHz
    return samples[max(0, loud[0] - pad): loud[-1] + 1 + pad]


def _to_wav(samples, sample_rate: int) -> bytes:
    import numpy as np

    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


class KokoroEngine:
    def __init__(self) -> None:
        self._model = None
        self._voices: list[str] = []
        self._lock = threading.Lock()  # one synthesis (and one load) at a time
        self._last_used = 0.0
        self._reaper: asyncio.Task | None = None
        self._closing = threading.Event()

    # ── blocking helpers (run in a worker thread) ──────────────────────────
    def _ensure_loaded(self):
        if self._model is None:
            d = model_dir()
            try:
                # The import is inside: find_spec can succeed for a package
                # whose import then fails (no onnxruntime wheel, numpy ABI).
                from kokoro_onnx import Kokoro

                model = Kokoro(str(d / MODEL_FILE), str(d / VOICES_FILE))
                voices = sorted(model.get_voices())
            except Exception as e:  # broken install, corrupt download, …
                raise KokoroError(f"kokoro failed to load: {e}") from e
            self._model, self._voices = model, voices
        self._last_used = time.monotonic()  # a load counts as use: don't reap it at once
        return self._model

    def _synth_blocking(self, text: str, voice: str, speed: float, cancel: threading.Event) -> bytes:
        import numpy as np

        with self._lock:
            model = self._ensure_loaded()
            if voice not in self._voices:
                raise KokoroError(f"unknown kokoro voice {voice!r}")
            parts, sr = [], 24000
            british = lang_for(voice) == "en-gb"
            for piece in split_sentences(text):
                # A request whose client went away (stop, voice change) — or a
                # gateway shutting down — gives the engine back here, not after
                # the rest of its text.
                if cancel.is_set() or self._closing.is_set():
                    raise KokoroError("cancelled")
                try:
                    # Numbers, money and shorthand as words: the phonemizer
                    # reads "2.5%" as "two, five percent" (speech_text).
                    samples, sr = model.create(normalize(piece, british=british), voice=voice, speed=speed,
                                               lang=lang_for(voice))
                except Exception as e:
                    raise KokoroError(f"kokoro synthesis failed: {e}") from e
                if samples is not None and len(samples):
                    samples = _trim(samples)
                    parts.append(samples)
                    tail = piece.rstrip()[-1:]
                    pause = PAUSE_SENTENCE_S if tail in ".!?" else PAUSE_CLAUSE_S if tail in ":;" else 0.12
                    # the pause keeps pace with the voice: 1.2x speech, 1.2x shorter gaps
                    parts.append(np.zeros(int(sr * pause / speed), dtype=samples.dtype))
            self._last_used = time.monotonic()
            if not parts:
                raise KokoroError("kokoro produced no audio")
            return _to_wav(np.concatenate(parts), sr)

    # ── async API ───────────────────────────────────────────────────────────
    async def synth(self, text: str, voice: str | None = None, speed: float = 1.0) -> bytes:
        self._closing.clear()
        if not installed():
            raise KokoroError("kokoro not installed — run `hearth voice --voice kokoro`")
        speed = min(max(float(speed), 0.5), 2.0)
        self._ensure_reaper()
        cancel = threading.Event()
        try:
            return await asyncio.to_thread(self._synth_blocking, text, voice or DEFAULT_VOICE, speed, cancel)
        except asyncio.CancelledError:
            cancel.set()  # the thread can't be killed; stop it at the next sentence
            raise

    def is_voice_id(self, voice: str | None) -> bool:
        """Kokoro voice ids look like 'af_heart'; Piper's like 'en_US-amy-medium'."""
        if not isinstance(voice, str) or len(voice) < 4 or voice[2] != "_":
            return False
        return voice[:2].isalpha() and voice[:2].islower() and voice[3:].isalpha()

    def _ensure_reaper(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        # A reaper left pending on a loop that has since closed never becomes
        # done(): replace any reaper that doesn't belong to the running loop.
        if self._reaper is None or self._reaper.done() or self._reaper.get_loop() is not loop:
            self._reaper = loop.create_task(self._reap())

    async def _reap(self) -> None:
        while True:
            await asyncio.sleep(min(60.0, IDLE_SECS))
            if self._model is None:
                return
            if time.monotonic() - self._last_used > IDLE_SECS and self._lock.acquire(blocking=False):
                try:
                    self._model = None
                    self._voices = []
                finally:
                    self._lock.release()
                return

    async def aclose(self) -> None:
        self._closing.set()  # an in-flight synthesis stops at its next sentence
        if self._reaper is not None:
            self._reaper.cancel()
            self._reaper = None
        # Wait out that sentence (off the event loop), then drop the model.
        await asyncio.to_thread(self._lock.acquire)
        try:
            self._model = None
            self._voices = []
        finally:
            self._lock.release()


engine = KokoroEngine()
