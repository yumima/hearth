"""Kokoro on /v1/audio/speech: engine selection, the Piper fallback, the voice
list, and (when the model is provisioned) a real synthesis."""

from __future__ import annotations

import asyncio
import io
import wave

import pytest
from fastapi.testclient import TestClient

from hearth import config as cfgmod
from hearth import kokoro_tts, tts
from hearth.app import create_app
from hearth.routes import v1

from .test_routes import FakeBackend


def _client():
    app = create_app(cfgmod.default_config(), allow_any_host=True, backends_override={"ollama": FakeBackend()})
    return TestClient(app)


@pytest.fixture
def engines(monkeypatch, tmp_path):
    """Both engines faked; records which one answered and with what voice."""
    calls = []

    async def kokoro_synth(text, voice=None, speed=1.0):
        calls.append(("kokoro", voice, speed))
        if voice == "broken":
            raise kokoro_tts.KokoroError("boom")
        return b"RIFFkokoro"

    async def piper_synth(piper, voice, text):
        calls.append(("piper", voice.stem, None))
        return b"RIFFpiper"

    (tmp_path / "en_US-amy-medium.onnx").write_bytes(b"x")
    monkeypatch.setattr(kokoro_tts.engine, "synth", kokoro_synth)
    monkeypatch.setattr(tts.pool, "synth", piper_synth)
    monkeypatch.setattr(v1, "_piper_bin", lambda: "/bin/piper")
    monkeypatch.setattr(v1.paths, "voice_dirs", lambda engine: [tmp_path])
    return calls


def test_model_kokoro_uses_kokoro(engines):
    r = _client().post("/v1/audio/speech", json={"model": "kokoro", "voice": "af_bella", "input": "hi", "speed": 0.9})
    assert r.status_code == 200
    assert r.content == b"RIFFkokoro"
    assert r.headers["X-Hearth-TTS-Engine"] == "kokoro"
    assert engines == [("kokoro", "af_bella", 0.9)]


def test_kokoro_voice_without_model_uses_kokoro(engines):
    r = _client().post("/v1/audio/speech", json={"voice": "af_heart", "input": "hi"})
    assert r.headers["X-Hearth-TTS-Engine"] == "kokoro"


def test_model_piper_stays_piper(engines):
    r = _client().post("/v1/audio/speech", json={"model": "piper", "input": "hi"})
    assert r.content == b"RIFFpiper"
    assert r.headers["X-Hearth-TTS-Engine"] == "piper"
    assert engines == [("piper", "en_US-amy-medium", None)]


def test_kokoro_failure_falls_back_to_default_piper_voice(engines):
    r = _client().post("/v1/audio/speech", json={"model": "kokoro", "voice": "broken", "input": "hi"})
    assert r.status_code == 200
    assert r.content == b"RIFFpiper"
    # The Kokoro id is dropped, not looked up as a Piper file.
    assert engines[-1] == ("piper", "en_US-amy-medium", None)


def test_voices_lists_kokoro_first_then_english_piper(engines, monkeypatch, tmp_path):
    (tmp_path / "de_DE-thorsten-medium.onnx").write_bytes(b"x")
    monkeypatch.setattr(kokoro_tts, "installed", lambda: True)
    body = _client().get("/v1/audio/voices").json()
    ids = [(v["model"], v["id"]) for v in body["voices"]]
    assert ids[0] == ("kokoro", "af_heart")
    assert ("piper", "en_US-amy-medium") in ids
    assert all(i != "de_DE-thorsten-medium" for _, i in ids)
    assert body["default"] == {"model": "kokoro", "id": "af_heart"}


def test_voices_without_kokoro_default_to_piper(engines, monkeypatch):
    monkeypatch.setattr(kokoro_tts, "installed", lambda: False)
    body = _client().get("/v1/audio/voices").json()
    assert body["default"] == {"model": "piper", "id": "en_US-amy-medium"}


@pytest.mark.parametrize("vid,ok", [("af_heart", True), ("bf_emma", True), ("en_US-amy-medium", False),
                                    ("", False), (None, False), ("AF_x", False)])
def test_is_voice_id(vid, ok):
    assert kokoro_tts.engine.is_voice_id(vid) is ok


_MODELS = kokoro_tts.model_dir()  # the machine's own, read before a test has another home


@pytest.mark.skipif(not kokoro_tts.installed(), reason="kokoro not provisioned")
def test_real_synthesis_is_a_wav(monkeypatch):
    monkeypatch.setenv("HEARTH_KOKORO_DIR", str(_MODELS))
    eng = kokoro_tts.KokoroEngine()
    data = asyncio.run(eng.synth("Markets were steady.", "af_heart"))
    with wave.open(io.BytesIO(data)) as w:
        assert w.getframerate() == 24000
        assert w.getsampwidth() == 2
        assert w.getnframes() > 24000 * 0.5
    with pytest.raises(kokoro_tts.KokoroError):
        asyncio.run(eng.synth("x", "zz_nobody"))


def test_split_sentences_merges_short_pieces():
    # A piece under min_len absorbs the next sentence (the model phrases them
    # as one passage); one past it stands alone.
    long = "Treasury yields climbed to their highest level in three weeks."  # 63 chars
    assert kokoro_tts.split_sentences(f"Hi. Markets fell. {long} {long}", min_len=60) == [
        f"Hi. Markets fell. {long}",
        long,
    ]
    # the default merges up to a passage (160 chars)
    assert kokoro_tts.split_sentences(f"Hi. Markets fell. {long} {long}") == [f"Hi. Markets fell. {long} {long}"]
    assert kokoro_tts.split_sentences("   ") == []


def test_route_keeps_a_chunks_trailing_break(monkeypatch):
    # The input is stripped, but a chunk ending a section must still reach the
    # engine ending in "\n\n" — that is what earns it the longer pause.
    seen = []

    async def synth(text, voice=None, speed=1.0):
        seen.append(text)
        return b"RIFF"

    monkeypatch.setattr(kokoro_tts.engine, "synth", synth)
    c = _client()
    for raw in ("  Fed holds rates.\n\n", "Defense.\n", "Brent rose.  "):
        assert c.post("/v1/audio/speech", json={"model": "kokoro", "voice": "af_heart", "input": raw}).status_code == 200
    assert seen == ["Fed holds rates.\n\n", "Defense.\n", "Brent rose."]


def test_split_pieces_never_merges_across_a_line_break():
    # "DEFENSE" ran into its first bullet when lines were flattened: a line is
    # its own piece however short, and a blank line marks a section.
    text = "Fed holds rates.\n\nDefense.\nFrance ordered jets. Paris confirmed."
    assert kokoro_tts.split_pieces(text) == [
        ("Fed holds rates.", kokoro_tts.BREAK_SECTION),
        ("Defense.", kokoro_tts.BREAK_PARAGRAPH),
        ("France ordered jets. Paris confirmed.", kokoro_tts.BREAK_PARAGRAPH),
    ]
    assert kokoro_tts.trailing_break("a.\n\n") == kokoro_tts.BREAK_SECTION
    assert kokoro_tts.trailing_break("a.\n ") == kokoro_tts.BREAK_PARAGRAPH
    assert kokoro_tts.trailing_break("a.") == kokoro_tts.BREAK_NONE


def test_line_breaks_get_longer_pauses():
    import threading

    import numpy as np

    class FakeModel:
        def create(self, text, voice, speed, lang):
            return np.full(1000, 0.1, dtype=np.float32), 24000

    eng = kokoro_tts.KokoroEngine()
    eng._model, eng._voices = FakeModel(), ["af_heart"]

    def frames(text):
        with wave.open(io.BytesIO(eng._synth_blocking(text, "af_heart", 1.0, threading.Event()))) as w:
            return w.getnframes()

    p = {k: int(24000 * v) for k, v in (("s", kokoro_tts.PAUSE_SENTENCE_S), ("p", kokoro_tts.PAUSE_PARAGRAPH_S),
                                         ("x", kokoro_tts.PAUSE_SECTION_S))}
    # section break after "rates", paragraph after the heading, sentence at the end
    assert frames("Fed holds rates.\n\nDefense.\nFrance ordered jets.") == 3000 + p["x"] + p["p"] + p["s"]
    # a chunk that ends a section owes the section pause at its end
    assert frames("Fed holds rates.\n\n") == 1000 + p["x"]
    assert kokoro_tts.PAUSE_SENTENCE_S < kokoro_tts.PAUSE_PARAGRAPH_S < kokoro_tts.PAUSE_SECTION_S


def test_cancel_stops_synthesis_at_the_next_sentence():
    """A cancelled request gives the engine back within one sentence: the
    engine lock is held for the whole text, so this bounds how long the next
    request (a restarted read-aloud) waits."""
    import threading

    import numpy as np

    cancel = threading.Event()
    calls = []

    class FakeModel:
        def create(self, text, voice, speed, lang):
            calls.append(text)
            cancel.set()  # the client hangs up while the first sentence synthesises
            return np.zeros(2400, dtype=np.float32), 24000

    eng = kokoro_tts.KokoroEngine()
    eng._model, eng._voices = FakeModel(), ["af_heart"]
    text = " ".join(["Treasury yields climbed to their highest level in three weeks."] * 5)
    with pytest.raises(kokoro_tts.KokoroError, match="cancelled"):
        eng._synth_blocking(text, "af_heart", 1.0, cancel)
    assert len(calls) == 1
    assert not eng._lock.locked()


def test_uncancelled_synthesis_joins_every_sentence():
    import threading

    import numpy as np

    class FakeModel:
        def create(self, text, voice, speed, lang):
            return np.full(1000, 0.1, dtype=np.float32), 24000

    eng = kokoro_tts.KokoroEngine()
    eng._model, eng._voices = FakeModel(), ["bf_emma"]
    sentence = "Treasury yields climbed to their highest level in three weeks as traders " * 3 + "waited."
    text = " ".join([sentence] * 3)  # each > min_len: three pieces
    data = eng._synth_blocking(text, "bf_emma", 1.0, threading.Event())
    pause = int(24000 * kokoro_tts.PAUSE_SENTENCE_S)
    with wave.open(io.BytesIO(data)) as w:
        # three pieces, all joined, each followed by a sentence pause (they
        # used to be butted together, so sentences ran into each other)
        assert w.getnframes() == 3 * (1000 + pause)
    # a faster voice gets proportionally shorter pauses
    data = eng._synth_blocking(text, "bf_emma", 1.25, threading.Event())
    with wave.open(io.BytesIO(data)) as w:
        assert w.getnframes() == 3 * (1000 + int(24000 * kokoro_tts.PAUSE_SENTENCE_S / 1.25))


def test_piece_edges_are_trimmed_to_even_gaps():
    import numpy as np

    s = np.concatenate([np.zeros(5000), np.full(1000, 0.2), np.zeros(5000)]).astype(np.float32)
    t = kokoro_tts._trim(s)
    assert len(t) == 1000 + 2 * 600  # 25 ms kept either side of the sound


def test_non_string_voice_falls_through_to_piper(engines):
    r = _client().post("/v1/audio/speech", json={"voice": 123, "input": "hi"})
    assert r.status_code == 200
    assert r.headers["X-Hearth-TTS-Engine"] == "piper"


def test_broken_kokoro_install_falls_back_to_piper(monkeypatch, tmp_path):
    """installed() can be true while loading fails (bad import, corrupt file):
    that must surface as KokoroError — the route's fallback — not a 500."""
    (tmp_path / "en_US-amy-medium.onnx").write_bytes(b"x")

    async def piper_synth(piper, voice, text):
        return b"RIFFpiper"

    monkeypatch.setattr(kokoro_tts, "installed", lambda: True)
    monkeypatch.setattr(kokoro_tts, "model_dir", lambda: tmp_path / "missing")
    monkeypatch.setattr(kokoro_tts, "engine", kokoro_tts.KokoroEngine())
    monkeypatch.setattr(tts.pool, "synth", piper_synth)
    monkeypatch.setattr(v1, "_piper_bin", lambda: "/bin/piper")
    monkeypatch.setattr(v1.paths, "voice_dirs", lambda engine: [tmp_path])
    r = _client().post("/v1/audio/speech", json={"model": "kokoro", "voice": "af_heart", "input": "hi"})
    assert r.status_code == 200
    assert r.headers["X-Hearth-TTS-Engine"] == "piper"


def test_voices_default_follows_the_configured_kokoro_voice(engines, monkeypatch):
    monkeypatch.setattr(kokoro_tts, "installed", lambda: True)
    monkeypatch.setattr(kokoro_tts, "DEFAULT_VOICE", "bf_emma")
    assert _client().get("/v1/audio/voices").json()["default"] == {"model": "kokoro", "id": "bf_emma"}


def test_piper_default_matches_what_speech_uses(engines, monkeypatch, tmp_path):
    # Sorts before amy, but a voice-less request speaks amy (HEARTH_TTS_VOICE default).
    (tmp_path / "en_GB-alan-medium.onnx").write_bytes(b"x")
    monkeypatch.setattr(kokoro_tts, "installed", lambda: False)
    assert _client().get("/v1/audio/voices").json()["default"] == {"model": "piper", "id": "en_US-amy-medium"}


def test_closing_engine_stops_synthesis_at_the_next_sentence():
    import threading

    import numpy as np

    eng = kokoro_tts.KokoroEngine()
    calls = []

    class FakeModel:
        def create(self, text, voice, speed, lang):
            calls.append(text)
            eng._closing.set()  # shutdown begins mid-text
            return np.zeros(10, dtype=np.float32), 24000

    eng._model, eng._voices = FakeModel(), ["af_heart"]
    text = " ".join(["Treasury yields climbed to their highest level in three weeks."] * 4)
    with pytest.raises(kokoro_tts.KokoroError):
        eng._synth_blocking(text, "af_heart", 1.0, threading.Event())
    assert len(calls) == 1
