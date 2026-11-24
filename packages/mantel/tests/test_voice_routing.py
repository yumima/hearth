"""Which voice reads a text — the page's own code, run by node."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

PAGE = Path(__file__).resolve().parents[1] / "src" / "mantel" / "webui" / "index.html"
FR, ES, DE, IT, ZH = ("fr_FR-siwis-medium", "es_ES-davefx-medium", "de_DE-thorsten-medium",
                      "it_IT-paola-medium", "zh_CN-huayan-medium")
EN = ""  # no lesson: the chosen voice, which the server fills in


def _between(js: str, start: str, end: str) -> str:
    i = js.index(start)
    return js[i:js.index(end, i)]


def _voices(texts: list[str], lang: dict | None = None) -> list[list[str]]:
    js = re.findall(r"<script(?![^>]*src)[^>]*>(.*?)</script>", PAGE.read_text(encoding="utf-8"), re.S)[0]
    code = ('let cfgInfo = {tts_voice: "af_heart"};\n'
            + _between(js, "const EN_VOICE_DEFAULT", "// ── tutor session state")
            + re.search(r"^const normBraces.*$|^function normBraces[\s\S]*?^}", js, re.M).group(0) + "\n"
            + _between(js, "function speakable(s){", "function voiceSegments(")
            + _between(js, "function voiceSegments(", "\n}\n") + "\n}\n"
            + f"console.log(JSON.stringify({json.dumps(texts)}.map("
              f"t => voiceSegments(t, {json.dumps(lang)}).map(s => s.voice))));")
    r = subprocess.run(["node", "-e", code], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


pytestmark = pytest.mark.skipif(not shutil.which("node"), reason="node is not installed")


def test_english_stays_with_the_chosen_voice():
    texts = ["There was no bread, so everyone was hungry.",
             "Who was Napoleon? He was a French general.",
             "The die was cast in Rome.",
             "Die Hard was the film he was in, and it was a hit.",
             "La La Land won; El Niño and Los Angeles did not.",
             "Yes."]
    assert _voices(texts) == [[EN]] * len(texts)


def test_a_text_plainly_in_another_language_gets_its_voice():
    texts = {"Le chat et la souris mangent avec nous dans la maison.": FR,
             "La casa es una maravilla y está muy bien para usted.": ES,
             "Ich bin nicht müde, aber das ist auch gut.": DE,
             "Il gatto non è molto grande, ma mi piace perché è bello.": IT}
    assert _voices(list(texts)) == [[v] for v in texts.values()]


def test_chinese_runs_get_the_mandarin_voice_and_japanese_does_not():
    assert _voices(["I say 你好世界 to you", "日本語です"]) == [[EN, ZH, EN], [EN]]


def test_a_lessons_english_is_not_read_by_a_third_language():
    lang = {"code": "it", "voice": IT}
    assert _voices(["Say {{ciao}}: there was no other word, and it was polite."], lang) == [
        ["af_heart", IT, "af_heart"]]
