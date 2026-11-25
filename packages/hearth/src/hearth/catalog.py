"""Which models suit this computer.

catalog.json is what hearth offers: measured sizes, licences, the exact build
that was tested. This says which of them fit, in order, and why. An estimate until
the model has been loaded here and measured (provision.py).

The rule is to fit the model entirely in the GPU's memory. One that spills runs
its overflow layers on the processor and leaves the GPU waiting for them: the
full 27B on a 12 GB card managed 3 tokens a second, its 2-bit build 27.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from importlib import resources
from typing import Iterable

from .hardware import Hardware

# Contexts worth choosing between; each model's ceiling is its max_context.
CONTEXT_STEPS = (65536, 40960, 32768, 24576, 16384, 12288, 8192, 4096)
# The least a chat with tools can do with, and what it is comfortable with.
LEAST_CONTEXT, GOOD_CONTEXT = 8192, 16384
# Held back for compute buffers and whatever else shares the card: a share and a
# fixed amount. Under-reserving is what turns "it fits" into layers silently
# running on the processor.
_OVERHEAD_MIB, _USABLE = 700, 0.92
# On the processor a model needs its own size in RAM, beside what is running.
_CPU_SPARE_MIB = 6000
# Disk: the download, its unpacking, and not filling the disk to the last byte.
DISK_SPARE_MIB = 4000


@dataclass(frozen=True)
class Model:
    model: str
    title: str
    quality: int
    download: int      # bytes
    weights_mib: int
    kv_kib: float      # GPU memory per token of context, q8_0 cache
    max_context: int
    digest: str
    license: str
    vision: bool = False
    tested: str = ""   # where model_check passed; "" where nobody has run it


@dataclass(frozen=True)
class Offer:
    model: str
    title: str
    fit: str           # "gpu" | "cpu" | "spills" | "no"
    context: int       # 0: it cannot be run usefully
    download: int
    license: str
    vision: bool
    recommended: bool
    note: str          # one sentence for the person choosing
    disk_ok: bool
    digest: str


@lru_cache(maxsize=1)
def _data() -> dict:
    return json.loads(resources.files("hearth").joinpath("catalog.json").read_text(encoding="utf-8"))


def chat_models() -> list[Model]:
    """Best first."""
    return sorted((Model(**m) for m in _data()["chat"]), key=lambda m: -m.quality)


def embedding() -> dict:
    return dict(_data()["embedding"])


def find(model: str) -> Model | None:
    return next((m for m in chat_models() if m.model == model), None)


def digest_of(model: str) -> str:
    m = find(model)
    return m.digest if m else embedding()["digest"] if model in (embedding()["model"], embedding()["model"] + ":latest") else ""


def context_in(m: Model, vram_mib: int) -> int:
    """The largest step of context at which m is entirely on the GPU, else 0."""
    budget = vram_mib * _USABLE - m.weights_mib - _OVERHEAD_MIB
    if budget <= 0:
        return 0
    affordable = min(int(budget * 1024 / m.kv_kib), m.max_context)
    return next((s for s in CONTEXT_STEPS if s <= affordable), 0)


def verdict(hw: Hardware) -> tuple[str, str]:
    """("recommended" | "works" | "slow" | "no", one sentence)."""
    g = hw.best_gpu()
    if g is not None and any(context_in(m, g.vram_mib) >= LEAST_CONTEXT for m in chat_models()):
        size = f"{g.name}, {'' if g.vendor == 'apple' else 'with '}{g.vram_mib / 1024:.0f} GB for models"
        # From here on (a 24 GB Mac: half of it) the 27B's 2-bit build fits.
        if g.vram_mib >= 11000:
            return "recommended", f"{size}: runs models entirely on the GPU."
        return "works", f"{size}: runs smaller models entirely on the GPU."
    ram = hw.ram_total_mib or 0
    if hw.avx2 is False:
        return "no", "This processor lacks AVX2, which models need to run at a usable speed."
    if ram >= 15000:
        return "slow", "No GPU that can run models was found; the processor can run a small one, slowly."
    return "no", f"No GPU that can run models, and {ram / 1024:.0f} GB of memory: too little for a useful model."


def offers(hw: Hardware, here: Iterable[str] = ()) -> list[Offer]:
    """Every model, those that suit first, the recommended one marked. `here`:
    models already downloaded, which need no disk."""
    g = hw.best_gpu()
    level, _ = verdict(hw)
    ram = hw.ram_total_mib or 0
    have = {m.removesuffix(":latest") for m in here}
    out: list[Offer] = []
    for m in chat_models():
        disk_ok = (m.model in have or hw.disk_free_mib is None
                   or hw.disk_free_mib >= m.download / (1 << 20) + DISK_SPARE_MIB)
        context = context_in(m, g.vram_mib) if g else 0
        if context >= LEAST_CONTEXT:
            fit, note = "gpu", "Runs entirely on the GPU."
            if context < GOOD_CONTEXT:
                note = "Runs entirely on the GPU, with a short memory for long chats."
        elif level == "slow" and m.weights_mib * 1.15 + _CPU_SPARE_MIB <= ram and m.weights_mib <= 5500:
            fit, context, note = "cpu", 8192, "Runs on the processor: expect a few words a second."
        elif g and m.weights_mib * 1.1 + _CPU_SPARE_MIB <= ram + g.vram_mib and m.weights_mib > g.vram_mib * 0.5:
            fit, context, note = "spills", 0, "Too large for the GPU: part would run on the processor, very slowly."
        else:
            fit, context, note = "no", 0, "Does not fit this computer."
        if fit in ("gpu", "cpu") and not m.tested:
            note += " Not tested with hearth yet; it is checked here before it is used."
        if not disk_ok:
            note += " Not enough free disk space."
        out.append(Offer(m.model, m.title, fit, context, m.download, m.license, m.vision, False, note, disk_ok, m.digest))
    # The best that fits comfortably; on the processor the smallest is the usable
    # one, so it is the best of those under 2.6 GB of weights.
    choice = None
    if level in ("recommended", "works"):
        fits = [o for o in out if o.fit == "gpu" and o.disk_ok]
        choice = next((o for o in fits if o.context >= GOOD_CONTEXT), fits[0] if fits else None)
    elif level == "slow":
        small = {m.model for m in chat_models() if m.weights_mib <= 2600}
        choice = next((o for o in out if o.fit == "cpu" and o.disk_ok and o.model in small), None)
    order = {"gpu": 0, "cpu": 0, "spills": 1, "no": 2}
    out = [Offer(**{**asdict(o), "recommended": o is choice}) for o in out]
    return sorted(out, key=lambda o: (not o.recommended, order[o.fit]))


def plan(hw: Hardware, here: Iterable[str] = ()) -> dict:
    """What the setup page and `hearth model suggest --json` show."""
    level, sentence = verdict(hw)
    g = hw.best_gpu()
    return {
        "hardware": {**asdict(hw), "best_gpu": asdict(g) if g else None},
        "verdict": level, "summary": sentence,
        "free_mib": g.free_mib if g else None,
        "offers": [asdict(o) for o in offers(hw, here)],
        "embedding": embedding(),
    }


def recommend_roles(hw: Hardware, here: Iterable[str] = ()) -> dict[str, str]:
    """Role → model for this computer; {} when none suits it. One model for every
    chat role, so a hop from one to another does not swap models."""
    best = next((o for o in offers(hw, here) if o.recommended), None)
    if best is None:
        return {}
    roles = {r: best.model for r in ("primary_chat", "fast_chat", "coding")}
    if best.vision:
        roles["vision"] = best.model
    return {**roles, "embedding": embedding()["model"]}
