"""Hardware probe — GPU type, VRAM, RAM (design doc §3.1 F9).

Drives ``GET /admin/hardware`` and (M3) the first-run model-fit wizard.
Pure best-effort: every probe degrades to ``None``/empty rather than
raising, so the engine starts on any box.

Note on this dev box: the NVIDIA dGPU is wedged for *display* (Qt falls
back to Mesa/iGPU) but ``nvidia-smi`` and CUDA compute are healthy — so we
report it as usable for inference. We never touch the GL/Vulkan display
path.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import asdict, dataclass


@dataclass
class Gpu:
    vendor: str  # "nvidia" | "intel" | "amd" | "apple"
    name: str
    vram_mib: int | None
    driver: str | None
    compute: str | None  # "cuda" | "rocm" | "metal" | "vulkan" | None


@dataclass
class Hardware:
    cpu_cores: int | None
    ram_total_mib: int | None
    ram_available_mib: int | None
    gpus: list[Gpu]


def _run(cmd: list[str], timeout: float = 4.0) -> str | None:
    if not shutil.which(cmd[0]):
        return None
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        return out.stdout if out.returncode == 0 else None
    except (subprocess.TimeoutExpired, OSError):
        return None


def _nvidia() -> list[Gpu]:
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if not out:
        return []
    gpus: list[Gpu] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        name, vram, driver = parts[0], parts[1], parts[2]
        try:
            vram_mib: int | None = int(float(vram))
        except ValueError:
            vram_mib = None
        gpus.append(Gpu("nvidia", name, vram_mib, driver, "cuda"))
    return gpus


def _intel() -> list[Gpu]:
    # No reliable VRAM number for an iGPU (shared system RAM). Report presence
    # so the fit logic knows an iGPU offload path exists (IPEX-LLM / Vulkan).
    out = _run(["lspci"])
    if not out:
        return []
    for line in out.splitlines():
        low = line.lower()
        if ("vga" in low or "display" in low or "3d" in low) and "intel" in low:
            # lspci line: "<slot> <class>: <vendor device>" — the slot has
            # colons but no colon-space, so split on the first ": ".
            name = line.split(": ", 1)[-1].strip()
            return [Gpu("intel", name, None, None, "vulkan")]
    return []


def _cpu_cores() -> int | None:
    try:
        import os

        return os.cpu_count()
    except Exception:
        return None


def _mem() -> tuple[int | None, int | None]:
    try:
        info: dict[str, int] = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, _, rest = line.partition(":")
                kb = int(rest.strip().split()[0])
                info[k] = kb // 1024  # MiB
        return info.get("MemTotal"), info.get("MemAvailable")
    except Exception:
        return None, None


def probe() -> Hardware:
    total, avail = _mem()
    gpus = _nvidia() + _intel()
    return Hardware(
        cpu_cores=_cpu_cores(),
        ram_total_mib=total,
        ram_available_mib=avail,
        gpus=gpus,
    )


def as_dict() -> dict:
    hw = probe()
    d = asdict(hw)
    # A coarse "best inference target" hint consumers can show in setup UX.
    best = None
    for g in hw.gpus:
        if g.compute == "cuda" and (g.vram_mib or 0) >= 6000:
            best = f"{g.name} ({g.vram_mib} MiB, CUDA)"
            break
    d["inference_target"] = best or "cpu"
    return d


def recommend_roles() -> dict[str, str]:
    """Pick a role→model set that fits the detected hardware (the M3 first-run
    wizard). Qwen3 generation.

    On a ~12 GB GPU with ample system RAM we prefer the **30B MoE**
    (qwen3:30b-a3b): only ~3B params are active per token, so the few GB that
    spill from VRAM to RAM cost little speed, and it beats a dense 14B on
    quality. Smaller GPUs get dense models that fit fully on-GPU; CPU-only
    boxes are sized by RAM.
    """
    hw = probe()
    vram = 0
    for g in hw.gpus:
        if g.compute == "cuda" and g.vram_mib:
            vram = max(vram, g.vram_mib)
    ram = hw.ram_total_mib or 0

    if vram >= 22000:
        primary, fast = "qwen3:32b", "qwen3:14b"
    elif vram >= 11000:
        # 12–16 GB GPU: dense 14B fits fully; with ≥24 GB RAM the 30B MoE
        # (3B active) gives more quality via cheap partial offload.
        primary = "qwen3:30b-a3b" if ram >= 24000 else "qwen3:14b"
        fast = "qwen3:8b"
    elif vram >= 7000:
        primary, fast = "qwen3:8b", "qwen3:4b"
    elif vram >= 4500:
        primary, fast = "qwen3:4b", "qwen3:1.7b"
    elif vram > 0:
        primary, fast = "qwen3:1.7b", "qwen3:0.6b"
    elif ram >= 32000:  # CPU-only — size by RAM
        primary, fast = "qwen3:8b", "qwen3:4b"
    elif ram >= 16000:
        primary, fast = "qwen3:4b", "qwen3:1.7b"
    else:
        primary, fast = "qwen3:1.7b", "qwen3:0.6b"

    return {
        "primary_chat": primary,
        "fast_chat": fast,
        "coding": fast,
        "embedding": "nomic-embed-text",
    }
