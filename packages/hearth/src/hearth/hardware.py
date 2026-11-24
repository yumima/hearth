"""What this computer has to run models with (design doc §3.1 F9).

Best-effort on every system: what cannot be found out is None or left out, never
an error, so the engine starts on any machine. The standard library only, psutil
aside. What fits on it is catalog.py's to say.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Gpu:
    vendor: str  # "nvidia" | "amd" | "apple" | "intel"
    name: str
    vram_mib: int | None       # Apple: what a model may have of the shared memory
    free_mib: int | None = None
    driver: str | None = None
    compute: str | None = None  # "cuda" | "rocm" | "vulkan" | "metal" | None: not for models


@dataclass
class Hardware:
    system: str = ""            # "linux" | "macos" | "windows"
    machine: str = ""           # "x86_64" | "arm64"
    cpu: str = ""
    cpu_cores: int | None = None
    avx2: bool | None = None    # x86 only; None: not known
    ram_total_mib: int | None = None
    ram_available_mib: int | None = None
    disk_free_mib: int | None = None  # where models go
    gpus: list[Gpu] = field(default_factory=list)

    def best_gpu(self) -> Gpu | None:
        """The one to size a model by. One: Ollama can split a model over several,
        at a cost that depends on them, so two small ones are not one large one."""
        usable = [g for g in self.gpus if g.compute and g.vram_mib]
        return max(usable, key=lambda g: g.vram_mib, default=None)


def _run(cmd: list[str], timeout: float = 6.0) -> str | None:
    if not shutil.which(cmd[0]):
        return None
    extra = {"creationflags": 0x08000000} if sys.platform == "win32" else {}  # no console window
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False, **extra)
        return out.stdout if out.returncode == 0 else None
    except (subprocess.TimeoutExpired, OSError, UnicodeDecodeError):
        return None


def _int(text) -> int | None:
    try:
        return int(float(str(text).strip()))
    except ValueError:
        return None


# ── GPUs ──────────────────────────────────────────────────────────────────────

def _nvidia() -> list[Gpu]:
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits"])
    gpus = []
    for line in (out or "").strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 4:
            gpus.append(Gpu("nvidia", p[0], _int(p[1]), _int(p[2]), p[3], "cuda"))
    return gpus


def _read(path: Path) -> str:
    try:
        return path.read_text().strip()
    except (OSError, UnicodeDecodeError):
        return ""


# A card below this is a processor's built-in graphics with a slice of the RAM.
_AMD_LEAST_MIB = 3500


def _amd_linux(drm: Path = Path("/sys/class/drm"), kfd: Path = Path("/dev/kfd")) -> list[Gpu]:
    gpus = []
    for dev in sorted(drm.glob("card[0-9]*/device")):
        if _read(dev / "vendor") != "0x1002":
            continue
        total, used = _int(_read(dev / "mem_info_vram_total") or ""), _int(_read(dev / "mem_info_vram_used") or "")
        mib = total // (1 << 20) if total else None
        big = bool(mib and mib >= _AMD_LEAST_MIB)
        name = _read(dev / "product_name") or f"AMD graphics ({_read(dev / 'device') or '?'})"
        gpus.append(Gpu("amd", name, mib, (total - used) // (1 << 20) if total and used is not None else None, None,
                        ("rocm" if kfd.exists() else "vulkan") if big else None))
    return gpus


def _intel_linux() -> list[Gpu]:
    for line in (_run(["lspci"]) or "").splitlines():
        low = line.lower()
        if ("vga" in low or "display" in low or "3d" in low) and "intel" in low:
            # lspci line: "<slot> <class>: <vendor device>" — the slot has
            # colons but no colon-space, so split on the first ": ".
            return [Gpu("intel", line.split(": ", 1)[-1].strip(), None)]
    return []


def _windows_adapters() -> list[tuple[str, str, int | None]]:
    """(hardware id, name, dedicated bytes) from the display class in the
    registry: what Windows itself records, 64-bit, for every vendor."""
    import winreg

    out = []
    root = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, root) as cls:
        for i in range(64):
            try:
                sub = winreg.EnumKey(cls, i)
                if not sub.isdigit():
                    continue
                with winreg.OpenKey(cls, sub) as k:
                    def value(name, k=k):
                        try:
                            return winreg.QueryValueEx(k, name)[0]
                        except OSError:
                            return None
                    size = value("HardwareInformation.qwMemorySize")
                    if isinstance(size, bytes):
                        size = int.from_bytes(size[:8], "little")
                    out.append((str(value("MatchingDeviceId") or "").lower(), str(value("DriverDesc") or ""), size))
            except OSError:
                if i:
                    break
    return out


def _from_windows(adapters: list[tuple[str, str, int | None]]) -> list[Gpu]:
    gpus, seen = [], set()
    for hwid, name, size in adapters:
        vendor = {"ven_1002": "amd", "ven_8086": "intel"}.get((re.search(r"ven_[0-9a-f]{4}", hwid) or [""])[0])
        if vendor is None or (hwid, name) in seen:
            continue
        seen.add((hwid, name))
        mib = size // (1 << 20) if size else None
        big = vendor == "amd" and bool(mib and mib >= _AMD_LEAST_MIB)
        gpus.append(Gpu(vendor, name, mib, None, None, "vulkan" if big else None))
    return gpus


def _apple(ram_mib: int | None, chip: str) -> list[Gpu]:
    """The GPU shares the memory. macOS would let it have two thirds or more; half
    is what a model may have here, so that the programs it helps with run too."""
    if not ram_mib:
        return []
    return [Gpu("apple", chip or "Apple Silicon", ram_mib // 2, None, None, "metal")]


# ── the rest ──────────────────────────────────────────────────────────────────

def _memory() -> tuple[int | None, int | None]:
    try:
        import psutil

        m = psutil.virtual_memory()
        return m.total // (1 << 20), m.available // (1 << 20)
    except Exception:
        return None, None


def _disk_free(path: Path) -> int | None:
    for p in (path, *path.parents):
        try:
            return shutil.disk_usage(p).free // (1 << 20)
        except OSError:
            continue
    return None


def _cpu(system: str) -> tuple[str, bool | None]:
    if system == "linux":
        info = _read(Path("/proc/cpuinfo"))
        name = re.search(r"^model name\s*:\s*(.+)$", info, re.M)
        flags = re.search(r"^flags\s*:\s*(.+)$", info, re.M)
        return (name.group(1) if name else platform.processor(),
                "avx2" in flags.group(1).split() if flags else None)
    if system == "macos":
        return (_run(["sysctl", "-n", "machdep.cpu.brand_string"]) or "").strip(), None
    if system == "windows":
        try:
            import ctypes

            avx2 = bool(ctypes.windll.kernel32.IsProcessorFeaturePresent(40))  # PF_AVX2_INSTRUCTIONS_AVAILABLE
        except Exception:
            avx2 = None
        return platform.processor(), avx2
    return platform.processor(), None


def probe() -> Hardware:
    from . import paths

    system = {"linux": "linux", "darwin": "macos", "win32": "windows"}.get(sys.platform, sys.platform)
    machine = {"amd64": "x86_64", "x86_64": "x86_64", "aarch64": "arm64", "arm64": "arm64"}.get(
        platform.machine().lower(), platform.machine().lower())
    total, available = _memory()
    cpu, avx2 = _cpu(system)
    hw = Hardware(system, machine, cpu, os.cpu_count(), avx2 if machine == "x86_64" else None, total, available,
                  _disk_free(paths.data_dir()))
    for find in {"linux": (_nvidia, _amd_linux, _intel_linux),
                 "windows": (_nvidia, lambda: _from_windows(_windows_adapters())),
                 "macos": (lambda: _apple(total, cpu) if machine == "arm64" else [],)}.get(system, (_nvidia,)):
        try:
            hw.gpus += find()
        except Exception:  # a probe is never why the engine does not start
            continue
    return hw


def as_dict() -> dict:
    hw = probe()
    d = asdict(hw)
    # A coarse "best inference target" hint consumers can show in setup UX.
    g = hw.best_gpu()
    d["inference_target"] = f"{g.name} ({g.vram_mib} MiB, {g.compute.upper()})" if g else "cpu"
    return d
