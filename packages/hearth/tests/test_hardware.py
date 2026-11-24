"""What each kind of computer is found to have, from what its system reports."""

from __future__ import annotations

from hearth import hardware
from hearth.hardware import Gpu, Hardware


def test_probe_returns_shape():
    hw = hardware.probe()
    assert hw.system in ("linux", "macos", "windows")
    assert hw.machine in ("x86_64", "arm64")
    assert hw.cpu_cores is None or hw.cpu_cores > 0
    assert hw.ram_total_mib > 500 and hw.disk_free_mib > 0  # every system, not Linux alone


def test_as_dict_has_inference_target():
    d = hardware.as_dict()
    assert "inference_target" in d
    assert "gpus" in d and "ram_total_mib" in d


def test_a_probe_that_breaks_is_left_out(monkeypatch):
    for name in ("_nvidia", "_amd_linux", "_intel_linux", "_windows_adapters", "_apple"):
        def broken(*a, **k):
            raise RuntimeError("no")
        monkeypatch.setattr(hardware, name, broken)
    assert hardware.probe().gpus == []


def test_nvidia(monkeypatch):
    monkeypatch.setattr(hardware, "_run", lambda cmd, **k:
                        "NVIDIA GeForce RTX 4090, 24564, 23000, 550.54\nNVIDIA GeForce GT 1030, [N/A], [N/A], 550.54\nbroken\n")
    assert hardware._nvidia() == [Gpu("nvidia", "NVIDIA GeForce RTX 4090", 24564, 23000, "550.54", "cuda"),
                                  Gpu("nvidia", "NVIDIA GeForce GT 1030", None, None, "550.54", "cuda")]
    monkeypatch.setattr(hardware, "_run", lambda cmd, **k: None)  # no driver
    assert hardware._nvidia() == []


def _card(drm, n, **files):
    d = drm / f"card{n}" / "device"
    d.mkdir(parents=True)
    for name, text in files.items():
        (d / name).write_text(text + "\n")


def test_amd_on_linux(tmp_path):
    _card(tmp_path, 0, vendor="0x1002", device="0x744c", mem_info_vram_total=str(24 << 30), mem_info_vram_used=str(1 << 30),
          product_name="Radeon RX 7900 XTX")
    _card(tmp_path, 1, vendor="0x1002", device="0x164e", mem_info_vram_total=str(512 << 20))  # a processor's own
    _card(tmp_path, 2, vendor="0x8086", device="0x7d55")
    (tmp_path / "card0-DP-1").mkdir()  # a connector, not a card
    kfd = tmp_path / "kfd"
    big, small = hardware._amd_linux(tmp_path, kfd)
    assert (big.name, big.vram_mib, big.free_mib, big.compute) == ("Radeon RX 7900 XTX", 24576, 23552, "vulkan")
    assert small.compute is None and small.vram_mib == 512 and "0x164e" in small.name
    kfd.write_text("")
    assert hardware._amd_linux(tmp_path, kfd)[0].compute == "rocm"


def test_windows_adapters():
    got = hardware._from_windows([
        (r"pci\ven_1002&dev_7550", "AMD Radeon RX 9070 XT", 16 << 30),
        (r"pci\ven_1002&dev_7550", "AMD Radeon RX 9070 XT", 16 << 30),  # an old driver's entry
        (r"pci\ven_1002&dev_164e", "AMD Radeon(TM) Graphics", 512 << 20),
        (r"pci\ven_10de&dev_2684", "NVIDIA GeForce RTX 4090", 24 << 30),  # nvidia-smi's to report
        (r"pci\ven_8086&dev_7d55", "Intel(R) Arc(TM) Graphics", 128 << 20),
        ("root\\basicdisplay", "Microsoft Basic Display Adapter", None)])
    assert [(g.vendor, g.vram_mib, g.compute) for g in got] == [
        ("amd", 16384, "vulkan"), ("amd", 512, None), ("intel", 128, None)]


def test_apple_shares_its_memory():
    assert hardware._apple(65536, "Apple M4 Max") == [Gpu("apple", "Apple M4 Max", 32768, None, None, "metal")]
    assert hardware._apple(None, "") == []


def test_the_gpu_to_size_by():
    a, b = Gpu("nvidia", "a", 8192, None, None, "cuda"), Gpu("nvidia", "b", 12288, None, None, "cuda")
    assert Hardware(gpus=[a, b, Gpu("intel", "i", 60000)]).best_gpu() is b  # not their sum, and not a built-in one
    assert Hardware(gpus=[Gpu("nvidia", "?", None, None, None, "cuda")]).best_gpu() is None
    assert Hardware().best_gpu() is None
