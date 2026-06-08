"""Desktop integration for the hearth chat UI.

The chat UI is served by the gateway at ``http://127.0.0.1:11435/app`` (see
``routes/webui.py``). This module turns it into a clickable desktop app:

  • ``open_window`` shows a chrome-less native WINDOW on that URL. It prefers
    ``pywebview`` — which wraps each OS's own webview (WebView2 on Windows,
    WKWebView on macOS, WebKitGTK on Linux), so there is no bundled browser —
    and falls back to a Chromium/Edge ``--app=`` window, then the default
    browser. It first ensures the gateway is running, starting it detached if
    not, so clicking the icon "just works".

  • ``install`` / ``uninstall`` create a launcher the OS shows in its app menu:
    an XDG ``.desktop`` file on Linux, a minimal ``.app`` bundle on macOS, and
    a Start-Menu ``.lnk`` on Windows — the clickable, GUI counterpart to the
    ``hearth start`` / ``hearth chat`` terminal commands.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import time
import webbrowser
from importlib import resources
from pathlib import Path

import httpx

from . import config as cfgmod

# ── URLs / gateway lifecycle ──────────────────────────────────────────────────


def app_url(cfg: cfgmod.Config | None = None) -> str:
    cfg = cfg or cfgmod.load()
    return f"http://{cfg.bind_host}:{cfg.bind_port}/app"


def _health_url(cfg: cfgmod.Config) -> str:
    return f"http://{cfg.bind_host}:{cfg.bind_port}/admin/health"


def gateway_up(cfg: cfgmod.Config) -> bool:
    try:
        # 503 = up but a backend is degraded; the UI still loads, so treat as up.
        return httpx.get(_health_url(cfg), timeout=1.5).status_code in (200, 503)
    except httpx.HTTPError:
        return False


def ensure_gateway_up(cfg: cfgmod.Config, hearth_exec: list[str], timeout: float = 25.0) -> bool:
    """Return True if the gateway is serving; if not, start it DETACHED (so the
    engine outlives this launcher and the window) and wait until it's healthy."""
    if gateway_up(cfg):
        return True
    if not hearth_exec:
        return False
    kwargs: dict = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if os.name == "posix":
        kwargs["start_new_session"] = True  # detach: survives window/launcher exit
    else:
        # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        kwargs["creationflags"] = 0x00000008 | 0x00000200
    try:
        subprocess.Popen([*hearth_exec, "start"], **kwargs)
    except OSError:
        return False
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if gateway_up(cfg):
            return True
        time.sleep(0.5)
    return False


# ── the window ────────────────────────────────────────────────────────────────


def _app_mode_browser() -> list[str] | None:
    """A Chromium-family browser that supports a chrome-less ``--app=`` window."""
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser",
                 "brave-browser", "microsoft-edge", "msedge"):
        path = shutil.which(name)
        if path:
            return [path]
    # macOS app bundles aren't on PATH; probe the usual spots.
    if sys.platform == "darwin":
        for p in ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                  "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
                  "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"):
            if Path(p).exists():
                return [p]
    return None


def _window_icon() -> str | None:
    """A PNG path for the pywebview window title-bar/taskbar icon, if we can
    produce one (pywebview's icon= wants a raster file, not SVG). Best-effort."""
    png = _user_data_dir() / "hearth.png"
    if png.exists():
        return str(png)
    svg = _bundled_asset("hearth.svg")
    if svg and _svg_to_png(svg, png):
        return str(png)
    return None


def open_window(cfg: cfgmod.Config | None = None, hearth_exec: list[str] | None = None) -> int:
    cfg = cfg or cfgmod.load()
    url = app_url(cfg)
    if not ensure_gateway_up(cfg, hearth_exec or []):
        print("warning: the hearth gateway is not responding; the window may show "
              "'no engine' until it starts (run `hearth start`).", file=sys.stderr)

    # 1) pywebview — a real chrome-less native window using the OS webview.
    try:
        import webview  # type: ignore

        webview.create_window("Hearth", url, width=1024, height=760, min_size=(420, 480))
        icon = _window_icon()
        try:
            webview.start(icon=icon) if icon else webview.start()
        except TypeError:
            webview.start()  # older pywebview has no icon= kwarg
        return 0
    except ImportError:
        pass  # pywebview not installed — fall through
    except Exception as e:  # backend missing / no display — fall through
        print(f"(pywebview unavailable: {e}; falling back to a browser window)", file=sys.stderr)

    # 2) Chromium/Edge --app= → still a chrome-less window, zero extra deps.
    browser = _app_mode_browser()
    if browser:
        try:
            kwargs = {"start_new_session": True} if os.name == "posix" else {}
            subprocess.Popen([*browser, f"--app={url}"], **kwargs)
            print(f"opened Hearth in an app window ({Path(browser[0]).name}).")
            return 0
        except OSError:
            pass

    # 3) Last resort: a normal browser tab.
    print(f"opening Hearth at {url} in your default browser "
          f"(install pywebview for a standalone window: pip install 'hearth[gui]').")
    webbrowser.open(url)
    return 0


# ── assets / icon helpers ─────────────────────────────────────────────────────


def _bundled_asset(name: str) -> Path | None:
    try:
        p = resources.files("hearth").joinpath("assets", name)
        return Path(str(p)) if p.is_file() else None
    except (FileNotFoundError, OSError, ModuleNotFoundError):
        return None


def _user_data_dir() -> Path:
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    d = Path(base) / "hearth"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _svg_to_png(svg: Path, png: Path, size: int = 256) -> bool:
    """Best-effort SVG→PNG using whatever converter is on the system. Returns
    False (no crash) if none is available — the icon is then just omitted."""
    png.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("rsvg-convert"):
        cmd = ["rsvg-convert", "-w", str(size), "-h", str(size), str(svg), "-o", str(png)]
    elif shutil.which("inkscape"):
        cmd = ["inkscape", str(svg), "-w", str(size), "-h", str(size), "-o", str(png)]
    elif shutil.which("convert"):  # ImageMagick
        cmd = ["convert", "-background", "none", "-resize", f"{size}x{size}", str(svg), str(png)]
    else:
        try:  # Pillow + cairosvg if either path is installed
            import cairosvg  # type: ignore
            cairosvg.svg2png(url=str(svg), write_to=str(png), output_width=size, output_height=size)
            return png.exists()
        except Exception:
            return False
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return png.exists()
    except (OSError, subprocess.CalledProcessError):
        return False


# ── install / uninstall (per-OS launcher) ─────────────────────────────────────


def install(exec_argv: list[str]) -> int:
    """Create a clickable desktop launcher. ``exec_argv`` is the UNQUOTED argv
    prefix that invokes hearth (e.g. ``["/home/u/.local/bin/hearth"]`` or
    ``["/usr/bin/python", "-m", "hearth"]``); the launcher runs it as
    ``<exec_argv...> gui``. Each OS quotes it for its own launcher format."""
    if sys.platform.startswith("linux"):
        return _install_linux(exec_argv)
    if sys.platform == "darwin":
        return _install_macos(exec_argv)
    if os.name == "nt":
        return _install_windows(exec_argv)
    print(f"desktop install not supported on {sys.platform!r}; run `hearth gui` directly.",
          file=sys.stderr)
    return 1


def uninstall() -> int:
    if sys.platform.startswith("linux"):
        return _uninstall_linux()
    if sys.platform == "darwin":
        return _uninstall_macos()
    if os.name == "nt":
        return _uninstall_windows()
    print(f"desktop uninstall not supported on {sys.platform!r}.", file=sys.stderr)
    return 1


# ---- Linux: XDG .desktop ----

_DESKTOP_NAME = "hearth-chat.desktop"
_DESKTOP_TEMPLATE = """\
[Desktop Entry]
Type=Application
Version=1.0
Name=Hearth Chat
GenericName=Local AI Chat
Comment=Chat with your local AI engine — runs entirely on this machine
Exec={exec} gui
Icon={icon}
Terminal=false
Categories=Utility;Network;
Keywords=ai;chat;llm;assistant;hearth;
StartupWMClass=hearth
"""


def _linux_apps_dir() -> Path:
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "applications"


def _remove_legacy_launcher() -> list[str]:
    """Remove the old ``make install-app`` engine launcher (hearth.desktop +
    hearth-app-launch.sh) if present. It's superseded by this chat client, which
    starts the engine on open and adds a window — so we migrate rather than leave
    two confusingly-similar entries. Returns the paths removed."""
    data = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    removed = []
    for p in (Path(data) / "applications" / "hearth.desktop",
              Path(data) / "hearth" / "hearth-app-launch.sh"):
        if p.exists():
            try:
                p.unlink()
                removed.append(str(p))
            except OSError:
                pass
    return removed


def _desktop_exec(tokens: list[str]) -> str:
    """Join argv into an XDG Desktop Entry Exec value. The spec uses double-quote
    quoting (with backslash escaping of " \\ ` $) for tokens that need it;
    no-space tokens (the common case) are emitted bare."""
    out = []
    for t in tokens:
        if any(c in t for c in ' \t\n"\'\\`$'):
            esc = (t.replace("\\", "\\\\").replace('"', '\\"')
                    .replace("`", "\\`").replace("$", "\\$"))
            out.append(f'"{esc}"')
        else:
            out.append(t)
    return " ".join(out)


def _install_linux(exec_argv: list[str]) -> int:
    apps = _linux_apps_dir()
    apps.mkdir(parents=True, exist_ok=True)
    # Icon: install the SVG into the hicolor theme; reference it by absolute path
    # (a bare name would require the theme cache to know it). Fall back to a
    # stock icon name if we somehow can't place the asset.
    icon_ref = "applications-internet"
    svg = _bundled_asset("hearth.svg")
    if svg:
        icon_dir = (Path(os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share"))
                    / "icons" / "hicolor" / "scalable" / "apps")
        icon_dir.mkdir(parents=True, exist_ok=True)
        dest = icon_dir / "hearth.svg"
        try:
            shutil.copyfile(svg, dest)
            icon_ref = str(dest)
        except OSError:
            pass

    exec_field = _desktop_exec(exec_argv)
    desktop = apps / _DESKTOP_NAME
    desktop.write_text(_DESKTOP_TEMPLATE.format(exec=exec_field, icon=icon_ref))
    desktop.chmod(0o644)
    for legacy in _remove_legacy_launcher():
        print(f"  (migrated: removed the old engine launcher {legacy})")
    if shutil.which("update-desktop-database"):
        subprocess.run(["update-desktop-database", str(apps)], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"✓ installed {desktop}")
    print(f"    Exec={exec_field} gui")
    print("Look for 'Hearth Chat' in your application menu (or run: hearth gui).")
    return 0


def _uninstall_linux() -> int:
    apps = _linux_apps_dir()
    removed = []
    desktop = apps / _DESKTOP_NAME
    if desktop.exists():
        desktop.unlink()
        removed.append(str(desktop))
    icon = (Path(os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share"))
            / "icons" / "hicolor" / "scalable" / "apps" / "hearth.svg")
    if icon.exists():
        icon.unlink()
        removed.append(str(icon))
    removed += _remove_legacy_launcher()
    if shutil.which("update-desktop-database"):
        subprocess.run(["update-desktop-database", str(apps)], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("✓ removed Hearth Chat launcher" + (":" if removed else " (nothing was installed)"))
    for r in removed:
        print(f"    {r}")
    return 0


# ---- macOS: .app bundle ----

_PLIST = """\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>Hearth Chat</string>
  <key>CFBundleDisplayName</key><string>Hearth</string>
  <key>CFBundleIdentifier</key><string>dev.hearth.chat</string>
  <key>CFBundleVersion</key><string>1.0</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleExecutable</key><string>hearth-chat</string>
  <key>CFBundleIconFile</key><string>hearth.icns</string>
  <key>LSMinimumSystemVersion</key><string>10.13</string>
</dict></plist>
"""


def _macos_app_dir() -> Path:
    return Path.home() / "Applications" / "Hearth.app"


def _install_macos(exec_argv: list[str]) -> int:
    app = _macos_app_dir()
    macos = app / "Contents" / "MacOS"
    resd = app / "Contents" / "Resources"
    macos.mkdir(parents=True, exist_ok=True)
    resd.mkdir(parents=True, exist_ok=True)

    launcher = macos / "hearth-chat"
    cmd = " ".join(shlex.quote(a) for a in [*exec_argv, "gui"])
    launcher.write_text(f"#!/bin/sh\nexec {cmd}\n")
    launcher.chmod(0o755)
    (app / "Contents" / "Info.plist").write_text(_PLIST)

    # Icon best-effort: SVG → PNG → .icns via the system `sips`/`iconutil`.
    svg = _bundled_asset("hearth.svg")
    if svg:
        png = resd / "hearth.png"
        if _svg_to_png(svg, png, size=512) and shutil.which("sips"):
            iconset = resd / "hearth.iconset"
            iconset.mkdir(exist_ok=True)
            ok = True
            for s in (16, 32, 64, 128, 256, 512):
                try:
                    subprocess.run(["sips", "-z", str(s), str(s), str(png),
                                    "--out", str(iconset / f"icon_{s}x{s}.png")],
                                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except (OSError, subprocess.CalledProcessError):
                    ok = False
                    break
            if ok and shutil.which("iconutil"):
                subprocess.run(["iconutil", "-c", "icns", str(iconset),
                                "-o", str(resd / "hearth.icns")], check=False,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            shutil.rmtree(iconset, ignore_errors=True)

    print(f"✓ installed {app}")
    print("Find 'Hearth' in Launchpad / ~/Applications (or run: hearth gui).")
    return 0


def _uninstall_macos() -> int:
    app = _macos_app_dir()
    if app.exists():
        shutil.rmtree(app, ignore_errors=True)
        print(f"✓ removed {app}")
    else:
        print("✓ nothing to remove (Hearth.app not installed)")
    return 0


# ---- Windows: Start-Menu .lnk via PowerShell (no extra deps) ----


def _windows_lnk_path() -> Path:
    base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    return Path(base) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Hearth Chat.lnk"


def _install_windows(exec_argv: list[str]) -> int:
    lnk = _windows_lnk_path()
    lnk.parent.mkdir(parents=True, exist_ok=True)
    icon = ""
    svg = _bundled_asset("hearth.svg")
    if svg:
        ico = _user_data_dir() / "hearth.png"
        if _svg_to_png(svg, ico):
            icon = str(ico)

    def psq(s: str) -> str:
        # PowerShell single-quoted literal: backslashes stay raw (correct for
        # Windows paths), and an embedded single quote is escaped by doubling.
        return "'" + str(s).replace("'", "''") + "'"

    target = exec_argv[0]
    arguments = " ".join([*exec_argv[1:], "gui"])
    # WScript.Shell creates a proper shortcut without pywin32.
    ps = (
        "$ws = New-Object -ComObject WScript.Shell; "
        f"$s = $ws.CreateShortcut({psq(str(lnk))}); "
        f"$s.TargetPath = {psq(target)}; "
        f"$s.Arguments = {psq(arguments)}; "
        "$s.Description = 'Chat with your local AI engine'; "
        + (f"$s.IconLocation = {psq(icon)}; " if icon else "")
        + "$s.Save()"
    )
    try:
        subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError) as e:
        print(f"could not create Start-Menu shortcut: {e}", file=sys.stderr)
        return 1
    print(f"✓ installed {lnk}")
    print("Find 'Hearth Chat' in the Start menu (or run: hearth gui).")
    return 0


def _uninstall_windows() -> int:
    lnk = _windows_lnk_path()
    if lnk.exists():
        lnk.unlink()
        print(f"✓ removed {lnk}")
    else:
        print("✓ nothing to remove (no Start-Menu shortcut)")
    return 0
