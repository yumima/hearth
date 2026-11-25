"""Desktop integration: a chrome-less native window + a per-OS launcher.

``open_window(url)`` shows the client's UI in a native window via each OS's own
webview (WebView2 / WKWebView / WebKitGTK through pywebview — no bundled
browser), falling back to a Chromium/Edge ``--app`` window, then the default
browser. ``install`` / ``uninstall`` register a clickable launcher (XDG
``.desktop`` on Linux, ``.app`` bundle on macOS, Start-Menu ``.lnk`` on Windows)
whose icon runs bare ``hearth client`` (which opens the window).
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import webbrowser
from importlib import resources
from pathlib import Path


# ── the window ────────────────────────────────────────────────────────────────

# Default app-window size, shared by the native webview and the --app fallback.
# Sized for a chat UI: comfortably under a laptop screen and roughly half of a
# typical desktop, instead of Chromium's oversized default for app windows.
_WIN_W, _WIN_H = 1100, 850


def open_window(url: str) -> int:
    # 1) pywebview — a real chrome-less native window using the OS webview.
    try:
        import webview  # type: ignore

        webview.create_window("Hearth", url, width=_WIN_W, height=_WIN_H, min_size=(420, 480))
        icon = _window_icon()
        try:
            webview.start(icon=icon) if icon else webview.start()
        except TypeError:
            webview.start()  # older pywebview has no icon= kwarg
        return 0
    except ImportError:
        pass
    except Exception as e:  # backend missing / no display — fall through
        print(f"(pywebview unavailable: {e}; falling back to a browser window)", file=sys.stderr)

    # 2) Chromium/Edge --app= → a chrome-less window. Use a DEDICATED profile so
    # the launched binary is the window's own process, and BLOCK on it — the client's
    # server is a daemon thread in THIS process, so if we returned here the
    # process would exit and the server would die under the window (→ connection
    # refused). Closing the window unblocks us and the client exits cleanly.
    browser = _app_mode_browser()
    if browser:
        try:
            profile = str(_user_data_dir() / "chrome-profile")
            print(f"Hearth — app window via {Path(browser[0]).name}; close it to quit.")
            cmd = [*browser, f"--app={url}", f"--user-data-dir={profile}",
                   "--no-first-run", "--no-default-browser-check",
                   f"--window-size={_WIN_W},{_WIN_H}"]
            # Linux/GNOME dock integration. A Chromium --app window on *Wayland*
            # invents its own app_id ("chrome-<url>-<profile>"), so the running
            # window never matches our launcher (hearth-client.desktop /
            # StartupWMClass): no "running" dot, dock clicks relaunch instead of focusing,
            # and the right-click menu shows no window list. --class sets the X11
            # WM_CLASS — which GNOME *does* match to StartupWMClass — but Chromium
            # only honours it under X11/XWayland, not native Wayland. So on Linux
            # we pin the class to "hearth-client" and route the window through XWayland
            # (when present), making the dock treat it as a first-class app the way
            # Claude's desktop client behaves.
            if sys.platform.startswith("linux"):
                cmd.append("--class=hearth-client")
                if os.environ.get("DISPLAY"):  # XWayland (or X11) available
                    cmd.append("--ozone-platform=x11")
            proc = subprocess.Popen(cmd)
            proc.wait()
            return 0
        except OSError:
            pass

    # 3) Last resort: a normal browser tab. Block to keep the server alive.
    import threading
    print(f"Hearth is serving at {url} (no app-window backend — opened a browser tab).")
    print("Leave this running; Ctrl-C to stop.")
    webbrowser.open(url)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    return 0


def _app_mode_browser() -> list[str] | None:
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser",
                 "brave-browser", "microsoft-edge", "msedge"):
        path = shutil.which(name)
        if path:
            return [path]
    if sys.platform == "darwin":
        for p in ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                  "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
                  "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"):
            if Path(p).exists():
                return [p]
    return None


# ── assets / icon ─────────────────────────────────────────────────────────────


def _bundled_asset(name: str) -> Path | None:
    try:
        p = resources.files("hearth_client").joinpath("assets", name)
        return Path(str(p)) if p.is_file() else None
    except (FileNotFoundError, OSError, ModuleNotFoundError):
        return None


def _user_data_dir() -> Path:
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    d = Path(base) / "hearth-client"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _window_icon() -> str | None:
    png = _user_data_dir() / "hearth-client.png"
    if png.exists():
        return str(png)
    svg = _bundled_asset("hearth-client.svg")
    if svg and _svg_to_png(svg, png):
        return str(png)
    return None


def _svg_to_png(svg: Path, png: Path, size: int = 256) -> bool:
    """Best-effort SVG→PNG via whatever converter exists; False (no crash) if none."""
    png.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("rsvg-convert"):
        cmd = ["rsvg-convert", "-w", str(size), "-h", str(size), str(svg), "-o", str(png)]
    elif shutil.which("inkscape"):
        cmd = ["inkscape", str(svg), "-w", str(size), "-h", str(size), "-o", str(png)]
    elif shutil.which("convert"):
        cmd = ["convert", "-background", "none", "-resize", f"{size}x{size}", str(svg), str(png)]
    else:
        try:
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
    if sys.platform.startswith("linux"):
        return _install_linux(exec_argv)
    if sys.platform == "darwin":
        return _install_macos(exec_argv)
    if os.name == "nt":
        return _install_windows(exec_argv)
    print(f"desktop install not supported on {sys.platform!r}; run `hearth client` directly.",
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

_DESKTOP_NAME = "hearth-client.desktop"
_DESKTOP_TEMPLATE = """\
[Desktop Entry]
Type=Application
Version=1.0
Name=Hearth
GenericName=AI Chat
Comment=Chat with the AI on your own computer
Exec={exec}
Icon={icon}
Terminal=false
Categories=Utility;Network;
Keywords=ai;chat;llm;mcp;assistant;hearth;
StartupWMClass=hearth-client
"""


def _linux_apps_dir() -> Path:
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "applications"


def _desktop_exec(tokens: list[str]) -> str:
    """Join argv into an XDG Exec value (double-quote quoting for tokens needing it)."""
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
    icon_ref = "applications-internet"
    svg = _bundled_asset("hearth-client.svg")
    if svg:
        icon_dir = (Path(os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share"))
                    / "icons" / "hicolor" / "scalable" / "apps")
        icon_dir.mkdir(parents=True, exist_ok=True)
        dest = icon_dir / "hearth-client.svg"
        try:
            shutil.copyfile(svg, dest)
            icon_ref = str(dest)
        except OSError:
            pass

    exec_field = _desktop_exec(exec_argv)
    entry = apps / _DESKTOP_NAME
    entry.write_text(_DESKTOP_TEMPLATE.format(exec=exec_field, icon=icon_ref))
    entry.chmod(0o644)
    if shutil.which("update-desktop-database"):
        subprocess.run(["update-desktop-database", str(apps)], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"✓ installed {entry}")
    print(f"    Exec={exec_field}")
    print("Look for 'Hearth' in your application menu (or run: hearth client).")
    return 0


def _uninstall_linux() -> int:
    apps = _linux_apps_dir()
    removed = []
    entry = apps / _DESKTOP_NAME
    if entry.exists():
        entry.unlink()
        removed.append(str(entry))
    icon = (Path(os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share"))
            / "icons" / "hicolor" / "scalable" / "apps" / "hearth-client.svg")
    if icon.exists():
        icon.unlink()
        removed.append(str(icon))
    if shutil.which("update-desktop-database"):
        subprocess.run(["update-desktop-database", str(apps)], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("✓ removed Hearth launcher" + (":" if removed else " (nothing was installed)"))
    for r in removed:
        print(f"    {r}")
    return 0


# ---- macOS: .app bundle ----

_PLIST = """\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>Hearth</string>
  <key>CFBundleDisplayName</key><string>Hearth</string>
  <key>CFBundleIdentifier</key><string>io.github.yumima.hearth.client</string>
  <key>CFBundleVersion</key><string>1.0</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleExecutable</key><string>hearth-client</string>
  <key>CFBundleIconFile</key><string>hearth-client.icns</string>
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

    launcher = macos / "hearth-client"
    cmd = " ".join(shlex.quote(a) for a in exec_argv)
    launcher.write_text(f"#!/bin/sh\nexec {cmd}\n")
    launcher.chmod(0o755)
    (app / "Contents" / "Info.plist").write_text(_PLIST)

    svg = _bundled_asset("hearth-client.svg")
    if svg:
        png = resd / "hearth-client.png"
        if _svg_to_png(svg, png, size=512) and shutil.which("sips"):
            iconset = resd / "hearth-client.iconset"
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
                                "-o", str(resd / "hearth-client.icns")], check=False,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            shutil.rmtree(iconset, ignore_errors=True)

    print(f"✓ installed {app}")
    print("Find 'Hearth' in Launchpad / ~/Applications (or run: hearth client).")
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
    return Path(base) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Hearth.lnk"


def _install_windows(exec_argv: list[str]) -> int:
    lnk = _windows_lnk_path()
    lnk.parent.mkdir(parents=True, exist_ok=True)
    icon = ""
    svg = _bundled_asset("hearth-client.svg")
    if svg:
        ico = _user_data_dir() / "hearth-client.png"
        if _svg_to_png(svg, ico):
            icon = str(ico)

    def psq(s: str) -> str:
        return "'" + str(s).replace("'", "''") + "'"

    argv = exec_argv
    target = argv[0]
    arguments = " ".join(argv[1:])
    ps = (
        "$ws = New-Object -ComObject WScript.Shell; "
        f"$s = $ws.CreateShortcut({psq(str(lnk))}); "
        f"$s.TargetPath = {psq(target)}; "
        f"$s.Arguments = {psq(arguments)}; "
        "$s.Description = 'Chat with the AI on your own computer'; "
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
    print("Find 'Hearth' in the Start menu (or run: hearth client).")
    return 0


def _uninstall_windows() -> int:
    lnk = _windows_lnk_path()
    if lnk.exists():
        lnk.unlink()
        print(f"✓ removed {lnk}")
    else:
        print("✓ nothing to remove (no Start-Menu shortcut)")
    return 0
