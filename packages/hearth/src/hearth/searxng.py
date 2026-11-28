"""A SearXNG of hearth's own, in a container, for web search.

Without it — or a Brave key — search is down to Brave's HTML page, which now
answers 429 on the first request, and Wikipedia. SearXNG is the one keyless
route that holds up (see websearch.py), so the installer offers it.

It runs in Podman or Docker, whichever is here and works without sudo; hearth
does not install either. The image is the release's pin, by digest. The
container listens on loopback only, is named and labelled as hearth's, and
`hearth uninstall` removes it with the rest; its settings are in the install
folder under searxng/.
"""

from __future__ import annotations

import secrets
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Callable

import httpx

NAME = "hearth-searxng"
LABEL = "io.hearth.part=searxng"
PORTS = range(8888, 8899)

Say = Callable[[str], None]

# use_default_settings keeps upstream's engines; this only changes what hearth
# needs: the JSON API on (websearch._searxng reads it), and the bot limiter off,
# since the only client is this machine and the limiter would want a Redis.
SETTINGS = """\
# Made by hearth's installer. `hearth uninstall` removes it.
use_default_settings: true
server:
  secret_key: "{secret}"
  limiter: false
  image_proxy: false
search:
  formats:
    - html
    - json
"""


class Unavailable(Exception):
    """No container runtime here that can be used."""


def runtime() -> str:
    """Podman first (rootless by design), then Docker — each only if it answers
    for this user without sudo."""
    for name in ("podman", "docker"):
        exe = shutil.which(name)
        if not exe:
            continue
        try:
            if subprocess.run([exe, "info"], capture_output=True, timeout=30).returncode == 0:
                return exe
        except (OSError, subprocess.SubprocessError):
            continue
    raise Unavailable("SearXNG needs Podman or Docker, usable without sudo; neither is here")


def _free_port() -> int:
    for port in PORTS:
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise Unavailable(f"no free port in {PORTS.start}-{PORTS.stop - 1}")


def write_settings(folder: Path) -> Path:
    """Its settings.yml; a secret already there is kept."""
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / "settings.yml"
    if not p.exists():
        p.write_text(SETTINGS.format(secret=secrets.token_hex(32)), encoding="utf-8")
    return p


def _ours(rt: str) -> bool:
    """A container by our name exists — and it is ours."""
    r = subprocess.run([rt, "inspect", "--format", "{{index .Config.Labels \"io.hearth.part\"}}", NAME],
                       capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        return False
    if r.stdout.strip() != "searxng":
        raise Unavailable(f"a container named {NAME} is here and is not hearth's; it was left alone")
    return True


def answers(url: str, timeout: float = 5.0) -> bool:
    try:
        r = httpx.get(f"{url}/search", params={"q": "hearth", "format": "json"}, timeout=timeout)
        return r.status_code == 200 and "results" in r.json()
    except (httpx.HTTPError, ValueError):
        return False


def setup(image: str, folder: Path, say: Say, wait_s: float = 90.0) -> str:
    """Pull and run it; return its URL once it answers JSON."""
    if not image:
        raise Unavailable("this release has no SearXNG image pinned")
    rt = runtime()
    settings = write_settings(folder)
    say(f"Getting SearXNG ({Path(rt).name})…")
    r = subprocess.run([rt, "pull", "--quiet", image], capture_output=True, text=True)
    if r.returncode != 0:
        raise Unavailable(f"the SearXNG image could not be pulled: {(r.stderr or r.stdout).strip()[:200]}")
    if _ours(rt):
        subprocess.run([rt, "rm", "-f", NAME], capture_output=True, timeout=60)
    port = _free_port()
    r = subprocess.run([
        rt, "run", "-d", "--name", NAME, "--label", LABEL, "--restart", "unless-stopped",
        "-p", f"127.0.0.1:{port}:8080",
        # :Z relabels for SELinux (Fedora); elsewhere it is ignored.
        "-v", f"{settings.parent}:/etc/searxng:Z",
        image], capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise Unavailable(f"SearXNG did not start: {(r.stderr or r.stdout).strip()[:200]}")
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if answers(url):
            return url
        time.sleep(1.0)
    raise Unavailable(f"SearXNG started but did not answer at {url}")


def remove() -> bool:
    """Take our container away, if there is one. Never someone else's."""
    try:
        rt = runtime()
        if not _ours(rt):
            return False
        subprocess.run([rt, "rm", "-f", NAME], capture_output=True, timeout=60)
        return True
    except (Unavailable, OSError, subprocess.SubprocessError):
        return False
