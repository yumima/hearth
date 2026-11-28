#!/usr/bin/env python3
"""Make the files of a release, in a folder.

    uv run --no-project tools/release.py OUT [--base-url URL]

Both wheels, what each needs (with hashes, from uv.lock), manifest.json naming
every file and its hash, and the install scripts, which carry the manifest's hash:
somebody who trusts the script trusts exactly these files.

And SearXNG, hearth's search backend: a wheel built here from the commit pinned
in installer/pins.json, with installer/searxng-requirements.txt (hashes, wheels
only). After moving the pin, remake that file from the commit's requirements.txt:

    uv pip compile requirements.txt --universal --python-version 3.12 \
        --generate-hashes --no-build --no-header --no-annotate -o installer/searxng-requirements.txt
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "3.12.11"
# Not the client's "gui": it needs a package that has no wheel, and nothing is built on
# the user's computer. Its window is a browser's, without the browser around it.
EXTRAS = {"hearth": ["tts", "stt"], "client": []}
PACKAGES = {"hearth": "hearth", "client": "hearth-client"}


def sh(*cmd: str) -> str:
    return subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True).stdout


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def version() -> str:
    text = (ROOT / "packages/hearth/src/hearth/__init__.py").read_text(encoding="utf-8")
    return re.search(r'^__version__ = "([^"]+)"', text, re.M).group(1)


def requirements(name: str) -> str:
    extras = [x for e in EXTRAS[name] for x in ("--extra", e)]
    text = sh("uv", "export", "--quiet", "--frozen", "--package", PACKAGES[name], "--no-emit-project", "--no-emit-workspace",
              "--no-dev", "--no-header", "--no-annotate", *extras)
    assert "-e " not in text and "file:" not in text, "something is not from the index"
    lock = {p["name"]: p for p in tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))["package"]}
    for line in re.findall(r"^([a-z0-9-]+)==", text, re.M):
        assert lock[line].get("wheels"), f"{line} has no wheel: it would be built on the user's computer"
    return text


def searxng_wheel(pin: dict, out: Path) -> Path:
    """Its wheel, built from the pinned commit. Its setup.py imports searx, so
    the build runs in an environment that has what it needs — the same hashed
    files the user's install gets."""
    reqs = ROOT / "installer" / "searxng-requirements.txt"
    with tempfile.TemporaryDirectory() as t:
        repo, src, env = Path(t) / "repo", Path(t) / "src", Path(t) / "env"
        subprocess.run(["git", "init", "-q", "--bare", str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "fetch", "-q", "--depth", "1", pin["repo"], pin["commit"]], check=True)
        got = subprocess.run(["git", "-C", str(repo), "rev-parse", "FETCH_HEAD"], check=True, capture_output=True, text=True)
        assert got.stdout.strip() == pin["commit"], got.stdout
        # Only what setup.py reads, out of the commit itself: elsewhere
        # (utils/templates) are file names with ":", which a Windows checkout
        # refuses even when sparse; this archive leaves them out.
        # (Git for Windows checks NTFS names whenever it reads a tree, so even an
        # archive needs that check off — safe: it writes nothing to disk.)
        tar = subprocess.run(["git", "-c", "core.protectNTFS=false", "-C", str(repo), "archive", "--format=tar", pin["commit"], "searx", "setup.py",
                              "README.rst", "LICENSE", "AUTHORS.rst", "requirements.txt", "requirements-dev.txt"],
                             check=True, stdout=subprocess.PIPE).stdout  # (its errors to the log)
        with tarfile.open(fileobj=io.BytesIO(tar)) as f:
            f.extractall(src, filter="data")
        # Frozen, as upstream's own archives are: unfrozen, searx asks git for its
        # version at import time — on a computer without git that is a crash. And
        # a plain date keeps "+" (a local version) out of the asset's file name.
        ymd = subprocess.run(["git", "-C", str(repo), "show", "-s", "--format=%cs", pin["commit"]],
                             check=True, capture_output=True, text=True).stdout.strip()
        day = ".".join(str(int(x)) for x in ymd.split("-"))  # as upstream tags it: 2026.11.11
        (src / "searx" / "version_frozen.py").write_text(
            f'VERSION_STRING = "{day}-{pin["commit"][:9]}"\nVERSION_TAG = "{day}"\nDOCKER_TAG = "{day}-{pin["commit"][:9]}"\n'
            f'GIT_URL = "{pin["repo"]}"\nGIT_BRANCH = "master"\n', encoding="utf-8")
        sh("uv", "venv", "--quiet", "--python", PYTHON, str(env))
        sh("uv", "pip", "install", "--quiet", "--python", str(env), "--require-hashes", "--only-binary", ":all:", "-r", str(reqs))
        sh("uv", "pip", "install", "--quiet", "--python", str(env), "setuptools", "wheel")
        before = set(out.glob("searxng-*.whl"))
        subprocess.run(["uv", "build", "--quiet", "--wheel", "--no-build-isolation", "--python", str(env), "-o", str(out),
                        str(src)], check=True, cwd=t)
        new = sorted(set(out.glob("searxng-*.whl")) - before)
        assert len(new) == 1, new
        return new[0]


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("out", type=Path)
    a.add_argument("--base-url", default="")
    a.add_argument("--pins", type=Path, default=ROOT / "installer/pins.json")
    a.add_argument("--no-extras", action="store_true", help="without voices, the window and SearXNG: for tests")
    args = a.parse_args()
    if args.no_extras:
        EXTRAS.update({k: [] for k in EXTRAS})
    out, v = args.out.resolve(), version()
    out.mkdir(parents=True, exist_ok=True)
    base = args.base_url or f"https://github.com/yumima/hearth/releases/download/v{v}/"
    sh("uv", "lock", "--check")
    files = {}
    for name, package in PACKAGES.items():
        sh("uv", "build", "--quiet", "--wheel", "--package", package, "-o", str(out))
        (out / f"{name}-requirements.txt").write_text(requirements(name), encoding="utf-8", newline="\n")
        wheel = out / f"{package.replace('-', '_')}-{v}-py3-none-any.whl"
        for kind, p in (("wheel", wheel), ("requirements", out / f"{name}-requirements.txt")):
            files[f"{name}-{kind}"] = {"name": p.name, "sha256": digest(p), "size": p.stat().st_size}
    pins = json.loads(args.pins.read_text(encoding="utf-8"))
    if not args.no_extras:
        wheel = searxng_wheel(pins["searxng"], out)
        reqs = shutil.copy(ROOT / "installer" / "searxng-requirements.txt", out / "searxng-requirements.txt")
        for kind, p in (("wheel", wheel), ("requirements", Path(reqs))):
            files[f"searxng-{kind}"] = {"name": p.name, "sha256": digest(p), "size": p.stat().st_size}
    manifest = {"schema": 1, "version": v, "python": PYTHON, "files": files, "ollama": pins["ollama"]}
    m = out / "manifest.json"
    m.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8", newline="\n")
    fill = {"@VERSION@": v, "@MANIFEST_URL@": base + "manifest.json", "@MANIFEST_SHA256@": digest(m), "@BASE_URL@": base,
            "@PYTHON@": PYTHON, "@UV_VERSION@": pins["uv"]["version"], "@WHEEL@": files["hearth-wheel"]["name"],
            "@WHEEL_SHA256@": files["hearth-wheel"]["sha256"], "@REQUIREMENTS_SHA256@": files["hearth-requirements"]["sha256"],
            **{f"@UV_{k}@": h for k, h in pins["uv"]["sha256"].items()}}
    for script in ("install.sh", "install.ps1"):
        text = (ROOT / "installer" / script).read_text(encoding="utf-8")
        for k, val in fill.items():
            text = text.replace(k, val)
        assert not re.search(r"@[A-Z0-9_]+@", text), re.search(r"@[A-Z0-9_]+@", text).group(0)
        (out / script).write_text(text, encoding="utf-8", newline="\n")
    (out / ".gitignore").unlink(missing_ok=True)  # uv's; not of the release
    sums = "".join(f"{digest(p)}  {p.name}\n" for p in sorted(out.iterdir()) if p.name != "SHA256SUMS")
    (out / "SHA256SUMS").write_text(sums, encoding="utf-8", newline="\n")
    print(out)


if __name__ == "__main__":
    main()
