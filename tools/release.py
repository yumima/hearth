#!/usr/bin/env python3
"""Make the files of a release, in a folder.

    uv run --no-project tools/release.py OUT [--base-url URL]

Both wheels, what each needs (with hashes, from uv.lock), manifest.json naming
every file and its hash, and the install scripts, which carry the manifest's hash:
somebody who trusts the script trusts exactly these files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "3.12.11"
# Not mantel's "gui": it needs a package that has no wheel, and nothing is built on
# the user's computer. Its window is a browser's, without the browser around it.
EXTRAS = {"hearth": ["tts", "stt"], "mantel": []}


def sh(*cmd: str) -> str:
    return subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True).stdout


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def version() -> str:
    text = (ROOT / "packages/hearth/src/hearth/__init__.py").read_text(encoding="utf-8")
    return re.search(r'^__version__ = "([^"]+)"', text, re.M).group(1)


def requirements(name: str) -> str:
    extras = [x for e in EXTRAS[name] for x in ("--extra", e)]
    text = sh("uv", "export", "--quiet", "--frozen", "--package", name, "--no-emit-project", "--no-emit-workspace",
              "--no-dev", "--no-header", "--no-annotate", *extras)
    assert "-e " not in text and "file:" not in text, "something is not from the index"
    lock = {p["name"]: p for p in tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))["package"]}
    for line in re.findall(r"^([a-z0-9-]+)==", text, re.M):
        assert lock[line].get("wheels"), f"{line} has no wheel: it would be built on the user's computer"
    return text


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("out", type=Path)
    a.add_argument("--base-url", default="")
    a.add_argument("--pins", type=Path, default=ROOT / "installer/pins.json")
    a.add_argument("--no-extras", action="store_true", help="without voices and the window: for tests")
    args = a.parse_args()
    if args.no_extras:
        EXTRAS.update({k: [] for k in EXTRAS})
    out, v = args.out.resolve(), version()
    out.mkdir(parents=True, exist_ok=True)
    base = args.base_url or f"https://github.com/yumima/hearth/releases/download/v{v}/"
    sh("uv", "lock", "--check")
    files = {}
    for name in ("hearth", "mantel"):
        sh("uv", "build", "--quiet", "--wheel", "--package", name, "-o", str(out))
        (out / f"{name}-requirements.txt").write_text(requirements(name), encoding="utf-8", newline="\n")
        wheel = out / f"{name}-{v}-py3-none-any.whl"
        for kind, p in (("wheel", wheel), ("requirements", out / f"{name}-requirements.txt")):
            files[f"{name}-{kind}"] = {"name": p.name, "sha256": digest(p), "size": p.stat().st_size}
    pins = json.loads(args.pins.read_text(encoding="utf-8"))
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
    sums = "".join(f"{digest(p)}  {p.name}\n" for p in sorted(out.iterdir()) if p.name != "SHA256SUMS")
    (out / "SHA256SUMS").write_text(sums, encoding="utf-8", newline="\n")
    print(out)


if __name__ == "__main__":
    main()
