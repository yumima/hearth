"""What holds across the two packages."""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = {"hearth": ROOT / "packages" / "hearth", "mantel": ROOT / "packages" / "mantel"}


def _imports(path: Path) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            out |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            out.add(node.module.split(".")[0])
    return out


@pytest.mark.parametrize("name,other", [("hearth", "mantel"), ("mantel", "hearth")])
def test_neither_imports_the_other(name, other):
    """They are two programs: the engine holds no tools, files or keys of the
    chat's, and the chat works with other engines. HTTP is all they share."""
    files = list((PACKAGES[name] / "src").rglob("*.py"))
    assert len(files) > 5
    assert not [str(f) for f in files if other in _imports(f)]
    project = tomllib.loads((PACKAGES[name] / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    deps = [*project["dependencies"], *(d for extra in project["optional-dependencies"].values() for d in extra)]
    assert len(deps) > 5
    assert [d for d in deps if re.split(r"[^a-z0-9-]", d.lower())[0] == other] == []


def _version(name: str) -> str:
    text = (PACKAGES[name] / "src" / name / "__init__.py").read_text(encoding="utf-8")
    return re.search(r'^__version__ = "([^"]+)"', text, re.M).group(1)


def test_one_version():
    assert _version("hearth") == _version("mantel")


def test_everything_in_the_lock_is_checked_by_hash():
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    packages = {p["name"]: p for p in lock["package"]}
    assert {"hearth", "mantel", "fastapi", "mcp", "kokoro-onnx"} <= set(packages)
    for p in packages.values():
        files = [*p.get("wheels", []), *([p["sdist"]] if "sdist" in p else [])]
        assert all(f["hash"].startswith("sha256:") for f in files), p["name"]
        assert files or "virtual" in p["source"] or "editable" in p["source"], p["name"]
    assert packages["mcp"]["version"].startswith("1.")  # 2.x renamed what mantel's servers are written for
