import sys
from pathlib import Path

# Make `hearth` importable from src/ even without an editable install, and the
# project root importable so `from tests.test_routes import ...` resolves.
ROOT = Path(__file__).resolve().parent
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


import pytest


@pytest.fixture(autouse=True)
def _a_home_of_its_own(tmp_path_factory, monkeypatch):
    """No test sees the real home: the code under test keeps links, launchers,
    services and settings there, and this machine's hearth runs from them."""
    home = tmp_path_factory.mktemp("home")
    for name in ("HOME", "USERPROFILE"):
        monkeypatch.setenv(name, str(home))
    for name in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME", "HEARTH_HOME", "HEARTH_CONFIG", "HEARTH_DATA",
                 "HEARTH_OLLAMA_BIN"):
        monkeypatch.delenv(name, raising=False)
