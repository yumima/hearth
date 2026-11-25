"""No test sees the real home: chats, lessons and settings are kept there."""

import pytest


@pytest.fixture(autouse=True)
def _a_home_of_its_own(tmp_path_factory, monkeypatch):
    home = tmp_path_factory.mktemp("home")
    for name in ("HOME", "USERPROFILE"):
        monkeypatch.setenv(name, str(home))
    for name in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME"):
        monkeypatch.delenv(name, raising=False)
