"""hearth code agent — tool execution + path-confinement (no network/LLM)."""

import os
import tempfile

from hearth.code_agent import CodeAgent


def _agent(root):
    return CodeAgent(root, "http://127.0.0.1:11435", "coding", auto=True)


def test_list_read_write_roundtrip():
    d = tempfile.mkdtemp()
    open(os.path.join(d, "a.py"), "w").write("x = 1\n")
    a = _agent(d)
    assert "a.py" in a._exec_tool("list_files", {})
    assert a._exec_tool("read_file", {"path": "a.py"}) == "x = 1\n"
    a._exec_tool("write_file", {"path": "sub/b.txt", "content": "hello"})
    assert open(os.path.join(d, "sub", "b.txt")).read() == "hello"


def test_path_escape_is_blocked():
    d = tempfile.mkdtemp()
    a = _agent(d)
    for bad in ("../../etc/passwd", "../outside.txt", "/etc/hosts"):
        assert a._exec_tool("read_file", {"path": bad}).startswith("ERROR")
        # a write that escapes must not create anything
        assert a._exec_tool("write_file", {"path": bad, "content": "x"}).startswith("ERROR")


def test_run_shell_runs_in_root():
    d = tempfile.mkdtemp()
    open(os.path.join(d, "marker.txt"), "w").write("")
    a = _agent(d)
    out = a._exec_tool("run_shell", {"command": "ls"})
    assert "marker.txt" in out and "[exit 0]" in out


def test_denied_when_not_auto(monkeypatch):
    d = tempfile.mkdtemp()
    a = CodeAgent(d, "http://127.0.0.1:11435", "coding", auto=False)
    monkeypatch.setattr("builtins.input", lambda *_: "n")
    assert a._exec_tool("write_file", {"path": "x.txt", "content": "y"}) == "DENIED by user"
    assert not os.path.exists(os.path.join(d, "x.txt"))
