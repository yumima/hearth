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
    # read_file is line-numbered: that is what lets the model quote an exact,
    # unique old_string back to edit_file and refer to code by line.
    assert a._exec_tool("read_file", {"path": "a.py"}) == "     1\tx = 1"
    a._exec_tool("write_file", {"path": "sub/b.txt", "content": "hello"})
    assert open(os.path.join(d, "sub", "b.txt")).read() == "hello"


def test_read_file_windows_large_files():
    d = tempfile.mkdtemp()
    open(os.path.join(d, "big.txt"), "w").write("\n".join(f"line {i}" for i in range(1, 101)))
    a = _agent(d)
    out = a._exec_tool("read_file", {"path": "big.txt", "offset": 10, "limit": 5})
    assert "    10\tline 10" in out and "    14\tline 14" in out
    assert "line 15" not in out
    assert "read on with offset=15" in out


def test_list_files_pattern_filters():
    d = tempfile.mkdtemp()
    open(os.path.join(d, "a.py"), "w").write("")
    open(os.path.join(d, "b.md"), "w").write("")
    a = _agent(d)
    out = a._exec_tool("list_files", {"pattern": "*.py"})
    assert "a.py" in out and "b.md" not in out


def test_grep_finds_content_with_location():
    d = tempfile.mkdtemp()
    open(os.path.join(d, "m.py"), "w").write("import os\n\ndef target():\n    pass\n")
    open(os.path.join(d, "n.txt"), "w").write("target\n")
    a = _agent(d)
    out = a._exec_tool("grep", {"pattern": r"def target", "glob": "*.py"})
    assert "m.py:3: def target():" in out
    assert "n.txt" not in out, "glob must restrict which files are searched"
    assert a._exec_tool("grep", {"pattern": "nothing-here"}) == "(no matches)"
    assert a._exec_tool("grep", {"pattern": "["}).startswith("ERROR")


def test_grep_skips_binary_files():
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "blob.bin"), "wb") as fh:
        fh.write(b"needle\x00\x01\x02")
    a = _agent(d)
    assert a._exec_tool("grep", {"pattern": "needle"}) == "(no matches)"


def test_edit_file_replaces_exact_string():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "c.py")
    open(p, "w").write("def add(a, b):\n    return a - b\n")
    a = _agent(d)
    out = a._exec_tool("edit_file", {"path": "c.py",
                                     "old_string": "return a - b",
                                     "new_string": "return a + b"})
    assert "edited" in out
    # The surgical edit must leave everything else byte-identical — the whole
    # reason to prefer it over a full rewrite.
    assert open(p).read() == "def add(a, b):\n    return a + b\n"


def test_edit_file_refuses_ambiguous_match():
    """Silently editing the first of several identical matches is how an agent
    corrupts a file while reporting success."""
    d = tempfile.mkdtemp()
    p = os.path.join(d, "d.py")
    open(p, "w").write("x = 1\nx = 1\n")
    a = _agent(d)
    out = a._exec_tool("edit_file", {"path": "d.py", "old_string": "x = 1", "new_string": "x = 2"})
    assert out.startswith("ERROR") and "2 times" in out
    assert open(p).read() == "x = 1\nx = 1\n", "file must be untouched"
    # …unless the model explicitly asks for every occurrence.
    a._exec_tool("edit_file", {"path": "d.py", "old_string": "x = 1",
                               "new_string": "x = 2", "replace_all": True})
    assert open(p).read() == "x = 2\nx = 2\n"


def test_edit_file_reports_missing_string_and_missing_file():
    d = tempfile.mkdtemp()
    open(os.path.join(d, "e.py"), "w").write("hello\n")
    a = _agent(d)
    assert a._exec_tool("edit_file", {"path": "e.py", "old_string": "nope",
                                      "new_string": "x"}).startswith("ERROR")
    assert a._exec_tool("edit_file", {"path": "gone.py", "old_string": "a",
                                      "new_string": "b"}).startswith("ERROR")


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
