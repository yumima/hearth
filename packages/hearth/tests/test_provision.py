"""Installing, checking, assigning and removing models — against a stand-in for
Ollama that does what the real one's API does, and no model."""

from __future__ import annotations

import http.server
import json
import socketserver
import threading
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from hearth import catalog, hardware, model_check, paths, provision
from hearth import config as cfgmod
from hearth.app import create_app
from hearth.hardware import Gpu, Hardware
from hearth.provision import Problem, Provisioner

BIT2 = "hf.co/unsloth/Qwen3.8-27B-GGUF:UD-IQ2_S"
EMBED = "nomic-embed-text"
GOOD_HW = Hardware("linux", "x86_64", "cpu", 8, True, 32000, 32000, 500_000,
                   [Gpu("nvidia", "RTX", 12227, 12000, None, "cuda")])


class FakeOllama:
    """What it holds and how it behaves; `self.models` is name → digest."""

    def __init__(self):
        self.models: dict[str, str] = {}
        self.loaded: dict[str, int] = {}  # model → context
        self.fits = 40960                 # the largest context wholly on the GPU
        self.speed = 27.0
        self.serves = {}                  # model → digest a pull gives, else the catalog's
        self.pull_error = ""
        self.slow_pull = False
        self.cannot = set()               # checks it fails
        self.chat_error = ""
        self.refuses_think = self.refuses_tools = self.stalls = False
        self.pulled: list[str] = []
        self.chats: list[dict] = []
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def _reply(self, obj, status=200):
                body = json.dumps(obj).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/api/tags":
                    self._reply({"models": [{"name": n if ":" in n.rsplit("/", 1)[-1] else n + ":latest", "digest": d, "size": 5_000_000_000}
                                            for n, d in outer.models.items()]})
                elif self.path == "/api/ps":
                    self._reply({"models": [{"name": n, "size": 1000, "size_vram": 1000 if c <= outer.fits else 900}
                                            for n, c in outer.loaded.items()]})
                else:
                    self._reply({"version": "fake"})

            def do_DELETE(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                found = outer.models.pop(body["model"], None)
                self._reply({} if found else {"error": "not found"}, 200 if found else 404)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/api/pull":
                    self._pull(body["model"])
                elif self.path == "/api/chat":
                    outer.chats.append(body)
                    if outer.chat_error:
                        return self._reply({"error": outer.chat_error}, 500)
                    if outer.refuses_think and "think" in body:
                        return self._reply({"error": "model does not support thinking"}, 400)
                    if outer.refuses_tools and body.get("tools"):
                        return self._reply({"error": "m does not support tools"}, 400)
                    outer.loaded = {body["model"]: body["options"]["num_ctx"]}
                    self._reply({"message": outer.answer(body), "load_duration": 2e9,
                                 "eval_count": 100, "eval_duration": int(100 / outer.speed * 1e9)})
                else:
                    self._reply({})

            def _pull(self, model):
                outer.pulled.append(model)
                if outer.pull_error:
                    return self._reply({"error": outer.pull_error}, 412)
                self.send_response(200)
                self.end_headers()

                def line(o):
                    self.wfile.write(json.dumps(o).encode() + b"\n")
                    self.wfile.flush()

                try:
                    for done in range(0, 101, 10):
                        while outer.stalls and done:
                            time.sleep(0.2)
                        line({"status": "pulling", "digest": "sha256:a", "total": 100, "completed": done})
                        if outer.slow_pull:
                            time.sleep(0.3)
                    line({"status": "success"})
                except OSError:
                    return  # the asker left
                outer.models[model] = outer.serves.get(model, catalog.digest_of(model) or "d" * 64)

        class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
            daemon_threads = True

            def server_bind(self):
                socketserver.TCPServer.server_bind(self)
                self.server_name, self.server_port = self.server_address

        self.server = Server(("127.0.0.1", 0), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def answer(self, body) -> dict:
        text = body["messages"][0]["content"]
        if body.get("tools"):
            bad = "tool" in self.cannot
            return {"tool_calls": [{"function": {"name": "send_mail" if bad else "get_weather",
                                                 "arguments": {"city": "Oslo", "unit": "celsius"}}}]}
        if body.get("format") == "json":
            return {"content": "sorry" if "json" in self.cannot else '{"city": "Lund", "population": 120000}'}
        if "three figures" in text:
            return {"content": "4,817.36; 23.5%; -0.072" if "figures" in self.cannot else "4,817.36; 23.45%; -0.072"}
        if "vault code" in text:
            return {"content": "I do not know" if "long" in self.cannot else "58213"}
        return {"content": "OK"}


@pytest.fixture
def ollama():
    f = FakeOllama()
    yield f
    f.server.shutdown()
    f.server.server_close()


@pytest.fixture
def prov(ollama, tmp_path, monkeypatch):
    monkeypatch.setenv("HEARTH_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HEARTH_DATA", str(tmp_path / "data"))
    monkeypatch.setattr(provision.hardware, "_disk_free", lambda p: 500_000)  # not this machine's
    cfg = cfgmod.Config(backends={"ollama": cfgmod.Backend("ollama", "ollama", ollama.url)},
                        path=tmp_path / "home" / "config.yaml")
    p = Provisioner(cfg, lambda: GOOD_HW)
    yield p
    p.close()


def _do(p, kind="model", what=BIT2, **kw):
    return p.wait(p.submit(kind, what, **kw))


# ── a model ───────────────────────────────────────────────────────────────────

def test_a_model_is_downloaded_checked_and_then_used(prov, ollama):
    job = _do(prov)
    assert job.stage == "done", job.message
    assert ollama.pulled == [BIT2, EMBED]  # and what memory search needs
    roles = {n: (r.model, r.context) for n, r in cfgmod.load().roles.items()}  # saved, not only in memory
    assert roles == {**{r: (BIT2, 40960) for r in ("primary_chat", "fast_chat", "coding", "vision")},
                     "embedding": (EMBED, 0)}
    assert "27 tokens a second" in job.message and "40960" in job.message
    assert job.result["ok"] and job.result["gpu_share"] == 1
    assert next(m for m in prov.installed() if m["model"] == BIT2)["measured"]["tokens_per_second"] == 27.0


def test_the_context_is_lowered_until_all_of_it_is_on_the_gpu(prov, ollama):
    ollama.fits = 20000
    job = _do(prov)
    assert job.stage == "done" and prov.cfg.roles["primary_chat"].context == 16384
    tried = [c["options"]["num_ctx"] for c in ollama.chats]
    assert tried[:5] == [40960, 32768, 24576, 16384, 16384]


def test_one_that_never_fits_is_used_at_the_least_context_and_that_is_said(prov, ollama):
    ollama.fits = 0
    job = _do(prov)
    assert job.stage == "done" and prov.cfg.roles["primary_chat"].context == catalog.LEAST_CONTEXT
    assert "90% of it is on the GPU" in job.message


def test_a_slow_one_is_used_and_a_faster_one_named(prov, ollama):
    ollama.speed = 3.0
    job = _do(prov)
    assert job.stage == "done" and "That is slow; Qwen 3.5 9B would be faster." in job.message


@pytest.mark.parametrize("check,words", [("json", "JSON"), ("tool", "tools"), ("figures", "figures"), ("long", "long text")])
def test_one_that_fails_a_check_is_not_used_unless_the_person_says_so(prov, ollama, check, words):
    prov.cfg.roles["primary_chat"] = cfgmod.RoleBinding("before", "ollama", 4096)
    ollama.cannot = {check}
    job = _do(prov)
    assert job.stage == "failed" and words in job.message and job.as_dict()["can_anyway"]
    assert prov.cfg.roles["primary_chat"].model == "before"  # what was used still is
    assert _do(prov, anyway=True).stage == "done"
    assert prov.cfg.roles["primary_chat"].model == BIT2


def test_one_that_cannot_be_run_is_never_used(prov, ollama):
    ollama.chat_error = "unable to load model"
    for anyway in (False, True):
        job = _do(prov, anyway=anyway)
        assert job.stage == "failed" and "unable to load model" in job.message and not job.as_dict()["can_anyway"]
    assert "primary_chat" not in prov.cfg.roles


def test_one_that_is_not_the_build_that_was_tested(prov, ollama):
    ollama.serves[BIT2] = "e" * 64
    job = _do(prov)
    assert job.stage == "failed" and "has changed since it was tested" in job.message and job.as_dict()["can_anyway"]
    assert "primary_chat" not in prov.cfg.roles and not ollama.chats  # not even run
    assert _do(prov, anyway=True).stage == "done"


def test_one_already_here_is_not_downloaded_again(prov, ollama):
    ollama.models = {BIT2: catalog.digest_of(BIT2), EMBED: catalog.digest_of(EMBED)}
    assert _do(prov).stage == "done" and ollama.pulled == []


def test_one_that_is_here_is_never_downloaded_over(prov, ollama):
    """A role may be using it as it is; the tag may be another model by now."""
    ollama.models = {BIT2: "0" * 64, EMBED: "1" * 64}  # other builds than the tested ones
    job = _do(prov)
    assert job.stage == "failed" and "that is here is not the build that was tested" in job.message
    assert _do(prov, anyway=True).stage == "done"
    assert ollama.pulled == [] and ollama.models == {BIT2: "0" * 64, EMBED: "1" * 64}


def test_going_on_in_spite_of_one_thing_is_not_in_spite_of_another(prov, ollama):
    ollama.serves[BIT2] = "e" * 64
    ollama.cannot = {"json"}
    assert "has changed" in _do(prov).message
    job = _do(prov, anyway=True)
    assert job.stage == "failed" and "JSON" in job.message and job.as_dict()["can_anyway"]
    job = _do(prov, anyway=True)  # and now of both
    assert job.stage == "done" and "In use although it does not answer in JSON" in job.message


def test_a_model_that_refuses_tools_can_still_be_chosen(prov, ollama):
    ollama.refuses_tools = True
    job = _do(prov, what="no-tools:1b")
    assert job.stage == "failed" and "does not use tools" in job.message and job.as_dict()["can_anyway"]
    assert _do(prov, what="no-tools:1b", anyway=True).stage == "done"


def test_a_model_for_the_processor_is_judged_as_one(prov, ollama):
    """A computer with a GPU too small for any model."""
    prov._find = lambda: Hardware("linux", "x86_64", "cpu", 8, True, 32000, 32000, 500_000,
                                  [Gpu("nvidia", "small", 2048, 2000, None, "cuda")])
    ollama.fits, ollama.speed = 0, 5.0
    job = _do(prov, what="qwen3.5:4b")
    assert job.stage == "done" and "slow" not in job.message and "GPU" not in job.message
    assert not any("vault code" in c["messages"][0]["content"] for c in ollama.chats)


def test_a_stalled_download_can_be_stopped(prov, ollama):
    ollama.stalls = True
    job = prov.submit("model", BIT2)
    deadline = time.monotonic() + 10
    while not job.total and time.monotonic() < deadline:
        time.sleep(0.05)
    t = time.monotonic()
    prov.stop_job(job.id)
    prov.wait(job)
    assert job.stage == "stopped" and time.monotonic() - t < 5


def test_the_disk_that_counts_is_the_one_models_go_to(prov, monkeypatch, tmp_path):
    asked = []
    monkeypatch.setattr(provision.hardware, "_disk_free", lambda p: asked.append(p) or 500_000)
    prov.cfg.backends["ollama"].models_dir = str(tmp_path / "big")
    _do(prov)
    assert set(asked) == {tmp_path / "big"}
    prov.cfg.backends["ollama"].models_dir = ""
    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path / "theirs"))
    assert prov.models_dir() == tmp_path / "theirs"


def test_any_other_model_by_name(prov, ollama):
    job = _do(prov, what="somebody/elses:1b")
    assert job.stage == "done" and prov.cfg.roles["primary_chat"].model == "somebody/elses:1b"
    assert "vision" not in prov.cfg.roles  # nothing is known of that
    ollama.refuses_think = True
    assert _do(prov, what="no-thinking:1b").stage == "done"


def test_names_that_are_none(prov):
    for bad in ("", "a b", "x" * 300, "a\nb", "a\x00"):
        with pytest.raises(Problem):
            prov.submit("model", bad)
    with pytest.raises(Problem):
        prov.submit("everything")


def test_a_bound_embedding_model_is_never_changed(prov, ollama):
    """What other programs have embedded is of no use to another model."""
    prov.cfg.roles["embedding"] = cfgmod.RoleBinding("mxbai-embed-large", "ollama")
    assert _do(prov).stage == "done"
    assert prov.cfg.roles["embedding"].model == "mxbai-embed-large" and EMBED not in ollama.pulled


def test_a_download_that_is_refused(prov, ollama):
    ollama.pull_error = "requires a newer version of Ollama"
    job = _do(prov)
    assert job.stage == "failed" and "newer version" in job.message


def test_no_ollama(prov, ollama):
    ollama.server.shutdown()
    ollama.server.server_close()
    job = _do(prov)
    assert job.stage == "failed" and "Start it again to go on" in job.message
    assert prov.installed() == []


def test_a_full_disk_is_said_before_the_download(prov, ollama, monkeypatch):
    monkeypatch.setattr(provision.hardware, "_disk_free", lambda p: 5000)
    job = _do(prov)
    assert job.stage == "failed" and "Not enough free disk space" in job.message and ollama.pulled == []


def test_stopping_and_starting_again(prov, ollama):
    ollama.slow_pull = True
    job = prov.submit("model", BIT2)
    assert prov.submit("model", BIT2) is job  # asked for twice is asked for once
    waiting = prov.submit("model", "another:1b")
    deadline = time.monotonic() + 10
    while not job.done and time.monotonic() < deadline:
        time.sleep(0.05)
    assert job.total == 100
    assert prov.stop_job(waiting.id) and waiting.stage == "stopped"
    assert prov.stop_job(job.id)
    prov.wait(job)
    assert job.stage == "stopped" and "primary_chat" not in prov.cfg.roles
    assert prov.stop_job(job.id) is False and prov.stop_job(999) is False
    ollama.slow_pull = False
    again = _do(prov)
    assert again is not job and again.stage == "done" and "another:1b" not in ollama.pulled


def test_one_job_failing_does_not_stop_the_next(prov, ollama):
    ollama.serves["bad:1b"] = ""
    ollama.cannot = {"json"}
    a, b = prov.submit("model", "bad:1b"), prov.submit("kokoro")
    prov.wait(a)
    ollama.cannot = set()
    prov.wait(b)
    assert a.stage == "failed" and b.finished
    assert _do(prov).stage == "done"


def test_on_the_processor_it_is_not_kept_busy_for_minutes(prov, ollama):
    prov._find = lambda: Hardware("linux", "x86_64", "cpu", 8, True, 32000, 32000, 500_000)
    job = _do(prov, what="qwen3.5:4b")
    assert job.stage == "done" and prov.cfg.roles["primary_chat"].context == 8192
    assert not any("vault code" in c["messages"][0]["content"] for c in ollama.chats)


# ── removing ──────────────────────────────────────────────────────────────────

def test_a_model_in_use_cannot_be_removed(prov, ollama):
    _do(prov)
    with pytest.raises(Problem, match="primary_chat"):
        prov.remove(BIT2)
    with pytest.raises(Problem, match="embedding"):
        prov.remove(EMBED + ":latest")  # the same model by its other name
    assert BIT2 in ollama.models
    assert _do(prov, what="qwen3.5:9b").stage == "done"
    prov.remove(BIT2)
    assert BIT2 not in ollama.models
    with pytest.raises(Problem, match="not installed"):
        prov.remove(BIT2)


def test_a_model_being_installed_cannot_be_removed(prov, ollama):
    ollama.slow_pull = True
    job = prov.submit("model", BIT2)
    with pytest.raises(Problem, match="being installed"):
        prov.remove(BIT2)
    prov.stop_job(job.id)
    prov.wait(job)


# ── files ─────────────────────────────────────────────────────────────────────

class _Files(http.server.BaseHTTPRequestHandler):
    data = bytes(range(256)) * 4000
    cut = 0

    def log_message(self, *_):
        pass

    def do_GET(self):
        start = int(self.headers["Range"][6:-1]) if self.headers.get("Range") else 0
        if start >= len(self.data):
            self.send_response(416)
            self.end_headers()
            return
        self.send_response(206 if start else 200)
        self.send_header("Content-Length", str(len(self.data) - start))
        self.end_headers()
        end = type(self).cut or len(self.data)
        type(self).cut = 0
        self.wfile.write(self.data[start:end])


@pytest.fixture
def files():
    s = socketserver.TCPServer(("127.0.0.1", 0), _Files)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{s.server_address[1]}/voice.bin"
    s.shutdown()
    s.server_close()


def test_a_download_goes_on_where_it_was_cut(files, tmp_path):
    import hashlib

    want = hashlib.sha256(_Files.data).hexdigest()
    dest = tmp_path / "d" / "voice.bin"
    _Files.cut = 300_000
    with pytest.raises(httpx.HTTPError):
        provision.fetch(files, dest, want)
    assert not dest.exists()  # nothing is there until it is the file it should be
    assert (tmp_path / "d" / "voice.bin.part").stat().st_size > 0
    provision.fetch(files, dest, want)
    assert dest.read_bytes() == _Files.data and not (tmp_path / "d" / "voice.bin.part").exists()
    # whole already, though its last step was lost
    dest.rename(tmp_path / "d" / "voice.bin.part")
    provision.fetch(files, dest, want)
    assert dest.read_bytes() == _Files.data


def test_a_file_that_is_not_the_published_one_is_not_kept(files, tmp_path):
    dest = tmp_path / "voice.bin"
    with pytest.raises(Problem, match="not the published file"):
        provision.fetch(files, dest, "0" * 64)
    assert list(tmp_path.iterdir()) == []


# ── over HTTP ─────────────────────────────────────────────────────────────────

@pytest.fixture
def client(prov):
    app = create_app(prov.cfg, backends_override={})
    app.state.provisioner = prov
    return TestClient(app, base_url="http://127.0.0.1:11435")


def test_the_page(client):
    r = client.get("/")
    assert r.status_code == 200 and "<title>hearth setup</title>" in r.text
    assert "default-src 'none'" in r.headers["content-security-policy"] and r.headers["x-frame-options"] == "DENY"
    assert 'src="http' not in r.text and "<link" not in r.text and "//cdn" not in r.text  # all of it is here
    assert client.get("/app", follow_redirects=False).headers["location"] == "/"


def test_setting_up_over_http(client, prov, ollama):
    s = client.get("/admin/setup").json()
    assert s["plan"]["verdict"] == "recommended" and s["installed"] == [] and s["jobs"] == []
    best = next(o for o in s["plan"]["offers"] if o["recommended"])
    job = client.post("/admin/jobs", json={"kind": "model", "what": best["model"]}).json()
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        s = client.get("/admin/setup").json()
        if s["jobs"][-1]["stage"] in ("done", "failed"):
            break
        time.sleep(0.1)
    assert s["jobs"][-1]["id"] == job["id"] and s["jobs"][-1]["stage"] == "done"
    assert s["roles"]["primary_chat"]["model"] == best["model"]
    assert client.get("/admin/roles").json()["roles"]["primary_chat"]["model"] == best["model"]  # what finterm reads
    r = client.delete("/admin/models/" + best["model"])
    assert r.status_code == 409 and "primary_chat" in r.json()["error"]
    ollama.models["extra/one:1b"] = "f" * 64
    assert client.delete("/admin/models/extra/one:1b").json() == {"deleted": "extra/one:1b"}


def test_what_is_not_a_job(client):
    for body in (b"", b"[]", b'"x"', b"{", b'{"kind":"model"}', b'{"kind":"model","what":["x"]}', b'{"kind":"nope","what":"x"}'):
        r = client.post("/admin/jobs", content=body)
        assert r.status_code in (200, 400), body
        if body != b'{"kind":"model","what":["x"]}':
            assert r.status_code == 400
    assert client.post("/admin/jobs/77/stop").json() == {"stopping": False}


def test_another_site_cannot_install_or_remove(client, ollama):
    evil = {"origin": "https://evil.example", "content-type": "text/plain"}
    assert client.post("/admin/jobs", content=b'{"kind":"model","what":"x:1b"}', headers=evil).status_code == 403
    assert client.delete("/admin/models/x", headers=evil).status_code == 403
    assert ollama.pulled == []


# ── the command line ──────────────────────────────────────────────────────────

def test_the_command_line_without_a_gateway(prov, ollama, monkeypatch, capsys):
    from hearth import cli

    cfgmod.save(prov.cfg)
    monkeypatch.setattr(cli, "_gateway_identity", lambda cfg: None)
    monkeypatch.setattr(hardware, "probe", lambda: GOOD_HW)
    monkeypatch.setattr(provision.hardware, "probe", lambda: GOOD_HW)
    assert cli.main(["plan"]) == 0 and "★ Qwen 3.8 27B, 2-bit" in capsys.readouterr().out
    assert cli.main(["plan", "--json"]) == 0 and json.loads(capsys.readouterr().out)["verdict"] == "recommended"
    assert cli.main(["setup", "--yes", "--no-voice"]) == 0
    assert "tokens a second" in capsys.readouterr().out
    assert cfgmod.load().roles["primary_chat"].model == BIT2
    assert cli.main(["remove", BIT2]) == 1 and "primary_chat" in capsys.readouterr().err
    ollama.cannot = {"json"}
    assert cli.main(["install", "qwen3.5:9b"]) == 1 and "--anyway" in capsys.readouterr().err
    assert cli.main(["install", "qwen3.5:9b", "--anyway"]) == 0
    assert cli.main(["remove", BIT2]) == 0
    assert cli.main(["install", "a b"]) == 1 and "a model's name" in capsys.readouterr().err
    ollama.server.shutdown()
    ollama.server.server_close()
    monkeypatch.setattr(cli, "_ensure_ollama", lambda cfg: object())
    assert cli.main(["remove", "x:1b"]) == 1  # said, not a traceback


def test_the_command_line_when_the_gateway_goes_away(monkeypatch, capsys):
    from hearth import cli

    monkeypatch.setattr(cli, "_gateway_identity", lambda cfg: "hearth/9")
    replies = [httpx.Response(200, json={"id": 5, "kind": "model", "what": "m:1b", "stage": "waiting", "message": "",
                                         "done": 0, "total": 0})]

    def request(method, url, **kw):
        if replies:
            return replies.pop(0)
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(cli.httpx, "request", request)
    assert cli.main(["install", "m:1b"]) == 1 and "stopped answering" in capsys.readouterr().err
    replies[:] = [httpx.Response(200, json={"id": 5, "kind": "model", "what": "m:1b", "stage": "waiting", "message": "",
                                            "done": 0, "total": 0}), httpx.Response(200, json={"jobs": []})]
    assert cli.main(["install", "m:1b"]) == 1 and "restarted" in capsys.readouterr().err


def test_what_was_measured_is_kept_whatever_is_in_the_file(prov, tmp_path):
    for text in ("", "[]", "not json"):
        provision._measured_path().parent.mkdir(parents=True, exist_ok=True)
        provision._measured_path().write_text(text)
        assert provision._measured() == {}
    provision._remember("d1", {"a": 1})
    provision._remember("d2", {"a": 2})
    provision._remember("", {"a": 3})
    assert provision._measured() == {"d1": {"a": 1}, "d2": {"a": 2}}


def test_a_figure_written_another_way_is_the_same_figure(ollama, monkeypatch):
    for text, ok in (("4817.36;23.45%;-0.072", True), ("4,817.36; 23.45 %; \u22120.072", True), ("4,817.4; 23.45%; -0.072", False),
                     ("4.817,36; 23,45%; -0,072", False), ("Revenue; 23.45%; -0.072", False)):
        monkeypatch.setattr(ollama, "answer", lambda body, text=text: {"content": text})
        c = model_check.Checker(ollama.url, "m:1b")
        assert c._figures(8192) is ok, text
        c.close()


def test_the_checks_know_a_wrong_answer(ollama):
    c = model_check.Checker(ollama.url, "m:1b")
    assert c.run(40960, True).failed == []
    ollama.cannot = {"json", "tool", "figures", "long"}
    assert len(c.run(40960, True).failed) == 4
    c.close()
    assert paths.home()
