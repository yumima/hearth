"""Putting models and voices on this computer, checking them, and taking them off.

One job at a time, in a thread, whoever asked (the setup page, the command line):
download → is it the build that was tested → does it work here → only then do the
roles get it. A job that fails leaves what was assigned before as it was, and the
others go on.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import httpx

from . import catalog, hardware, paths
from . import config as cfgmod
from .model_check import SLOW_CPU, SLOW_GPU, Checker, Stopped

log = logging.getLogger("hearth.provision")

CHAT_ROLES = ("primary_chat", "fast_chat", "coding")
WHISPER_SIZE = "small"


class Problem(Exception):
    """Said to the person as it is."""


@dataclass
class Job:
    id: int
    kind: str              # "model" | "kokoro" | "stt"
    what: str              # the model's name; "" for the others
    anyway: bool = False   # assign it although it changed, or failed a check
    stage: str = "waiting"  # waiting | download | check | assign | done | failed | stopped
    message: str = ""
    done: int = 0          # bytes
    total: int = 0
    result: dict | None = None
    can_anyway: bool = False  # failed in a way the person may overrule
    stop: threading.Event = field(default_factory=threading.Event)

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in
                ("id", "kind", "what", "stage", "message", "done", "total", "result", "can_anyway")}

    @property
    def finished(self) -> bool:
        return self.stage in ("done", "failed", "stopped")


# ── files ─────────────────────────────────────────────────────────────────────

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, dest: Path, digest: str, job: Job | None = None) -> None:
    """Download to dest, whole and checked, or not at all; an interrupted one
    goes on where it was. Nothing is at `dest` until it is the file it should be."""
    if dest.is_file() and sha256(dest) == digest:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with httpx.stream("GET", url, follow_redirects=True, headers=headers,
                      timeout=httpx.Timeout(60.0, connect=15.0)) as r:
        if r.status_code == 416:  # nothing after what we have: whole, or not this file
            pass
        elif r.status_code not in (200, 206):
            raise Problem(f"{url.rsplit('/', 1)[-1]}: the download answered HTTP {r.status_code}")
        else:
            if r.status_code == 200:
                have = 0
            size = int(r.headers.get("content-length") or 0) + have
            base = job.done if job else 0
            with open(part, "ab" if have else "wb") as f:
                for chunk in r.iter_bytes(1 << 18):
                    if job and job.stop.is_set():
                        raise Stopped
                    f.write(chunk)
                    have += len(chunk)
                    if job:
                        job.done = base + have
                        job.total = max(job.total, base + size)
    if sha256(part) != digest:
        part.unlink(missing_ok=True)
        raise Problem(f"{dest.name} is not the published file — removed")
    part.replace(dest)


# ── Ollama ────────────────────────────────────────────────────────────────────

def _models(base_url: str) -> dict[str, dict]:
    r = httpx.get(f"{base_url}/api/tags", timeout=10.0)
    return {m["name"]: m for m in r.json().get("models", [])}


def _named(models: dict[str, dict], model: str) -> dict | None:
    return models.get(model) or models.get(model + ":latest")


def same_model(a: str, b: str) -> bool:
    return a.removesuffix(":latest") == b.removesuffix(":latest")


def pull(base_url: str, model: str, job: Job) -> None:
    """Ollama keeps what it has of an interrupted download and goes on from it."""
    layers: dict[str, tuple[int, int]] = {}
    with httpx.stream("POST", f"{base_url}/api/pull", json={"model": model, "stream": True},
                      timeout=httpx.Timeout(None, connect=10.0)) as r:
        if r.status_code >= 400:
            r.read()
            raise Problem(_error(r) or f"the download answered HTTP {r.status_code}")
        for line in r.iter_lines():
            if job.stop.is_set():
                raise Stopped
            try:
                o = json.loads(line) if line else {}
            except ValueError:
                continue
            if o.get("error"):
                raise Problem(str(o["error"]))
            if o.get("digest") and o.get("total"):
                layers[o["digest"]] = (int(o.get("completed") or 0), int(o["total"]))
                job.done = sum(d for d, _ in layers.values())
                job.total = max(job.total, sum(t for _, t in layers.values()))


def _error(r: httpx.Response) -> str:
    try:
        return str(r.json().get("error") or "")
    except ValueError:
        return r.text.strip()[:200]


# ── the provisioner ───────────────────────────────────────────────────────────

class Provisioner:
    """`cfg` is the gateway's own, so what is assigned is used at once."""

    def __init__(self, cfg: cfgmod.Config, probe: Callable[[], hardware.Hardware] = hardware.probe):
        self.cfg = cfg
        self._find = probe
        self._found: tuple[float, hardware.Hardware] | None = None
        self._jobs: list[Job] = []
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._thread: threading.Thread | None = None
        self._next = 1
        self._closed = False

    # ── what there is ────────────────────────────────────────────────────────
    def _probe(self) -> hardware.Hardware:
        """Kept for a while: the page asks every second, and it runs programs."""
        if self._found is None or time.monotonic() - self._found[0] > 15:
            self._found = (time.monotonic(), self._find())
        return self._found[1]

    def ollama(self) -> cfgmod.Backend:
        for b in self.cfg.backends.values():
            if b.type == "ollama":
                return b
        raise Problem("no Ollama is configured")

    def roles_of(self, model: str) -> list[str]:
        return sorted(n for n, r in self.cfg.roles.items() if same_model(r.model, model))

    def installed(self) -> list[dict]:
        try:
            models = _models(self.ollama().base_url)
        except (Problem, httpx.HTTPError, ValueError):
            return []
        measured = _measured()
        out = []
        for name, m in sorted(models.items()):
            shown = name.removesuffix(":latest")
            c = catalog.find(shown)
            out.append({"model": shown, "title": c.title if c else shown, "size": m.get("size") or 0,
                        "roles": self.roles_of(shown), "measured": measured.get(m.get("digest", "")),
                        "offered": c is not None, "embedding": same_model(shown, catalog.embedding()["model"])})
        return out

    def voices(self) -> dict:
        from . import kokoro_tts

        return {"kokoro": kokoro_tts.installed(), "kokoro_possible": kokoro_tts.package_present(),
                "stt": whisper_present(WHISPER_SIZE), "stt_possible": stt_possible()}

    def state(self) -> dict:
        with self._lock:
            jobs = [j.as_dict() for j in self._jobs[-20:]]
        return {"plan": catalog.plan(self._probe()), "installed": self.installed(), "voices": self.voices(),
                "roles": {n: {"model": r.model, "backend": r.backend, "context": r.context}
                          for n, r in self.cfg.roles.items()},
                "jobs": jobs}

    # ── jobs ─────────────────────────────────────────────────────────────────
    def submit(self, kind: str, what: str = "", anyway: bool = False) -> Job:
        if kind not in ("model", "kokoro", "stt"):
            raise Problem(f"unknown kind {kind!r}")
        if kind == "model" and not (what and len(what) < 200 and all(c.isprintable() and not c.isspace() for c in what)):
            raise Problem("a model's name is needed")
        with self._lock:
            for j in self._jobs:
                if not j.finished and (j.kind, j.what) == (kind, what):
                    return j  # asked for twice is asked for once
            job = Job(self._next, kind, what, anyway)
            self._next += 1
            self._jobs.append(job)
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._work, name="provision", daemon=True)
                self._thread.start()
            self._wake.notify_all()
            return job

    def stop_job(self, job_id: int) -> bool:
        with self._lock:
            for j in self._jobs:
                if j.id == job_id and not j.finished:
                    j.stop.set()
                    if j.stage == "waiting":
                        j.stage, j.message = "stopped", "Stopped."
                    return True
        return False

    def close(self) -> None:
        with self._lock:
            self._closed = True
            for j in self._jobs:
                j.stop.set()
            self._wake.notify_all()

    def wait(self, job: Job, poll: Callable[[Job], None] = lambda j: None) -> Job:
        while not job.finished:
            poll(job)
            time.sleep(0.2)
        poll(job)
        return job

    def _work(self) -> None:
        while True:
            with self._lock:
                job = next((j for j in self._jobs if j.stage == "waiting"), None)
                if job is None:
                    if self._closed or not self._wake.wait(timeout=60):
                        self._thread = None
                        return
                    continue
                job.stage = "download"
            try:
                {"model": self._model, "kokoro": self._kokoro, "stt": self._stt}[job.kind](job)
                job.stage = "done"
            except Stopped:
                job.stage, job.message = "stopped", "Stopped. What was downloaded is kept, and used if it is started again."
            except Problem as e:
                job.stage, job.message = "failed", str(e)
            except httpx.HTTPError as e:
                job.stage, job.message = "failed", f"The connection failed ({e or type(e).__name__}). Start it again to go on."
            except Exception as e:  # one job's bug is not the gateway's end
                log.exception("job %s", job.id)
                job.stage, job.message = "failed", f"{type(e).__name__}: {e}"

    # ── a model ──────────────────────────────────────────────────────────────
    def _need_disk(self, bytes_wanted: int, job: Job) -> None:
        free = hardware._disk_free(paths.data_dir())
        if free is not None and free < bytes_wanted / (1 << 20) + catalog.DISK_SPARE_MIB:
            raise Problem(f"Not enough free disk space: it needs {bytes_wanted / 1e9:.1f} GB and some to spare, "
                          f"and {free / 1024:.1f} GB are free.")

    def _get(self, base: str, model: str, job: Job, size: int) -> dict:
        have = _named(_models(base), model)
        want = catalog.digest_of(model)
        if have is None or (want and have.get("digest") != want):
            if have is None:
                self._need_disk(size, job)
            job.message = f"Downloading {model}…"
            pull(base, model, job)
            have = _named(_models(base), model)
        if have is None:
            raise Problem(f"{model} was downloaded, and Ollama does not list it")
        if want and have.get("digest") != want and not job.anyway:
            job.can_anyway = True
            raise Problem(f"{model} has changed since it was tested with hearth. Update hearth, or use it anyway.")
        return have

    def _model(self, job: Job) -> None:
        base = self.ollama().base_url
        hw = self._probe()
        embed = catalog.embedding()
        offer = next((o for o in catalog.offers(hw) if o.model == job.what), None)
        if job.what == embed["model"]:
            self._embedding(base, job)
            job.message = f"{embed['title']} is ready."
            return
        info = self._get(base, job.what, job, offer.download if offer else 0)
        self._embedding(base, job)

        job.stage = "check"
        on_gpu = hw.best_gpu() is not None
        bound = self.cfg.context_of_model(job.what)
        context = (offer.context if offer and offer.context else 0) or bound or (16384 if on_gpu else 8192)

        def say(text: str) -> None:
            job.message = text

        checker = Checker(base, job.what, job.stop.is_set, say)
        try:
            res = checker.run(context, on_gpu)
        finally:
            checker.close()
        job.result = res.as_dict()
        if res.error:
            raise Problem(f"{job.what} could not be run here: {res.error}")
        _remember(info.get("digest", ""), job.result)
        if res.failed and not job.anyway:
            job.can_anyway = True
            raise Problem(f"{job.what} runs, but " + "; ".join(res.failed) + ". It was not chosen for chat.")

        job.stage = "assign"
        roles = [*CHAT_ROLES, *(["vision"] if offer and offer.vision else [])]
        for role in roles:
            self.cfg.roles[role] = cfgmod.RoleBinding(model=job.what, backend=self.ollama().name, context=res.context)
        cfgmod.save(self.cfg)
        job.message = self._outcome(res, on_gpu, hw, job.what)

    def _outcome(self, res, on_gpu: bool, hw: hardware.Hardware, model: str) -> str:
        text = f"Ready: {res.tokens_per_second:.0f} tokens a second, {res.context} tokens of context."
        if on_gpu and res.gpu_share < 0.995:
            text += f" {res.gpu_share * 100:.0f}% of it is on the GPU; the rest runs on the processor."
        if res.tokens_per_second < (SLOW_GPU if on_gpu else SLOW_CPU):
            smaller = next((o for o in catalog.offers(hw) if o.fit in ("gpu", "cpu") and (m := catalog.find(o.model))
                            and (c := catalog.find(model)) and m.weights_mib < c.weights_mib * 0.8), None)
            text += " That is slow" + (f"; {smaller.title} would be faster." if smaller else ".")
        return text

    def _embedding(self, base: str, job: Job) -> None:
        """Needed by every program that searches its memory. One that is bound is
        kept: what it has embedded is of no use to another."""
        e = catalog.embedding()
        bound = self.cfg.roles.get("embedding")
        if bound is not None and not same_model(bound.model, e["model"]):
            return
        self._get(base, e["model"], job, e["download"])
        if bound is None:
            self.cfg.roles["embedding"] = cfgmod.RoleBinding(model=e["model"], backend=self.ollama().name)
            cfgmod.save(self.cfg)

    def remove(self, model: str) -> None:
        roles = self.roles_of(model)
        if roles:
            raise Problem(f"{model} is used for {', '.join(roles)}. Choose another model for it first.")
        with self._lock:
            if any(not j.finished and same_model(j.what, model) for j in self._jobs):
                raise Problem(f"{model} is being installed.")
        r = httpx.request("DELETE", f"{self.ollama().base_url}/api/delete", json={"model": model}, timeout=60.0)
        if r.status_code == 404:
            raise Problem(f"{model} is not installed")
        if r.status_code >= 400:
            raise Problem(_error(r) or f"HTTP {r.status_code}")

    # ── voices ───────────────────────────────────────────────────────────────
    def _kokoro(self, job: Job) -> None:
        from . import kokoro_tts

        if not kokoro_tts.package_present():
            raise Problem("This hearth was installed without natural voices — run its installer again.")
        d = kokoro_tts.override_dir() or paths.voices_dir("kokoro")
        self._need_disk(400_000_000, job)
        job.message = "Downloading the natural voices…"
        for name, digest in kokoro_tts.FILES.items():
            fetch(kokoro_tts.RELEASE + name, d / name, digest, job)
        job.message = "Natural voices are ready."

    def _stt(self, job: Job) -> None:
        if not stt_possible():
            raise Problem("This hearth was installed without speech recognition — run its installer again.")
        self._need_disk(600_000_000, job)
        job.message = "Downloading speech recognition…"
        from faster_whisper.utils import download_model

        download_model(WHISPER_SIZE, cache_dir=str(paths.voices_dir("whisper")))
        job.message = "Speech recognition is ready."


def stt_possible() -> bool:
    from importlib.util import find_spec

    return find_spec("faster_whisper") is not None


def whisper_present(size: str) -> bool:
    d = paths.voices_dir("whisper") / f"models--Systran--faster-whisper-{size}" / "snapshots"
    return any(d.glob("*/model.bin")) if d.is_dir() else False


# ── what was measured here, by the model's digest ─────────────────────────────

def _measured_path() -> Path:
    return paths.home() / "measured.json"


def _measured() -> dict:
    try:
        d = json.loads(_measured_path().read_text())
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _remember(digest: str, result: dict) -> None:
    if not digest:
        return
    p = _measured_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps({**_measured(), digest: result}, indent=1))
        tmp.replace(p)
    except OSError:
        pass
