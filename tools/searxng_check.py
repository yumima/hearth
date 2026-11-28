"""CI: the release's SearXNG starts here and answers JSON, as hearth runs it.

    <searxng env python> tools/searxng_check.py

Stdlib only: it runs in SearXNG's own environment. A search that reaches no
engine still answers (with no results) — what is checked is that it runs.
"""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

PORT = 8897

with tempfile.TemporaryDirectory() as t:
    settings = Path(t) / "settings.yml"
    settings.write_text(f"""use_default_settings: true
server:
  secret_key: "{secrets.token_hex(32)}"
  bind_address: "127.0.0.1"
  port: {PORT}
  limiter: false
search:
  formats: [html, json]
""", encoding="utf-8")
    p = subprocess.Popen([sys.executable, "-I", "-m", "searx.webapp"], cwd=t,
                         env={**os.environ, "SEARXNG_SETTINGS_PATH": str(settings)})
    try:
        deadline = time.monotonic() + 90
        while True:
            if p.poll() is not None:
                sys.exit(f"searx.webapp ended ({p.returncode})")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/search?q=hearth&format=json", timeout=30) as r:
                    body = json.load(r)
                break
            except OSError:
                if time.monotonic() > deadline:
                    sys.exit("searx.webapp did not answer")
                time.sleep(1)
        assert "results" in body, body
        print(f"searxng answers: {len(body['results'])} results")
    finally:
        p.terminate()
        p.wait(30)
