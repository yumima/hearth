"""Live smoke against a running hearth gateway (needs Ollama + pulled models).

    $ hearth start &                 # in one shell
    $ python tests/smoke_live.py     # in another

Exercises the exact surface finterm depends on: /v1/models, a streaming
/v1/chat/completions, and /v1/embeddings. Not a pytest test — it requires the
full stack up and real models pulled.
"""

from __future__ import annotations

import os
import sys

import httpx

BASE = os.environ.get("HEARTH_BASE", "http://127.0.0.1:11435")


def main() -> int:
    ok = True

    print("== /admin/health ==")
    r = httpx.get(f"{BASE}/admin/health", timeout=10)
    print(r.status_code, r.text)

    print("== /admin/hardware ==")
    print(httpx.get(f"{BASE}/admin/hardware", timeout=10).text)

    print("== /v1/models ==")
    r = httpx.get(f"{BASE}/v1/models", timeout=10)
    ids = [m["id"] for m in r.json().get("data", [])]
    print(ids)
    ok &= "primary_chat" in ids

    print("== /v1/chat/completions (stream, role=primary_chat) ==")
    with httpx.stream(
        "POST", f"{BASE}/v1/chat/completions",
        json={"model": "primary_chat", "stream": True,
              "messages": [{"role": "user", "content": "Reply with exactly: pong"}]},
        timeout=120,
    ) as resp:
        got = ""
        for line in resp.iter_lines():
            if line.startswith("data: ") and "[DONE]" not in line:
                got += line
        print("stream bytes:", len(got))
        ok &= len(got) > 0

    print("== /v1/embeddings (role=embedding) ==")
    r = httpx.post(f"{BASE}/v1/embeddings", json={"model": "embedding", "input": "hello world"}, timeout=60)
    dim = len(r.json()["data"][0]["embedding"]) if r.status_code == 200 else 0
    print("status", r.status_code, "dim", dim)
    ok &= dim > 0

    print("\nSMOKE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
