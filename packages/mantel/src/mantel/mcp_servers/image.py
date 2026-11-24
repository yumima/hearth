"""Image generation — a local MCP server (ComfyUI/SDXL via hearth).

``generate_image(prompt)`` renders an image and returns it as a ``data:`` URL,
which the chat UI displays inline. Backed by the backend's OpenAI
``/v1/images/generations`` (hearth → ComfyUI). The user enables it from the Tools
& Settings panel; generation temporarily swaps the chat model out of VRAM.

NB: mantel substitutes a short placeholder for the data URL in the text fed back
to the model (a 2 MB base64 string would blow up the context) — see app.py.

Run standalone over stdio:  python -m mantel.mcp_servers.image
"""

from __future__ import annotations

import os

import httpx
from mcp.server.fastmcp import FastMCP

IMAGES_URL = os.environ.get("MANTEL_IMAGES_URL", "http://127.0.0.1:11435/v1/images/generations")


def _do_generate(prompt: str, width: int = 1024, height: int = 1024) -> dict:
    try:
        with httpx.Client(timeout=300.0) as client:
            r = client.post(IMAGES_URL, json={"prompt": prompt, "size": f"{int(width)}x{int(height)}"})
    except httpx.HTTPError as e:
        return {"error": f"image backend unreachable: {e}"}
    if r.status_code != 200:
        try:
            msg = (r.json().get("error") or {}).get("message") or r.text[:200]
        except Exception:
            msg = r.text[:200]
        return {"error": f"generation failed: {msg}"}
    data = (r.json().get("data") or [{}])[0]
    b64 = data.get("b64_json")
    if not b64:
        return {"error": "no image returned by the backend"}
    return {"data_url": "data:image/png;base64," + b64}


mcp = FastMCP("image")


@mcp.tool()
def generate_image(prompt: str, width: int = 1024, height: int = 1024) -> str:
    """Generate an image from a text PROMPT (local SDXL via ComfyUI) and return it
    as a data: URL the chat renders inline. Use when the user asks to draw, create,
    paint, or generate a picture/image. Square sizes work best (e.g. 1024x1024)."""
    r = _do_generate(prompt, width, height)
    return r["error"] if "error" in r else r["data_url"]


if __name__ == "__main__":
    mcp.run()
