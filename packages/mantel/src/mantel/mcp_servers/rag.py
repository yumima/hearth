"""RAG over the user's own files — a local MCP server.

Embeds documents with the backend's OpenAI-compatible ``/v1/embeddings`` (hearth's
``embedding`` role → nomic-embed-text by default), stores normalized vectors in a
small SQLite file, and exposes agentic retrieval tools so the chat model can ground
answers in the user's notes/docs:

  index(path)        — chunk + embed a file or directory into the store
  search(query, k)   — return the top-k most relevant chunks
  stats()            — how much is indexed
  forget(source)     — drop a file (or everything under a dir) from the store

Run standalone over stdio:  python -m mantel.mcp_servers.rag
Everything is local: the only network call is to the loopback embeddings endpoint.
"""

from __future__ import annotations

import math
import os
import sqlite3
from array import array
from pathlib import Path

import httpx
from mcp.server.fastmcp import FastMCP

# ── config (env-overridable; loopback defaults match hearth) ──────────────────
EMBED_URL = os.environ.get("MANTEL_EMBED_URL", "http://127.0.0.1:11435/v1/embeddings")
EMBED_MODEL = os.environ.get("MANTEL_EMBED_MODEL", "embedding")
MAX_FILE_BYTES = 20 * 1024 * 1024
CHUNK_CHARS = 1000
CHUNK_OVERLAP = 150
EMBED_BATCH = 64

_TEXT_EXTS = {
    ".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".yaml", ".yml", ".log",
    ".py", ".js", ".ts", ".tsx", ".html", ".css", ".scss", ".sh", ".c", ".cpp",
    ".h", ".hpp", ".rs", ".go", ".java", ".rb", ".php", ".toml", ".ini", ".cfg",
    ".xml", ".sql", ".r", ".lua", ".pl", ".kt", ".swift", ".rst", ".tex",
}
_INDEXABLE = _TEXT_EXTS | {".pdf", ".docx"}


def _db_path() -> Path:
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    d = Path(base) / "mantel"
    d.mkdir(parents=True, exist_ok=True)
    return d / "rag.db"


def _db() -> sqlite3.Connection:
    con = sqlite3.connect(_db_path(), timeout=10.0)
    con.execute("PRAGMA busy_timeout=5000")  # wait, don't error, on concurrent writes
    con.execute("PRAGMA journal_mode=WAL")    # concurrent readers + one writer
    con.execute(
        "CREATE TABLE IF NOT EXISTS chunks("
        "id INTEGER PRIMARY KEY, source TEXT, chunk_idx INTEGER, text TEXT, "
        "dim INTEGER, vec BLOB, mtime REAL)"
    )
    con.execute("CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source)")
    return con


# ── embeddings + vector math (pure-Python; numpy not required) ────────────────
def _embed(texts: list[str]) -> list[list[float]]:
    out: list[list[float]] = []
    with httpx.Client(timeout=120.0) as client:
        for i in range(0, len(texts), EMBED_BATCH):
            grp = texts[i : i + EMBED_BATCH]
            r = client.post(EMBED_URL, json={"model": EMBED_MODEL, "input": grp})
            r.raise_for_status()
            data = r.json().get("data", [])
            # Reorder by 'index' only when every item carries one; otherwise trust
            # arrival order (Ollama's OpenAI-compat omits 'index') — a partial sort
            # would scramble the chunk↔vector alignment.
            if data and all("index" in d for d in data):
                data.sort(key=lambda d: d["index"])
            out.extend(d["embedding"] for d in data)
    return out


def _normalize(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _pack(v: list[float]) -> bytes:
    return array("f", v).tobytes()


def _unpack(b: bytes) -> array:
    a = array("f")
    a.frombytes(b)
    return a


def _chunk(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    chunks, i, n = [], 0, len(text)
    while i < n:
        end = min(i + CHUNK_CHARS, n)
        if end < n:  # prefer breaking at a newline/space near the window edge
            brk = text.rfind("\n", i + CHUNK_CHARS - 200, end)
            if brk <= i:
                brk = text.rfind(" ", i + CHUNK_CHARS - 100, end)
            if brk > i:
                end = brk
        piece = text[i:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        i = max(end - CHUNK_OVERLAP, i + 1)
    return chunks


def _read_text(p: Path) -> str | None:
    ext = p.suffix.lower()
    try:
        if ext == ".pdf":
            from pypdf import PdfReader
            return "\n\n".join((pg.extract_text() or "") for pg in PdfReader(str(p)).pages)
        if ext == ".docx":
            import docx
            return "\n".join(par.text for par in docx.Document(str(p)).paragraphs)
        if ext in _TEXT_EXTS or ext == "":
            return p.read_text("utf-8", errors="replace")
    except Exception:
        return None
    return None


# ── core operations (importable for tests; tools below just format these) ─────
def _do_index(path: str) -> dict:
    root = Path(path).expanduser()
    if root.is_file():
        files = [root]
    elif root.is_dir():
        files = [p for p in root.rglob("*")
                 if p.is_file() and p.suffix.lower() in _INDEXABLE]
    else:
        return {"error": f"no such path: {path}"}

    con = _db()
    indexed = skipped = total_chunks = 0
    try:
        for p in files:
            try:
                st = p.stat()
            except OSError:
                continue
            if st.st_size > MAX_FILE_BYTES:
                skipped += 1
                continue
            src = str(p.resolve())
            row = con.execute("SELECT mtime FROM chunks WHERE source=? LIMIT 1", (src,)).fetchone()
            if row and abs((row[0] or 0) - st.st_mtime) < 1e-6:
                skipped += 1
                continue  # unchanged since last index
            text = _read_text(p)
            if not text:
                skipped += 1
                continue
            chunks = _chunk(text)
            if not chunks:
                skipped += 1
                continue
            vecs = _embed(chunks)
            if len(vecs) != len(chunks):  # never store misaligned text↔vector pairs
                skipped += 1
                continue
            con.execute("DELETE FROM chunks WHERE source=?", (src,))
            con.executemany(
                "INSERT INTO chunks(source,chunk_idx,text,dim,vec,mtime) VALUES(?,?,?,?,?,?)",
                [(src, idx, c, len(v), _pack(_normalize(v)), st.st_mtime)
                 for idx, (c, v) in enumerate(zip(chunks, vecs))],
            )
            indexed += 1
            total_chunks += len(chunks)
        con.commit()
    finally:
        con.close()
    return {"files": indexed, "chunks": total_chunks, "skipped": skipped, "candidates": len(files)}


def _do_search(query: str, k: int = 5) -> list[dict]:
    q = _normalize(_embed([query])[0])
    con = _db()
    try:
        rows = con.execute("SELECT source, chunk_idx, text, vec FROM chunks").fetchall()
    finally:
        con.close()
    scored = []
    for src, idx, text, blob in rows:
        try:
            v = _unpack(blob)
        except (ValueError, TypeError):
            continue  # corrupt/partial blob — skip this row, don't kill the search
        if len(v) != len(q):
            continue
        score = sum(a * b for a, b in zip(q, v))  # cosine (both normalized)
        scored.append((score, src, idx, text))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [{"score": round(s, 4), "source": src, "chunk": idx, "text": txt}
            for s, src, idx, txt in scored[: max(1, k)]]


def _do_stats() -> dict:
    con = _db()
    try:
        chunks = con.execute("SELECT count(*) FROM chunks").fetchone()[0]
        sources = con.execute("SELECT count(DISTINCT source) FROM chunks").fetchone()[0]
    finally:
        con.close()
    return {"sources": sources, "chunks": chunks}


def _do_forget(source: str) -> dict:
    target = str(Path(source).expanduser().resolve())
    con = _db()
    try:
        cur = con.execute("DELETE FROM chunks WHERE source=? OR source LIKE ?",
                          (target, target.rstrip("/") + "/%"))
        con.commit()
        return {"removed_chunks": cur.rowcount}
    finally:
        con.close()


# ── MCP tools ─────────────────────────────────────────────────────────────────
mcp = FastMCP("rag")


@mcp.tool()
def search(query: str, k: int = 5) -> str:
    """Search the user's indexed documents/notes for passages relevant to QUERY.
    Returns the top-k matching chunks with their source file and a relevance score.
    Use this to ground answers in the user's own files before answering."""
    try:
        hits = _do_search(query, k)
    except Exception as e:
        return f"search failed: {e}"
    if not hits:
        return "No indexed content yet (or no matches). Use `index` to add files first."
    return "\n\n".join(
        f"[{i+1}] {h['source']} (chunk {h['chunk']}, score {h['score']})\n{h['text']}"
        for i, h in enumerate(hits)
    )


@mcp.tool()
def index(path: str) -> str:
    """Index a file or directory into the RAG store (chunk + embed) so `search`
    can retrieve from it. PATH is a filesystem path (~ and absolute ok). Re-indexing
    skips files unchanged since last time. Supports text files, PDF, and DOCX."""
    r = _do_index(path)
    if "error" in r:
        return r["error"]
    return (f"Indexed {r['files']} file(s) ({r['chunks']} chunks); "
            f"skipped {r['skipped']} of {r['candidates']} candidates (unchanged/empty/too-large).")


@mcp.tool()
def stats() -> str:
    """Report how many sources and chunks are currently in the RAG store."""
    r = _do_stats()
    return f"{r['sources']} source(s), {r['chunks']} chunk(s) indexed."


@mcp.tool()
def forget(source: str) -> str:
    """Remove a previously-indexed file (or everything under a directory) from the
    RAG store. Does NOT delete the file on disk — only its index entries."""
    r = _do_forget(source)
    return f"Removed {r['removed_chunks']} chunk(s) for {source}."


if __name__ == "__main__":
    mcp.run()
