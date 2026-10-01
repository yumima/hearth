"""Language tutor: the persistent student model, spaced-repetition review, and
lesson history.

The student model used to be ONE JSON blob that an LLM rewrote after every
lesson. That was lossy (lists capped at 40, "drop oldest", the model free to
forget or invent), it had no notion of *when* to review, and only voice lessons
were ever assessed. Here the model is structured data that code owns:

* ``tutor_items`` — every word / phrase / correction the learner has met, each
  on a Leitner box with a due time. Reviewing an item (flashcard drill, or the
  tutor's warm-up) moves it up a box (longer interval) or back to box 0.
* ``tutor_sessions`` — one row per lesson: mode, focus, summary, and what was
  new, so the next lesson can pick up where the last one ended.
* ``tutor_profiles`` — the small remaining blob: level, goals, notes, struggles.

The LLM is only asked to *extract* what happened in a lesson (new vocabulary,
corrections, topics, a summary, a level judgement); merging that into the
store is deterministic code here, so nothing the learner earned can be dropped.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time

# Leitner intervals by box (seconds). Box 0 = "again soon"; 6+ = long-term.
DAY = 86400.0
INTERVALS = [10 * 60, 1 * DAY, 3 * DAY, 7 * DAY, 16 * DAY, 35 * DAY, 90 * DAY]
MAX_BOX = len(INTERVALS) - 1
MASTERED_BOX = 4          # ≥ 16-day interval survived ⇒ "mastered"
NEW_ITEM_DELAY = 60 * 60  # don't drill a word in the same lesson that taught it
KINDS = ("vocab", "phrase", "correction")
CEFR = ("A1", "A2", "B1", "B2", "C1", "C2")
SAME_LESSON = 2 * 60 * 60  # sightings closer than this belong to one lesson
LIST_CAP = 30             # struggles / topics kept on the profile (most recent)


def ensure_schema(con: sqlite3.Connection) -> None:
    con.execute("CREATE TABLE IF NOT EXISTS tutor_profiles("
                "lang TEXT PRIMARY KEY, profile TEXT, sessions INTEGER DEFAULT 0, updated REAL)")
    con.execute("CREATE TABLE IF NOT EXISTS tutor_items("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, lang TEXT NOT NULL, kind TEXT NOT NULL, "
                "text TEXT NOT NULL, gloss TEXT DEFAULT '', wrong TEXT DEFAULT '', note TEXT DEFAULT '', "
                "box INTEGER DEFAULT 0, due REAL, reps INTEGER DEFAULT 0, lapses INTEGER DEFAULT 0, "
                "seen INTEGER DEFAULT 1, created REAL, last_seen REAL)")
    con.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_items_key "
                "ON tutor_items(lang, kind, text COLLATE NOCASE)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_items_due ON tutor_items(lang, due)")
    con.execute("CREATE TABLE IF NOT EXISTS tutor_sessions("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, lang TEXT NOT NULL, mode TEXT, focus TEXT, "
                "started REAL, ended REAL, turns INTEGER, summary TEXT, next_focus TEXT, "
                "new_items INTEGER DEFAULT 0, corrections INTEGER DEFAULT 0)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_sessions_lang ON tutor_sessions(lang, ended)")


# ── profile ───────────────────────────────────────────────────────────────────

def default_profile(lang: str) -> dict:
    return {"lang": lang, "level": "", "goals": "", "notes": "", "struggles": [],
            "topics_covered": [], "sessions": 0}


def level_code(level) -> str:
    """'A1 (beginner)' / 'b2' / 'B2 upper-intermediate' → 'B2'; anything else → ''."""
    m = re.match(r"\s*([ABC][12])\b", str(level or ""), re.I)
    return m.group(1).upper() if m else ""


def _merge_list(old, new, cap=LIST_CAP) -> list:
    if isinstance(new, str):          # a model answered "x" where ["x"] was asked
        new = [new]
    if not isinstance(new, (list, tuple)):
        new = []
    out = [x for x in (old or []) if isinstance(x, str) and x.strip()]
    seen = {x.lower() for x in out}
    for x in new or []:
        if isinstance(x, str) and x.strip() and x.strip().lower() not in seen:
            out.append(x.strip())
            seen.add(x.strip().lower())
    return out[-cap:]


def load_profile(con: sqlite3.Connection, lang: str) -> dict:
    row = con.execute("SELECT profile, sessions FROM tutor_profiles WHERE lang=?", (lang,)).fetchone()
    if not row:
        return default_profile(lang)
    try:
        p = json.loads(row[0]) or {}
    except (ValueError, TypeError):
        p = {}
    prof = {**default_profile(lang), **p}
    prof["lang"] = lang
    prof["sessions"] = row[1] or 0
    prof["level"] = level_code(prof.get("level"))
    if not p.get("migrated_v2"):
        _migrate_v1(con, lang, p, prof)
    return prof


def _migrate_v1(con, lang: str, old: dict, prof: dict) -> None:
    """Carry a v1 blob's vocab / mistakes into tutor_items exactly once, so a
    learner's earlier lessons keep counting."""
    now = time.time()
    for word in old.get("vocab_known") or []:
        _upsert_item(con, lang, "vocab", word, box=3, now=now)
    for word in old.get("vocab_learning") or []:
        _upsert_item(con, lang, "vocab", word, box=1, now=now)
    for m in old.get("recurring_mistakes") or []:
        _upsert_item(con, lang, "correction", m, box=0, now=now)
    for k in ("vocab_known", "vocab_learning", "recurring_mistakes"):
        prof.pop(k, None)
    save_profile(con, lang, prof)


def save_profile(con: sqlite3.Connection, lang: str, prof: dict) -> None:
    blob = {k: v for k, v in prof.items() if k not in ("lang", "sessions")}
    blob["migrated_v2"] = True
    con.execute("INSERT INTO tutor_profiles(lang,profile,sessions,updated) VALUES(?,?,?,?) "
                "ON CONFLICT(lang) DO UPDATE SET profile=excluded.profile, updated=excluded.updated",
                (lang, json.dumps(blob), prof.get("sessions") or 0, time.time()))


def update_profile_fields(con, lang: str, fields: dict) -> dict:
    prof = load_profile(con, lang)
    if "goals" in fields:
        prof["goals"] = str(fields["goals"] or "")[:500]
    if "level" in fields:
        prof["level"] = level_code(fields["level"])
    save_profile(con, lang, prof)
    con.commit()
    return prof


# ── items + spaced repetition ─────────────────────────────────────────────────

def _upsert_item(con, lang, kind, text, gloss="", wrong="", note="", box=0, now=None, due=None):
    """Insert a new item, or — if the learner meets it again — refresh it. A
    correction that recurs is a lapse: back to box 0 and due now."""
    text = str(text or "").strip()[:200]
    if not text or kind not in KINDS:
        return None, False
    now = now or time.time()
    row = con.execute("SELECT id, box, lapses, last_seen FROM tutor_items WHERE lang=? AND kind=? "
                      "AND text=? COLLATE NOCASE", (lang, kind, text)).fetchone()
    if row:
        iid = row[0]
        # The same correction reported twice within one lesson (the tutor's
        # inline ✏️ line, then the end-of-lesson assessment) is one mistake.
        if kind == "correction" and now - (row[3] or 0) > SAME_LESSON:
            con.execute("UPDATE tutor_items SET box=0, due=?, lapses=lapses+1, seen=seen+1, "
                        "last_seen=? WHERE id=?", (now, now, iid))
        elif kind != "correction":
            con.execute("UPDATE tutor_items SET seen=seen+1, last_seen=?, "
                        "gloss=CASE WHEN gloss='' THEN ? ELSE gloss END WHERE id=?",
                        (now, str(gloss or "")[:200], iid))
        return iid, False
    cur = con.execute(
        "INSERT INTO tutor_items(lang,kind,text,gloss,wrong,note,box,due,created,last_seen) "
        "VALUES(?,?,?,?,?,?,?,?,?,?)",
        (lang, kind, text, str(gloss or "")[:200], str(wrong or "")[:200], str(note or "")[:300],
         box, due if due is not None else now + (INTERVALS[box] if box else NEW_ITEM_DELAY),
         now, now))
    return cur.lastrowid, True


def add_item(con, lang: str, body: dict) -> dict | None:
    kind = body.get("kind") if body.get("kind") in KINDS else "vocab"
    iid, _ = _upsert_item(con, lang, kind, body.get("text"), body.get("gloss"),
                          body.get("wrong"), body.get("note"))
    con.commit()
    return get_item(con, iid) if iid else None


def _row_item(r) -> dict:
    return {"id": r[0], "kind": r[1], "text": r[2], "gloss": r[3], "wrong": r[4], "note": r[5],
            "box": r[6], "due": r[7], "reps": r[8], "lapses": r[9], "seen": r[10],
            "created": r[11], "last_seen": r[12]}


_ITEM_COLS = "id,kind,text,gloss,wrong,note,box,due,reps,lapses,seen,created,last_seen"


def get_item(con, iid: int) -> dict | None:
    r = con.execute(f"SELECT {_ITEM_COLS} FROM tutor_items WHERE id=?", (iid,)).fetchone()
    return _row_item(r) if r else None


def list_items(con, lang: str, kind: str | None = None) -> list[dict]:
    q, args = f"SELECT {_ITEM_COLS} FROM tutor_items WHERE lang=?", [lang]
    if kind in KINDS:
        q += " AND kind=?"
        args.append(kind)
    q += " ORDER BY last_seen DESC"
    return [_row_item(r) for r in con.execute(q, args).fetchall()]


def due_items(con, lang: str, limit: int = 10, now: float | None = None) -> list[dict]:
    now = now or time.time()
    rows = con.execute(f"SELECT {_ITEM_COLS} FROM tutor_items WHERE lang=? AND due<=? "
                       "ORDER BY box ASC, due ASC LIMIT ?", (lang, now, limit)).fetchall()
    return [_row_item(r) for r in rows]


def review_item(con, lang: str, iid: int, passed: bool, now: float | None = None) -> dict | None:
    now = now or time.time()
    r = con.execute("SELECT box FROM tutor_items WHERE id=? AND lang=?", (iid, lang)).fetchone()
    if not r:
        return None
    box = min(r[0] + 1, MAX_BOX) if passed else 0
    con.execute("UPDATE tutor_items SET box=?, due=?, reps=reps+1, lapses=lapses+?, last_seen=? "
                "WHERE id=?", (box, now + INTERVALS[box], 0 if passed else 1, now, iid))
    con.commit()
    return get_item(con, iid)


def delete_item(con, lang: str, iid: int) -> None:
    con.execute("DELETE FROM tutor_items WHERE id=? AND lang=?", (iid, lang))
    con.commit()


def languages(con) -> list[str]:
    """Every language key the learner has any history in."""
    rows = con.execute("SELECT lang FROM tutor_profiles UNION SELECT lang FROM tutor_items "
                       "UNION SELECT lang FROM tutor_sessions").fetchall()
    return sorted({r[0] for r in rows if r[0]})


# ── overview (the Learn home + the tutor's lesson plan) ───────────────────────

def _streak(days: list[str], today: str, yesterday: str) -> int:
    """Consecutive study days ending today (or yesterday — a streak isn't lost
    until a whole day is missed). ``days`` are distinct YYYY-MM-DD, newest first."""
    if not days or days[0] not in (today, yesterday):
        return 0
    n, expect = 0, days[0]
    for d in days:
        if d != expect:
            break
        n += 1
        expect = time.strftime("%Y-%m-%d", time.localtime(
            time.mktime(time.strptime(d, "%Y-%m-%d")) - DAY / 2))
    return n


def overview(con, lang: str, now: float | None = None) -> dict:
    now = now or time.time()
    prof = load_profile(con, lang)
    counts = {"total": 0, "new": 0, "learning": 0, "mastered": 0, "due": 0,
              "vocab": 0, "phrase": 0, "correction": 0}
    for kind, box, due in con.execute("SELECT kind, box, due FROM tutor_items WHERE lang=?", (lang,)):
        counts["total"] += 1
        counts[kind] = counts.get(kind, 0) + 1
        if box >= MASTERED_BOX:
            counts["mastered"] += 1
        elif box == 0:
            counts["new"] += 1
        else:
            counts["learning"] += 1
        if due is not None and due <= now:
            counts["due"] += 1
    sessions = [
        {"id": r[0], "mode": r[1], "focus": r[2], "started": r[3], "ended": r[4], "turns": r[5],
         "summary": r[6], "next_focus": r[7], "new_items": r[8], "corrections": r[9]}
        for r in con.execute("SELECT id,mode,focus,started,ended,turns,summary,next_focus,"
                             "new_items,corrections FROM tutor_sessions WHERE lang=? "
                             "ORDER BY ended DESC LIMIT 8", (lang,))]
    days = [r[0] for r in con.execute(
        "SELECT DISTINCT date(ended, 'unixepoch', 'localtime') d FROM tutor_sessions "
        "WHERE lang=? ORDER BY d DESC LIMIT 400", (lang,))]
    today = time.strftime("%Y-%m-%d", time.localtime(now))
    yesterday = time.strftime("%Y-%m-%d", time.localtime(now - DAY))
    recent_corr = [_row_item(r) for r in con.execute(
        f"SELECT {_ITEM_COLS} FROM tutor_items WHERE lang=? AND kind='correction' "
        "ORDER BY lapses DESC, last_seen DESC LIMIT 6", (lang,))]
    con.commit()  # load_profile may have migrated
    return {"profile": prof, "counts": counts, "sessions": sessions,
            "streak": _streak(days, today, yesterday), "studied_today": bool(days and days[0] == today),
            "due": due_items(con, lang, 12, now), "corrections": recent_corr,
            "recent_vocab": [i for i in list_items(con, lang) if i["kind"] != "correction"][:12]}


# ── end-of-lesson assessment (LLM extracts; code merges) ──────────────────────

ASSESS_SYSTEM = (
    "You review a language lesson between a TUTOR and a LEARNER and extract what "
    "happened, as ONE JSON object with exactly these keys:\n"
    '  "level": the learner\'s CEFR level (A1, A2, B1, B2, C1 or C2) judged from what '
    "the LEARNER produced; use \"\" if there is too little evidence to judge,\n"
    '  "vocab": target-language words or short phrases the learner was taught or used '
    'this lesson, as [{"text": "<target language>", "gloss": "<English meaning>"}],\n'
    '  "corrections": mistakes the LEARNER made, as [{"wrong": "<what they said>", '
    '"right": "<correct target-language form>", "note": "<short English why>"}],\n'
    '  "struggles": short names of grammar or skills they found hard, e.g. "past tense",\n'
    '  "topics": short names of subjects covered, e.g. "ordering coffee",\n'
    '  "summary": one or two plain English sentences on what the lesson covered and how it went,\n'
    '  "next_focus": one short suggestion for what the next lesson should practise.\n'
    "Only include things that actually appear in the transcript — never invent words, "
    "mistakes or progress. Empty lists are fine. Output ONLY the JSON object."
)


def assess_prompt(lang_name: str, prof: dict, transcript: str) -> list[dict]:
    known = {k: prof.get(k) for k in ("level", "struggles", "topics_covered")}
    user = (f"TARGET LANGUAGE: {lang_name}\nPRIOR PROFILE: {json.dumps(known)}\n\n"
            f"LESSON TRANSCRIPT:\n{transcript}")
    return [{"role": "system", "content": ASSESS_SYSTEM}, {"role": "user", "content": user}]


def transcript_of(messages: list, limit: int = 9000) -> str:
    def text(c):
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            return " ".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
        return ""
    lines = [f"{'LEARNER' if m.get('role') == 'user' else 'TUTOR'}: {text(m.get('content'))}"
             for m in messages if isinstance(m, dict) and m.get("role") in ("user", "assistant")]
    return "\n".join(lines)[-limit:]


def _text(v, cap: int) -> str:
    """Models sometimes return a list where a string was asked for."""
    if isinstance(v, (list, tuple)):
        v = "; ".join(str(x) for x in v if x)
    return str(v or "").strip()[:cap]


def merge_assessment(con, lang: str, data: dict, session: dict, now: float | None = None) -> dict:
    """Fold an extracted lesson into the store. Returns counts for the UI."""
    now = now or time.time()
    prof = load_profile(con, lang)
    new_items = corrections = 0
    for v in (data.get("vocab") or [])[:40]:
        if isinstance(v, dict):
            text = v.get("text") or ""
            kind = "phrase" if len(str(text).split()) > 2 else "vocab"
            _, created = _upsert_item(con, lang, kind, text, v.get("gloss"), now=now)
            new_items += created
    for c in (data.get("corrections") or [])[:20]:
        if isinstance(c, dict) and c.get("right"):
            _upsert_item(con, lang, "correction", c.get("right"), gloss=c.get("note") or "",
                         wrong=c.get("wrong") or "", note=c.get("note") or "", now=now, due=now)
            corrections += 1
    lvl = level_code(data.get("level"))
    if lvl:
        prof["level"] = lvl
    prof["struggles"] = _merge_list(prof.get("struggles"), data.get("struggles"))
    prof["topics_covered"] = _merge_list(prof.get("topics_covered"), data.get("topics"))
    prof["sessions"] = (prof.get("sessions") or 0) + 1
    save_profile(con, lang, prof)
    con.execute("UPDATE tutor_profiles SET sessions=? WHERE lang=?", (prof["sessions"], lang))
    con.execute("INSERT INTO tutor_sessions(lang,mode,focus,started,ended,turns,summary,next_focus,"
                "new_items,corrections) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (lang, str(session.get("mode") or "")[:40], str(session.get("focus") or "")[:200],
                 session.get("started") or now, now, int(session.get("turns") or 0),
                 _text(data.get("summary"), 600), _text(data.get("next_focus"), 200),
                 new_items, corrections))
    con.commit()
    return {"profile": prof, "new_items": new_items, "corrections": corrections,
            "summary": _text(data.get("summary"), 600), "next_focus": _text(data.get("next_focus"), 200)}


def record_drill(con, lang: str, mode: str, reviewed: int, passed: int, now: float | None = None) -> None:
    """A flashcard / pronunciation drill counts toward the streak. It has no
    summary, so the Learn page's "last lesson" stays the last real lesson."""
    now = now or time.time()
    con.execute("INSERT INTO tutor_sessions(lang,mode,focus,started,ended,turns,summary,next_focus,"
                "new_items,corrections) VALUES(?,?,?,?,?,?,?,?,0,0)",
                (lang, mode[:40], f"{passed}/{reviewed} correct", now, now, reviewed, "", ""))
    con.commit()
