"""Tutor store: v1 migration, spaced repetition, assessment merge, streaks,
and the HTTP surface — all against an isolated data dir."""

from __future__ import annotations

import json
import sqlite3
import time

from fastapi.testclient import TestClient

from hearth_client import config as cfgmod
from hearth_client import tutor
from hearth_client.app import create_app


def _con():
    con = sqlite3.connect(":memory:")
    tutor.ensure_schema(con)
    return con


def test_v1_profile_migrates_into_items_once():
    con = _con()
    v1 = {"lang": "it", "level": "A1 (beginner)", "vocab_known": ["questa", "è"],
          "vocab_learning": ["sottotitoli"], "recurring_mistakes": ["nel mondo (not 'in il mondo')"],
          "struggles": ["articles"], "topics_covered": ["Italy"], "notes": "n"}
    con.execute("INSERT INTO tutor_profiles VALUES(?,?,?,?)", ("it", json.dumps(v1), 1, 0))
    p = tutor.load_profile(con, "it")
    assert p["level"] == "A1" and p["sessions"] == 1 and p["topics_covered"] == ["Italy"]
    assert "vocab_known" not in p
    items = {i["text"]: i for i in tutor.list_items(con, "it")}
    assert items["questa"]["box"] == 3 and items["sottotitoli"]["box"] == 1
    assert items["nel mondo (not 'in il mondo')"]["kind"] == "correction"
    tutor.load_profile(con, "it")                      # second load: no duplicates
    assert len(tutor.list_items(con, "it")) == 4


def test_review_moves_boxes_and_schedules():
    con = _con()
    item = tutor.add_item(con, "fr", {"text": "bonjour", "gloss": "hello"})
    now = time.time()
    assert item["due"] > now                          # new words wait for the next lesson
    up = tutor.review_item(con, "fr", item["id"], True, now=now)
    assert up["box"] == 1 and abs(up["due"] - (now + tutor.INTERVALS[1])) < 1
    down = tutor.review_item(con, "fr", item["id"], False, now=now)
    assert down["box"] == 0 and down["lapses"] == 1 and down["reps"] == 2
    assert tutor.review_item(con, "it", item["id"], True) is None   # wrong language


def test_merge_assessment_is_additive_and_recurring_mistakes_lapse():
    con = _con()
    tutor.add_item(con, "it", {"text": "ciao", "gloss": "hi"})
    data = {"level": "A2", "vocab": [{"text": "ciao", "gloss": "hi"},
                                     {"text": "un caffè per favore", "gloss": "a coffee please"}],
            "corrections": [{"wrong": "in il bar", "right": "nel bar", "note": "in+il=nel"}],
            "struggles": ["articulated prepositions"], "topics": ["café"], "summary": "Ordered coffee.",
            "next_focus": "numbers"}
    out = tutor.merge_assessment(con, "it", data, {"mode": "conversation", "turns": 6})
    assert out["new_items"] == 1 and out["corrections"] == 1   # 'ciao' already known
    assert out["profile"]["level"] == "A2" and out["profile"]["sessions"] == 1
    kinds = {i["text"]: i["kind"] for i in tutor.list_items(con, "it")}
    assert kinds["un caffè per favore"] == "phrase" and kinds["nel bar"] == "correction"
    # reported again in the same lesson: still one mistake
    tutor.merge_assessment(con, "it", {"corrections": [{"wrong": "in il bar", "right": "nel bar"}]}, {})
    assert [i["lapses"] for i in tutor.list_items(con, "it") if i["kind"] == "correction"] == [0]
    # the same mistake in a later lesson is a lapse, and it is due right away
    later = time.time() + tutor.SAME_LESSON + 60
    tutor.merge_assessment(con, "it", {"corrections": [{"wrong": "in il bar", "right": "Nel bar"}]}, {}, now=later)
    corr = [i for i in tutor.list_items(con, "it") if i["kind"] == "correction"]
    assert len(corr) == 1 and corr[0]["lapses"] == 1 and corr[0]["due"] <= later
    # an assessment without a level keeps the old one
    out = tutor.merge_assessment(con, "it", {"level": "", "next_focus": ["numbers", "time"],
                                             "struggles": "past tense", "topics": 7}, {})
    assert out["profile"]["struggles"] == ["articulated prepositions", "past tense"]
    assert out["profile"]["level"] == "A2" and out["next_focus"] == "numbers; time"
    ov = tutor.overview(con, "it", now=later)
    assert ov["sessions"][0]["summary"] == "" and len(ov["sessions"]) == 4
    assert ov["counts"]["correction"] == 1 and ov["counts"]["due"] >= 1


def test_streak_counts_consecutive_days():
    assert tutor._streak(["2026-10-01", "2026-09-30", "2026-09-28"], "2026-10-01", "2026-09-30") == 2
    assert tutor._streak(["2026-09-30", "2026-09-29"], "2026-10-01", "2026-09-30") == 2
    assert tutor._streak(["2026-09-28"], "2026-10-01", "2026-09-30") == 0
    assert tutor._streak([], "2026-10-01", "2026-09-30") == 0


def test_level_code_parsing():
    assert tutor.level_code("A1 (beginner)") == "A1"
    assert tutor.level_code("b2") == "B2"
    assert tutor.level_code("beginner") == ""


def test_tutor_http_surface(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    cfg = cfgmod.Config()
    cfg.providers = {"hearth": cfgmod.Provider(base_url="http://127.0.0.1:1/v1")}
    cfg.provider = "hearth"
    c = TestClient(create_app(cfg), base_url="http://localhost")
    assert c.post("/api/tutor/es/items", json={}).status_code == 400
    it = c.post("/api/tutor/es/items", json={"text": "hola", "gloss": "hi"}).json()
    assert c.post("/api/tutor/es/review", json={"id": it["id"], "passed": True}).json()["box"] == 1
    assert c.post("/api/tutor/es/review", json={"id": 999, "passed": True}).status_code == 404
    assert c.put("/api/tutor/es/profile", json={"goals": "travel", "level": "a2"}).json()["level"] == "A2"
    c.post("/api/tutor/es/drill", json={"mode": "flashcards", "reviewed": 3, "passed": 2})
    assert c.get("/api/tutor/languages").json()["languages"] == ["es"]
    ov = c.get("/api/tutor/es/overview").json()
    assert ov["profile"]["goals"] == "travel" and ov["counts"]["total"] == 1 and ov["streak"] == 1
    # too short to assess → skipped without calling the (unreachable) backend
    assert c.post("/api/tutor/es/assess", json={"messages": [{"role": "user", "content": "hola"}]}).json() == {"skipped": True}
    assert c.post("/api/tutor/es/assess", json={"messages": [
        {"role": "user", "content": "hola"}, {"role": "assistant", "content": "¡Hola!"},
        {"role": "user", "content": "bien"}]}).status_code == 502
    c.delete(f"/api/tutor/es/items/{it['id']}")
    assert c.get("/api/tutor/es/items").json()["items"] == []


def test_warmup_review_items_are_graded_by_the_assessment():
    con = _con()
    now = time.time()
    a = tutor.add_item(con, "it", {"text": "grazie", "gloss": "thanks"})
    b = tutor.add_item(con, "it", {"text": "prego", "gloss": "you're welcome"})
    c = tutor.add_item(con, "it", {"text": "ciao"})
    review = [{"id": a["id"], "text": "grazie"}, {"id": b["id"], "text": "prego"}]
    out = tutor.merge_assessment(con, "it", {"reviewed": [
        {"text": "Grazie", "correct": True}, {"text": "prego", "correct": False},
        {"text": "ciao", "correct": True},          # not in the warm-up list: ignored
        {"text": "grazie", "correct": True}]},      # graded once only
        {}, now=now, review=review)
    assert out["reviewed"] == 2
    items = {i["text"]: i for i in tutor.list_items(con, "it")}
    assert items["grazie"]["box"] == 1 and items["grazie"]["reps"] == 1
    assert items["prego"]["box"] == 0 and items["prego"]["lapses"] == 1
    assert items["ciao"]["reps"] == 0 and c


def test_a_drill_does_not_mask_a_repeat_mistake():
    con = _con()
    t0 = time.time()
    tutor.merge_assessment(con, "it", {"corrections": [{"wrong": "in il bar", "right": "al bar"}]}, {}, now=t0)
    item = [i for i in tutor.list_items(con, "it") if i["kind"] == "correction"][0]
    later = t0 + tutor.SAME_LESSON + 600
    tutor.review_item(con, "it", item["id"], True, now=later)          # drilled it...
    tutor.merge_assessment(con, "it", {"corrections": [{"wrong": "in il bar", "right": "al bar"}]},
                           {}, now=later + 60)                          # ...then made it again
    item = tutor.get_item(con, item["id"])
    assert item["lapses"] == 1 and item["box"] == 0


def test_schema_upgrade_adds_last_error(tmp_path):
    con = sqlite3.connect(tmp_path / "old.db")
    con.execute("CREATE TABLE tutor_items(id INTEGER PRIMARY KEY AUTOINCREMENT, lang TEXT NOT NULL, "
                "kind TEXT NOT NULL, text TEXT NOT NULL, gloss TEXT DEFAULT '', wrong TEXT DEFAULT '', "
                "note TEXT DEFAULT '', box INTEGER DEFAULT 0, due REAL, reps INTEGER DEFAULT 0, "
                "lapses INTEGER DEFAULT 0, seen INTEGER DEFAULT 1, created REAL, last_seen REAL)")
    tutor.ensure_schema(con)
    assert "last_error" in {r[1] for r in con.execute("PRAGMA table_info(tutor_items)")}
    tutor.ensure_schema(con)   # idempotent
