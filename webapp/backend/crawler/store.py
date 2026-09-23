import hashlib
import os
import sqlite3
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DB_PATH = Path(os.environ.get("COURTVISION_DB", ROOT / "data" / "courtvision.db"))

TRACKED = ["name", "start_date", "end_date", "venue", "divisions", "registration"]
FIELDS = ["source", "source_id", "name", "start_date", "end_date", "venue", "divisions",
          "registration", "ranking_group", "category", "url"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS tournaments (
    uid TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    source_id TEXT DEFAULT '',
    name TEXT NOT NULL,
    start_date TEXT NOT NULL,
    end_date TEXT NOT NULL,
    venue TEXT DEFAULT '',
    divisions TEXT DEFAULT '',
    registration TEXT DEFAULT '',
    ranking_group TEXT DEFAULT '',
    category TEXT DEFAULT '',
    url TEXT DEFAULT '',
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tournament_changes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uid TEXT NOT NULL,
    changed_at TEXT NOT NULL,
    field TEXT NOT NULL,
    old_value TEXT,
    new_value TEXT
);
CREATE INDEX IF NOT EXISTS idx_tournaments_start ON tournaments(start_date);
"""


def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def make_uid(row):
    if row.get("source_id"):
        return f"{row['source']}:{row['source_id']}"
    key = f"{row['name']}|{row['start_date']}"
    return f"{row['source']}:" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def upsert(rows):
    """새 대회는 추가, 기존 대회는 바뀐 항목만 이력에 기록. 결과 통계를 반환"""
    now = datetime.now().isoformat(timespec="seconds")
    stats = {"new": 0, "changed": 0, "unchanged": 0}
    con = connect()
    try:
        for r in rows:
            uid = make_uid(r)
            cur = con.execute("SELECT * FROM tournaments WHERE uid = ?", (uid,)).fetchone()
            values = {f: (r.get(f) or "") for f in FIELDS}

            if cur is None:
                cols = ["uid"] + FIELDS + ["first_seen", "last_seen"]
                con.execute(
                    f"INSERT INTO tournaments ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                    [uid] + [values[f] for f in FIELDS] + [now, now],
                )
                stats["new"] += 1
                continue

            changes = [(f, cur[f], values[f]) for f in TRACKED if (cur[f] or "") != values[f]]
            for f, old, new in changes:
                con.execute(
                    "INSERT INTO tournament_changes (uid, changed_at, field, old_value, new_value) VALUES (?,?,?,?,?)",
                    (uid, now, f, old, new),
                )
            stats["changed" if changes else "unchanged"] += 1

            sets = ", ".join(f"{f} = ?" for f in FIELDS)
            con.execute(f"UPDATE tournaments SET {sets}, last_seen = ? WHERE uid = ?",
                        [values[f] for f in FIELDS] + [now, uid])
        con.commit()
    finally:
        con.close()
    return stats


def latest_seen():
    con = connect()
    try:
        return con.execute("SELECT MAX(last_seen) FROM tournaments").fetchone()[0]
    finally:
        con.close()


def compute_phase(start_date, end_date, today=None):
    today = (today or date.today()).isoformat()
    if end_date < today:
        return "종료"
    if start_date > today:
        return "예정"
    return "진행중"


def query_tournaments(upcoming_only=True, source=None, division=None, limit=200):
    today = date.today().isoformat()
    sql = "SELECT * FROM tournaments WHERE 1=1"
    params = []
    if upcoming_only:
        sql += " AND end_date >= ?"
        params.append(today)
    if source:
        sql += " AND source = ?"
        params.append(source)
    if division:
        sql += " AND divisions LIKE ?"
        params.append(f"%{division}%")
    sql += " ORDER BY start_date LIMIT ?"
    params.append(limit)

    con = connect()
    try:
        rows = [dict(r) for r in con.execute(sql, params).fetchall()]
    finally:
        con.close()
    for r in rows:
        r["phase"] = compute_phase(r["start_date"], r["end_date"])
    return rows