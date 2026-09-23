"""회원/로그인 세션/분석 기록 저장 (SQLite). 크롤러와 같은 DB 파일을 쓴다."""
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
from datetime import datetime, timedelta

from crawler.store import DB_PATH

SESSION_DAYS = 14
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL UNIQUE,
    nickname TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS login_sessions (
    token_hash TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    expires_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    video_id TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    recorded_date TEXT NOT NULL,
    created_at TEXT NOT NULL,
    stats_json TEXT NOT NULL,
    UNIQUE(user_id, video_id)
);
CREATE INDEX IF NOT EXISTS idx_analyses_user ON analyses(user_id, recorded_date);
"""


def _conn():
    os.makedirs(os.path.dirname(str(DB_PATH)), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init():
    with _conn() as c:
        c.executescript(SCHEMA)


# ---- 비밀번호 (scrypt + 사용자별 salt) ----
def _hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    dk = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1)
    return salt.hex() + "$" + dk.hex()


def _verify_password(password: str, stored: str) -> bool:
    salt_hex, dk_hex = stored.split("$")
    dk = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt_hex), n=2**14, r=8, p=1)
    return hmac.compare_digest(dk.hex(), dk_hex)


def _now():
    return datetime.now().isoformat(timespec="seconds")


def create_user(email: str, nickname: str, password: str) -> int:
    email = email.strip().lower()
    nickname = nickname.strip()
    if not EMAIL_RE.match(email):
        raise ValueError("이메일 형식이 올바르지 않습니다.")
    if not (1 <= len(nickname) <= 20):
        raise ValueError("닉네임은 1~20자로 입력해주세요.")
    if len(password) < 8:
        raise ValueError("비밀번호는 8자 이상이어야 합니다.")
    try:
        with _conn() as c:
            cur = c.execute(
                "INSERT INTO users(email, nickname, password_hash, created_at) VALUES(?,?,?,?)",
                (email, nickname, _hash_password(password), _now()))
            return cur.lastrowid
    except sqlite3.IntegrityError:
        raise ValueError("이미 가입된 이메일입니다.")


def authenticate(email: str, password: str):
    with _conn() as c:
        row = c.execute("SELECT * FROM users WHERE email=?", (email.strip().lower(),)).fetchone()
    if row is None:
        _hash_password(password)  # 존재 여부와 상관없이 비슷한 시간이 걸리도록
        return None
    return dict(row) if _verify_password(password, row["password_hash"]) else None


# ---- 로그인 세션 (쿠키에는 무작위 토큰, DB에는 그 해시만 저장) ----
def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def create_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    expires = (datetime.now() + timedelta(days=SESSION_DAYS)).isoformat(timespec="seconds")
    with _conn() as c:
        c.execute("DELETE FROM login_sessions WHERE expires_at < ?", (_now(),))
        c.execute("INSERT INTO login_sessions VALUES(?,?,?)", (_token_hash(token), user_id, expires))
    return token


def user_from_token(token: str | None):
    if not token:
        return None
    with _conn() as c:
        row = c.execute(
            """SELECT u.id, u.email, u.nickname FROM login_sessions s
               JOIN users u ON u.id = s.user_id
               WHERE s.token_hash=? AND s.expires_at > ?""",
            (_token_hash(token), _now())).fetchone()
    return dict(row) if row else None


def delete_session(token: str | None):
    if token:
        with _conn() as c:
            c.execute("DELETE FROM login_sessions WHERE token_hash=?", (_token_hash(token),))


# ---- 분석 기록 ----
def save_analysis(user_id: int, video_id: str, recorded_date: str, title: str, stats: dict) -> int:
    datetime.strptime(recorded_date, "%Y-%m-%d")  # 형식 검증
    with _conn() as c:
        c.execute(
            """INSERT INTO analyses(user_id, video_id, title, recorded_date, created_at, stats_json)
               VALUES(?,?,?,?,?,?)
               ON CONFLICT(user_id, video_id) DO UPDATE SET
                 title=excluded.title, recorded_date=excluded.recorded_date,
                 created_at=excluded.created_at, stats_json=excluded.stats_json""",
            (user_id, video_id, title.strip()[:60], recorded_date, _now(), json.dumps(stats)))
        return c.execute("SELECT id FROM analyses WHERE user_id=? AND video_id=?",
                         (user_id, video_id)).fetchone()["id"]


def _row_to_analysis(r):
    d = dict(r)
    d["stats"] = json.loads(d.pop("stats_json"))
    vid = d["video_id"]
    d["trajectory_url"] = f"/results/{vid}/trajectory.jpg"
    d["heatmap_url"] = f"/results/{vid}/heatmap.png"
    return d


def list_analyses(user_id: int, limit: int = 200):
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM analyses WHERE user_id=? ORDER BY recorded_date DESC, id DESC LIMIT ?",
            (user_id, limit)).fetchall()
    return [_row_to_analysis(r) for r in rows]


def get_analysis(user_id: int, analysis_id: int):
    with _conn() as c:
        r = c.execute("SELECT * FROM analyses WHERE id=? AND user_id=?",
                      (analysis_id, user_id)).fetchone()
    return _row_to_analysis(r) if r else None


def delete_analysis(user_id: int, analysis_id: int) -> bool:
    with _conn() as c:
        return c.execute("DELETE FROM analyses WHERE id=? AND user_id=?",
                         (analysis_id, user_id)).rowcount > 0
