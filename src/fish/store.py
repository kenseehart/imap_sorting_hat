from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

import sqlite_vec
from sqlite_vec import serialize_float32

from fish.config import DB_PATH, EMBED_DIM, ensure_config_dir
from fish.parse import ParsedMessage


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


SCHEMA = f"""
CREATE TABLE IF NOT EXISTS accounts (
    id INTEGER PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    imap_host TEXT NOT NULL,
    smtp_host TEXT NOT NULL,
    username TEXT NOT NULL,
    archive_folder TEXT DEFAULT 'Archive',
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS sync_state (
    account_id INTEGER NOT NULL,
    folder TEXT NOT NULL,
    uidvalidity INTEGER,
    last_uid INTEGER,
    last_sync_at TEXT,
    since_date TEXT,
    PRIMARY KEY (account_id, folder),
    FOREIGN KEY (account_id) REFERENCES accounts(id)
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY,
    account_id INTEGER NOT NULL,
    folder TEXT NOT NULL,
    uid INTEGER NOT NULL,
    message_id TEXT,
    in_reply_to TEXT,
    subject TEXT,
    from_addr TEXT,
    to_addrs TEXT,
    cc_addrs TEXT,
    date TEXT,
    flags TEXT,
    body_text TEXT,
    body_for_embed TEXT,
    content_hash TEXT,
    embedded_at TEXT,
    UNIQUE(account_id, folder, uid),
    FOREIGN KEY (account_id) REFERENCES accounts(id)
);

CREATE INDEX IF NOT EXISTS idx_messages_message_id ON messages(message_id);
CREATE INDEX IF NOT EXISTS idx_messages_in_reply_to ON messages(in_reply_to);
CREATE INDEX IF NOT EXISTS idx_messages_from ON messages(from_addr);
CREATE INDEX IF NOT EXISTS idx_messages_date ON messages(date);
CREATE INDEX IF NOT EXISTS idx_messages_account_folder ON messages(account_id, folder);

CREATE TABLE IF NOT EXISTS topics (
    id INTEGER PRIMARY KEY,
    label TEXT NOT NULL,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS message_topics (
    message_id INTEGER NOT NULL,
    topic_id INTEGER NOT NULL,
    score REAL,
    PRIMARY KEY (message_id, topic_id),
    FOREIGN KEY (message_id) REFERENCES messages(id),
    FOREIGN KEY (topic_id) REFERENCES topics(id)
);

CREATE TABLE IF NOT EXISTS importance (
    message_id INTEGER PRIMARY KEY,
    score REAL NOT NULL,
    signals_json TEXT,
    computed_at TEXT,
    FOREIGN KEY (message_id) REFERENCES messages(id)
);

CREATE TABLE IF NOT EXISTS drafts (
    id INTEGER PRIMARY KEY,
    account_email TEXT NOT NULL,
    to_addrs TEXT,
    cc_addrs TEXT,
    subject TEXT,
    body TEXT,
    in_reply_to TEXT,
    created_at TEXT
);
"""


def connect() -> sqlite3.Connection:
    ensure_config_dir()
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    return db


@contextmanager
def db_conn() -> Iterator[sqlite3.Connection]:
    db = connect()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db() -> None:
    with db_conn() as db:
        db.executescript(SCHEMA)
        row = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='message_vec'"
        ).fetchone()
        if not row:
            db.execute(
                f"CREATE VIRTUAL TABLE message_vec USING vec0(embedding float[{EMBED_DIM}])"
            )


def upsert_account(
    db: sqlite3.Connection,
    account_id: int,
    email: str,
    imap_host: str,
    smtp_host: str,
    username: str,
    archive_folder: str,
) -> int:
    db.execute(
        """
        INSERT INTO accounts (id, email, imap_host, smtp_host, username, archive_folder, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(email) DO UPDATE SET
            imap_host=excluded.imap_host,
            smtp_host=excluded.smtp_host,
            username=excluded.username,
            archive_folder=excluded.archive_folder
        """,
        (account_id, email, imap_host, smtp_host, username, archive_folder, _utcnow()),
    )
    row = db.execute("SELECT id FROM accounts WHERE email = ?", (email,)).fetchone()
    return int(row["id"])


def get_message_by_id(db: sqlite3.Connection, message_id: int) -> dict[str, Any] | None:
    row = db.execute("SELECT * FROM messages WHERE id = ?", (message_id,)).fetchone()
    return dict(row) if row else None


def get_message_by_uid(
    db: sqlite3.Connection, account_id: int, folder: str, uid: int
) -> dict[str, Any] | None:
    row = db.execute(
        "SELECT * FROM messages WHERE account_id = ? AND folder = ? AND uid = ?",
        (account_id, folder, uid),
    ).fetchone()
    return dict(row) if row else None


def upsert_message(
    db: sqlite3.Connection,
    account_id: int,
    folder: str,
    uid: int,
    parsed: ParsedMessage,
) -> tuple[int, bool]:
    existing = get_message_by_uid(db, account_id, folder, uid)
    if existing and existing["content_hash"] == parsed.content_hash:
        return int(existing["id"]), False

    db.execute(
        """
        INSERT INTO messages (
            account_id, folder, uid, message_id, in_reply_to, subject, from_addr,
            to_addrs, cc_addrs, date, flags, body_text, body_for_embed, content_hash
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(account_id, folder, uid) DO UPDATE SET
            message_id=excluded.message_id,
            in_reply_to=excluded.in_reply_to,
            subject=excluded.subject,
            from_addr=excluded.from_addr,
            to_addrs=excluded.to_addrs,
            cc_addrs=excluded.cc_addrs,
            date=excluded.date,
            flags=excluded.flags,
            body_text=excluded.body_text,
            body_for_embed=excluded.body_for_embed,
            content_hash=excluded.content_hash,
            embedded_at=NULL
        """,
        (
            account_id,
            folder,
            uid,
            parsed.message_id,
            parsed.in_reply_to,
            parsed.subject,
            parsed.from_addrs[0] if parsed.from_addrs else "",
            json.dumps(parsed.to_addrs),
            json.dumps(parsed.cc_addrs),
            parsed.date,
            json.dumps(parsed.flags),
            parsed.body_text,
            parsed.body_for_embed,
            parsed.content_hash,
        ),
    )
    row = get_message_by_uid(db, account_id, folder, uid)
    return int(row["id"]), True


def set_embedding(db: sqlite3.Connection, message_id: int, embedding: list[float]) -> None:
    db.execute("DELETE FROM message_vec WHERE rowid = ?", (message_id,))
    db.execute(
        "INSERT INTO message_vec(rowid, embedding) VALUES (?, ?)",
        (message_id, serialize_float32(embedding)),
    )
    db.execute(
        "UPDATE messages SET embedded_at = ? WHERE id = ?",
        (_utcnow(), message_id),
    )


def count_messages_needing_embedding(db: sqlite3.Connection) -> int:
    row = db.execute(
        "SELECT COUNT(*) FROM messages WHERE embedded_at IS NULL"
    ).fetchone()
    return int(row[0])


def messages_needing_embedding(db: sqlite3.Connection, limit: int = 100) -> list[dict[str, Any]]:
    rows = db.execute(
        """
        SELECT id, body_for_embed FROM messages
        WHERE embedded_at IS NULL
        ORDER BY date DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def vector_search(
    db: sqlite3.Connection, query_embedding: list[float], limit: int = 20
) -> list[tuple[int, float]]:
    rows = db.execute(
        """
        SELECT rowid, distance
        FROM message_vec
        WHERE embedding MATCH ?
        ORDER BY distance
        LIMIT ?
        """,
        (serialize_float32(query_embedding), limit),
    ).fetchall()
    return [(int(r[0]), float(r[1])) for r in rows]


def keyword_search(
    db: sqlite3.Connection,
    query: str,
    account_email: str | None = None,
    folder: str | None = None,
    limit: int = 50,
) -> list[int]:
    pattern = f"%{query}%"
    sql = """
        SELECT m.id FROM messages m
        JOIN accounts a ON a.id = m.account_id
        WHERE (
            m.subject LIKE ? OR m.from_addr LIKE ? OR m.body_text LIKE ?
            OR m.body_for_embed LIKE ?
        )
    """
    params: list[Any] = [pattern, pattern, pattern, pattern]
    if account_email:
        sql += " AND a.email = ?"
        params.append(account_email)
    if folder:
        sql += " AND m.folder = ?"
        params.append(folder)
    sql += " ORDER BY m.date DESC LIMIT ?"
    params.append(limit)
    rows = db.execute(sql, params).fetchall()
    return [int(r[0]) for r in rows]


def update_sync_state(
    db: sqlite3.Connection,
    account_id: int,
    folder: str,
    uidvalidity: int | None,
    last_uid: int | None,
    since_date: str | None,
) -> None:
    db.execute(
        """
        INSERT INTO sync_state (account_id, folder, uidvalidity, last_uid, last_sync_at, since_date)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(account_id, folder) DO UPDATE SET
            uidvalidity=excluded.uidvalidity,
            last_uid=excluded.last_uid,
            last_sync_at=excluded.last_sync_at,
            since_date=excluded.since_date
        """,
        (account_id, folder, uidvalidity, last_uid, _utcnow(), since_date),
    )


def sync_status(db: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = db.execute(
        """
        SELECT a.email, s.folder, s.uidvalidity, s.last_uid, s.last_sync_at, s.since_date,
               (SELECT COUNT(*) FROM messages m WHERE m.account_id = s.account_id AND m.folder = s.folder) AS message_count
        FROM sync_state s
        JOIN accounts a ON a.id = s.account_id
        ORDER BY a.email, s.folder
        """
    ).fetchall()
    return [dict(r) for r in rows]


def update_message_folder(
    db: sqlite3.Connection, message_id: int, folder: str, uid: int | None = None
) -> None:
    if uid is not None:
        db.execute(
            "UPDATE messages SET folder = ?, uid = ? WHERE id = ?",
            (folder, uid, message_id),
        )
    else:
        db.execute("UPDATE messages SET folder = ? WHERE id = ?", (folder, message_id))


def update_message_flags(db: sqlite3.Connection, message_id: int, flags: list[str]) -> None:
    db.execute(
        "UPDATE messages SET flags = ? WHERE id = ?",
        (json.dumps(flags), message_id),
    )


def delete_message(db: sqlite3.Connection, message_id: int) -> None:
    db.execute("DELETE FROM message_vec WHERE rowid = ?", (message_id,))
    db.execute("DELETE FROM messages WHERE id = ?", (message_id,))


def get_thread_messages(db: sqlite3.Connection, message_id: int) -> list[dict[str, Any]]:
    root = get_message_by_id(db, message_id)
    if not root:
        return []
    ids = {message_id}
    mid = root.get("message_id") or ""
    irt = root.get("in_reply_to") or ""

    rows = db.execute("SELECT * FROM messages").fetchall()
    by_mid = {r["message_id"]: dict(r) for r in rows if r["message_id"]}
    by_irt: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        key = r["in_reply_to"] or ""
        by_irt.setdefault(key, []).append(dict(r))

    def collect(mid_key: str) -> None:
        if mid_key in by_irt:
            for child in by_irt[mid_key]:
                if child["id"] not in ids:
                    ids.add(child["id"])
                    collect(child["message_id"] or "")

    if mid:
        collect(mid)
    if irt and irt in by_mid:
        parent = by_mid[irt]
        if parent["id"] not in ids:
            ids.add(parent["id"])
            collect(parent["message_id"] or "")

    result = []
    for mid_id in ids:
        row = get_message_by_id(db, mid_id)
        if row:
            result.append(row)
    result.sort(key=lambda r: r.get("date") or "")
    return result


def save_draft(
    db: sqlite3.Connection,
    account_email: str,
    to_addrs: list[str],
    subject: str,
    body: str,
    cc_addrs: list[str] | None = None,
    in_reply_to: str | None = None,
) -> int:
    cur = db.execute(
        """
        INSERT INTO drafts (account_email, to_addrs, cc_addrs, subject, body, in_reply_to, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            account_email,
            json.dumps(to_addrs),
            json.dumps(cc_addrs or []),
            subject,
            body,
            in_reply_to,
            _utcnow(),
        ),
    )
    return int(cur.lastrowid)


def get_draft(db: sqlite3.Connection, draft_id: int) -> dict[str, Any] | None:
    row = db.execute("SELECT * FROM drafts WHERE id = ?", (draft_id,)).fetchone()
    return dict(row) if row else None
