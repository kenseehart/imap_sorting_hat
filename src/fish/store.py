from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator, Literal

import sqlite_vec
from sqlite_vec import serialize_float32

from fish.config import EMBED_DIM, db_path, ensure_config_dir
from fish.corpus import CorpusItem, corpus_row_to_dict, email_corpus_from_message
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

CREATE TABLE IF NOT EXISTS corpus_items (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,
    source TEXT NOT NULL,
    source_key TEXT NOT NULL UNIQUE,
    text_for_embed TEXT NOT NULL,
    body_text TEXT,
    occurred_at TEXT,
    ingested_at TEXT,
    embedded_at TEXT,
    content_hash TEXT,
    payload TEXT,
    tags TEXT
);

CREATE INDEX IF NOT EXISTS idx_corpus_kind ON corpus_items(kind);
CREATE INDEX IF NOT EXISTS idx_corpus_occurred ON corpus_items(occurred_at);
CREATE INDEX IF NOT EXISTS idx_corpus_source ON corpus_items(source);

CREATE TABLE IF NOT EXISTS training_queries (
    id INTEGER PRIMARY KEY,
    text TEXT NOT NULL,
    origin TEXT NOT NULL,
    parent_query_id INTEGER,
    context_json TEXT,
    synthesis_method TEXT,
    embed_model TEXT,
    query_embedding BLOB,
    created_at TEXT NOT NULL,
    text_hash TEXT NOT NULL,
    UNIQUE(text_hash, origin),
    FOREIGN KEY (parent_query_id) REFERENCES training_queries(id)
);

CREATE INDEX IF NOT EXISTS idx_training_queries_origin ON training_queries(origin);

CREATE TABLE IF NOT EXISTS training_samples (
    id INTEGER PRIMARY KEY,
    query_id INTEGER NOT NULL,
    corpus_item_id INTEGER NOT NULL,
    source_key TEXT NOT NULL,
    kind TEXT NOT NULL,
    occurred_at TEXT,
    content_hash TEXT,
    retriever TEXT NOT NULL,
    retrieval_similarity REAL,
    retrieval_rank INTEGER,
    query_embedding BLOB,
    message_embedding BLOB NOT NULL,
    target_relevance REAL,
    relevance_agent_version TEXT,
    relevance_model TEXT,
    labeled_at TEXT,
    created_at TEXT NOT NULL,
    superseded_at TEXT,
    pair_hash TEXT NOT NULL UNIQUE,
    FOREIGN KEY (query_id) REFERENCES training_queries(id),
    FOREIGN KEY (corpus_item_id) REFERENCES corpus_items(id)
);

CREATE INDEX IF NOT EXISTS idx_training_samples_kind ON training_samples(kind);
CREATE INDEX IF NOT EXISTS idx_training_samples_occurred ON training_samples(occurred_at);
CREATE INDEX IF NOT EXISTS idx_training_samples_retriever ON training_samples(retriever);
CREATE INDEX IF NOT EXISTS idx_training_samples_agent ON training_samples(relevance_agent_version);
CREATE INDEX IF NOT EXISTS idx_training_samples_superseded ON training_samples(superseded_at);
CREATE INDEX IF NOT EXISTS idx_training_samples_corpus ON training_samples(corpus_item_id);
"""


def connect() -> sqlite3.Connection:
    ensure_config_dir()
    db = sqlite3.connect(db_path(), timeout=30)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA busy_timeout=30000")
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
        cols = {row[1] for row in db.execute("PRAGMA table_info(messages)")}
        if "gmail_labels" not in cols:
            db.execute("ALTER TABLE messages ADD COLUMN gmail_labels TEXT")
        _ensure_vec_table(db, "corpus_vec")
        _ensure_vec_table(db, "message_vec")
        _migrate_messages_to_corpus(db)


def _ensure_vec_table(db: sqlite3.Connection, name: str) -> None:
    row = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    if not row:
        db.execute(
            f"CREATE VIRTUAL TABLE {name} USING vec0(embedding float[{EMBED_DIM}])"
        )


def _vec_table(db: sqlite3.Connection) -> str:
    row = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='corpus_vec'"
    ).fetchone()
    return "corpus_vec" if row else "message_vec"


def _migrate_messages_to_corpus(db: sqlite3.Connection) -> None:
    meta = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='corpus_items'"
    ).fetchone()
    if not meta:
        return
    count = db.execute("SELECT COUNT(*) FROM corpus_items WHERE kind='email'").fetchone()[0]
    msg_count = db.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    if msg_count and count < msg_count:
        rows = db.execute("SELECT * FROM messages").fetchall()
        for row in rows:
            _upsert_email_corpus_from_row(db, dict(row))
    if _vec_table(db) == "corpus_vec":
        legacy = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='message_vec'"
        ).fetchone()
        if legacy:
            copied = db.execute("SELECT COUNT(*) FROM corpus_vec").fetchone()[0]
            if copied == 0:
                for row in db.execute("SELECT rowid, embedding FROM message_vec").fetchall():
                    db.execute(
                        "INSERT INTO corpus_vec(rowid, embedding) VALUES (?, ?)",
                        (int(row["rowid"]), row["embedding"]),
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
        acct = db.execute(
            "SELECT email FROM accounts WHERE id = ?", (account_id,)
        ).fetchone()
        _upsert_email_corpus(
            db,
            int(existing["id"]),
            account_id,
            folder,
            uid,
            parsed,
            acct["email"] if acct else None,
            embedded_at=existing.get("embedded_at"),
        )
        return int(existing["id"]), False

    db.execute(
        """
        INSERT INTO messages (
            account_id, folder, uid, message_id, in_reply_to, subject, from_addr,
            to_addrs, cc_addrs, date, flags, body_text, body_for_embed, content_hash,
            gmail_labels
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            gmail_labels=excluded.gmail_labels,
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
            json.dumps(parsed.gmail_labels) if parsed.gmail_labels else None,
        ),
    )
    row = get_message_by_uid(db, account_id, folder, uid)
    msg_id = int(row["id"])
    acct = db.execute("SELECT email FROM accounts WHERE id = ?", (account_id,)).fetchone()
    account_email = acct["email"] if acct else None
    _upsert_email_corpus(db, msg_id, account_id, folder, uid, parsed, account_email)
    return msg_id, True


def _upsert_email_corpus_from_row(db: sqlite3.Connection, row: dict[str, Any]) -> int:
    from fish.parse import ParsedMessage

    parsed = ParsedMessage(
        subject=row.get("subject") or "",
        from_addrs=[row.get("from_addr") or ""] if row.get("from_addr") else [],
        to_addrs=json.loads(row.get("to_addrs") or "[]"),
        cc_addrs=json.loads(row.get("cc_addrs") or "[]"),
        message_id=row.get("message_id") or "",
        in_reply_to=row.get("in_reply_to") or "",
        date=row.get("date"),
        flags=json.loads(row.get("flags") or "[]"),
        body_text=row.get("body_text") or "",
        body_for_embed=row.get("body_for_embed") or "",
        content_hash=row.get("content_hash") or "",
        gmail_labels=json.loads(row["gmail_labels"]) if row.get("gmail_labels") else None,
    )
    acct = db.execute(
        "SELECT email FROM accounts WHERE id = ?", (row["account_id"],)
    ).fetchone()
    return _upsert_email_corpus(
        db,
        int(row["id"]),
        int(row["account_id"]),
        row["folder"],
        int(row["uid"]),
        parsed,
        acct["email"] if acct else None,
        embedded_at=row.get("embedded_at"),
    )


def _upsert_email_corpus(
    db: sqlite3.Connection,
    message_id: int,
    account_id: int,
    folder: str,
    uid: int,
    parsed: ParsedMessage,
    account_email: str | None,
    *,
    embedded_at: str | None = None,
) -> int:
    item = email_corpus_from_message(
        message_id, account_id, folder, uid, parsed, account_email
    )
    existing = get_corpus_by_source_key(db, item.source_key)
    preserve_embedded = embedded_at
    if existing and existing.get("embedded_at") and parsed.content_hash == existing.get("content_hash"):
        preserve_embedded = existing["embedded_at"]
    elif existing and existing.get("embedded_at") and parsed.content_hash != existing.get("content_hash"):
        preserve_embedded = None
        db.execute(f"DELETE FROM {_vec_table(db)} WHERE rowid = ?", (message_id,))
    return upsert_corpus_item(db, item, item_id=message_id, embedded_at=preserve_embedded)


def upsert_corpus_item(
    db: sqlite3.Connection,
    item: CorpusItem,
    *,
    item_id: int | None = None,
    embedded_at: str | None = None,
) -> int:
    now = _utcnow()
    embedded = embedded_at if embedded_at is not None else item.embedded_at
    if item_id is not None:
        db.execute(
            """
            INSERT INTO corpus_items (
                id, kind, source, source_key, text_for_embed, body_text,
                occurred_at, ingested_at, embedded_at, content_hash, payload, tags
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_key) DO UPDATE SET
                kind=excluded.kind,
                source=excluded.source,
                text_for_embed=excluded.text_for_embed,
                body_text=excluded.body_text,
                occurred_at=excluded.occurred_at,
                content_hash=excluded.content_hash,
                payload=excluded.payload,
                tags=excluded.tags,
                embedded_at=CASE
                    WHEN corpus_items.content_hash != excluded.content_hash THEN NULL
                    ELSE COALESCE(corpus_items.embedded_at, excluded.embedded_at)
                END
            """,
            (
                item_id,
                item.kind,
                item.source,
                item.source_key,
                item.text_for_embed,
                item.body_text,
                item.occurred_at,
                now,
                embedded,
                item.content_hash_value,
                json.dumps(item.payload),
                json.dumps(item.tags),
            ),
        )
    else:
        db.execute(
            """
            INSERT INTO corpus_items (
                kind, source, source_key, text_for_embed, body_text,
                occurred_at, ingested_at, embedded_at, content_hash, payload, tags
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_key) DO UPDATE SET
                kind=excluded.kind,
                source=excluded.source,
                text_for_embed=excluded.text_for_embed,
                body_text=excluded.body_text,
                occurred_at=excluded.occurred_at,
                content_hash=excluded.content_hash,
                payload=excluded.payload,
                tags=excluded.tags,
                embedded_at=CASE
                    WHEN corpus_items.content_hash != excluded.content_hash THEN NULL
                    ELSE COALESCE(corpus_items.embedded_at, excluded.embedded_at)
                END
            """,
            (
                item.kind,
                item.source,
                item.source_key,
                item.text_for_embed,
                item.body_text,
                item.occurred_at,
                now,
                embedded,
                item.content_hash_value,
                json.dumps(item.payload),
                json.dumps(item.tags),
            ),
        )
    row = db.execute(
        "SELECT id FROM corpus_items WHERE source_key = ?", (item.source_key,)
    ).fetchone()
    return int(row["id"])


def get_corpus_by_id(db: sqlite3.Connection, item_id: int) -> dict[str, Any] | None:
    row = db.execute("SELECT * FROM corpus_items WHERE id = ?", (item_id,)).fetchone()
    return corpus_row_to_dict(row) if row else None


def get_corpus_by_source_key(db: sqlite3.Connection, source_key: str) -> dict[str, Any] | None:
    row = db.execute(
        "SELECT * FROM corpus_items WHERE source_key = ?", (source_key,)
    ).fetchone()
    return corpus_row_to_dict(row) if row else None


def corpus_payload(row: dict[str, Any]) -> dict[str, Any]:
    payload = row.get("payload")
    if isinstance(payload, str):
        return json.loads(payload)
    return payload if isinstance(payload, dict) else {}


def memory_is_active(row: dict[str, Any]) -> bool:
    if row.get("kind") != "memory":
        return True
    return corpus_payload(row).get("superseded_by") is None


def mark_memory_superseded(
    db: sqlite3.Connection, item_id: int, superseded_by: int
) -> None:
    row = get_corpus_by_id(db, item_id)
    if not row:
        raise ValueError(f"Memory {item_id} not found")
    payload = corpus_payload(row)
    payload["superseded_by"] = superseded_by
    payload["superseded_at"] = _utcnow()
    db.execute(
        "UPDATE corpus_items SET payload = ? WHERE id = ?",
        (json.dumps(payload), item_id),
    )
    table = _vec_table(db)
    db.execute(f"DELETE FROM {table} WHERE rowid = ?", (item_id,))
    mark_samples_superseded_for_corpus(db, item_id)


def get_embedding(db: sqlite3.Connection, item_id: int) -> list[float] | None:
    import numpy as np

    table = _vec_table(db)
    row = db.execute(
        f"SELECT embedding FROM {table} WHERE rowid = ?", (item_id,)
    ).fetchone()
    if not row:
        return None
    return np.frombuffer(row["embedding"], dtype=np.float32).tolist()


def set_embedding(db: sqlite3.Connection, message_id: int, embedding: list[float]) -> None:
    set_corpus_embedding(db, message_id, embedding)


def set_corpus_embedding(db: sqlite3.Connection, item_id: int, embedding: list[float]) -> None:
    table = _vec_table(db)
    db.execute(f"DELETE FROM {table} WHERE rowid = ?", (item_id,))
    db.execute(
        f"INSERT INTO {table}(rowid, embedding) VALUES (?, ?)",
        (item_id, serialize_float32(embedding)),
    )
    now = _utcnow()
    db.execute(
        "UPDATE corpus_items SET embedded_at = ? WHERE id = ?",
        (now, item_id),
    )
    db.execute(
        "UPDATE messages SET embedded_at = ? WHERE id = ?",
        (now, item_id),
    )


def count_messages_needing_embedding(db: sqlite3.Connection) -> int:
    return count_corpus_needing_embedding(db)


def count_corpus_needing_embedding(db: sqlite3.Connection) -> int:
    row = db.execute(
        "SELECT COUNT(*) FROM corpus_items WHERE embedded_at IS NULL"
    ).fetchone()
    return int(row[0])


def messages_needing_embedding(db: sqlite3.Connection, limit: int = 100) -> list[dict[str, Any]]:
    return corpus_needing_embedding(db, limit=limit)


def corpus_needing_embedding(db: sqlite3.Connection, limit: int = 100) -> list[dict[str, Any]]:
    rows = db.execute(
        """
        SELECT id, text_for_embed FROM corpus_items
        WHERE embedded_at IS NULL
        ORDER BY occurred_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def vector_search(
    db: sqlite3.Connection, query_embedding: list[float], limit: int = 20
) -> list[tuple[int, float]]:
    return corpus_vector_search(db, query_embedding, limit=limit)


def corpus_vector_search(
    db: sqlite3.Connection,
    query_embedding: list[float],
    limit: int = 20,
    kinds: list[str] | None = None,
) -> list[tuple[int, float]]:
    k = int(limit)
    if k <= 0:
        raise ValueError(f"corpus_vector_search limit must be positive, got {limit!r}")
    table = _vec_table(db)
    rows = db.execute(
        f"""
        SELECT rowid, distance
        FROM {table}
        WHERE embedding MATCH ?
          AND k = ?
        ORDER BY distance
        """,
        (serialize_float32(query_embedding), k * 3 if kinds else k),
    ).fetchall()
    hits = [(int(r[0]), float(r[1])) for r in rows]
    if not kinds:
        return hits[:k]
    allowed = set(kinds)
    filtered: list[tuple[int, float]] = []
    for item_id, dist in hits:
        row = db.execute(
            "SELECT kind FROM corpus_items WHERE id = ?", (item_id,)
        ).fetchone()
        if row and row["kind"] in allowed:
            filtered.append((item_id, dist))
        if len(filtered) >= k:
            break
    return filtered


def keyword_search(
    db: sqlite3.Connection,
    query: str,
    account_email: str | None = None,
    folder: str | None = None,
    limit: int = 50,
    kinds: list[str] | None = None,
) -> list[int]:
    return corpus_keyword_search(
        db, query, account_email=account_email, folder=folder, limit=limit, kinds=kinds
    )


def corpus_keyword_search(
    db: sqlite3.Connection,
    query: str,
    account_email: str | None = None,
    folder: str | None = None,
    limit: int = 50,
    kinds: list[str] | None = None,
) -> list[int]:
    pattern = f"%{query}%"
    sql = """
        SELECT c.id FROM corpus_items c
        LEFT JOIN messages m ON m.id = c.id AND c.kind = 'email'
        LEFT JOIN accounts a ON a.id = m.account_id
        WHERE (
            c.text_for_embed LIKE ? OR c.body_text LIKE ?
            OR json_extract(c.payload, '$.subject') LIKE ?
            OR json_extract(c.payload, '$.from_addr') LIKE ?
        )
    """
    params: list[Any] = [pattern, pattern, pattern, pattern]
    if kinds:
        placeholders = ",".join("?" for _ in kinds)
        sql += f" AND c.kind IN ({placeholders})"
        params.extend(kinds)
    if account_email:
        sql += " AND c.kind = 'email' AND a.email = ?"
        params.append(account_email)
    if folder:
        sql += " AND c.kind = 'email' AND m.folder = ?"
        params.append(folder)
    sql += " ORDER BY c.occurred_at DESC LIMIT ?"
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
    mark_samples_superseded_for_corpus(db, message_id)
    table = _vec_table(db)
    db.execute(f"DELETE FROM {table} WHERE rowid = ?", (message_id,))
    db.execute("DELETE FROM corpus_items WHERE id = ?", (message_id,))
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


QueryOrigin = Literal["real", "synthetic"]


def normalize_query_text(text: str) -> str:
    return " ".join(text.split()).strip().lower()


def query_text_hash(text: str) -> str:
    return hashlib.sha256(normalize_query_text(text).encode()).hexdigest()


def sample_pair_hash(query_id: int, corpus_item_id: int, retriever: str) -> str:
    payload = f"{query_id}\0{corpus_item_id}\0{retriever}"
    return hashlib.sha256(payload.encode()).hexdigest()


def embedding_to_blob(vec: list[float]) -> bytes:
    return serialize_float32(vec)


def blob_to_embedding(blob: bytes | None) -> list[float] | None:
    if blob is None:
        return None
    import numpy as np

    return np.frombuffer(blob, dtype=np.float32).tolist()


def training_query_row_to_dict(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    data = dict(row)
    if data.get("query_embedding"):
        data["query_embedding"] = blob_to_embedding(data["query_embedding"])
    return data


def training_sample_row_to_dict(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    data = dict(row)
    if data.get("query_embedding"):
        data["query_embedding"] = blob_to_embedding(data["query_embedding"])
    if data.get("message_embedding"):
        data["message_embedding"] = blob_to_embedding(data["message_embedding"])
    return data


def insert_training_query(
    db: sqlite3.Connection,
    *,
    text: str,
    origin: QueryOrigin,
    context_json: str | None = None,
    parent_query_id: int | None = None,
    synthesis_method: str | None = None,
    embed_model: str | None = None,
    query_embedding: list[float] | None = None,
) -> int | None:
    """Insert a training query. Returns id, or None if duplicate."""
    now = _utcnow()
    thash = query_text_hash(text)
    blob = embedding_to_blob(query_embedding) if query_embedding else None
    try:
        cur = db.execute(
            """
            INSERT INTO training_queries (
                text, origin, parent_query_id, context_json, synthesis_method,
                embed_model, query_embedding, created_at, text_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                text,
                origin,
                parent_query_id,
                context_json,
                synthesis_method,
                embed_model,
                blob,
                now,
                thash,
            ),
        )
        return int(cur.lastrowid)
    except sqlite3.IntegrityError:
        return None


def get_training_query(db: sqlite3.Connection, query_id: int) -> dict[str, Any] | None:
    row = db.execute(
        "SELECT * FROM training_queries WHERE id = ?", (query_id,)
    ).fetchone()
    return training_query_row_to_dict(row) if row else None


def count_training_queries(
    db: sqlite3.Connection, *, origin: QueryOrigin | None = None
) -> int:
    if origin:
        row = db.execute(
            "SELECT COUNT(*) FROM training_queries WHERE origin = ?", (origin,)
        ).fetchone()
    else:
        row = db.execute("SELECT COUNT(*) FROM training_queries").fetchone()
    return int(row[0])


def list_training_queries(
    db: sqlite3.Connection,
    *,
    origin: QueryOrigin | None = None,
    limit: int | None = None,
    require_embedding: bool = False,
) -> list[dict[str, Any]]:
    sql = "SELECT * FROM training_queries WHERE 1=1"
    params: list[Any] = []
    if origin:
        sql += " AND origin = ?"
        params.append(origin)
    if require_embedding:
        sql += " AND query_embedding IS NOT NULL"
    sql += " ORDER BY id"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    rows = db.execute(sql, params).fetchall()
    return [training_query_row_to_dict(r) for r in rows]


def pick_random_training_queries(
    db: sqlite3.Connection,
    *,
    origin: QueryOrigin,
    limit: int,
) -> list[dict[str, Any]]:
    rows = db.execute(
        """
        SELECT * FROM training_queries
        WHERE origin = ?
        ORDER BY RANDOM()
        LIMIT ?
        """,
        (origin, limit),
    ).fetchall()
    return [training_query_row_to_dict(r) for r in rows]


def update_training_query_embedding(
    db: sqlite3.Connection, query_id: int, embedding: list[float], embed_model: str
) -> None:
    db.execute(
        """
        UPDATE training_queries
        SET query_embedding = ?, embed_model = ?
        WHERE id = ?
        """,
        (embedding_to_blob(embedding), embed_model, query_id),
    )


def insert_training_sample(
    db: sqlite3.Connection,
    *,
    query_id: int,
    corpus_item_id: int,
    source_key: str,
    kind: str,
    occurred_at: str | None,
    content_hash: str | None,
    retriever: str,
    retrieval_similarity: float,
    retrieval_rank: int,
    query_embedding: list[float],
    message_embedding: list[float],
) -> int | None:
    """Insert sample. Returns id, or None if pair_hash duplicate."""
    now = _utcnow()
    phash = sample_pair_hash(query_id, corpus_item_id, retriever)
    try:
        cur = db.execute(
            """
            INSERT INTO training_samples (
                query_id, corpus_item_id, source_key, kind, occurred_at, content_hash,
                retriever, retrieval_similarity, retrieval_rank,
                query_embedding, message_embedding, created_at, pair_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                query_id,
                corpus_item_id,
                source_key,
                kind,
                occurred_at,
                content_hash,
                retriever,
                retrieval_similarity,
                retrieval_rank,
                embedding_to_blob(query_embedding),
                embedding_to_blob(message_embedding),
                now,
                phash,
            ),
        )
        return int(cur.lastrowid)
    except sqlite3.IntegrityError:
        return None


def get_training_sample(db: sqlite3.Connection, sample_id: int) -> dict[str, Any] | None:
    row = db.execute(
        "SELECT * FROM training_samples WHERE id = ?", (sample_id,)
    ).fetchone()
    return training_sample_row_to_dict(row) if row else None


def list_unlabeled_samples(
    db: sqlite3.Connection,
    *,
    limit: int,
    agent_version: str | None = None,
    force: bool = False,
) -> list[dict[str, Any]]:
    if force:
        sql = """
            SELECT * FROM training_samples
            WHERE superseded_at IS NULL
            ORDER BY id
            LIMIT ?
        """
        rows = db.execute(sql, (limit,)).fetchall()
    else:
        sql = """
            SELECT * FROM training_samples
            WHERE superseded_at IS NULL
              AND (target_relevance IS NULL OR relevance_agent_version IS NULL
                   OR relevance_agent_version != ?)
            ORDER BY id
            LIMIT ?
        """
        rows = db.execute(sql, (agent_version or "", limit)).fetchall()
    return [training_sample_row_to_dict(r) for r in rows]


def update_sample_relevance(
    db: sqlite3.Connection,
    sample_id: int,
    *,
    target_relevance: float,
    agent_version: str,
    relevance_model: str,
) -> None:
    db.execute(
        """
        UPDATE training_samples
        SET target_relevance = ?, relevance_agent_version = ?,
            relevance_model = ?, labeled_at = ?
        WHERE id = ?
        """,
        (target_relevance, agent_version, relevance_model, _utcnow(), sample_id),
    )


def training_corpus_stats(db: sqlite3.Connection) -> dict[str, Any]:
    queries = {
        row["origin"]: row["n"]
        for row in db.execute(
            "SELECT origin, COUNT(*) AS n FROM training_queries GROUP BY origin"
        ).fetchall()
    }
    samples_total = db.execute(
        "SELECT COUNT(*) FROM training_samples WHERE superseded_at IS NULL"
    ).fetchone()[0]
    labeled = db.execute(
        """
        SELECT COUNT(*) FROM training_samples
        WHERE superseded_at IS NULL AND target_relevance IS NOT NULL
        """
    ).fetchone()[0]
    by_retriever = {
        row["retriever"]: row["n"]
        for row in db.execute(
            """
            SELECT retriever, COUNT(*) AS n FROM training_samples
            WHERE superseded_at IS NULL
            GROUP BY retriever
            """
        ).fetchall()
    }
    by_kind = {
        row["kind"]: row["n"]
        for row in db.execute(
            """
            SELECT kind, COUNT(*) AS n FROM training_samples
            WHERE superseded_at IS NULL
            GROUP BY kind
            """
        ).fetchall()
    }
    stale = db.execute(
        """
        SELECT COUNT(*) FROM training_samples s
        JOIN corpus_items c ON c.id = s.corpus_item_id
        WHERE s.superseded_at IS NULL
          AND s.content_hash IS NOT NULL
          AND c.content_hash IS NOT NULL
          AND s.content_hash != c.content_hash
        """
    ).fetchone()[0]
    return {
        "queries": queries,
        "samples_total": int(samples_total),
        "samples_labeled": int(labeled),
        "samples_unlabeled": int(samples_total) - int(labeled),
        "samples_by_retriever": by_retriever,
        "samples_by_kind": by_kind,
        "samples_stale": int(stale),
    }


def mark_samples_superseded_for_corpus(db: sqlite3.Connection, corpus_item_id: int) -> int:
    cur = db.execute(
        """
        UPDATE training_samples
        SET superseded_at = ?
        WHERE corpus_item_id = ? AND superseded_at IS NULL
        """,
        (_utcnow(), corpus_item_id),
    )
    return cur.rowcount


def mark_stale_samples(db: sqlite3.Connection) -> int:
    cur = db.execute(
        """
        UPDATE training_samples
        SET superseded_at = ?
        WHERE superseded_at IS NULL
          AND id IN (
            SELECT s.id FROM training_samples s
            JOIN corpus_items c ON c.id = s.corpus_item_id
            WHERE s.content_hash IS NOT NULL
              AND c.content_hash IS NOT NULL
              AND s.content_hash != c.content_hash
          )
        """,
        (_utcnow(),),
    )
    return cur.rowcount


def purge_training_samples(
    db: sqlite3.Connection,
    *,
    stale: bool = False,
    kind: str | None = None,
    before: str | None = None,
    retriever: str | None = None,
    superseded_only: bool = False,
) -> int:
    sql = "DELETE FROM training_samples WHERE 1=1"
    params: list[Any] = []
    if superseded_only:
        sql += " AND superseded_at IS NOT NULL"
    if stale:
        sql += """
            AND id IN (
                SELECT s.id FROM training_samples s
                JOIN corpus_items c ON c.id = s.corpus_item_id
                WHERE s.content_hash IS NOT NULL
                  AND c.content_hash IS NOT NULL
                  AND s.content_hash != c.content_hash
            )
        """
    if kind:
        sql += " AND kind = ?"
        params.append(kind)
    if before:
        sql += " AND occurred_at < ?"
        params.append(before)
    if retriever:
        sql += " AND retriever = ?"
        params.append(retriever)
    cur = db.execute(sql, params)
    return cur.rowcount


def load_labeled_training_pairs(
    db: sqlite3.Connection,
    *,
    exclude_superseded: bool = True,
    retriever: str | None = None,
) -> list[dict[str, Any]]:
    sql = """
        SELECT s.*, q.text AS query_text, q.context_json
        FROM training_samples s
        JOIN training_queries q ON q.id = s.query_id
        WHERE s.target_relevance IS NOT NULL
    """
    params: list[Any] = []
    if exclude_superseded:
        sql += " AND s.superseded_at IS NULL"
    if retriever:
        sql += " AND s.retriever = ?"
        params.append(retriever)
    rows = db.execute(sql, params).fetchall()
    return [training_sample_row_to_dict(r) for r in rows]
