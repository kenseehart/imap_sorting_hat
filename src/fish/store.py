from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator, Literal

import numpy as np

from fish.config import db_path, ensure_config_dir
from fish.corpus import (
    EMAIL_CORPUS_MESSAGE_JOIN,
    CorpusItem,
    corpus_row_to_dict,
    email_corpus_from_message,
    parse_imap_source_key,
)
from fish.parse import ParsedMessage


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def embedding_to_blob(vec: list[float] | np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).reshape(-1).tobytes()


def blob_to_embedding(blob: bytes | None) -> np.ndarray | None:
    """Decode a float32 embedding blob without expanding to Python floats.

    Returns a contiguous ``float32`` copy so the array outlives the SQLite
    buffer. Call ``embedding_as_list`` only at API edges that require lists
    (e.g. some Qdrant client paths).
    """
    if blob is None:
        return None
    arr = np.frombuffer(blob, dtype=np.float32)
    if arr.size == 0:
        return None
    return np.array(arr, dtype=np.float32, copy=True)


def embedding_as_list(vec: list[float] | np.ndarray | None) -> list[float] | None:
    """Convert an embedding to ``list[float]`` at an API boundary only."""
    if vec is None:
        return None
    if isinstance(vec, list):
        return vec
    return np.asarray(vec, dtype=np.float32).reshape(-1).tolist()


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
    header_json TEXT,
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

CREATE TABLE IF NOT EXISTS corpus_raw_embeddings (
    item_id INTEGER PRIMARY KEY,
    embedding BLOB NOT NULL,
    header_embedding BLOB,
    body_embedding BLOB,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (item_id) REFERENCES corpus_items(id)
);

CREATE TABLE IF NOT EXISTS retrieval_models (
    model_id TEXT PRIMARY KEY,
    config_name TEXT NOT NULL,
    vec_table TEXT NOT NULL UNIQUE,
    prz_name TEXT,
    created_at TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 0,
    meta_json TEXT
);

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
    source TEXT,
    meta_json TEXT,
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
    return db


def is_sqlite_locked(exc: BaseException) -> bool:
    return isinstance(exc, sqlite3.OperationalError) and "locked" in str(exc).lower()


def format_sqlite_locked_error(exc: BaseException) -> RuntimeError:
    """Actionable error when SQLite busy_timeout is exhausted."""
    from fish.write_lock import read_lock_status

    status = read_lock_status()
    if status.held:
        holder = f"Fish write lock held by pid={status.pid} op={status.operation!r}"
    else:
        holder = (
            "Fish write lock is free — another process may be writing "
            "(label/embed/memory) or a long SQLite transaction is open"
        )
    return RuntimeError(
        f"SQLite database is locked after busy_timeout ({exc}). {holder}. "
        f"Retry shortly, or stop overlapping heavy writers (sync/import/reembed)."
    )


@contextmanager
def db_conn() -> Iterator[sqlite3.Connection]:
    db = connect()
    try:
        yield db
        db.commit()
    except Exception as exc:
        db.rollback()
        if is_sqlite_locked(exc):
            raise format_sqlite_locked_error(exc) from exc
        raise
    finally:
        db.close()


def init_db() -> None:
    with db_conn() as db:
        db.executescript(SCHEMA)
        cols = {row[1] for row in db.execute("PRAGMA table_info(messages)")}
        if "gmail_labels" not in cols:
            db.execute("ALTER TABLE messages ADD COLUMN gmail_labels TEXT")
        if "gm_msgid" not in cols:
            db.execute("ALTER TABLE messages ADD COLUMN gm_msgid TEXT")
        if "canonical_id" not in cols:
            db.execute("ALTER TABLE messages ADD COLUMN canonical_id TEXT")
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_canonical_id ON messages(canonical_id)"
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_gm_msgid ON messages(gm_msgid)"
        )
        tq_cols = {row[1] for row in db.execute("PRAGMA table_info(training_queries)")}
        if "source" not in tq_cols:
            db.execute("ALTER TABLE training_queries ADD COLUMN source TEXT")
        if "meta_json" not in tq_cols:
            db.execute("ALTER TABLE training_queries ADD COLUMN meta_json TEXT")
        ci_cols = {row[1] for row in db.execute("PRAGMA table_info(corpus_items)")}
        if "header_json" not in ci_cols:
            db.execute("ALTER TABLE corpus_items ADD COLUMN header_json TEXT")
        raw_cols = {row[1] for row in db.execute("PRAGMA table_info(corpus_raw_embeddings)")}
        if "header_embedding" not in raw_cols:
            db.execute("ALTER TABLE corpus_raw_embeddings ADD COLUMN header_embedding BLOB")
        if "body_embedding" not in raw_cols:
            db.execute("ALTER TABLE corpus_raw_embeddings ADD COLUMN body_embedding BLOB")
        from fish.prism.registry import ensure_legacy_model, ensure_model_vec_tables

        ensure_legacy_model(db)
        ensure_model_vec_tables(db)
        _migrate_messages_to_corpus(db)
        migrate_training_query_origins(db)


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
    # Historical message_vec → corpus_vec copy removed: never COUNT/scan huge
    # sqlite-vec tables (hangs). Live DBs already completed that migration.



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
    from fish.identity import email_canonical_id, normalize_gm_msgid

    acct = db.execute(
        "SELECT email FROM accounts WHERE id = ?", (account_id,)
    ).fetchone()
    account_email = acct["email"] if acct else None
    if not account_email:
        raise RuntimeError(f"No account email for account_id={account_id}")

    gm_msgid = normalize_gm_msgid(getattr(parsed, "gm_msgid", None))
    parsed.gm_msgid = gm_msgid
    from_addr = parsed.from_addrs[0] if parsed.from_addrs else ""
    canon, _synthetic = email_canonical_id(
        account_email=account_email,
        rfc_message_id=parsed.message_id,
        gm_msgid=gm_msgid,
        from_addr=from_addr,
        date=parsed.date,
        subject=parsed.subject or "",
        body=parsed.body_text or "",
    )

    existing = get_message_by_uid(db, account_id, folder, uid)
    if existing and existing["content_hash"] == parsed.content_hash:
        _upsert_email_corpus(
            db,
            int(existing["id"]),
            account_id,
            folder,
            uid,
            parsed,
            account_email,
            embedded_at=existing.get("embedded_at"),
            canonical_id=canon,
            gm_msgid=gm_msgid,
        )
        return int(existing["id"]), False

    db.execute(
        """
        INSERT INTO messages (
            account_id, folder, uid, message_id, in_reply_to, subject, from_addr,
            to_addrs, cc_addrs, date, flags, body_text, body_for_embed, content_hash,
            gmail_labels, gm_msgid, canonical_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            gm_msgid=excluded.gm_msgid,
            canonical_id=excluded.canonical_id,
            embedded_at=NULL
        """,
        (
            account_id,
            folder,
            uid,
            parsed.message_id,
            parsed.in_reply_to,
            parsed.subject,
            from_addr,
            json.dumps(parsed.to_addrs),
            json.dumps(parsed.cc_addrs),
            parsed.date,
            json.dumps(parsed.flags),
            parsed.body_text,
            parsed.body_for_embed,
            parsed.content_hash,
            json.dumps(parsed.gmail_labels) if parsed.gmail_labels else None,
            gm_msgid,
            canon,
        ),
    )
    row = get_message_by_uid(db, account_id, folder, uid)
    msg_id = int(row["id"])
    _upsert_email_corpus(
        db,
        msg_id,
        account_id,
        folder,
        uid,
        parsed,
        account_email,
        canonical_id=canon,
        gm_msgid=gm_msgid,
    )
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
        gm_msgid=row.get("gm_msgid"),
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
        canonical_id=row.get("canonical_id"),
        gm_msgid=row.get("gm_msgid"),
        allow_gmail_rfc_fallback=True,
    )


def _upsert_email_corpus(
    db: sqlite3.Connection,
    message_pk: int,
    account_id: int,
    folder: str,
    uid: int,
    parsed: ParsedMessage,
    account_email: str | None,
    *,
    embedded_at: str | None = None,
    canonical_id: str | None = None,
    gm_msgid: str | None = None,
    allow_gmail_rfc_fallback: bool = False,
) -> int:
    if gm_msgid and not getattr(parsed, "gm_msgid", None):
        parsed.gm_msgid = gm_msgid
    item = email_corpus_from_message(
        message_pk,
        account_id,
        folder,
        uid,
        parsed,
        account_email,
        allow_gmail_rfc_fallback=allow_gmail_rfc_fallback,
    )
    if canonical_id:
        item.source_key = canonical_id
        item.payload["canonical_id"] = canonical_id
    existing = get_corpus_by_source_key(db, item.source_key)
    preserve_embedded = embedded_at
    if existing and existing.get("embedded_at") and parsed.content_hash == existing.get(
        "content_hash"
    ):
        preserve_embedded = existing["embedded_at"]
    elif (
        existing
        and existing.get("embedded_at")
        and parsed.content_hash != existing.get("content_hash")
    ):
        preserve_embedded = None
        unindex_corpus_item(db, int(existing["id"]))
    # Never pass messages.id as preferred corpus PK (cross-source collisions).
    corpus_id = upsert_corpus_item(db, item, item_id=None, embedded_at=preserve_embedded)
    db.execute(
        "UPDATE messages SET canonical_id = ?, gm_msgid = COALESCE(?, gm_msgid) WHERE id = ?",
        (item.source_key, gm_msgid or getattr(parsed, "gm_msgid", None), message_pk),
    )
    return corpus_id


def upsert_corpus_item(
    db: sqlite3.Connection,
    item: CorpusItem,
    *,
    item_id: int | None = None,
    embedded_at: str | None = None,
) -> int:
    now = _utcnow()
    embedded = embedded_at if embedded_at is not None else item.embedded_at
    existing_key = get_corpus_by_source_key(db, item.source_key)
    if existing_key is not None:
        db.execute(
            """
            UPDATE corpus_items SET
                kind=?,
                source=?,
                text_for_embed=?,
                body_text=?,
                header_json=?,
                occurred_at=?,
                content_hash=?,
                payload=?,
                tags=?,
                embedded_at=CASE
                    WHEN content_hash != ? THEN NULL
                    ELSE COALESCE(embedded_at, ?)
                END
            WHERE source_key=?
            """,
            (
                item.kind,
                item.source,
                item.text_for_embed,
                item.body_text,
                item.header_json,
                item.occurred_at,
                item.content_hash_value,
                json.dumps(item.payload),
                json.dumps(item.tags),
                item.content_hash_value,
                embedded,
                item.source_key,
            ),
        )
        return int(existing_key["id"])

    use_id = item_id
    if use_id is not None:
        existing_id = db.execute(
            "SELECT id, source_key FROM corpus_items WHERE id = ?", (use_id,)
        ).fetchone()
        if existing_id is not None and existing_id["source_key"] != item.source_key:
            # Primary-key collision with a different corpus row — allocate a new id.
            use_id = None

    if use_id is not None:
        db.execute(
            """
            INSERT INTO corpus_items (
                id, kind, source, source_key, text_for_embed, body_text, header_json,
                occurred_at, ingested_at, embedded_at, content_hash, payload, tags
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                use_id,
                item.kind,
                item.source,
                item.source_key,
                item.text_for_embed,
                item.body_text,
                item.header_json,
                item.occurred_at,
                now,
                embedded,
                item.content_hash_value,
                json.dumps(item.payload),
                json.dumps(item.tags),
            ),
        )
        return use_id

    db.execute(
        """
        INSERT INTO corpus_items (
            kind, source, source_key, text_for_embed, body_text, header_json,
            occurred_at, ingested_at, embedded_at, content_hash, payload, tags
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            item.kind,
            item.source,
            item.source_key,
            item.text_for_embed,
            item.body_text,
            item.header_json,
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
    unindex_corpus_item(db, item_id)
    mark_samples_superseded_for_corpus(db, item_id)


def get_embedding(db: sqlite3.Connection, item_id: int) -> np.ndarray | None:
    """Frozen OpenAI raw embedding c (SQLite corpus_raw_embeddings)."""
    return get_raw_embedding(db, item_id)


def get_raw_embedding(db: sqlite3.Connection, item_id: int) -> np.ndarray | None:
    """Frozen OpenAI combined embedding c from SQLite (durable; not Qdrant-only)."""
    row = db.execute(
        "SELECT embedding FROM corpus_raw_embeddings WHERE item_id = ?",
        (item_id,),
    ).fetchone()
    if not row:
        return None
    return blob_to_embedding(row["embedding"])


def get_raw_field_embeddings(
    db: sqlite3.Connection, item_id: int
) -> dict[str, np.ndarray | None]:
    """Return durable raw vectors: combined, header, body (SQLite only for fields)."""
    row = db.execute(
        """
        SELECT embedding, header_embedding, body_embedding
        FROM corpus_raw_embeddings WHERE item_id = ?
        """,
        (item_id,),
    ).fetchone()
    if not row:
        return {"combined": None, "header": None, "body": None}
    return {
        "combined": blob_to_embedding(row["embedding"]),
        "header": blob_to_embedding(row["header_embedding"]),
        "body": blob_to_embedding(row["body_embedding"]),
    }


def set_raw_field_embeddings(
    db: sqlite3.Connection,
    item_id: int,
    *,
    header_embedding: list[float] | np.ndarray | None = None,
    body_embedding: list[float] | np.ndarray | None = None,
) -> None:
    """Write header/body raw vectors to SQLite only (no Qdrant upsert)."""
    if header_embedding is None and body_embedding is None:
        return
    combined = get_raw_embedding(db, item_id)
    if combined is None:
        raise ValueError(f"corpus item {item_id} has no combined embedding")
    _store_raw_embedding(
        db,
        item_id,
        combined,
        header_embedding=header_embedding,
        body_embedding=body_embedding,
    )


def _header_json_from_message_row(row: sqlite3.Row | dict[str, Any]) -> str:
    header_obj = {
        "subject": row["subject"] or "",
        "from": [row["from_addr"]] if row["from_addr"] else [],
        "to": json.loads(row["to_addrs"] or "[]"),
        "cc": json.loads(row["cc_addrs"] or "[]"),
        "message_id": row["message_id"] or "",
        "in_reply_to": row["in_reply_to"] or "",
        "date": row["date"],
        "account_email": row["account_email"],
        "folder": row["folder"],
    }
    return json.dumps(header_obj, sort_keys=True, ensure_ascii=False)


def _header_json_from_non_email_payload(kind: str, payload: dict[str, Any]) -> str | None:
    header_obj: dict[str, Any] = {"kind": kind}
    for key in (
        "phone",
        "direction",
        "contact_name",
        "sms_id",
        "title",
        "role",
        "conversation_id",
        "memory_key",
        "source",
    ):
        if key in payload and payload[key] is not None:
            header_obj[key] = payload[key]
    if len(header_obj) <= 1:
        return None
    return json.dumps(header_obj, sort_keys=True, ensure_ascii=False)


def backfill_corpus_header_json(
    db: sqlite3.Connection, *, training_only: bool = False
) -> int:
    """Fill missing corpus_items.header_json from messages / payload (no OpenAI).

    Email rows are resolved via ``source_key`` → messages (account/folder/uid),
    never ``m.id = c.id`` (id skew poisons neighbor headers).

    ``training_only=True`` limits to labeled training_samples (smoke / train prep).
    """
    return repair_corpus_header_json(
        db, training_only=training_only, missing_only=True
    )["updated"]


def repair_corpus_header_json(
    db: sqlite3.Connection,
    *,
    training_only: bool = False,
    missing_only: bool = False,
    dry_run: bool = False,
    min_corpus_id: int | None = None,
) -> dict[str, int]:
    """Rebuild corpus_items.header_json from the correctly resolved message.

    Resolves email via payload IMAP locator (or legacy ``imap:`` source_key) +
    indexed ``messages(account_id, folder, uid)`` — never ``m.id = c.id``.

    Scans email corpus rows (default ``id >= 110920``, the known SMS collision)
    rather than a full expression-join over messages (too slow on cloud DBs).

    Clears ``header_embedding`` for rewritten rows (``fish embed --fields``).
    """
    from fish.corpus import email_locator_from_payload

    train_ids: set[int] | None = None
    if training_only:
        train_ids = {
            int(r[0])
            for r in db.execute(
                """
                SELECT DISTINCT corpus_item_id FROM training_samples
                WHERE superseded_at IS NULL AND target_relevance IS NOT NULL
                """
            ).fetchall()
        }

    account_email = {
        int(r["id"]): r["email"]
        for r in db.execute("SELECT id, email FROM accounts").fetchall()
    }
    msg_stmt = """
        SELECT id, subject, from_addr, to_addrs, cc_addrs, message_id, in_reply_to,
               date, folder, account_id
        FROM messages
        WHERE account_id = ? AND folder = ? AND uid = ?
    """

    # Default: only rows at/after the known SMS PK collision (test:sms:1 @ 110920).
    if min_corpus_id is None and not missing_only and not training_only:
        min_corpus_id = 110920

    clauses = ["c.kind = 'email'"]
    params: list[Any] = []
    if missing_only:
        clauses.append("(c.header_json IS NULL OR c.header_json = '')")
    if min_corpus_id is not None:
        clauses.append("c.id >= ?")
        params.append(min_corpus_id)

    updated = 0
    header_embeds_cleared = 0
    unmatched_email = 0
    changed_ids: list[int] = []

    email_rows = db.execute(
        f"""
        SELECT c.id, c.source_key, c.header_json, c.payload
        FROM corpus_items c
        WHERE {' AND '.join(clauses)}
        ORDER BY c.id
        """,
        params,
    )
    for crow in email_rows:
        cid = int(crow["id"])
        if train_ids is not None and cid not in train_ids:
            continue
        locator = email_locator_from_payload(crow["payload"])
        if locator is None:
            locator = parse_imap_source_key(crow["source_key"] or "")
        if locator is None:
            unmatched_email += 1
            continue
        account_id, folder, uid = locator
        m = db.execute(msg_stmt, (account_id, folder, uid)).fetchone()
        if m is None:
            unmatched_email += 1
            continue
        msg = {
            "subject": m["subject"],
            "from_addr": m["from_addr"],
            "to_addrs": m["to_addrs"],
            "cc_addrs": m["cc_addrs"],
            "message_id": m["message_id"],
            "in_reply_to": m["in_reply_to"],
            "date": m["date"],
            "folder": m["folder"],
            "account_email": account_email.get(int(m["account_id"])),
        }
        new_header = _header_json_from_message_row(msg)
        old_header = crow["header_json"] or ""
        if old_header == new_header:
            continue
        if not missing_only and old_header:
            try:
                old = json.loads(old_header)
            except json.JSONDecodeError:
                old = {}
            if (
                (old.get("subject") or "") == (msg["subject"] or "")
                and (old.get("message_id") or "") == (msg["message_id"] or "")
                and (old.get("folder") or "") == (msg["folder"] or "")
            ):
                continue
        changed_ids.append(cid)
        if not dry_run:
            db.execute(
                "UPDATE corpus_items SET header_json = ? WHERE id = ?",
                (new_header, cid),
            )
            if len(changed_ids) % 500 == 0:
                db.commit()
                print(f"repair progress updated={len(changed_ids)}", flush=True)
        updated += 1

    if changed_ids and not dry_run:
        now = _utcnow()
        chunk = 500
        for i in range(0, len(changed_ids), chunk):
            batch = changed_ids[i : i + chunk]
            placeholders = ",".join("?" * len(batch))
            cur = db.execute(
                f"""
                UPDATE corpus_raw_embeddings
                SET header_embedding = NULL, updated_at = ?
                WHERE item_id IN ({placeholders}) AND header_embedding IS NOT NULL
                """,
                [now, *batch],
            )
            header_embeds_cleared += cur.rowcount

    other_updated = 0
    other = db.execute(
        """
        SELECT c.id, c.kind, c.payload, c.header_json FROM corpus_items c
        WHERE c.kind != 'email'
          AND (c.header_json IS NULL OR c.header_json = '')
          AND c.payload IS NOT NULL AND c.payload != ''
        """
    ).fetchall()
    for row in other:
        if train_ids is not None and int(row["id"]) not in train_ids:
            continue
        try:
            payload = json.loads(row["payload"] or "{}")
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or not payload:
            continue
        new_header = _header_json_from_non_email_payload(row["kind"], payload)
        if new_header is None:
            continue
        old_header = row["header_json"] or ""
        if old_header == new_header:
            continue
        if not dry_run:
            db.execute(
                "UPDATE corpus_items SET header_json = ? WHERE id = ?",
                (new_header, row["id"]),
            )
            cur = db.execute(
                """
                UPDATE corpus_raw_embeddings
                SET header_embedding = NULL, updated_at = ?
                WHERE item_id = ? AND header_embedding IS NOT NULL
                """,
                (_utcnow(), row["id"]),
            )
            header_embeds_cleared += cur.rowcount
        other_updated += 1
        updated += 1

    return {
        "updated": updated,
        "email_updated": updated - other_updated,
        "other_updated": other_updated,
        "header_embeds_cleared": header_embeds_cleared,
        "unmatched_email": unmatched_email,
        "dry_run": int(dry_run),
        "min_corpus_id": int(min_corpus_id or 0),
    }



def neutralize_test_sms_collision(
    db: sqlite3.Connection,
    *,
    source_key: str = "test:sms:1",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Remove a synthetic SMS row that occupied a messages-range integer id.

    Historically email upsert preferred ``corpus_items.id = messages.id``; a
    non-email row at that id forced skewed ids. Current ingest never prefers
    that mapping; this helper still cleans leftover collision rows.
    Deletes embeddings, Qdrant points, training samples, then the corpus row.
    Does not touch ``messages`` (ids may collide but are unrelated rows).
    """
    row = get_corpus_by_source_key(db, source_key)
    if row is None:
        return {
            "deleted": False,
            "source_key": source_key,
            "reason": "not_found",
        }
    item_id = int(row["id"])
    msg_collision = db.execute(
        "SELECT 1 FROM messages WHERE id = ?", (item_id,)
    ).fetchone()
    sample_n = int(
        db.execute(
            "SELECT COUNT(*) FROM training_samples WHERE corpus_item_id = ?",
            (item_id,),
        ).fetchone()[0]
    )
    if dry_run:
        return {
            "deleted": False,
            "dry_run": True,
            "source_key": source_key,
            "id": item_id,
            "kind": row.get("kind"),
            "collided_with_message": msg_collision is not None,
            "training_samples": sample_n,
        }

    from fish.prism.registry import list_retrieval_models
    from fish.qdrant_store import delete_point

    db.execute("DELETE FROM corpus_raw_embeddings WHERE item_id = ?", (item_id,))
    qdrant_errors: list[str] = []
    for model in list_retrieval_models(db):
        try:
            delete_point(model["vec_table"], item_id)
        except Exception as exc:  # pragma: no cover - best-effort ANN cleanup
            qdrant_errors.append(f"{model['vec_table']}: {exc}")
    db.execute("DELETE FROM training_samples WHERE corpus_item_id = ?", (item_id,))
    db.execute("DELETE FROM corpus_items WHERE id = ?", (item_id,))
    return {
        "deleted": True,
        "source_key": source_key,
        "id": item_id,
        "kind": row.get("kind"),
        "collided_with_message": msg_collision is not None,
        "qdrant_errors": qdrant_errors,
        "training_samples_removed": sample_n,
    }


def corpus_needing_field_embeddings(
    db: sqlite3.Connection,
    limit: int = 100,
    *,
    training_only: bool = False,
) -> list[dict[str, Any]]:
    """Items with combined raw embed but missing header and/or body field embeds."""
    train_join = ""
    if training_only:
        train_join = """
        JOIN (
            SELECT DISTINCT corpus_item_id AS id
            FROM training_samples
            WHERE superseded_at IS NULL AND target_relevance IS NOT NULL
        ) t ON t.id = c.id
        """
    rows = db.execute(
        f"""
        SELECT c.id, c.text_for_embed, c.body_text, c.header_json,
               r.header_embedding, r.body_embedding
        FROM corpus_items c
        JOIN corpus_raw_embeddings r ON r.item_id = c.id
        {train_join}
        WHERE r.header_embedding IS NULL OR r.body_embedding IS NULL
        ORDER BY c.id
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def count_corpus_needing_field_embeddings(
    db: sqlite3.Connection, *, training_only: bool = False
) -> int:
    if training_only:
        row = db.execute(
            """
            SELECT COUNT(DISTINCT s.corpus_item_id)
            FROM training_samples s
            JOIN corpus_raw_embeddings r ON r.item_id = s.corpus_item_id
            WHERE s.superseded_at IS NULL
              AND s.target_relevance IS NOT NULL
              AND (r.header_embedding IS NULL OR r.body_embedding IS NULL)
            """
        ).fetchone()
    else:
        row = db.execute(
            """
            SELECT COUNT(*) FROM corpus_raw_embeddings
            WHERE header_embedding IS NULL OR body_embedding IS NULL
            """
        ).fetchone()
    return int(row[0])


def get_model_embedding(
    db: sqlite3.Connection, item_id: int, model_id: str
) -> list[float] | None:
    from fish.prism.registry import get_retrieval_model
    from fish.qdrant_store import get_point_vector

    model = get_retrieval_model(db, model_id)
    if model is None:
        raise KeyError(f"Unknown model_id {model_id!r}")
    return get_point_vector(model["vec_table"], item_id)


def set_embedding(
    db: sqlite3.Connection, message_id: int, embedding: list[float] | np.ndarray
) -> None:
    set_corpus_embedding(db, message_id, embedding)


def _store_raw_embedding(
    db: sqlite3.Connection,
    item_id: int,
    embedding: list[float] | np.ndarray,
    *,
    header_embedding: list[float] | np.ndarray | None = None,
    body_embedding: list[float] | np.ndarray | None = None,
) -> None:
    """Persist durable OpenAI raw vectors in SQLite (never Qdrant-only).

    ``embedding`` is the combined text_for_embed vector (also copied to Qdrant
    fish_legacy). ``header_embedding`` / ``body_embedding`` are field vectors
    for PRISM composition; they stay in SQLite only.
    """
    db.execute(
        """
        INSERT INTO corpus_raw_embeddings (
            item_id, embedding, header_embedding, body_embedding, updated_at
        )
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(item_id) DO UPDATE SET
            embedding = excluded.embedding,
            header_embedding = COALESCE(excluded.header_embedding, corpus_raw_embeddings.header_embedding),
            body_embedding = COALESCE(excluded.body_embedding, corpus_raw_embeddings.body_embedding),
            updated_at = excluded.updated_at
        """,
        (
            item_id,
            embedding_to_blob(embedding),
            embedding_to_blob(header_embedding) if header_embedding is not None else None,
            embedding_to_blob(body_embedding) if body_embedding is not None else None,
            _utcnow(),
        ),
    )


def _payload_for_item(db: sqlite3.Connection, item_id: int) -> dict[str, Any]:
    from fish.qdrant_store import build_payload

    row = get_corpus_by_id(db, item_id)
    if row is None:
        raise KeyError(f"corpus item {item_id} not found")
    return build_payload(row)


def set_corpus_embedding(
    db: sqlite3.Connection,
    item_id: int,
    embedding: list[float] | np.ndarray,
    *,
    header_embedding: list[float] | np.ndarray | None = None,
    body_embedding: list[float] | np.ndarray | None = None,
    model_embeddings: dict[str, list[float] | np.ndarray] | None = None,
) -> None:
    """Persist raw vectors in SQLite; upsert combined vector to Qdrant ANN.

    Durable store (SQLite ``corpus_raw_embeddings``):
      - ``embedding`` — combined ``text_for_embed`` (also → fish_legacy)
      - ``header_embedding`` / ``body_embedding`` — field vectors for PRISM
        composition; **not** written to Qdrant

    ``model_embeddings`` maps model_id → Ac(c) for registered PRISM indexes.
    """
    from fish.prism.configs import LEGACY_MODEL_ID
    from fish.prism.registry import get_retrieval_model, list_retrieval_models
    from fish.qdrant_store import upsert_point

    _store_raw_embedding(
        db,
        item_id,
        embedding,
        header_embedding=header_embedding,
        body_embedding=body_embedding,
    )
    payload = _payload_for_item(db, item_id)
    legacy = get_retrieval_model(db, LEGACY_MODEL_ID)
    if legacy is None:
        raise RuntimeError("legacy retrieval model missing — run init_db")
    upsert_point(legacy["vec_table"], item_id, embedding, payload)

    extras = model_embeddings or {}
    for model in list_retrieval_models(db):
        mid = model["model_id"]
        if mid == LEGACY_MODEL_ID:
            continue
        vec = extras.get(mid)
        if vec is not None:
            upsert_point(model["vec_table"], item_id, vec, payload)
    now = _utcnow()
    db.execute(
        "UPDATE corpus_items SET embedded_at = ? WHERE id = ?",
        (now, item_id),
    )
    db.execute(
        "UPDATE messages SET embedded_at = ? WHERE id = ?",
        (now, item_id),
    )


def set_model_embedding(
    db: sqlite3.Connection,
    item_id: int,
    model_id: str,
    embedding: list[float] | np.ndarray,
) -> None:
    from fish.prism.configs import LEGACY_MODEL_ID
    from fish.prism.registry import get_retrieval_model
    from fish.qdrant_store import upsert_point

    if model_id == LEGACY_MODEL_ID:
        _store_raw_embedding(db, item_id, embedding)
        model = get_retrieval_model(db, LEGACY_MODEL_ID)
        if model is None:
            raise RuntimeError("legacy retrieval model missing — run init_db")
        upsert_point(
            model["vec_table"], item_id, embedding, _payload_for_item(db, item_id)
        )
        return
    model = get_retrieval_model(db, model_id)
    if model is None:
        raise KeyError(f"Unknown model_id {model_id!r}")
    upsert_point(
        model["vec_table"], item_id, embedding, _payload_for_item(db, item_id)
    )


def unindex_corpus_item(db: sqlite3.Connection, item_id: int) -> int:
    """Remove item from SQLite raw store and every Qdrant collection."""
    from fish.corpus import email_locator_from_payload, parse_imap_source_key
    from fish.prism.registry import list_retrieval_models
    from fish.qdrant_store import delete_point

    n = 0
    row = db.execute(
        "SELECT kind, source_key, payload FROM corpus_items WHERE id = ?", (item_id,)
    ).fetchone()
    db.execute("DELETE FROM corpus_raw_embeddings WHERE item_id = ?", (item_id,))
    for model in list_retrieval_models(db):
        delete_point(model["vec_table"], item_id)
        n += 1
    db.execute(
        "UPDATE corpus_items SET embedded_at = NULL WHERE id = ?", (item_id,)
    )
    # Clear matching message via payload locator or legacy imap source_key —
    # never assume messages.id = corpus.id.
    if row and row["kind"] == "email":
        locator = email_locator_from_payload(row["payload"])
        if locator is None:
            locator = parse_imap_source_key(row["source_key"] or "")
        if locator is not None:
            account_id, folder, uid = locator
            db.execute(
                """
                UPDATE messages SET embedded_at = NULL
                WHERE account_id = ? AND folder = ? AND uid = ?
                """,
                (account_id, folder, uid),
            )
    return n


def cleanup_index_orphans(db: sqlite3.Connection) -> dict[str, int]:
    """Delete Qdrant points / raw rows whose id is not a live corpus_items.id."""
    from fish.prism.registry import list_retrieval_models
    from fish.qdrant_store import delete_point, scroll_ids

    removed: dict[str, int] = {}
    # Raw SQLite orphans
    raw_orphans = db.execute(
        """
        SELECT r.item_id FROM corpus_raw_embeddings r
        LEFT JOIN corpus_items c ON c.id = r.item_id
        WHERE c.id IS NULL
        """
    ).fetchall()
    for row in raw_orphans:
        db.execute(
            "DELETE FROM corpus_raw_embeddings WHERE item_id = ?", (int(row[0]),)
        )
    removed["corpus_raw_embeddings"] = len(raw_orphans)

    for model in list_retrieval_models(db):
        collection = model["vec_table"]
        count = 0
        for rid in scroll_ids(collection, limit=10_000_000):
            exists = db.execute(
                "SELECT 1 FROM corpus_items WHERE id = ?", (rid,)
            ).fetchone()
            if not exists:
                delete_point(collection, rid)
                count += 1
        removed[model["model_id"]] = count
    return removed


def wipe_all_vector_indexes(db: sqlite3.Connection) -> dict[str, str]:
    """Discard all Qdrant ANN indexes (corrupt recovery). Keeps corpus text + raw embeds.

    Recreates empty collections for each registered model. Re-run
    ``fish qdrant-reindex`` / ``fish prism-reembed`` to rebuild ANN from
    ``corpus_raw_embeddings``.
    """
    from fish.prism.configs import vec_table_for_model_id
    from fish.prism.registry import list_retrieval_models
    from fish.qdrant_store import delete_collection, ensure_collection

    cleared: dict[str, str] = {}
    for model in list_retrieval_models(db):
        mid = model["model_id"]
        old = model["vec_table"]
        expected = vec_table_for_model_id(mid)
        delete_collection(old)
        if old != expected:
            delete_collection(expected)
            db.execute(
                "UPDATE retrieval_models SET vec_table = ? WHERE model_id = ?",
                (expected, mid),
            )
        ensure_collection(expected)
        cleared[mid] = f"wiped:{expected}"
    return cleared


def list_corpus_with_raw_embedding(
    db: sqlite3.Connection,
    *,
    kinds: list[str] | None = None,
    limit: int | None = None,
    like: list[str] | None = None,
    since: str | None = None,
) -> list[dict[str, Any]]:
    """Items with a stored raw embedding (eligible for PRISM / Qdrant re-index)."""
    sql = """
        SELECT c.id, c.kind, r.embedding
        FROM corpus_items c
        JOIN corpus_raw_embeddings r ON r.item_id = c.id
        WHERE 1=1
    """
    params: list[Any] = []
    if kinds:
        placeholders = ",".join("?" for _ in kinds)
        sql += f" AND c.kind IN ({placeholders})"
        params.extend(kinds)
    if since:
        sql += " AND c.occurred_at >= ?"
        params.append(since)
    if like:
        like_clauses = []
        for pattern in like:
            like_clauses.append(
                "(ifnull(c.text_for_embed,'') LIKE ? OR ifnull(c.body_text,'') LIKE ?)"
            )
            params.extend([pattern, pattern])
        sql += " AND (" + " OR ".join(like_clauses) + ")"
    sql += " ORDER BY c.occurred_at DESC"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    rows = db.execute(sql, params).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        raw = blob_to_embedding(row["embedding"])
        if raw is None:
            continue
        out.append(
            {"id": int(row["id"]), "kind": row["kind"], "raw_embedding": raw}
        )
    return out


def count_messages_needing_embedding(db: sqlite3.Connection) -> int:
    return count_corpus_needing_embedding(db)


def count_corpus_needing_embedding(db: sqlite3.Connection) -> int:
    """Items missing a row in corpus_raw_embeddings."""
    row = db.execute(
        """
        SELECT COUNT(*) FROM corpus_items
        WHERE id NOT IN (SELECT item_id FROM corpus_raw_embeddings)
        """
    ).fetchone()
    return int(row[0])


def messages_needing_embedding(db: sqlite3.Connection, limit: int = 100) -> list[dict[str, Any]]:
    return corpus_needing_embedding(db, limit=limit)


def corpus_needing_embedding(
    db: sqlite3.Connection,
    limit: int = 100,
    *,
    kinds: list[str] | None = None,
    like: list[str] | None = None,
    since: str | None = None,
) -> list[dict[str, Any]]:
    sql = """
        SELECT id, text_for_embed, body_text, header_json FROM corpus_items
        WHERE id NOT IN (SELECT item_id FROM corpus_raw_embeddings)
    """
    params: list[Any] = []
    if kinds:
        placeholders = ",".join("?" for _ in kinds)
        sql += f" AND kind IN ({placeholders})"
        params.extend(kinds)
    if since:
        sql += " AND occurred_at >= ?"
        params.append(since)
    if like:
        clauses = []
        for pattern in like:
            clauses.append(
                "(ifnull(text_for_embed,'') LIKE ? OR ifnull(body_text,'') LIKE ?)"
            )
            params.extend([pattern, pattern])
        sql += " AND (" + " OR ".join(clauses) + ")"
    sql += " ORDER BY occurred_at DESC LIMIT ?"
    params.append(limit)
    rows = db.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def vector_search(
    db: sqlite3.Connection,
    query_embedding: list[float] | np.ndarray,
    limit: int = 20,
) -> list[tuple[int, float]]:
    return corpus_vector_search(db, query_embedding, limit=limit)


def corpus_vector_search(
    db: sqlite3.Connection,
    query_embedding: list[float] | np.ndarray,
    limit: int = 20,
    kinds: list[str] | None = None,
    *,
    model_id: str = "legacy",
    since: str | None = None,
    until: str | None = None,
    from_contains: str | None = None,
    account_email: str | None = None,
    folder: str | None = None,
    unread_only: bool = False,
) -> list[tuple[int, float]]:
    """ANN search against the Qdrant collection for ``model_id`` (default legacy/raw)."""
    from fish.prism.registry import get_retrieval_model
    from fish.qdrant_store import search as qdrant_search

    k = int(limit)
    if k <= 0:
        raise ValueError(f"corpus_vector_search limit must be positive, got {limit!r}")
    model = get_retrieval_model(db, model_id)
    if model is None:
        raise KeyError(f"Unknown model_id {model_id!r}")
    return qdrant_search(
        model["vec_table"],
        query_embedding,
        limit=k,
        kinds=kinds,
        since=since,
        until=until,
        from_contains=from_contains,
        account_email=account_email,
        folder=folder,
        unread_only=unread_only,
    )


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
    sql = f"""
        SELECT c.id FROM corpus_items c
        LEFT JOIN messages m ON {EMAIL_CORPUS_MESSAGE_JOIN} AND c.kind = 'email'
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


def get_sync_state(
    db: sqlite3.Connection, account_id: int, folder: str
) -> dict[str, Any] | None:
    row = db.execute(
        """
        SELECT account_id, folder, uidvalidity, last_uid, last_sync_at, since_date
        FROM sync_state
        WHERE account_id = ? AND folder = ?
        """,
        (account_id, folder),
    ).fetchone()
    return dict(row) if row else None


def newest_sync_at(db: sqlite3.Connection) -> str | None:
    row = db.execute("SELECT MAX(last_sync_at) AS ts FROM sync_state").fetchone()
    return row["ts"] if row and row["ts"] else None


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
    """Delete a messages row and its corpus item (via canonical_id / locator)."""
    row = get_message_by_id(db, message_id)
    corpus_id: int | None = None
    if row:
        cid = row.get("canonical_id")
        if cid:
            c = get_corpus_by_source_key(db, cid)
            if c:
                corpus_id = int(c["id"])
        if corpus_id is None:
            from fish.corpus import imap_source_key

            legacy = get_corpus_by_source_key(
                db,
                imap_source_key(int(row["account_id"]), row["folder"], int(row["uid"])),
            )
            if legacy:
                corpus_id = int(legacy["id"])
    if corpus_id is not None:
        mark_samples_superseded_for_corpus(db, corpus_id)
        unindex_corpus_item(db, corpus_id)
        db.execute("DELETE FROM corpus_items WHERE id = ?", (corpus_id,))
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


QueryOrigin = Literal["gold", "curated", "synth"]

# Applied only while legacy labels (real/synthetic) still exist.
_QUERY_ORIGIN_RENAMES: tuple[tuple[str, str], ...] = (
    ("gold", "curated"),  # old JSONL seeds were mislabeled "gold"
    ("real", "gold"),  # logged searches
    ("synthetic", "synth"),
)


def migrate_training_query_origins(db: sqlite3.Connection) -> dict[str, int]:
    """Rename origin values to gold / curated / synth. Idempotent.

    Runs the legacy rename only when ``real``/``synthetic`` rows still exist.
    Otherwise repairs logged rows wrongly folded into ``curated`` (source not
    ``curated:*``).
    """
    legacy = db.execute(
        "SELECT 1 FROM training_queries WHERE origin IN ('real', 'synthetic') LIMIT 1"
    ).fetchone()
    if not legacy:
        repaired = db.execute(
            """
            UPDATE training_queries
            SET origin = 'gold',
                source = COALESCE(NULLIF(source, ''), 'logged')
            WHERE origin = 'curated'
              AND (
                source IS NULL
                OR source = ''
                OR source = 'logged'
                OR source NOT LIKE 'curated:%'
              )
            """
        )
        return {"repaired_logged_from_curated": int(repaired.rowcount)}

    updated: dict[str, int] = {}
    for old, new in _QUERY_ORIGIN_RENAMES:
        n_old = db.execute(
            "SELECT COUNT(*) FROM training_queries WHERE origin = ?", (old,)
        ).fetchone()[0]
        if not n_old:
            updated[f"{old}->{new}"] = 0
            continue
        cur = db.execute(
            """
            UPDATE training_queries
            SET origin = ?
            WHERE origin = ?
              AND text_hash NOT IN (
                SELECT text_hash FROM training_queries WHERE origin = ?
              )
            """,
            (new, old, new),
        )
        moved = int(cur.rowcount)
        leftover = [
            int(r[0])
            for r in db.execute(
                "SELECT id FROM training_queries WHERE origin = ?", (old,)
            ).fetchall()
        ]
        dropped = 0
        if leftover:
            ph = ",".join("?" for _ in leftover)
            db.execute(
                f"UPDATE training_queries SET parent_query_id = NULL "
                f"WHERE parent_query_id IN ({ph})",
                leftover,
            )
            for qid in leftover:
                row = db.execute(
                    "SELECT text_hash FROM training_queries WHERE id = ?", (qid,)
                ).fetchone()
                if not row:
                    continue
                survivor = db.execute(
                    "SELECT id FROM training_queries WHERE text_hash = ? AND origin = ?",
                    (row[0], new),
                ).fetchone()
                if survivor:
                    db.execute(
                        "UPDATE training_samples SET query_id = ? WHERE query_id = ?",
                        (int(survivor[0]), qid),
                    )
                db.execute("DELETE FROM training_samples WHERE query_id = ?", (qid,))
                db.execute("DELETE FROM training_queries WHERE id = ?", (qid,))
                dropped += 1
        updated[f"{old}->{new}"] = moved
        updated[f"{old}->dropped_dup"] = dropped
    return updated


def normalize_query_text(text: str) -> str:
    return " ".join(text.split()).strip().lower()


def query_text_hash(text: str) -> str:
    return hashlib.sha256(normalize_query_text(text).encode()).hexdigest()


def sample_pair_hash(query_id: int, corpus_item_id: int) -> str:
    """Stable identity for a training pair: (query, corpus item).

    Retriever is provenance/metadata only — cycling retrievers must not mint
    duplicate rows or invalidate existing RA labels.
    """
    payload = f"{query_id}\0{corpus_item_id}"
    return hashlib.sha256(payload.encode()).hexdigest()


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
    query_embedding: list[float] | np.ndarray | None = None,
    source: str | None = None,
    meta_json: str | None = None,
) -> int | None:
    """Insert a training query. Returns id, or None if duplicate."""
    now = _utcnow()
    thash = query_text_hash(text)
    blob = embedding_to_blob(query_embedding) if query_embedding is not None else None
    try:
        cur = db.execute(
            """
            INSERT INTO training_queries (
                text, origin, parent_query_id, context_json, synthesis_method,
                embed_model, query_embedding, created_at, text_hash, source, meta_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                source,
                meta_json,
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


def get_training_query_by_text(
    db: sqlite3.Connection,
    text: str,
    *,
    origin: QueryOrigin | None = "gold",
) -> dict[str, Any] | None:
    thash = query_text_hash(text)
    if origin is None:
        row = db.execute(
            "SELECT * FROM training_queries WHERE text_hash = ? ORDER BY id LIMIT 1",
            (thash,),
        ).fetchone()
    else:
        row = db.execute(
            """
            SELECT * FROM training_queries
            WHERE text_hash = ? AND origin = ?
            ORDER BY id LIMIT 1
            """,
            (thash, origin),
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
    source: str | None = None,
    limit: int | None = None,
    require_embedding: bool = False,
    include_embeddings: bool = True,
) -> list[dict[str, Any]]:
    sql = "SELECT * FROM training_queries WHERE 1=1"
    params: list[Any] = []
    if origin:
        sql += " AND origin = ?"
        params.append(origin)
    if source:
        sql += " AND source = ?"
        params.append(source)
    if require_embedding:
        sql += " AND query_embedding IS NOT NULL"
    sql += " ORDER BY created_at DESC, id DESC"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(int(limit))
    rows = db.execute(sql, params).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        data = training_query_row_to_dict(row)
        if not include_embeddings:
            data.pop("query_embedding", None)
        if data.get("meta_json"):
            try:
                data["meta"] = json.loads(data["meta_json"])
            except json.JSONDecodeError:
                data["meta"] = None
        out.append(data)
    return out


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
    db: sqlite3.Connection, query_id: int, embedding: list[float] | np.ndarray, embed_model: str
) -> None:
    db.execute(
        """
        UPDATE training_queries
        SET query_embedding = ?, embed_model = ?
        WHERE id = ?
        """,
        (embedding_to_blob(embedding), embed_model, query_id),
    )


def get_active_training_sample_for_pair(
    db: sqlite3.Connection, query_id: int, corpus_item_id: int
) -> dict[str, Any] | None:
    row = db.execute(
        """
        SELECT * FROM training_samples
        WHERE query_id = ? AND corpus_item_id = ? AND superseded_at IS NULL
        ORDER BY id DESC LIMIT 1
        """,
        (query_id, corpus_item_id),
    ).fetchone()
    return training_sample_row_to_dict(row) if row else None


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
    query_embedding: list[float] | np.ndarray,
    message_embedding: list[float] | np.ndarray,
) -> int | None:
    """Insert sample. Returns id, or None if (query, item) already exists.

    On duplicate: refresh retrieval provenance only — never clear RA labels.
    """
    existing = get_active_training_sample_for_pair(db, query_id, corpus_item_id)
    if existing is not None:
        db.execute(
            """
            UPDATE training_samples
            SET retriever = ?, retrieval_similarity = ?, retrieval_rank = ?,
                query_embedding = COALESCE(?, query_embedding),
                message_embedding = COALESCE(?, message_embedding)
            WHERE id = ?
            """,
            (
                retriever,
                retrieval_similarity,
                retrieval_rank,
                embedding_to_blob(query_embedding),
                embedding_to_blob(message_embedding),
                int(existing["id"]),
            ),
        )
        return None

    now = _utcnow()
    phash = sample_pair_hash(query_id, corpus_item_id)
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
        existing = get_active_training_sample_for_pair(db, query_id, corpus_item_id)
        if existing is not None:
            db.execute(
                """
                UPDATE training_samples
                SET retriever = ?, retrieval_similarity = ?, retrieval_rank = ?
                WHERE id = ?
                """,
                (retriever, retrieval_similarity, retrieval_rank, int(existing["id"])),
            )
        return None


def backfill_null_relevance_agent_versions(db: sqlite3.Connection) -> int:
    """Labeled rows with null agent version were pre-versioning → treat as 1.0.0."""
    cur = db.execute(
        """
        UPDATE training_samples
        SET relevance_agent_version = '1.0.0'
        WHERE superseded_at IS NULL
          AND target_relevance IS NOT NULL
          AND relevance_agent_version IS NULL
        """
    )
    return int(cur.rowcount)


def _training_sample_keep_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """Higher is better: labeled > unlabeled; 2.0.0 > 1.0.0; newer labeled_at."""
    labeled = row.get("target_relevance") is not None
    ver = row.get("relevance_agent_version")
    if labeled and not ver:
        ver = "1.0.0"
    is_v2 = ver == "2.0.0"
    labeled_at = row.get("labeled_at") or ""
    created_at = row.get("created_at") or ""
    return (labeled, is_v2, labeled_at, created_at, int(row["id"]))


def dedupe_training_sample_pairs(
    db: sqlite3.Connection, *, dry_run: bool = False
) -> dict[str, Any]:
    """One active row per (query_id, corpus_item_id); keep best labeled row.

    Preference: any label over none; among labels prefer relevance_agent_version
    2.0.0 over 1.0.0 (null version on labeled rows counts as 1.0.0); then most
    recent labeled_at. Losers get superseded_at. Winner pair_hash rewritten to
    the retriever-free identity.
    """
    if dry_run:
        backfilled = int(
            db.execute(
                """
                SELECT COUNT(*) FROM training_samples
                WHERE superseded_at IS NULL
                  AND target_relevance IS NOT NULL
                  AND relevance_agent_version IS NULL
                """
            ).fetchone()[0]
        )
    else:
        backfilled = backfill_null_relevance_agent_versions(db)

    rows = db.execute(
        """
        SELECT * FROM training_samples
        WHERE superseded_at IS NULL
        ORDER BY query_id, corpus_item_id, id
        """
    ).fetchall()
    groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        d = training_sample_row_to_dict(row)
        key = (int(d["query_id"]), int(d["corpus_item_id"]))
        groups.setdefault(key, []).append(d)

    now = _utcnow()
    groups_deduped = 0
    superseded = 0
    hashes_rewritten = 0
    kept_labeled = 0
    kept_unlabeled = 0

    for (_qid, _cid), members in groups.items():
        winner = max(members, key=_training_sample_keep_key)
        losers = [m for m in members if int(m["id"]) != int(winner["id"])]
        if losers:
            groups_deduped += 1
        if winner.get("target_relevance") is not None:
            kept_labeled += 1
        else:
            kept_unlabeled += 1

        new_hash = sample_pair_hash(int(winner["query_id"]), int(winner["corpus_item_id"]))
        if dry_run:
            superseded += len(losers)
            if winner.get("pair_hash") != new_hash:
                hashes_rewritten += 1
            continue

        for loser in losers:
            db.execute(
                "UPDATE training_samples SET superseded_at = ? WHERE id = ?",
                (now, int(loser["id"])),
            )
            superseded += 1
        if winner.get("pair_hash") != new_hash:
            db.execute(
                "UPDATE training_samples SET pair_hash = ? WHERE id = ?",
                (new_hash, int(winner["id"])),
            )
            hashes_rewritten += 1

    return {
        "dry_run": dry_run,
        "null_versions_backfilled_to_1_0_0": backfilled,
        "active_pairs": len(groups),
        "groups_with_duplicates": groups_deduped,
        "rows_superseded": superseded,
        "hashes_rewritten": hashes_rewritten,
        "kept_labeled": kept_labeled,
        "kept_unlabeled": kept_unlabeled,
    }


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
    """List samples for labeling.

    Default: only rows with no ``target_relevance`` (incremental labeling).
    ``force=True``: re-score every active sample (optional full refresh).
    ``agent_version`` is unused for selection; kept for call-site compat.
    """
    del agent_version  # selection is by label presence, not version mismatch
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
              AND target_relevance IS NULL
            ORDER BY id
            LIMIT ?
        """
        rows = db.execute(sql, (limit,)).fetchall()
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


def update_sample_relevance_with_retry(
    sample_id: int,
    *,
    target_relevance: float,
    agent_version: str,
    relevance_model: str,
    retries: int = 3,
    delay_sec: float = 0.5,
) -> None:
    """Short retry for single-row label UPDATEs under concurrent writers."""
    import time

    last: BaseException | None = None
    for attempt in range(retries):
        try:
            with db_conn() as db:
                update_sample_relevance(
                    db,
                    sample_id,
                    target_relevance=target_relevance,
                    agent_version=agent_version,
                    relevance_model=relevance_model,
                )
            return
        except RuntimeError as exc:
            # db_conn wraps OperationalError locked → RuntimeError
            if "database is locked" not in str(exc).lower():
                raise
            last = exc
            if attempt + 1 >= retries:
                break
            time.sleep(delay_sec * (attempt + 1))
        except sqlite3.OperationalError as exc:
            if not is_sqlite_locked(exc):
                raise
            last = exc
            if attempt + 1 >= retries:
                break
            time.sleep(delay_sec * (attempt + 1))
    assert last is not None
    if isinstance(last, RuntimeError):
        raise last
    raise format_sqlite_locked_error(last)


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
