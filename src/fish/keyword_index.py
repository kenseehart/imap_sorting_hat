"""Keyword (BM25) index over the corpus — SQLite FTS5 beside ``corpus_items``.

The keyword half of hybrid search (search.py). Email search leans on exact
tokens - names, addresses, subjects, order numbers - that embeddings blur, so
the two arms are fused by rank (reciprocal rank fusion) rather than either one
alone.

``corpus_fts`` is a plain FTS5 table keyed by ``rowid = corpus_items.id`` and
kept in step by triggers on ``corpus_items`` (insert / update / delete), so no
writer can forget it. Rows that existed before the index was created are added
by ``backfill`` (``fish fts-reindex``); search refuses to run until that has
finished once, rather than silently returning keyword results for part of the
corpus.

Columns (BM25 weights in ``WEIGHTS``): ``subject`` and ``people`` (from / to /
cc) for email, ``body`` = the full body text (else ``text_for_embed``).
"""

from __future__ import annotations

import re
import sqlite3
import sys
import time
from datetime import datetime, timezone

TABLE = "corpus_fts"
WEIGHTS = (3.0, 2.0, 1.0)  # subject, people, body
BUILT_KEY = "fts_backfilled_at"

_ROW_SELECT = """
    json_extract(c.payload, '$.subject'),
    trim(coalesce(json_extract(c.payload, '$.from_addr'), '') || ' ' ||
         coalesce(json_extract(c.payload, '$.to_addrs'), '') || ' ' ||
         coalesce(json_extract(c.payload, '$.cc_addrs'), '')),
    coalesce(c.body_text, c.text_for_embed)
"""

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS fish_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS {TABLE} USING fts5(
    subject, people, body,
    tokenize = 'porter unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS corpus_fts_ai AFTER INSERT ON corpus_items BEGIN
    INSERT INTO {TABLE}(rowid, subject, people, body)
    SELECT c.id, {_ROW_SELECT} FROM corpus_items c WHERE c.id = new.id;
END;

CREATE TRIGGER IF NOT EXISTS corpus_fts_ad AFTER DELETE ON corpus_items BEGIN
    DELETE FROM {TABLE} WHERE rowid = old.id;
END;

CREATE TRIGGER IF NOT EXISTS corpus_fts_au
AFTER UPDATE OF id, payload, body_text, text_for_embed ON corpus_items BEGIN
    DELETE FROM {TABLE} WHERE rowid = old.id;
    INSERT INTO {TABLE}(rowid, subject, people, body)
    SELECT c.id, {_ROW_SELECT} FROM corpus_items c WHERE c.id = new.id;
END;
"""

_TOKEN = re.compile(r"\w+", re.UNICODE)


def ensure_schema(db: sqlite3.Connection) -> None:
    db.executescript(SCHEMA)


def is_built(db: sqlite3.Connection) -> bool:
    row = db.execute("SELECT value FROM fish_meta WHERE key = ?", (BUILT_KEY,)).fetchone()
    return row is not None


def require_built(db: sqlite3.Connection) -> None:
    if not is_built(db):
        raise RuntimeError(
            "Keyword index not built yet: run `fish fts-reindex` once on the host that "
            "holds fish.db (gcp-e2-mcp). New items are indexed automatically after that."
        )


def backfill(db: sqlite3.Connection, *, batch: int = 5000, progress: bool = True) -> dict:
    """Index every corpus item the triggers have not seen. Resumable: each batch
    commits, and re-running skips rows already indexed. ``--rebuild`` callers
    empty the table first (``rebuild``)."""
    total = db.execute("SELECT count(*) FROM corpus_items").fetchone()[0]
    done = db.execute(f"SELECT count(*) FROM {TABLE}").fetchone()[0]
    start = time.monotonic()
    added = 0
    last_id = 0
    while True:
        ids = [r[0] for r in db.execute(
            f"""SELECT c.id FROM corpus_items c
                WHERE c.id > ? AND c.id NOT IN (SELECT rowid FROM {TABLE})
                ORDER BY c.id LIMIT ?""", (last_id, batch))]
        if not ids:
            break
        marks = ",".join("?" * len(ids))
        db.execute(
            f"INSERT INTO {TABLE}(rowid, subject, people, body) "
            f"SELECT c.id, {_ROW_SELECT} FROM corpus_items c WHERE c.id IN ({marks})", ids)
        db.commit()
        added += len(ids)
        last_id = ids[-1]
        if progress:
            n = done + added
            rate = added / max(time.monotonic() - start, 1e-6)
            eta = (total - n) / rate if rate else 0.0
            print(f"[fts {n}/{total}] {rate:.0f} rows/s, ~{eta:.0f}s left", file=sys.stderr)
    db.execute(
        "INSERT INTO fish_meta(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (BUILT_KEY, datetime.now(timezone.utc).isoformat()))
    db.commit()
    indexed = db.execute(f"SELECT count(*) FROM {TABLE}").fetchone()[0]
    if indexed != total:
        raise RuntimeError(f"keyword index holds {indexed} rows, corpus has {total}")
    return {"corpus_items": total, "indexed": indexed, "added": added,
            "seconds": round(time.monotonic() - start, 1)}


def rebuild(db: sqlite3.Connection, *, progress: bool = True) -> dict:
    db.execute(f"DELETE FROM {TABLE}")
    db.execute("DELETE FROM fish_meta WHERE key = ?", (BUILT_KEY,))
    db.commit()
    return backfill(db, progress=progress)


def match_expression(query: str) -> str | None:
    """FTS5 MATCH text for a free-text query: every word token, quoted (so
    FTS5 operators and punctuation in the query are plain text), OR-ed so BM25
    ranks documents by how many and how rare the matched words are. None when
    the query has no word tokens."""
    tokens = list(dict.fromkeys(t.lower() for t in _TOKEN.findall(query)))
    if not tokens:
        return None
    return " OR ".join(f'"{t}"' for t in tokens)


def search(db: sqlite3.Connection, query: str, *, limit: int) -> list[int]:
    """Corpus item ids, best BM25 first."""
    require_built(db)
    expr = match_expression(query)
    if expr is None:
        return []
    rows = db.execute(
        f"SELECT rowid FROM {TABLE} WHERE {TABLE} MATCH ? "
        f"ORDER BY bm25({TABLE}, ?, ?, ?) LIMIT ?",
        (expr, *WEIGHTS, int(limit)))
    return [int(r[0]) for r in rows]
