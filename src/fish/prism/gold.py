"""Curated training queries — load JSONL (hand-authored seeds), dump, replace."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fish.config import CONFIG_DIR, embedding_model
from fish.embed import embed_texts
from fish.store import (
    db_conn,
    init_db,
    insert_training_query,
    list_training_queries,
    update_training_query_embedding,
)

# Package seed + optional user override
_PACKAGE_CURATED = Path(__file__).resolve().parents[3] / "config" / "gold_queries.jsonl"
USER_CURATED = CONFIG_DIR / "gold_queries.jsonl"
# Back-compat aliases
_PACKAGE_GOLD = _PACKAGE_CURATED
USER_GOLD = USER_CURATED

DEFAULT_SOURCE = "curated:email-kb"
CURATED_ORIGIN = "curated"


def load_gold_file(path: Path | None = None) -> list[dict[str, Any]]:
    """Load JSONL curated queries. Each line: {text, source?, meta?}."""
    target = path or (_PACKAGE_CURATED if _PACKAGE_CURATED.is_file() else USER_CURATED)
    if not target.is_file():
        raise FileNotFoundError(f"Curated query file not found: {target}")
    rows: list[dict[str, Any]] = []
    for i, line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{target}:{i}: invalid JSON: {exc}") from exc
        if not isinstance(obj, dict) or not str(obj.get("text", "")).strip():
            raise ValueError(f"{target}:{i}: each line needs a non-empty 'text' field")
        rows.append(obj)
    return rows


def delete_gold_queries(
    db: Any,
    *,
    source: str | None = None,
) -> dict[str, int]:
    """Permanently delete curated queries (and their training_samples)."""
    if source:
        qids = [
            int(r[0])
            for r in db.execute(
                "SELECT id FROM training_queries WHERE origin = ? AND source = ?",
                (CURATED_ORIGIN, source),
            ).fetchall()
        ]
    else:
        qids = [
            int(r[0])
            for r in db.execute(
                "SELECT id FROM training_queries WHERE origin = ?",
                (CURATED_ORIGIN,),
            ).fetchall()
        ]
    samples_deleted = 0
    if qids:
        placeholders = ",".join("?" for _ in qids)
        db.execute(
            f"UPDATE training_queries SET parent_query_id = NULL "
            f"WHERE parent_query_id IN ({placeholders})",
            qids,
        )
        cur = db.execute(
            f"DELETE FROM training_samples WHERE query_id IN ({placeholders})",
            qids,
        )
        samples_deleted = int(cur.rowcount)
        db.execute(
            f"DELETE FROM training_queries WHERE id IN ({placeholders})",
            qids,
        )
    return {"queries_deleted": len(qids), "samples_deleted": samples_deleted}


def add_gold_queries(
    entries: list[dict[str, Any]],
    *,
    embed: bool = True,
    default_source: str = DEFAULT_SOURCE,
) -> dict[str, Any]:
    """Insert origin=curated queries. Skips duplicates. Optionally embeds."""
    init_db()
    inserted: list[int] = []
    skipped = 0
    pending_embed: list[tuple[int, str]] = []

    with db_conn() as db:
        for entry in entries:
            text = str(entry["text"]).strip()
            source = str(entry.get("source") or default_source).strip() or default_source
            meta = entry.get("meta")
            meta_json = (
                json.dumps(meta, ensure_ascii=False)
                if isinstance(meta, dict)
                else (str(meta) if meta is not None else None)
            )
            qid = insert_training_query(
                db,
                text=text,
                origin=CURATED_ORIGIN,
                source=source,
                meta_json=meta_json,
            )
            if qid is None:
                skipped += 1
                continue
            inserted.append(qid)
            pending_embed.append((qid, text))

    embedded = 0
    if embed and pending_embed:
        batch = 64
        for start in range(0, len(pending_embed), batch):
            chunk = pending_embed[start : start + batch]
            vectors = embed_texts([t for _, t in chunk])
            with db_conn() as db:
                for (qid, _), vec in zip(chunk, vectors):
                    update_training_query_embedding(db, qid, vec, embedding_model())
                    embedded += 1

    return {
        "inserted": len(inserted),
        "skipped_duplicates": skipped,
        "embedded": embedded,
        "ids": inserted,
        "origin": CURATED_ORIGIN,
    }


def replace_gold_queries(
    entries: list[dict[str, Any]],
    *,
    embed: bool = True,
    default_source: str = DEFAULT_SOURCE,
) -> dict[str, Any]:
    """Delete all curated queries, then insert ``entries``."""
    init_db()
    with db_conn() as db:
        deleted = delete_gold_queries(db)
    added = add_gold_queries(
        entries, embed=embed, default_source=default_source
    )
    return {**deleted, **added, "file_count": len(entries)}


def dump_queries(
    *,
    origin: str | None = None,
    source: str | None = None,
    limit: int | None = None,
    include_embeddings: bool = False,
) -> list[dict[str, Any]]:
    init_db()
    with db_conn() as db:
        return list_training_queries(
            db,
            origin=origin,  # type: ignore[arg-type]
            source=source,
            limit=limit,
            include_embeddings=include_embeddings,
        )
