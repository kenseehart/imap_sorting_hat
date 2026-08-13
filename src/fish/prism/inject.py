"""Inject hard-positive (query, corpus) pairs for PRISM cold-start training."""

from __future__ import annotations

from typing import Any

from fish.embed import embed_text
from fish.prism.inference import cosine_similarity
from fish.prism.queries import ensure_query_embedding, log_real_query
from fish.store import (
    db_conn,
    get_corpus_by_id,
    get_training_query_by_text,
    init_db,
    insert_training_sample,
)


def find_corpus_ids_by_likes(
    db: Any,
    *,
    likes: list[str],
    since: str | None = None,
    kinds: list[str] | None = None,
    limit: int = 200,
) -> list[int]:
    """Match corpus rows whose text_for_embed or payload subject matches any LIKE pattern."""
    if not likes:
        raise ValueError("At least one --like pattern is required")
    clauses: list[str] = []
    params: list[Any] = []
    for pattern in likes:
        clauses.append(
            "(lower(c.text_for_embed) LIKE lower(?) OR "
            "lower(COALESCE(json_extract(c.payload, '$.subject'), '')) LIKE lower(?))"
        )
        params.extend([pattern, pattern])
    sql = f"""
        SELECT c.id FROM corpus_items c
        WHERE ({' OR '.join(clauses)})
    """
    if since:
        sql += " AND c.occurred_at >= ?"
        params.append(since)
    if kinds:
        placeholders = ",".join("?" for _ in kinds)
        sql += f" AND c.kind IN ({placeholders})"
        params.extend(kinds)
    sql += " ORDER BY c.occurred_at DESC LIMIT ?"
    params.append(limit)
    rows = db.execute(sql, params).fetchall()
    return [int(r["id"]) for r in rows]


def inject_positives(
    *,
    query: str,
    likes: list[str],
    since: str | None = None,
    kinds: list[str] | None = None,
    limit: int = 200,
    retriever: str = "legacy",
) -> dict[str, Any]:
    """Ensure query exists, find matching corpus items, insert training_samples with raw embeds."""
    init_db()
    kinds = kinds or ["email"]
    log_real_query(query)
    samples_created = 0
    sample_ids: list[int] = []
    matched_ids: list[int] = []
    with db_conn() as db:
        qrow = get_training_query_by_text(db, query, origin="real")
        if not qrow:
            qrow = get_training_query_by_text(db, query, origin=None)
        if not qrow:
            raise RuntimeError(f"Failed to create/find training query for {query!r}")
        qvec = ensure_query_embedding(db, qrow)
        matched_ids = find_corpus_ids_by_likes(
            db, likes=likes, since=since, kinds=kinds, limit=limit
        )
        for rank, item_id in enumerate(matched_ids, start=1):
            corpus = get_corpus_by_id(db, item_id)
            if not corpus:
                continue
            text = corpus.get("text_for_embed") or ""
            if not text:
                continue
            raw_c = embed_text(text)
            sim = cosine_similarity(qvec, raw_c)
            sid = insert_training_sample(
                db,
                query_id=int(qrow["id"]),
                corpus_item_id=item_id,
                source_key=corpus["source_key"],
                kind=corpus["kind"],
                occurred_at=corpus.get("occurred_at"),
                content_hash=corpus.get("content_hash"),
                retriever=retriever,
                retrieval_similarity=sim,
                retrieval_rank=rank,
                query_embedding=qvec,
                message_embedding=raw_c,
            )
            if sid is not None:
                samples_created += 1
                sample_ids.append(sid)
    return {
        "query": query,
        "retriever": retriever,
        "matched_corpus_ids": len(matched_ids),
        "samples_created": samples_created,
        "sample_ids": sample_ids[:50],
        "likes": likes,
        "since": since,
    }
