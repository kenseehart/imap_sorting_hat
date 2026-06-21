from __future__ import annotations

import json
from typing import Any

from fish.config import embedding_model
from fish.context import augment_query, parse_context
from fish.embed import embed_text
from fish.store import (
    db_conn,
    init_db,
    insert_training_query,
    update_training_query_embedding,
)


def query_text_for_search(query: str, context_json: str | dict[str, Any] | None) -> str:
    ctx = parse_context(context_json)
    return augment_query(query, ctx)


def log_real_query(
    query: str,
    context_json: str | dict[str, Any] | None = None,
    *,
    query_embedding: list[float] | None = None,
) -> int | None:
    """Record a real search query. Returns query id or None if duplicate."""
    init_db()
    ctx_str: str | None
    if context_json is None:
        ctx_str = None
    elif isinstance(context_json, str):
        ctx_str = context_json.strip() or None
    else:
        ctx_str = json.dumps(context_json)

    with db_conn() as db:
        query_id = insert_training_query(
            db,
            text=query,
            origin="real",
            context_json=ctx_str,
        )
        if query_id is None:
            return None
        if query_embedding is None:
            augmented = query_text_for_search(query, ctx_str)
            query_embedding = embed_text(augmented)
        update_training_query_embedding(
            db, query_id, query_embedding, embedding_model()
        )
        return query_id


def ensure_query_embedding(db: Any, query_row: dict[str, Any]) -> list[float]:
    existing = query_row.get("query_embedding")
    if isinstance(existing, list) and existing:
        return existing
    augmented = query_text_for_search(
        query_row["text"], query_row.get("context_json")
    )
    vec = embed_text(augmented)
    update_training_query_embedding(
        db, int(query_row["id"]), vec, embedding_model()
    )
    return vec
