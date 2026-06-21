from __future__ import annotations

import json
from typing import Any

from fish.context import augment_query, compute_context_boosts, parse_context
from fish.corpus import corpus_row_to_dict
from fish.embed import embed_text
from fish.prism.inference import adapt_query_embedding
from fish.store import (
    corpus_keyword_search,
    corpus_vector_search,
    db_conn,
    get_corpus_by_id,
    init_db,
    memory_is_active,
)


def _merge_scores(
    vector_hits: list[tuple[int, float]],
    keyword_ids: list[int],
    vector_weight: float = 0.7,
    context_boosts: dict[int, float] | None = None,
) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    boosts = context_boosts or {}
    if vector_hits:
        max_dist = max((d for _, d in vector_hits), default=1.0) or 1.0
        for item_id, dist in vector_hits:
            sim = 1.0 - (dist / max_dist)
            scores[item_id] = scores.get(item_id, 0.0) + sim * vector_weight
    for item_id in keyword_ids:
        scores[item_id] = scores.get(item_id, 0.0) + (1.0 - vector_weight)
    for item_id, boost in boosts.items():
        scores[item_id] = scores.get(item_id, 0.0) + boost
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _item_to_result(row: dict[str, Any], score: float) -> dict[str, Any]:
    item = corpus_row_to_dict(row)
    item["score"] = round(score, 4)
    if item.get("kind") == "email":
        payload = item.get("payload") or {}
        item["subject"] = payload.get("subject")
        item["from_addr"] = payload.get("from_addr")
        item["account_email"] = payload.get("account_email")
        item["folder"] = payload.get("folder")
        item["date"] = item.get("occurred_at")
        item["flags"] = payload.get("flags", [])
    return item


def search_corpus(
    query: str,
    kinds: list[str] | None = None,
    context: dict[str, Any] | str | None = None,
    account_email: str | None = None,
    folder: str | None = None,
    unread_only: bool = False,
    limit: int = 20,
) -> dict[str, Any]:
    """Hybrid semantic + keyword search over the unified corpus."""
    init_db()
    ctx = parse_context(context)
    augmented = augment_query(query, ctx)
    raw_query_embedding = embed_text(augmented)
    from fish.prism.queries import log_real_query

    ctx_json = json.dumps(ctx) if ctx else None
    log_real_query(query, ctx_json, query_embedding=raw_query_embedding)
    with db_conn() as db:
        query_embedding = adapt_query_embedding(raw_query_embedding)
        vector_hits = corpus_vector_search(
            db, query_embedding, limit=limit, kinds=kinds
        )
        keyword_ids = corpus_keyword_search(
            db,
            query,
            account_email=account_email,
            folder=folder,
            limit=limit * 3,
            kinds=kinds,
        )
        boosts: dict[int, float] = {}
        candidate_ids = {i for i, _ in vector_hits} | set(keyword_ids)
        for item_id in candidate_ids:
            row = get_corpus_by_id(db, item_id)
            if not row or not memory_is_active(row):
                continue
            tags = row.get("tags")
            if isinstance(tags, str):
                try:
                    tags = json.loads(tags)
                except json.JSONDecodeError:
                    tags = []
            boost = compute_context_boosts(ctx, kind=row["kind"], tags=tags or [])
            if boost:
                boosts[item_id] = boost
        merged = _merge_scores(vector_hits, keyword_ids, context_boosts=boosts)[:limit]

        results = []
        for item_id, score in merged:
            row = get_corpus_by_id(db, item_id)
            if not row:
                continue
            if not memory_is_active(row):
                continue
            if account_email and row.get("kind") == "email":
                payload = row.get("payload") or {}
                if isinstance(payload, str):
                    payload = json.loads(payload)
                if payload.get("account_email") != account_email:
                    continue
            if folder and row.get("kind") == "email":
                payload = row.get("payload") or {}
                if isinstance(payload, str):
                    payload = json.loads(payload)
                if payload.get("folder") != folder:
                    continue
            if unread_only and row.get("kind") == "email":
                payload = row.get("payload") or {}
                if isinstance(payload, str):
                    payload = json.loads(payload)
                flags = payload.get("flags") or []
                if "\\Seen" in flags:
                    continue
            results.append(_item_to_result(row, score))

        prompt = None
        if ctx:
            from fish.context import format_prompt

            prompt = format_prompt(query, results, ctx)

        return {
            "query": query,
            "context": ctx,
            "results": results,
            "prompt": prompt,
        }


def search_messages(
    query: str,
    account_email: str | None = None,
    folder: str | None = None,
    unread_only: bool = False,
    limit: int = 20,
    kinds: list[str] | None = None,
    context: dict[str, Any] | str | None = None,
) -> list[dict[str, Any]]:
    """Backward-compatible wrapper returning result list only."""
    kinds = kinds or ["email"]
    payload = search_corpus(
        query,
        kinds=kinds,
        context=context,
        account_email=account_email,
        folder=folder,
        unread_only=unread_only,
        limit=limit,
    )
    return payload["results"]
