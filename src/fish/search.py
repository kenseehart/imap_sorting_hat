"""Corpus search — PRISM / legacy ANN with hard metadata filters (no hybrid ranking)."""

from __future__ import annotations

import json
from typing import Any

from fish.context import augment_query, parse_context
from fish.corpus import corpus_row_to_dict
from fish.embed import embed_text
from fish.prism.inference import adapt_query_for_model
from fish.store import (
    corpus_vector_search,
    db_conn,
    get_corpus_by_id,
    init_db,
    memory_is_active,
)


def _payload(row: dict[str, Any]) -> dict[str, Any]:
    payload = row.get("payload") or {}
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return {}
    return payload if isinstance(payload, dict) else {}


def _item_to_result(row: dict[str, Any], score: float) -> dict[str, Any]:
    item = corpus_row_to_dict(row)
    item["score"] = round(score, 4)
    if item.get("kind") == "email":
        payload = _payload(item)
        item["subject"] = payload.get("subject")
        item["from_addr"] = payload.get("from_addr")
        item["account_email"] = payload.get("account_email")
        item["folder"] = payload.get("folder")
        item["date"] = item.get("occurred_at")
        item["flags"] = payload.get("flags", [])
    return item


def _passes_metadata_filters(
    row: dict[str, Any],
    *,
    kinds: list[str] | None,
    since: str | None,
    until: str | None,
    from_contains: str | None,
    account_email: str | None,
    folder: str | None,
    unread_only: bool,
) -> bool:
    if not memory_is_active(row):
        return False
    if kinds and row.get("kind") not in kinds:
        return False
    occurred = (row.get("occurred_at") or "")[:32]
    if since and (not occurred or occurred < since):
        return False
    if until and occurred and occurred > until:
        return False

    payload = _payload(row)
    kind = row.get("kind")
    if kind != "email":
        # Email-only filters reject non-email rows
        if from_contains or account_email or folder or unread_only:
            return False
        return True

    if account_email and payload.get("account_email") != account_email:
        return False
    if folder and payload.get("folder") != folder:
        return False
    if from_contains:
        needle = from_contains.lower()
        frm = (payload.get("from_addr") or "").lower()
        if needle not in frm:
            return False
    if unread_only:
        flags = payload.get("flags") or []
        if "\\Seen" in flags:
            return False
    return True


def search_corpus(
    query: str,
    kinds: list[str] | None = None,
    context: dict[str, Any] | str | None = None,
    account_email: str | None = None,
    folder: str | None = None,
    unread_only: bool = False,
    limit: int = 20,
    *,
    model_id: str | None = None,
    since: str | None = None,
    until: str | None = None,
    from_contains: str | None = None,
    # Deprecated: ignored. Hybrid keyword ranking removed.
    keyword: bool | None = None,
    vector_weight: float | None = None,
) -> dict[str, Any]:
    """ANN search (PRISM adapted cosine when a model is active).

    Ranking is pure vector distance — no keyword hybrid and no score weighting.
    Metadata filters (since/until/from/account/folder/kinds/unread) are hard
    constraints applied after ANN (with over-fetch to fill ``limit``).
    """
    if keyword is not None or vector_weight is not None:
        # Accept old kwargs for callers; do not blend.
        pass

    init_db()
    ctx = parse_context(context)
    augmented = augment_query(query, ctx)
    raw_query_embedding = embed_text(augmented)
    from fish.prism.queries import log_real_query

    ctx_json = json.dumps(ctx) if ctx else None
    log_real_query(query, ctx_json, query_embedding=raw_query_embedding)

    filters_active = bool(
        kinds
        or since
        or until
        or from_contains
        or account_email
        or folder
        or unread_only
    )
    # Over-fetch when filtering so we can still return ``limit`` matches
    fetch_k = limit * 10 if filters_active else limit
    fetch_k = max(fetch_k, limit)

    with db_conn() as db:
        from fish.config import active_prism_model_id
        from fish.prism.configs import LEGACY_MODEL_ID

        mid = model_id or active_prism_model_id()
        if mid and mid != LEGACY_MODEL_ID:
            query_embedding = adapt_query_for_model(raw_query_embedding, mid)
            search_model_id = mid
        else:
            query_embedding = raw_query_embedding
            search_model_id = LEGACY_MODEL_ID

        # kinds filtered in Python with other metadata (avoid double filter path)
        vector_hits = corpus_vector_search(
            db,
            query_embedding,
            limit=fetch_k,
            kinds=None,
            model_id=search_model_id,
        )

        results: list[dict[str, Any]] = []
        for item_id, dist in vector_hits:
            row = get_corpus_by_id(db, item_id)
            if not row:
                continue
            if not _passes_metadata_filters(
                row,
                kinds=kinds,
                since=since,
                until=until,
                from_contains=from_contains,
                account_email=account_email,
                folder=folder,
                unread_only=unread_only,
            ):
                continue
            # Lower distance = closer; expose as similarity-ish score for display
            score = 1.0 / (1.0 + float(dist))
            results.append(_item_to_result(row, score))
            if len(results) >= limit:
                break

        prompt = None
        if ctx:
            from fish.context import format_prompt

            prompt = format_prompt(query, results, ctx)

        return {
            "query": query,
            "context": ctx,
            "model_id": search_model_id,
            "filters": {
                "kinds": kinds,
                "since": since,
                "until": until,
                "from": from_contains,
                "account_email": account_email,
                "folder": folder,
                "unread_only": unread_only,
            },
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
    *,
    model_id: str | None = None,
    since: str | None = None,
    until: str | None = None,
    from_contains: str | None = None,
    keyword: bool | None = None,
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
        model_id=model_id,
        since=since,
        until=until,
        from_contains=from_contains,
    )
    return payload["results"]
