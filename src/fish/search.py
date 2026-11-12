"""Corpus search — hybrid: vector ANN + BM25 keywords, fused by rank, then reranked.

Ken, 2026-11-12: PRISM is shelved; search is plain RAG done well.

1. Vector: raw OpenAI embedding against Qdrant ``fish_legacy`` (a PRISM
   ``model_id`` is used only when a caller passes one explicitly).
2. Keyword: SQLite FTS5 / BM25 over subject, people and body
   (``fish.keyword_index``), filtered through the same Qdrant metadata filter
   as the vector arm so both obey one definition of since/until/from/...
3. Fusion: reciprocal rank fusion (``RRF_K``), no tuning weight.
4. Rerank: the RelevanceAgent prompt scores the top ``FISH_RERANK_K`` fused
   candidates (``fish.rerank``); ``FISH_RERANK=off`` skips it.

Each result carries ``score`` (relevance when reranked, else the fused RRF
score), ``fused_score``, ``vector_rank`` / ``keyword_rank`` (1-based, None
when that arm did not return it) and ``relevance`` (None when not reranked).
Real queries are still logged for a later PRISM revisit.
"""

from __future__ import annotations

import json
from typing import Any

from fish.context import augment_query, parse_context
from fish.corpus import corpus_row_to_dict
from fish.embed import embed_text
from fish.store import (
    corpus_vector_search,
    db_conn,
    get_corpus_by_id,
    init_db,
    memory_is_active,
)

RRF_K = 60
# Candidates each arm contributes before fusion (at least the page size).
ARM_DEPTH = 50
# Keyword hits fetched before the metadata filter trims them.
KEYWORD_PREFILTER = 200


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


def rrf_fuse(ranked_lists: list[list[int]], *, k: int = RRF_K) -> list[tuple[int, float]]:
    """Reciprocal rank fusion: sum of 1 / (k + rank) over the lists an id is
    in (rank 1-based). Returns (id, score), best first; ties keep first-seen
    order."""
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, item_id in enumerate(ranked, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])


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
    keyword: bool | None = None,
    rerank: bool | None = None,
    vector_weight: float | None = None,
    _scorer=None,
) -> dict[str, Any]:
    """Hybrid search (see module docstring). ``keyword=False`` drops the BM25
    arm; ``rerank`` overrides ``FISH_RERANK``. Metadata filters apply to both
    arms. Superseded memories are dropped after fetch."""
    from fish import keyword_index, rerank as rerank_mod
    from fish.prism.configs import LEGACY_MODEL_ID
    from fish.prism.queries import log_real_query
    from fish.qdrant_store import filter_ids

    if vector_weight is not None:
        raise TypeError("vector_weight is gone: hybrid search fuses by rank (RRF), not by weight")
    use_keyword = True if keyword is None else bool(keyword)
    use_rerank = rerank_mod.rerank_enabled() if rerank is None else bool(rerank)
    k_rerank = rerank_mod.rerank_k() if use_rerank else 0

    init_db()
    ctx = parse_context(context)
    augmented = augment_query(query, ctx)
    raw_query_embedding = embed_text(augmented)
    ctx_json = json.dumps(ctx) if ctx else None
    log_real_query(query, ctx_json, query_embedding=raw_query_embedding)
    filters = dict(kinds=kinds, since=since, until=until, from_contains=from_contains,
                   account_email=account_email, folder=folder, unread_only=unread_only)

    with db_conn() as db:
        if model_id and model_id != LEGACY_MODEL_ID:
            from fish.prism.inference import adapt_query_for_model

            query_embedding = adapt_query_for_model(raw_query_embedding, model_id)
            search_model_id = model_id
        else:
            query_embedding = raw_query_embedding
            search_model_id = LEGACY_MODEL_ID
        from fish.prism.registry import get_retrieval_model

        model = get_retrieval_model(db, search_model_id)
        if model is None:
            raise KeyError(f"Unknown model_id {search_model_id!r}")
        collection = model["vec_table"]

        depth = max(limit, k_rerank, ARM_DEPTH)
        vector_ids = [i for i, _ in corpus_vector_search(
            db, query_embedding, limit=depth, model_id=search_model_id, **filters)]
        keyword_ids: list[int] = []
        if use_keyword:
            # The keyword arm reads the query as typed: augment_query's
            # "[context: projects: ...]" wrapper would add literal words
            # ("context", "projects", "intent") as BM25 terms.
            hits = keyword_index.search(db, query, limit=max(depth, KEYWORD_PREFILTER))
            passing = filter_ids(collection, hits, **filters)
            keyword_ids = [i for i in hits if i in passing][:depth]

        vector_rank = {i: r for r, i in enumerate(vector_ids, start=1)}
        keyword_rank = {i: r for r, i in enumerate(keyword_ids, start=1)}
        pool = max(limit, k_rerank)
        candidates: list[tuple[dict[str, Any], float]] = []
        for item_id, fused in rrf_fuse([vector_ids, keyword_ids]):
            row = get_corpus_by_id(db, item_id)
            if not row:
                continue  # orphan index point; `fish index-cleanup` removes them
            if not memory_is_active(row):
                continue
            candidates.append((row, fused))
            if len(candidates) >= pool:
                break

    relevance: dict[int, float] = {}
    if use_rerank and candidates:
        head = candidates[:k_rerank]
        scores = rerank_mod.score_rows(query, [r for r, _ in head],
                                       context_json=ctx_json, scorer=_scorer)
        relevance = {int(r["id"]): s for (r, _), s in zip(head, scores)}
        head = sorted(head, key=lambda rf: -relevance[int(rf[0]["id"])])
        candidates = head + candidates[k_rerank:]

    results: list[dict[str, Any]] = []
    for row, fused in candidates[:limit]:
        rid = int(row["id"])
        rel = relevance.get(rid)
        item = _item_to_result(row, rel if rel is not None else fused)
        item["fused_score"] = round(fused, 6)
        item["vector_rank"] = vector_rank.get(rid)
        item["keyword_rank"] = keyword_rank.get(rid)
        item["relevance"] = None if rel is None else round(rel, 4)
        results.append(item)

    prompt = None
    if ctx:
        from fish.context import format_prompt

        prompt = format_prompt(query, results, ctx)

    return {
        "query": query,
        "context": ctx,
        "model_id": search_model_id,
        "ranking": {"keyword": use_keyword, "rerank": use_rerank, "rerank_k": k_rerank,
                    "rrf_k": RRF_K},
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
    rerank: bool | None = None,
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
        keyword=keyword,
        rerank=rerank,
    )
    return payload["results"]
