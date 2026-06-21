from __future__ import annotations

from typing import Any, Literal

from fish.embed import embed_text
from fish.prism.inference import (
    cosine_similarity,
    load_prism_model,
)
from fish.prism.queries import ensure_query_embedding
from fish.prism.query_synthesis import ensure_query_count
from fish.prism.relevance import label_batch
from fish.store import (
    corpus_vector_search,
    db_conn,
    get_corpus_by_id,
    init_db,
    insert_training_sample,
    list_training_queries,
    memory_is_active,
)

Retriever = Literal["legacy"] | str


def _raw_chunk_embedding_cache() -> dict[int, list[float]]:
    return {}


def get_raw_chunk_embedding(
    item_id: int,
    text_for_embed: str,
    cache: dict[int, list[float]],
) -> list[float]:
    if item_id not in cache:
        cache[item_id] = embed_text(text_for_embed)
    return cache[item_id]


def retrieve_top_k(
    db: Any,
    query_embedding: list[float],
    *,
    retriever: str,
    top_k: int,
    raw_cache: dict[int, list[float]],
    prism_model: Any | None = None,
) -> list[tuple[int, float]]:
    """Return (corpus_item_id, retrieval_similarity) sorted best-first."""
    candidate_k = max(top_k * 5, top_k)
    if retriever == "legacy":
        search_vec = query_embedding
    else:
        if prism_model is None:
            prism_model = load_prism_model(retriever)
        search_vec = prism_model.adapt_query(query_embedding).tolist()

    hits = corpus_vector_search(db, search_vec, limit=candidate_k)
    scored: list[tuple[int, float]] = []
    for item_id, _dist in hits:
        row = get_corpus_by_id(db, item_id)
        if not row or not memory_is_active(row):
            continue
        text = row.get("text_for_embed") or ""
        if not text:
            continue
        raw_c = get_raw_chunk_embedding(item_id, text, raw_cache)
        if retriever == "legacy":
            sim = cosine_similarity(query_embedding, raw_c)
        else:
            if prism_model is None:
                prism_model = load_prism_model(retriever)
            aq = prism_model.adapt_query(query_embedding).tolist()
            ac = prism_model.adapt_chunk(raw_c).tolist()
            sim = cosine_similarity(aq, ac)
        scored.append((item_id, sim))

    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_k]


def collect_samples(
    *,
    retriever: str,
    min_queries: int = 50,
    synthesis_batch: int = 5,
    top_k: int = 20,
    label: bool = False,
    label_limit: int = 500,
) -> dict[str, Any]:
    if not retriever:
        raise ValueError("--retriever is required (legacy or model stem e.g. personal)")

    init_db()
    ensure_query_count(min_queries=min_queries, synthesis_batch=synthesis_batch)

    prism_model = None
    if retriever != "legacy":
        prism_model = load_prism_model(retriever)

    samples_created = 0
    queries_processed = 0
    raw_cache: dict[int, list[float]] = {}

    with db_conn() as db:
        queries = list_training_queries(db, require_embedding=False)
        for query_row in queries:
            qvec = ensure_query_embedding(db, query_row)
            hits = retrieve_top_k(
                db,
                qvec,
                retriever=retriever,
                top_k=top_k,
                raw_cache=raw_cache,
                prism_model=prism_model,
            )
            queries_processed += 1
            for rank, (item_id, sim) in enumerate(hits, start=1):
                corpus = get_corpus_by_id(db, item_id)
                if not corpus:
                    continue
                text = corpus.get("text_for_embed") or ""
                raw_c = get_raw_chunk_embedding(item_id, text, raw_cache)
                sample_id = insert_training_sample(
                    db,
                    query_id=int(query_row["id"]),
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
                if sample_id is not None:
                    samples_created += 1

    result: dict[str, Any] = {
        "retriever": retriever,
        "queries_processed": queries_processed,
        "samples_created": samples_created,
    }
    if label:
        result["labeling"] = label_batch(limit=label_limit)
    return result
