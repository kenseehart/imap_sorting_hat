from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from fish.config import openai_api_key
from fish.prism.queries import query_text_for_search
from fish.store import (
    db_conn,
    get_corpus_by_id,
    get_training_query,
    init_db,
    list_unlabeled_samples,
    update_sample_relevance,
)

RELEVANCE_AGENT_VERSION = "1.0.0"
DEFAULT_RELEVANCE_MODEL = "gpt-4o-mini"


def relevance_model() -> str:
    import os

    from fish.config import load_env

    load_env()
    return os.getenv("FISH_RELEVANCE_MODEL", DEFAULT_RELEVANCE_MODEL)


def score(
    query_text: str,
    document_text: str,
    *,
    context_json: str | None = None,
) -> float:
    augmented = query_text_for_search(query_text, context_json)
    doc = (document_text or "")[:1500]
    client = OpenAI(api_key=openai_api_key())
    prompt = (
        "Rate relevance of this document to the query on a 0.0-1.0 scale.\n"
        f"Query: {augmented}\n\nDocument:\n{doc}\n\n"
        'Respond with JSON only: {"relevance": <number>}'
    )
    response = client.chat.completions.create(
        model=relevance_model(),
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_tokens=32,
    )
    raw = (response.choices[0].message.content or "").strip()
    data = json.loads(raw)
    value = float(data.get("relevance", 0.0))
    return max(0.0, min(1.0, value))


def label_sample(sample_id: int, *, force: bool = False) -> float | None:
    init_db()
    with db_conn() as db:
        row = db.execute(
            "SELECT * FROM training_samples WHERE id = ?", (sample_id,)
        ).fetchone()
        if not row:
            raise ValueError(f"Sample {sample_id} not found")
        sample = dict(row)
        if (
            not force
            and sample.get("target_relevance") is not None
            and sample.get("relevance_agent_version") == RELEVANCE_AGENT_VERSION
        ):
            return float(sample["target_relevance"])

        query = get_training_query(db, int(sample["query_id"]))
        if not query:
            raise ValueError(f"Query {sample['query_id']} not found")
        corpus = get_corpus_by_id(db, int(sample["corpus_item_id"]))
        if not corpus:
            raise ValueError(f"Corpus item {sample['corpus_item_id']} not found")

        doc_text = corpus.get("text_for_embed") or corpus.get("body_text") or ""
        rel = score(
            query["text"],
            doc_text,
            context_json=query.get("context_json"),
        )
        update_sample_relevance(
            db,
            sample_id,
            target_relevance=rel,
            agent_version=RELEVANCE_AGENT_VERSION,
            relevance_model=relevance_model(),
        )
        return rel


def label_batch(*, limit: int = 500, force: bool = False) -> dict[str, Any]:
    init_db()
    labeled = 0
    errors: list[str] = []
    with db_conn() as db:
        samples = list_unlabeled_samples(
            db,
            limit=limit,
            agent_version=RELEVANCE_AGENT_VERSION,
            force=force,
        )
    for sample in samples:
        try:
            label_sample(int(sample["id"]), force=force)
            labeled += 1
        except Exception as exc:
            errors.append(f"sample {sample['id']}: {exc}")
    return {"labeled": labeled, "errors": errors}
