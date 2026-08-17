"""Query synthesis — extend gold (logged) queries with synth variants."""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from fish.config import embedding_model, openai_api_key
from fish.store import (
    db_conn,
    init_db,
    insert_training_query,
    pick_random_training_queries,
    update_training_query_embedding,
)
from fish.embed import embed_text

QUERY_SYNTHESIS_AGENT_VERSION = "2.0.0"
DEFAULT_SYNTHESIS_MODEL = "gpt-4o-mini"


def synthesis_model() -> str:
    import os

    from fish.config import load_env

    load_env()
    return os.getenv("FISH_SYNTHESIS_MODEL", DEFAULT_SYNTHESIS_MODEL)


def synthesize_queries(seed_texts: list[str], *, n: int | None = None) -> list[str]:
    if not seed_texts:
        raise RuntimeError("No seed queries for synthesis")
    n_out = n if n is not None else len(seed_texts)
    seeds_block = "\n".join(f"- {t}" for t in seed_texts)
    client = OpenAI(api_key=openai_api_key())
    prompt = (
        f"Generate exactly {n_out} new retrieval queries for a personal corpus "
        "(email, SMS, chat, memory).\n\n"
        "Each query must be an explicit information request: something someone "
        "would ask when they need specific content they can use. Prefer queries "
        "concrete enough that a relevance judge can decide whether a document "
        "actually answers them, not merely whether it shares a topic.\n"
        "Vary phrasing and specificity. Stay in the same general domains as the "
        "seed list below. Do not copy seeds verbatim.\n"
        "The seeds are domain and phrasing anchors only; do not treat them as "
        "templates or as definitions of good vs bad queries.\n\n"
        f"Seeds:\n{seeds_block}\n\n"
        'Respond with JSON only: {"queries": ["...", ...]}'
    )
    response = client.chat.completions.create(
        model=synthesis_model(),
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_tokens=800,
    )
    raw = (response.choices[0].message.content or "").strip()
    data = json.loads(raw)
    queries = data.get("queries") or data.get("query") or []
    if isinstance(queries, str):
        queries = [queries]
    out = [str(q).strip() for q in queries if str(q).strip()]
    return out[:n_out]


def ensure_query_count(
    *,
    min_queries: int,
    synthesis_batch: int = 5,
    seed_batch: int = 10,
) -> dict[str, Any]:
    """Grow synth until gold+synth >= min_queries.

    Algorithm: while gold+synth < desired, sample ``seed_batch`` gold queries,
    generate ``synthesis_batch`` new synth queries like the examples but different.
    Curated queries are not counted toward the target and are not used as seeds.
    """
    init_db()
    created = 0
    with db_conn() as db:

        def gold_synth_count() -> int:
            return int(
                db.execute(
                    "SELECT COUNT(*) FROM training_queries "
                    "WHERE origin IN ('gold', 'synth')"
                ).fetchone()[0]
            )

        total = gold_synth_count()
        while total < min_queries:
            gold_count = db.execute(
                "SELECT COUNT(*) FROM training_queries WHERE origin = 'gold'"
            ).fetchone()[0]
            if gold_count == 0:
                raise RuntimeError(
                    "No gold (logged) queries yet — run fish search / fish_search first"
                )
            seeds = pick_random_training_queries(
                db, origin="gold", limit=min(seed_batch, int(gold_count))
            )
            new_texts = synthesize_queries(
                [s["text"] for s in seeds], n=synthesis_batch
            )
            parent_id = int(seeds[0]["id"]) if seeds else None
            round_created = 0
            for text in new_texts:
                query_id = insert_training_query(
                    db,
                    text=text,
                    origin="synth",
                    parent_query_id=parent_id,
                    synthesis_method="style_match",
                    source=f"synth:{QUERY_SYNTHESIS_AGENT_VERSION}",
                )
                if query_id is None:
                    continue
                vec = embed_text(text)
                update_training_query_embedding(
                    db, query_id, vec, embedding_model()
                )
                created += 1
                round_created += 1
                total = gold_synth_count()
                if total >= min_queries:
                    break
            if round_created == 0:
                break
    return {
        "created": created,
        "gold_plus_synth": total,
        "min_queries": min_queries,
    }
