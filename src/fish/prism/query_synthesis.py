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

QUERY_SYNTHESIS_AGENT_VERSION = "1.0.0"
DEFAULT_SYNTHESIS_MODEL = "gpt-4o-mini"


def synthesis_model() -> str:
    import os

    from fish.config import load_env

    load_env()
    return os.getenv("FISH_SYNTHESIS_MODEL", DEFAULT_SYNTHESIS_MODEL)


def synthesize_queries(seed_texts: list[str]) -> list[str]:
    if not seed_texts:
        raise RuntimeError("No seed queries for synthesis")
    n = len(seed_texts)
    seeds_block = "\n".join(f"- {t}" for t in seed_texts)
    client = OpenAI(api_key=openai_api_key())
    prompt = (
        f"Generate exactly {n} new search queries in a similar style to these examples.\n"
        "Vary phrasing, specificity, tense, and intent while keeping the same general domain.\n"
        "Do not copy the examples verbatim.\n\n"
        f"Examples:\n{seeds_block}\n\n"
        f'Respond with JSON only: {{"queries": ["...", ...]}} with exactly {n} strings.'
    )
    response = client.chat.completions.create(
        model=synthesis_model(),
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_tokens=800,
    )
    raw = (response.choices[0].message.content or "").strip()
    data = json.loads(raw)
    queries = data.get("queries") or []
    if not isinstance(queries, list):
        raise RuntimeError("QuerySynthesisAgent returned invalid queries list")
    out = [str(q).strip() for q in queries if str(q).strip()]
    if len(out) < n:
        raise RuntimeError(
            f"QuerySynthesisAgent returned {len(out)} queries, expected {n}"
        )
    return out[:n]


def ensure_query_count(
    *,
    min_queries: int,
    synthesis_batch: int = 5,
) -> dict[str, Any]:
    """Extend synthetic queries until total count >= min_queries."""
    init_db()
    created = 0
    with db_conn() as db:
        total = db.execute("SELECT COUNT(*) FROM training_queries").fetchone()[0]
        while total < min_queries:
            real_count = db.execute(
                "SELECT COUNT(*) FROM training_queries WHERE origin = 'real'"
            ).fetchone()[0]
            if real_count == 0:
                raise RuntimeError(
                    "No real queries logged yet — run fish search or fish_search MCP first"
                )
            batch = min(synthesis_batch, min_queries - total)
            seeds = pick_random_training_queries(db, origin="real", limit=batch)
            new_texts = synthesize_queries([s["text"] for s in seeds])
            parent_id = int(seeds[0]["id"]) if seeds else None
            round_created = 0
            for text in new_texts:
                query_id = insert_training_query(
                    db,
                    text=text,
                    origin="synthetic",
                    parent_query_id=parent_id,
                    synthesis_method="style_match",
                )
                if query_id is None:
                    continue
                vec = embed_text(text)
                update_training_query_embedding(
                    db, query_id, vec, embedding_model()
                )
                created += 1
                round_created += 1
                total += 1
                if total >= min_queries:
                    break
            if round_created == 0:
                break
    return {"created": created, "total_queries": total}
