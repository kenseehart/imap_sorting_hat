"""Re-index a registered PRISM model's Qdrant collection from corpus_raw_embeddings."""

from __future__ import annotations

from typing import Any

from cmdline.progress import progress_bar

from fish.prism.configs import LEGACY_MODEL_ID
from fish.prism.inference import adapt_chunk_for_model, clear_model_cache
from fish.store import (
    db_conn,
    init_db,
    list_corpus_with_raw_embedding,
    set_model_embedding,
)
from fish.write_lock import fish_write_lock


def prism_reembed(
    *,
    model_id: str | None = None,
    kinds: list[str] | None = None,
    show_progress: bool = True,
    limit: int | None = None,
    like: list[str] | None = None,
    since: str | None = None,
) -> dict[str, Any]:
    """Rewrite one PRISM model's Qdrant collection from stored raw embeddings."""
    init_db()
    clear_model_cache()
    with fish_write_lock("train"):
        with db_conn() as db:
            from fish.prism.registry import active_prism_model, get_retrieval_model

            if model_id is None:
                active = active_prism_model(db)
                if active is None:
                    raise RuntimeError(
                        "No active PRISM model. Pass --model-id or activate one "
                        "after fish prism-train."
                    )
                model_id = active["model_id"]
            if model_id == LEGACY_MODEL_ID:
                raise ValueError("Cannot re-index legacy/raw — it is the source index")
            model = get_retrieval_model(db, model_id)
            if model is None:
                raise KeyError(f"Unknown model_id {model_id!r}")

            items = list_corpus_with_raw_embedding(
                db, kinds=kinds, limit=limit, like=like, since=since
            )
            if not items:
                return {
                    "model_id": model_id,
                    "collection": model["vec_table"],
                    "adapted": 0,
                    "openai_calls": 0,
                    "note": (
                        "No raw vectors in corpus_raw_embeddings — "
                        "run fish embed / qdrant-migrate first"
                    ),
                }

            bar = progress_bar(
                total=len(items),
                desc=f"index {model_id}",
                unit="msg",
                disable=not show_progress,
            )
            adapted = 0
            for row in items:
                raw = row["raw_embedding"]
                ac = adapt_chunk_for_model(raw, model_id)
                set_model_embedding(db, int(row["id"]), model_id, ac)
                adapted += 1
                bar.update(1)
            bar.close()

    return {
        "model_id": model_id,
        "collection": model["vec_table"],
        "adapted": adapted,
        "limit": limit,
        "like": like,
        "since": since,
        "openai_calls": 0,
    }
