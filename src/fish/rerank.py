"""Search-time reranking: the RelevanceAgent prompt applied to the top hits.

PRISM tried to bake RelevanceAgent's judgement into the embeddings by
training adapters (shelved 2026-11-12: the training pipeline cost more in
time and RAM than its unmeasured gain). This asks the same prompt
(``fish.prism.relevance.score``) about the top ``k`` fused candidates at
search time instead: no training, a fraction of a cent per search on
gpt-4o-mini, and the scores stay comparable with the labels already
collected.

Settings (``fish.env``):
  FISH_RERANK      on | off   (default on)
  FISH_RERANK_K    candidates scored per search (default 20)
A failed relevance call fails the search; it is never skipped.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

DEFAULT_RERANK_K = 20


def rerank_enabled() -> bool:
    from fish.config import load_env

    load_env()
    raw = os.getenv("FISH_RERANK", "on").strip().lower()
    if raw not in ("on", "off"):
        raise ValueError(f"FISH_RERANK must be 'on' or 'off', got {raw!r}")
    return raw == "on"


def rerank_k() -> int:
    from fish.config import load_env

    load_env()
    raw = os.getenv("FISH_RERANK_K", str(DEFAULT_RERANK_K))
    try:
        k = int(raw)
    except ValueError as exc:
        raise ValueError(f"FISH_RERANK_K must be an int, got {raw!r}") from exc
    if k < 1:
        raise ValueError(f"FISH_RERANK_K must be >= 1, got {k}")
    return k


def score_rows(
    query: str,
    rows: list[dict[str, Any]],
    *,
    context_json: str | None = None,
    scorer=None,
    concurrency: int | None = None,
) -> list[float]:
    """Relevance in [0, 1] for each corpus row, in order. ``scorer`` is
    ``fish.prism.relevance.score`` unless a test passes another."""
    from fish.prism.relevance import default_label_concurrency

    if scorer is None:
        from openai import OpenAI

        from fish.config import openai_api_key
        from fish.prism.relevance import score as relevance_score

        # One client per worker thread, as corpus labelling does (the OpenAI
        # client is not guaranteed thread-safe).
        local = threading.local()

        def scorer(q: str, doc: str, *, context_json: str | None) -> float:
            if not hasattr(local, "client"):
                local.client = OpenAI(api_key=openai_api_key())
            return relevance_score(q, doc, context_json=context_json, client=local.client)

    docs = [r.get("text_for_embed") or r.get("body_text") or "" for r in rows]
    workers = min(len(docs), concurrency or default_label_concurrency()) or 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(scorer, query, d, context_json=context_json) for d in docs]
        return [f.result() for f in futures]
