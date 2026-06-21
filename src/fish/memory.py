from __future__ import annotations

import json
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Literal

from openai import OpenAI

from fish.config import (
    MEMORY_DEDUP_THRESHOLD,
    MEMORY_MERGE_THRESHOLD,
    memory_llm_model,
    openai_api_key,
)
from fish.corpus import memory_corpus_item
from fish.embed import embed_text
from fish.prism.inference import cosine_similarity
from fish.store import (
    corpus_payload,
    corpus_vector_search,
    db_conn,
    get_corpus_by_id,
    get_embedding,
    init_db,
    mark_memory_superseded,
    memory_is_active,
    upsert_corpus_item,
)
from fish.sync import embed_pending

MemoryAction = Literal["duplicate", "merge", "distinct", "insert"]


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _memory_source_key(fact: str) -> str:
    return f"memory:{sha256(fact.strip().lower().encode()).hexdigest()[:24]}"


def _memory_fact(row: dict[str, Any]) -> str:
    payload = corpus_payload(row)
    return (payload.get("fact") or row.get("body_text") or "").strip()


def _row_tags(row: dict[str, Any]) -> list[str]:
    payload = corpus_payload(row)
    tags = payload.get("tags")
    if isinstance(tags, list):
        return tags
    raw = row.get("tags")
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []


def _merge_tags(existing: list[str], new: list[str] | None) -> list[str]:
    out: list[str] = []
    for tag in existing + (new or []):
        if tag and tag not in out:
            out.append(tag)
    return out


def _find_similar_memories(
    db: Any, fact: str, *, limit: int = 5
) -> list[tuple[dict[str, Any], float]]:
    query_vec = embed_text(f"Memory: {fact}")
    hits = corpus_vector_search(db, query_vec, limit=limit * 2, kinds=["memory"])
    results: list[tuple[dict[str, Any], float]] = []
    for item_id, _dist in hits:
        row = get_corpus_by_id(db, item_id)
        if not row or not memory_is_active(row):
            continue
        stored = get_embedding(db, item_id)
        if not stored:
            continue
        score = cosine_similarity(query_vec, stored)
        if score >= MEMORY_MERGE_THRESHOLD:
            results.append((row, score))
        if len(results) >= limit:
            break
    results.sort(key=lambda pair: pair[1], reverse=True)
    return results


def _classify_memory(
    existing_fact: str, new_fact: str
) -> dict[str, Any]:
    client = OpenAI(api_key=openai_api_key())
    prompt = (
        "You reconcile two memory facts stored about the same person.\n\n"
        f'Existing memory: "{existing_fact}"\n'
        f'New memory: "{new_fact}"\n\n'
        "Respond with JSON only:\n"
        '{"action": "duplicate"|"merge"|"distinct", "merged_fact": "..."}\n\n'
        "Rules:\n"
        "- duplicate: same information; keep existing, discard new\n"
        "- merge: related facts that should be one memory; write combined fact in merged_fact\n"
        "- distinct: unrelated; keep both separate\n"
        "- merged_fact is required when action is merge; omit otherwise"
    )
    response = client.chat.completions.create(
        model=memory_llm_model(),
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_tokens=200,
    )
    raw = (response.choices[0].message.content or "").strip()
    data = json.loads(raw)
    action = data.get("action")
    if action not in ("duplicate", "merge", "distinct"):
        raise RuntimeError(f"Invalid memory merge action from LLM: {action!r}")
    merged = (data.get("merged_fact") or "").strip()
    if action == "merge" and not merged:
        raise RuntimeError("LLM returned merge without merged_fact")
    return {"action": action, "merged_fact": merged}


def _insert_memory(
    db: Any,
    fact: str,
    *,
    tags: list[str] | None,
    confidence: float | None,
    provenance: str | None,
    expires_at: str | None,
    supersedes_id: int | None,
) -> int:
    item = memory_corpus_item(
        fact=fact,
        source_key=_memory_source_key(fact),
        tags=tags,
        confidence=confidence,
        provenance=provenance,
        supersedes_id=supersedes_id,
        expires_at=expires_at,
        occurred_at=_utcnow(),
    )
    return upsert_corpus_item(db, item)


def upsert_memory(
    fact: str,
    *,
    tags: list[str] | None = None,
    confidence: float | None = None,
    provenance: str | None = None,
    expires_at: str | None = None,
    embed: bool = True,
    agentic: bool = True,
) -> dict[str, Any]:
    init_db()
    fact = fact.strip()
    if not fact:
        raise ValueError("Memory fact cannot be empty")

    action: MemoryAction = "insert"
    merged_fact: str | None = None
    similar_id: int | None = None
    similar_score: float | None = None
    item_id: int

    with db_conn() as db:
        candidates = _find_similar_memories(db, fact)
        if not candidates:
            action = "insert"
            item_id = _insert_memory(
                db,
                fact,
                tags=tags,
                confidence=confidence,
                provenance=provenance,
                expires_at=expires_at,
                supersedes_id=None,
            )
        else:
            best_row, best_score = candidates[0]
            similar_id = int(best_row["id"])
            similar_score = best_score
            existing_fact = _memory_fact(best_row)

            if best_score >= MEMORY_DEDUP_THRESHOLD:
                action = "duplicate"
                item_id = similar_id
            elif agentic:
                decision = _classify_memory(existing_fact, fact)
                action = decision["action"]
                if action == "duplicate":
                    item_id = similar_id
                elif action == "distinct":
                    item_id = _insert_memory(
                        db,
                        fact,
                        tags=tags,
                        confidence=confidence,
                        provenance=provenance,
                        expires_at=expires_at,
                        supersedes_id=None,
                    )
                else:
                    action = "merge"
                    merged_fact = decision["merged_fact"]
                    merged_tags = _merge_tags(_row_tags(best_row), tags)
                    item_id = _insert_memory(
                        db,
                        merged_fact,
                        tags=merged_tags,
                        confidence=confidence,
                        provenance=provenance,
                        expires_at=expires_at,
                        supersedes_id=similar_id,
                    )
                    mark_memory_superseded(db, similar_id, item_id)
            else:
                action = "insert"
                item_id = _insert_memory(
                    db,
                    fact,
                    tags=tags,
                    confidence=confidence,
                    provenance=provenance,
                    expires_at=expires_at,
                    supersedes_id=similar_id,
                )

        result = get_corpus_by_id(db, item_id) or {}

    if embed and action != "duplicate":
        embed_pending(batch_size=1)
        with db_conn() as db:
            result = get_corpus_by_id(db, item_id) or result

    return {
        "id": item_id,
        "action": action,
        "similar_id": similar_id,
        "similar_score": round(similar_score, 4) if similar_score is not None else None,
        "merged_fact": merged_fact,
        "supersedes_id": similar_id if action == "merge" else None,
        "item": result,
    }
