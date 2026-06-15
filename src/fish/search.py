from __future__ import annotations

import json
from typing import Any

from fish.embed import embed_text
from fish.parse import message_row_to_dict
from fish.store import db_conn, get_message_by_id, init_db, keyword_search, vector_search


def _merge_scores(
    vector_hits: list[tuple[int, float]],
    keyword_ids: list[int],
    vector_weight: float = 0.7,
) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    if vector_hits:
        max_dist = max((d for _, d in vector_hits), default=1.0) or 1.0
        for msg_id, dist in vector_hits:
            sim = 1.0 - (dist / max_dist)
            scores[msg_id] = scores.get(msg_id, 0.0) + sim * vector_weight
    for msg_id in keyword_ids:
        scores[msg_id] = scores.get(msg_id, 0.0) + (1.0 - vector_weight)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def search_messages(
    query: str,
    account_email: str | None = None,
    folder: str | None = None,
    unread_only: bool = False,
    limit: int = 20,
) -> list[dict[str, Any]]:
    init_db()
    with db_conn() as db:
        query_embedding = embed_text(query)
        vector_hits = vector_search(db, query_embedding, limit=limit * 3)
        keyword_ids = keyword_search(
            db, query, account_email=account_email, folder=folder, limit=limit * 3
        )
        merged = _merge_scores(vector_hits, keyword_ids)[:limit]

        results = []
        for msg_id, score in merged:
            row = get_message_by_id(db, msg_id)
            if not row:
                continue
            acct = db.execute(
                "SELECT email FROM accounts WHERE id = ?", (row["account_id"],)
            ).fetchone()
            acct_email = acct["email"] if acct else None
            if account_email and acct_email != account_email:
                continue
            if folder and row["folder"] != folder:
                continue
            flags = json.loads(row.get("flags") or "[]")
            if unread_only and "\\Seen" in flags:
                continue
            item = message_row_to_dict(row)
            item["score"] = round(score, 4)
            item["account_email"] = acct_email
            results.append(item)
        return results
