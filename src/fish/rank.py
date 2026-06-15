from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from fish.parse import message_row_to_dict
from fish.store import db_conn, get_message_by_id, init_db


def _parse_flags(flags_raw: str | None) -> list[str]:
    try:
        return json.loads(flags_raw or "[]")
    except json.JSONDecodeError:
        return []


def _recency_score(date_str: str | None) -> float:
    if not date_str:
        return 0.0
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        age_days = (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).days
        return max(0.0, 1.0 - age_days / 90.0)
    except (TypeError, ValueError):
        return 0.0


def compute_importance(limit: int = 500) -> int:
    init_db()
    updated = 0
    with db_conn() as db:
        rows = db.execute(
            """
            SELECT m.*, a.email AS account_email
            FROM messages m
            JOIN accounts a ON a.id = m.account_id
            ORDER BY m.date DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        sender_counts: dict[str, int] = {}
        for row in rows:
            sender = row["from_addr"] or ""
            sender_counts[sender] = sender_counts.get(sender, 0) + 1

        now = datetime.now(timezone.utc).isoformat()
        for row in rows:
            flags = _parse_flags(row["flags"])
            signals = {
                "recency": _recency_score(row["date"]),
                "unread": 1.0 if "\\Seen" not in flags else 0.0,
                "starred": 1.0 if "\\Flagged" in flags else 0.0,
                "sender_frequency": min(1.0, sender_counts.get(row["from_addr"] or "", 0) / 10.0),
                "has_reply_chain": 1.0 if row["in_reply_to"] else 0.0,
            }
            score = (
                signals["recency"] * 0.25
                + signals["unread"] * 0.35
                + signals["starred"] * 0.2
                + signals["sender_frequency"] * 0.1
                + signals["has_reply_chain"] * 0.1
            )
            db.execute(
                """
                INSERT INTO importance (message_id, score, signals_json, computed_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(message_id) DO UPDATE SET
                    score=excluded.score,
                    signals_json=excluded.signals_json,
                    computed_at=excluded.computed_at
                """,
                (row["id"], score, json.dumps(signals), now),
            )
            updated += 1
    return updated


def priority_inbox(limit: int = 20) -> list[dict[str, Any]]:
    init_db()
    compute_importance(limit=limit * 10)
    with db_conn() as db:
        rows = db.execute(
            """
            SELECT m.*, i.score, i.signals_json, a.email AS account_email
            FROM importance i
            JOIN messages m ON m.id = i.message_id
            JOIN accounts a ON a.id = m.account_id
            ORDER BY i.score DESC, m.date DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        results = []
        for row in rows:
            item = message_row_to_dict(dict(row))
            item["importance_score"] = row["score"]
            try:
                signals = json.loads(row["signals_json"] or "{}")
            except json.JSONDecodeError:
                signals = {}
            reasons = []
            if signals.get("unread"):
                reasons.append("unread")
            if signals.get("starred"):
                reasons.append("starred")
            if signals.get("recency", 0) > 0.7:
                reasons.append("recent")
            if signals.get("sender_frequency", 0) > 0.5:
                reasons.append("frequent sender")
            if signals.get("has_reply_chain"):
                reasons.append("part of thread")
            item["reasons"] = reasons or ["general relevance"]
            results.append(item)
        return results


def digest(limit: int = 10) -> dict[str, Any]:
    top = priority_inbox(limit=limit)
    senders: dict[str, int] = {}
    for item in top:
        sender = item.get("from_addr") or "unknown"
        senders[sender] = senders.get(sender, 0) + 1
    return {
        "summary": f"{len(top)} high-priority messages surfaced.",
        "top_senders": sorted(senders.items(), key=lambda x: x[1], reverse=True)[:5],
        "messages": [
            {
                "id": m["id"],
                "subject": m.get("subject"),
                "from_addr": m.get("from_addr"),
                "account_email": m.get("account_email"),
                "importance_score": m.get("importance_score"),
                "reasons": m.get("reasons"),
            }
            for m in top
        ],
    }
