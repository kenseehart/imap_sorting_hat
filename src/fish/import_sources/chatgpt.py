from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from fish.corpus import chat_corpus_item
from fish.store import db_conn, get_corpus_by_source_key, init_db, upsert_corpus_item


def _ts(value: float | int | None) -> str | None:
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        return None


def _iter_chatgpt_messages(conversations: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    for conv in conversations:
        conv_id = conv.get("id") or conv.get("conversation_id") or ""
        title = conv.get("title") or "untitled"
        mapping = conv.get("mapping") or {}
        for node_id, node in mapping.items():
            message = node.get("message")
            if not message:
                continue
            author = message.get("author") or {}
            role = author.get("role") or "unknown"
            if role not in ("user", "assistant"):
                continue
            content = message.get("content") or {}
            parts = content.get("parts") or []
            text = "\n".join(p for p in parts if isinstance(p, str)).strip()
            if not text:
                continue
            create_time = message.get("create_time")
            yield {
                "platform": "chatgpt",
                "conversation_id": conv_id,
                "title": title,
                "role": role,
                "content": text,
                "turn_index": hash(node_id) % 10_000_000,
                "node_id": node_id,
                "occurred_at": _ts(create_time),
                "model": (message.get("metadata") or {}).get("model_slug"),
            }


def _load_conversations_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "conversations" in data:
        return data["conversations"]
    return [data]


def import_chatgpt_export(
    path: Path,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    init_db()
    stats = {"conversations": 0, "turns": 0, "inserted": 0, "updated": 0}

    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            names = [n for n in zf.namelist() if n.endswith("conversations.json")]
            if not names:
                raise ValueError(f"No conversations.json in {path}")
            raw = zf.read(names[0])
        conversations = json.loads(raw)
    else:
        conversations = _load_conversations_json(path)

    if isinstance(conversations, dict):
        conversations = [conversations]
    stats["conversations"] = len(conversations)

    with db_conn() as db:
        for turn in _iter_chatgpt_messages(conversations):
            stats["turns"] += 1
            source_key = (
                f"chatgpt:{turn['conversation_id']}:{turn['node_id']}"
            )
            item = chat_corpus_item(
                platform="chatgpt",
                source_key=source_key,
                conversation_id=turn["conversation_id"],
                title=turn["title"],
                role=turn["role"],
                content=turn["content"],
                turn_index=int(turn["turn_index"]),
                occurred_at=turn["occurred_at"],
                model=turn.get("model"),
            )
            if dry_run:
                continue
            existing = get_corpus_by_source_key(db, source_key)
            upsert_corpus_item(db, item)
            if existing:
                stats["updated"] += 1
            else:
                stats["inserted"] += 1

    return stats


def import_chatgpt_memory_json(path: Path, *, dry_run: bool = False) -> dict[str, Any]:
    from fish.corpus import content_hash, memory_corpus_item

    init_db()
    data = json.loads(path.read_text())
    memories = data if isinstance(data, list) else data.get("memories") or []
    stats = {"seen": 0, "inserted": 0, "updated": 0}
    with db_conn() as db:
        for entry in memories:
            stats["seen"] += 1
            fact = entry.get("content") or entry.get("text") or ""
            if not fact:
                continue
            source_key = f"chatgpt_memory:{entry.get('id') or content_hash(fact)}"
            item = memory_corpus_item(
                fact=fact,
                source_key=source_key,
                provenance="chatgpt_export",
                occurred_at=_ts(entry.get("updated_at") or entry.get("created_at")),
                source="chatgpt_memory",
            )
            if dry_run:
                continue
            existing = get_corpus_by_source_key(db, source_key)
            upsert_corpus_item(db, item)
            if existing:
                stats["updated"] += 1
            else:
                stats["inserted"] += 1
    return stats
