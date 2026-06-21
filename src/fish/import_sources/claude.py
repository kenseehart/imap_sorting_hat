from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fish.corpus import chat_corpus_item
from fish.store import db_conn, get_corpus_by_source_key, init_db, upsert_corpus_item


def _ts(value: str | None) -> str | None:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value).astimezone(timezone.utc).isoformat()
    except ValueError:
        return value


def _load_claude_conversations(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("conversations", "chat_sessions"):
            if key in data:
                return data[key]
    return [data]


def import_claude_export(
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
        conversations = _load_claude_conversations(path)

    if isinstance(conversations, dict):
        conversations = [conversations]
    stats["conversations"] = len(conversations)

    with db_conn() as db:
        for conv in conversations:
            conv_id = conv.get("uuid") or conv.get("id") or ""
            title = conv.get("name") or conv.get("title") or "untitled"
            messages = conv.get("chat_messages") or conv.get("messages") or []
            for turn_index, msg in enumerate(messages):
                role = msg.get("sender") or msg.get("role") or "unknown"
                if role in ("human", "Human"):
                    role = "user"
                elif role in ("assistant", "Assistant"):
                    role = "assistant"
                content = msg.get("text") or msg.get("content") or ""
                if isinstance(content, list):
                    content = "\n".join(
                        block.get("text", "") if isinstance(block, dict) else str(block)
                        for block in content
                    )
                content = str(content).strip()
                if not content:
                    continue
                msg_uuid = msg.get("uuid") or msg.get("id") or f"{turn_index}"
                source_key = f"claude:{conv_id}:{msg_uuid}"
                stats["turns"] += 1
                item = chat_corpus_item(
                    platform="claude",
                    source_key=source_key,
                    conversation_id=conv_id,
                    title=title,
                    role=role,
                    content=content,
                    turn_index=turn_index,
                    occurred_at=_ts(msg.get("created_at") or msg.get("updated_at")),
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
