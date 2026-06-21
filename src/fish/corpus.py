from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Literal

CorpusKind = Literal["email", "sms", "chat", "memory"]
CorpusSource = Literal[
    "imap", "android_sms", "claude_export", "chatgpt_export", "agent", "chatgpt_memory"
]

KINDS: tuple[CorpusKind, ...] = ("email", "sms", "chat", "memory")
SOURCES: tuple[CorpusSource, ...] = (
    "imap",
    "android_sms",
    "claude_export",
    "chatgpt_export",
    "agent",
    "chatgpt_memory",
)

PHONE_FILTER_DEFAULT = "8315352442"


def normalize_phone(value: str) -> str:
    digits = re.sub(r"\D", "", value or "")
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    return digits


def content_hash(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class CorpusItem:
    kind: CorpusKind
    source: CorpusSource
    source_key: str
    text_for_embed: str
    occurred_at: str | None
    payload: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    id: int | None = None
    ingested_at: str | None = None
    embedded_at: str | None = None
    body_text: str = ""
    content_hash_value: str | None = None

    def __post_init__(self) -> None:
        if self.content_hash_value is None:
            self.content_hash_value = content_hash(self.text_for_embed)


def email_corpus_from_message(
    message_id: int,
    account_id: int,
    folder: str,
    uid: int,
    parsed: Any,
    account_email: str | None = None,
) -> CorpusItem:
    payload = {
        "account_id": account_id,
        "account_email": account_email,
        "folder": folder,
        "uid": uid,
        "message_id": parsed.message_id,
        "in_reply_to": parsed.in_reply_to,
        "subject": parsed.subject,
        "from_addr": parsed.from_addrs[0] if parsed.from_addrs else "",
        "to_addrs": parsed.to_addrs,
        "cc_addrs": parsed.cc_addrs,
        "flags": parsed.flags,
        "gmail_labels": parsed.gmail_labels,
    }
    source_key = f"imap:{account_id}:{folder}:{uid}"
    return CorpusItem(
        id=message_id,
        kind="email",
        source="imap",
        source_key=source_key,
        text_for_embed=parsed.body_for_embed,
        occurred_at=parsed.date,
        payload=payload,
        body_text=parsed.body_text,
        content_hash_value=parsed.content_hash,
    )


def sms_corpus_item(
    *,
    source_key: str,
    phone: str,
    direction: str,
    body: str,
    occurred_at: str | None,
    contact_name: str | None = None,
    sms_id: str | None = None,
) -> CorpusItem:
    phone_norm = normalize_phone(phone)
    dir_label = "from" if direction in ("in", "received", "1") else "to"
    text_for_embed = f"SMS {dir_label} {phone_norm}: {body}"
    payload = {
        "phone": phone_norm,
        "direction": direction,
        "contact_name": contact_name,
        "sms_id": sms_id,
        "body": body,
    }
    return CorpusItem(
        kind="sms",
        source="android_sms",
        source_key=source_key,
        text_for_embed=text_for_embed,
        occurred_at=occurred_at,
        payload=payload,
        body_text=body,
    )


def chat_corpus_item(
    *,
    platform: Literal["claude", "chatgpt"],
    source_key: str,
    conversation_id: str,
    title: str,
    role: str,
    content: str,
    turn_index: int,
    occurred_at: str | None,
    model: str | None = None,
) -> CorpusItem:
    source: CorpusSource = "claude_export" if platform == "claude" else "chatgpt_export"
    text_for_embed = f"Chat {platform} {title}: {role}: {content}"
    payload = {
        "platform": platform,
        "conversation_id": conversation_id,
        "title": title,
        "role": role,
        "turn_index": turn_index,
        "model": model,
        "content": content,
    }
    return CorpusItem(
        kind="chat",
        source=source,
        source_key=source_key,
        text_for_embed=text_for_embed,
        occurred_at=occurred_at,
        payload=payload,
        body_text=content,
    )


def memory_corpus_item(
    *,
    fact: str,
    source_key: str,
    tags: list[str] | None = None,
    confidence: float | None = None,
    provenance: str | None = None,
    supersedes_id: int | None = None,
    expires_at: str | None = None,
    occurred_at: str | None = None,
    source: CorpusSource = "agent",
) -> CorpusItem:
    tag_list = tags or []
    tag_suffix = f" Tags: {', '.join(tag_list)}." if tag_list else ""
    text_for_embed = f"Memory: {fact}.{tag_suffix}"
    payload = {
        "fact": fact,
        "confidence": confidence,
        "provenance": provenance,
        "supersedes_id": supersedes_id,
        "superseded_by": None,
        "expires_at": expires_at,
    }
    return CorpusItem(
        kind="memory",
        source=source,
        source_key=source_key,
        text_for_embed=text_for_embed,
        occurred_at=occurred_at,
        payload=payload,
        tags=tag_list,
        body_text=fact,
    )


def corpus_row_to_dict(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    for key in ("payload", "tags"):
        if key in out and isinstance(out[key], str):
            try:
                out[key] = json.loads(out[key])
            except json.JSONDecodeError:
                pass
    return out
