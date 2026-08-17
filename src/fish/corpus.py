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


def imap_source_key(account_id: int, folder: str, uid: int) -> str:
    """Legacy locator key ``imap:{account_id}:{folder}:{uid}`` (not canonical).

    Prefer ``fish.identity.email_canonical_id`` for ``corpus_items.source_key``.
    Kept for parsing old rows during migration.
    """
    return f"imap:{account_id}:{folder}:{uid}"


def parse_imap_source_key(source_key: str) -> tuple[int, str, int] | None:
    """Parse legacy ``imap:{account_id}:{folder}:{uid}`` (folder may contain ``:``)."""
    if not source_key.startswith("imap:"):
        return None
    rest = source_key[5:]
    first = rest.find(":")
    last = rest.rfind(":")
    if first < 0 or last <= first:
        return None
    try:
        account_id = int(rest[:first])
        uid = int(rest[last + 1 :])
    except ValueError:
        return None
    folder = rest[first + 1 : last]
    if not folder:
        return None
    return account_id, folder, uid


def email_locator_from_payload(payload: dict[str, Any] | str | None) -> tuple[int, str, int] | None:
    """IMAP locator from corpus payload (account_id, folder, uid)."""
    if payload is None:
        return None
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return None
    if not isinstance(payload, dict):
        return None
    try:
        account_id = int(payload["account_id"])
        folder = str(payload["folder"] or "")
        uid = int(payload["uid"])
    except (KeyError, TypeError, ValueError):
        return None
    if not folder:
        return None
    return account_id, folder, uid


# SQL: email corpus ↔ messages by IMAP locator in payload (never m.id = c.id).
EMAIL_CORPUS_MESSAGE_JOIN = """(
  m.account_id = CAST(json_extract(c.payload, '$.account_id') AS INTEGER)
  AND m.folder = json_extract(c.payload, '$.folder')
  AND m.uid = CAST(json_extract(c.payload, '$.uid') AS INTEGER)
)"""

# messages → corpus via messages.canonical_id = corpus.source_key (UNIQUE).
EMAIL_MESSAGE_TO_CORPUS_JOIN = "c.source_key = m.canonical_id"


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
    header_json: str | None = None
    content_hash_value: str | None = None

    def __post_init__(self) -> None:
        if self.content_hash_value is None:
            self.content_hash_value = content_hash(self.text_for_embed)


def email_corpus_from_message(
    message_pk: int | None,
    account_id: int,
    folder: str,
    uid: int,
    parsed: Any,
    account_email: str | None = None,
    *,
    allow_gmail_rfc_fallback: bool = False,
) -> CorpusItem:
    """Build email corpus item with canonical ``source_key`` (not IMAP UID).

    ``message_pk`` is the ``messages.id`` surrogate only — never preferred as
    ``corpus_items.id`` (cross-source PK collisions).
    """
    from fish.identity import email_canonical_id

    if not account_email:
        raise ValueError("account_email is required for canonical email source_key")
    from_addr = parsed.from_addrs[0] if parsed.from_addrs else ""
    source_key, synthetic = email_canonical_id(
        account_email=account_email,
        rfc_message_id=getattr(parsed, "message_id", None),
        gm_msgid=getattr(parsed, "gm_msgid", None),
        from_addr=from_addr,
        date=parsed.date,
        subject=parsed.subject or "",
        body=parsed.body_text or "",
        allow_gmail_rfc_fallback=allow_gmail_rfc_fallback,
    )
    payload = {
        "account_id": account_id,
        "account_email": account_email,
        "folder": folder,
        "uid": uid,
        "message_id": parsed.message_id,
        "gm_msgid": getattr(parsed, "gm_msgid", None),
        "canonical_id": source_key,
        "synthetic": synthetic,
        "in_reply_to": parsed.in_reply_to,
        "subject": parsed.subject,
        "from_addr": from_addr,
        "to_addrs": parsed.to_addrs,
        "cc_addrs": parsed.cc_addrs,
        "flags": parsed.flags,
        "gmail_labels": parsed.gmail_labels,
        "messages_pk": message_pk,
    }
    header_obj = {
        "subject": parsed.subject,
        "from": list(parsed.from_addrs),
        "to": list(parsed.to_addrs),
        "cc": list(parsed.cc_addrs),
        "message_id": parsed.message_id,
        "gm_msgid": getattr(parsed, "gm_msgid", None),
        "canonical_id": source_key,
        "synthetic": synthetic,
        "in_reply_to": parsed.in_reply_to,
        "date": parsed.date,
        "account_email": account_email,
        "folder": folder,
    }
    return CorpusItem(
        id=None,
        kind="email",
        source="imap",
        source_key=source_key,
        text_for_embed=parsed.body_for_embed,
        occurred_at=parsed.date,
        payload=payload,
        body_text=parsed.body_text,
        header_json=json.dumps(header_obj, sort_keys=True, ensure_ascii=False),
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
    header_json = json.dumps(
        {
            "kind": "sms",
            "phone": phone_norm,
            "direction": direction,
            "contact_name": contact_name,
            "sms_id": sms_id,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return CorpusItem(
        kind="sms",
        source="android_sms",
        source_key=source_key,
        text_for_embed=text_for_embed,
        occurred_at=occurred_at,
        payload=payload,
        body_text=body,
        header_json=header_json,
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
    header_json = json.dumps(
        {
            "kind": "chat",
            "platform": platform,
            "conversation_id": conversation_id,
            "title": title,
            "role": role,
            "turn_index": turn_index,
            "model": model,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return CorpusItem(
        kind="chat",
        source=source,
        source_key=source_key,
        text_for_embed=text_for_embed,
        occurred_at=occurred_at,
        payload=payload,
        body_text=content,
        header_json=header_json,
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
    header_json = json.dumps(
        {
            "kind": "memory",
            "tags": tag_list,
            "confidence": confidence,
            "provenance": provenance,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return CorpusItem(
        kind="memory",
        source=source,
        source_key=source_key,
        text_for_embed=text_for_embed,
        occurred_at=occurred_at,
        payload=payload,
        tags=tag_list,
        body_text=fact,
        header_json=header_json,
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
