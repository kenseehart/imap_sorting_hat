"""Canonical message identity: ``{source_id}.{message_id}``.

See ``docs/identity.md``. Integers are never cross-source identities; ingest
dedups on this composite string (stored as ``corpus_items.source_key``).
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

# 5-char source codes (account + platform).
SOURCE_EMKSC = "emksc"  # email: ken@seehart.com
SOURCE_EMKAG = "emkag"  # email: ken@agi.green
SOURCE_EMKGC = "emkgc"  # email: kenseehart@gmail.com
SOURCE_CCKAG = "cckag"  # claude_code: ken@agi.green
SOURCE_CUKSC = "cuksc"  # cursor: ken@seehart.com (format TBD — see docs)
SOURCE_SMSKN = "smskn"  # android SMS (default personal filter)
SOURCE_MEMAG = "memag"  # agent-written memory

EMAIL_SOURCE_BY_ADDRESS: dict[str, str] = {
    "ken@seehart.com": SOURCE_EMKSC,
    "ken@agi.green": SOURCE_EMKAG,
    "kenseehart@gmail.com": SOURCE_EMKGC,
}

GMAIL_SOURCES = frozenset({SOURCE_EMKGC})
DOVECOT_SOURCES = frozenset({SOURCE_EMKSC, SOURCE_EMKAG})

# Back-compat aliases (pre-freeze typos / 6-char drafts)
SOURCE_SMSKEN = SOURCE_SMSKN
SOURCE_MEMAGN = SOURCE_MEMAG

SYNTHETIC_BODY_BYTES = 4096
SOURCE_ID_RE = re.compile(r"^[a-z0-9]{5}$")


def canonical_id(source_id: str, message_id: str) -> str:
    """Build ``{source_id}.{message_id}`` (message_id must not be empty)."""
    sid = (source_id or "").strip().lower()
    mid = (message_id or "").strip()
    if not SOURCE_ID_RE.match(sid):
        raise ValueError(f"source_id must be 5 alphanumeric chars, got {source_id!r}")
    if not mid:
        raise ValueError("message_id must be non-empty")
    if mid.startswith("."):
        raise ValueError(f"message_id must not start with '.': {message_id!r}")
    return f"{sid}.{mid}"


def parse_canonical_id(value: str) -> tuple[str, str] | None:
    """Split ``source_id.message_id`` (message_id may contain ``.``)."""
    if not value or "." not in value:
        return None
    sid, mid = value.split(".", 1)
    if not SOURCE_ID_RE.match(sid) or not mid:
        return None
    return sid, mid


def source_id_for_email(account_email: str | None) -> str:
    email = (account_email or "").strip().lower()
    if not email:
        raise ValueError("account_email is required to derive email source_id")
    sid = EMAIL_SOURCE_BY_ADDRESS.get(email)
    if sid is None:
        raise ValueError(
            f"No source_id mapping for {account_email!r}. "
            f"Known: {sorted(EMAIL_SOURCE_BY_ADDRESS)}"
        )
    return sid


def strip_rfc_message_id(raw: str | None) -> str:
    """Normalize RFC 5322 Message-ID: strip whitespace and angle brackets."""
    s = (raw or "").strip()
    if s.startswith("<") and s.endswith(">"):
        s = s[1:-1].strip()
    return s


def synthetic_email_message_id(
    *,
    from_addr: str,
    date: str | None,
    subject: str,
    body: str,
) -> str:
    """Fallback when Message-ID is missing (emksc/emkag only).

    ``syn:`` + sha256(from|date|subject|body[:4096])[:32] — not re-fetchable
    from IMAP; callers must set ``synthetic: true`` in payload.
    """
    body_bytes = (body or "").encode("utf-8")[:SYNTHETIC_BODY_BYTES]
    material = "|".join(
        [
            (from_addr or "").strip().lower(),
            (date or "").strip(),
            (subject or "").strip(),
            body_bytes.decode("utf-8", errors="replace"),
        ]
    )
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]
    return f"syn:{digest}"


def is_synthetic_message_id(message_id: str) -> bool:
    return (message_id or "").startswith("syn:")


def email_message_id_parts(
    *,
    account_email: str,
    rfc_message_id: str | None,
    gm_msgid: str | int | None = None,
    from_addr: str = "",
    date: str | None = None,
    subject: str = "",
    body: str = "",
    allow_gmail_rfc_fallback: bool = False,
) -> tuple[str, str, bool]:
    """Return ``(source_id, message_id, synthetic)`` for an email."""
    source_id = source_id_for_email(account_email)
    if source_id in GMAIL_SOURCES:
        gm = str(gm_msgid).strip() if gm_msgid is not None else ""
        if gm:
            return source_id, gm, False
        if not allow_gmail_rfc_fallback:
            raise ValueError(
                f"Gmail account {account_email!r} requires X-GM-MSGID for canonical id"
            )
        # Backfill only: use RFC Message-ID or synthetic until re-sync fills gm_msgid.
        rfc = strip_rfc_message_id(rfc_message_id)
        if rfc:
            return source_id, f"rfc:{rfc}", False
        syn = synthetic_email_message_id(
            from_addr=from_addr, date=date, subject=subject, body=body
        )
        return source_id, syn, True

    rfc = strip_rfc_message_id(rfc_message_id)
    if rfc:
        return source_id, rfc, False
    syn = synthetic_email_message_id(
        from_addr=from_addr, date=date, subject=subject, body=body
    )
    return source_id, syn, True


def email_canonical_id(
    *,
    account_email: str,
    rfc_message_id: str | None,
    gm_msgid: str | int | None = None,
    from_addr: str = "",
    date: str | None = None,
    subject: str = "",
    body: str = "",
    allow_gmail_rfc_fallback: bool = False,
) -> tuple[str, bool]:
    """Return ``(canonical_id, synthetic)`` for an email message."""
    source_id, mid, synthetic = email_message_id_parts(
        account_email=account_email,
        rfc_message_id=rfc_message_id,
        gm_msgid=gm_msgid,
        from_addr=from_addr,
        date=date,
        subject=subject,
        body=body,
        allow_gmail_rfc_fallback=allow_gmail_rfc_fallback,
    )
    return canonical_id(source_id, mid), synthetic


def sms_canonical_id(*, sms_id: str | None, address: str, date: str | None, body: str) -> str:
    """Android SMS → ``smskn.{native_id|hash}``."""
    if sms_id:
        mid = str(sms_id).strip()
    else:
        material = f"{address}|{date or ''}|{body or ''}"
        mid = "syn:" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]
    return canonical_id(SOURCE_SMSKN, mid)


def claude_code_canonical_id(session_uuid: str, message_uuid: str) -> str:
    """Claude Code → ``cckag.{session_uuid}:{message_uuid}``."""
    sess = (session_uuid or "").strip()
    msg = (message_uuid or "").strip()
    if not sess or not msg:
        raise ValueError("session_uuid and message_uuid are required")
    return canonical_id(SOURCE_CCKAG, f"{sess}:{msg}")


def memory_canonical_id(fact: str) -> str:
    """Agent memory → ``memag.{sha256(fact.lower())[:24]}``."""
    digest = hashlib.sha256((fact or "").lower().encode("utf-8")).hexdigest()[:24]
    return canonical_id(SOURCE_MEMAG, digest)


def cursor_canonical_id(_composer_id: str, _message_id: str) -> str:
    """Cursor ``cuksc`` — format not frozen yet (see docs/identity.md)."""
    raise NotImplementedError(
        "cuksc message_id format is pending Cursor storage schema inspection "
        "(docs/identity.md open item)"
    )


def normalize_gm_msgid(raw: Any) -> str | None:
    """Decode IMAP ``X-GM-MSGID`` (int / bytes / str) to a decimal string."""
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    if isinstance(raw, (list, tuple)) and raw:
        return normalize_gm_msgid(raw[0])
    s = str(raw).strip()
    if not s:
        return None
    # Some servers return parenthesized form
    s = s.strip("()")
    if not s.isdigit():
        # still accept as opaque token if alphanumeric
        if not re.fullmatch(r"[0-9A-Za-z]+", s):
            return None
    return s
