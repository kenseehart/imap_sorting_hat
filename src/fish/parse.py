from __future__ import annotations

import email
import json
import re
from dataclasses import dataclass
from datetime import datetime
from email.utils import parsedate_to_datetime
from hashlib import sha256
from typing import Any

import bs4

from fish.config import MAX_EMBED_BODY_CHARS

re_header_item = re.compile(r"([\w-]+): (.*)")
re_address = re.compile(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)")
re_newline = re.compile(r"[\r\n]+")
re_symbol_sequence = re.compile(r"(?<=\s)\W+(?=\s)")
re_whitespace = re.compile(r"\s+")


def html2text(html: str) -> str:
    soup = bs4.BeautifulSoup(html, "html.parser")
    return soup.get_text(separator=" ")


def mesg_to_text(mesg: email.message.Message) -> str:
    text = ""
    for part in mesg.walk():
        if part.get_content_type() == "text/plain":
            payload = part.get_payload(decode=True)
            if payload:
                text += payload.decode("utf-8", errors="ignore")
        elif part.get_content_type() == "text/html":
            payload = part.get_payload(decode=True)
            if payload:
                text += html2text(payload.decode("utf-8", errors="ignore"))
    text = re_symbol_sequence.sub("", text)
    text = re_whitespace.sub(" ", text)
    return text.strip()


def parse_addresses(header_value: str) -> list[str]:
    return [m.group(1) for m in re_address.finditer(header_value or "")]


def parse_header_fields(header_bytes: bytes) -> dict[str, str]:
    header = header_bytes.decode("utf-8", errors="ignore")
    header_dict: dict[str, str] = {}
    for item in re_newline.split(header):
        m = re_header_item.match(item)
        if m:
            key = m.group(1)
            header_dict[key] = header_dict.get(key, "") + (" " if key in header_dict else "") + m.group(2)
    return header_dict


@dataclass
class ParsedMessage:
    subject: str
    from_addrs: list[str]
    to_addrs: list[str]
    cc_addrs: list[str]
    message_id: str
    in_reply_to: str
    date: str | None
    flags: list[str]
    body_text: str
    body_for_embed: str
    content_hash: str
    gmail_labels: list[str] | None = None


def parse_raw_message(
    header_bytes: bytes,
    body_bytes: bytes,
    flags: list[str] | None = None,
) -> ParsedMessage:
    header_dict = parse_header_fields(header_bytes)
    payload = email.message_from_bytes(body_bytes)
    body_text = mesg_to_text(payload)

    subject = header_dict.get("Subject", "").removeprefix("**SPAM**").strip()
    from_addrs = parse_addresses(header_dict.get("From", ""))
    to_addrs = parse_addresses(header_dict.get("To", ""))
    cc_addrs = parse_addresses(header_dict.get("Cc", ""))
    message_id = header_dict.get("Message-ID", "").strip()
    in_reply_to = header_dict.get("In-Reply-To", "").strip()

    date_raw = header_dict.get("Date", "")
    date_iso: str | None = None
    if date_raw:
        try:
            date_iso = parsedate_to_datetime(date_raw).isoformat()
        except (TypeError, ValueError, IndexError):
            date_iso = date_raw

    from_display = ", ".join(from_addrs)
    to_display = ", ".join(to_addrs + cc_addrs)
    body_for_embed = (
        f"From: {from_display}. To: {to_display}. Subject: {subject}. {body_text}"
    )[:MAX_EMBED_BODY_CHARS]
    content_hash = sha256(body_for_embed.encode("utf-8")).hexdigest()[:16]

    return ParsedMessage(
        subject=subject,
        from_addrs=from_addrs,
        to_addrs=to_addrs,
        cc_addrs=cc_addrs,
        message_id=message_id,
        in_reply_to=in_reply_to,
        date=date_iso,
        flags=flags or [],
        body_text=body_text,
        body_for_embed=body_for_embed,
        content_hash=content_hash,
    )


def parse_indexed_message(
    header_bytes: bytes,
    text_parts: dict[str, bytes],
    flags: list[str] | None = None,
) -> ParsedMessage:
    """Parse a message indexed from IMAP text parts (no attachment download)."""
    header_dict = parse_header_fields(header_bytes)
    body_text = ""
    if "plain" in text_parts:
        body_text = text_parts["plain"].decode("utf-8", errors="ignore")
    if "html" in text_parts:
        html_text = html2text(text_parts["html"].decode("utf-8", errors="ignore"))
        body_text = f"{body_text} {html_text}".strip() if body_text else html_text
    body_text = re_symbol_sequence.sub("", body_text)
    body_text = re_whitespace.sub(" ", body_text).strip()

    subject = header_dict.get("Subject", "").removeprefix("**SPAM**").strip()
    from_addrs = parse_addresses(header_dict.get("From", ""))
    to_addrs = parse_addresses(header_dict.get("To", ""))
    cc_addrs = parse_addresses(header_dict.get("Cc", ""))
    message_id = header_dict.get("Message-ID", "").strip()
    in_reply_to = header_dict.get("In-Reply-To", "").strip()

    date_raw = header_dict.get("Date", "")
    date_iso: str | None = None
    if date_raw:
        try:
            date_iso = parsedate_to_datetime(date_raw).isoformat()
        except (TypeError, ValueError, IndexError):
            date_iso = date_raw

    from_display = ", ".join(from_addrs)
    to_display = ", ".join(to_addrs + cc_addrs)
    body_for_embed = (
        f"From: {from_display}. To: {to_display}. Subject: {subject}. {body_text}"
    )[:MAX_EMBED_BODY_CHARS]
    content_hash = sha256(body_for_embed.encode("utf-8")).hexdigest()[:16]

    return ParsedMessage(
        subject=subject,
        from_addrs=from_addrs,
        to_addrs=to_addrs,
        cc_addrs=cc_addrs,
        message_id=message_id,
        in_reply_to=in_reply_to,
        date=date_iso,
        flags=flags or [],
        body_text=body_text,
        body_for_embed=body_for_embed,
        content_hash=content_hash,
    )


def parse_fetched_message(raw: dict) -> ParsedMessage:
    if raw.get("text_parts") is not None:
        parsed = parse_indexed_message(raw["header"], raw["text_parts"], raw.get("flags", []))
    else:
        parsed = parse_raw_message(raw["header"], raw["body"], raw.get("flags", []))
    labels = raw.get("gmail_labels")
    if labels:
        parsed.gmail_labels = labels
    return parsed


def message_row_to_dict(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    for key in ("to_addrs", "cc_addrs", "flags", "gmail_labels"):
        if key in out and isinstance(out[key], str):
            try:
                out[key] = json.loads(out[key])
            except json.JSONDecodeError:
                pass
    return out
