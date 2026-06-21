from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fish.corpus import PHONE_FILTER_DEFAULT, normalize_phone, sms_corpus_item
from fish.store import db_conn, get_corpus_by_source_key, init_db, upsert_corpus_item


def _sms_timestamp_ms(raw: str) -> str | None:
    try:
        ms = int(raw)
        return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat()
    except (TypeError, ValueError):
        return None


def _direction(sms_type: str) -> str:
    return "in" if sms_type == "1" else "out"


def import_android_sms(
    path: Path,
    *,
    phone_filter: str = PHONE_FILTER_DEFAULT,
    dry_run: bool = False,
) -> dict[str, Any]:
    init_db()
    target = normalize_phone(phone_filter)
    tree = ET.parse(path)
    root = tree.getroot()
    stats = {"seen": 0, "matched": 0, "inserted": 0, "updated": 0, "skipped": 0}

    with db_conn() as db:
        for elem in root.findall("sms"):
            stats["seen"] += 1
            address = elem.get("address") or ""
            if normalize_phone(address) != target:
                continue
            stats["matched"] += 1
            body = elem.get("body") or ""
            sms_id = elem.get("_id") or elem.get("id")
            source_key = f"android_sms:{sms_id or hash((address, elem.get('date'), body))}"
            item = sms_corpus_item(
                source_key=source_key,
                phone=address,
                direction=_direction(elem.get("type") or "1"),
                body=body,
                occurred_at=_sms_timestamp_ms(elem.get("date") or ""),
                contact_name=elem.get("contact_name"),
                sms_id=sms_id,
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
