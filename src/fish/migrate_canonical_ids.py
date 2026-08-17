"""Migrate corpus_items.source_key / messages.canonical_id to composite ids.

Keeps integer surrogate PKs (Qdrant + training_samples unchanged).
Streams in batches — never loads full email bodies into memory.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from fish.corpus import email_locator_from_payload, parse_imap_source_key
from fish.identity import (
    email_canonical_id,
    memory_canonical_id,
    parse_canonical_id,
    sms_canonical_id,
    strip_rfc_message_id,
)
from fish.store import connect, get_corpus_by_source_key


BATCH_SIZE = 2000


def _ensure_identity_schema(db) -> None:
    """Add identity columns/indexes without full init_db (avoids Qdrant round-trip)."""
    cols = {row[1] for row in db.execute("PRAGMA table_info(messages)")}
    if "gm_msgid" not in cols:
        db.execute("ALTER TABLE messages ADD COLUMN gm_msgid TEXT")
    if "canonical_id" not in cols:
        db.execute("ALTER TABLE messages ADD COLUMN canonical_id TEXT")
    db.execute(
        "CREATE INDEX IF NOT EXISTS idx_messages_canonical_id ON messages(canonical_id)"
    )
    db.execute(
        "CREATE INDEX IF NOT EXISTS idx_messages_gm_msgid ON messages(gm_msgid)"
    )


def _account_email_map(db) -> dict[int, str]:
    return {
        int(r["id"]): r["email"]
        for r in db.execute("SELECT id, email FROM accounts").fetchall()
    }


def _progress(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def migrate_canonical_ids(
    *,
    dry_run: bool = False,
    limit: int | None = None,
    batch_size: int = BATCH_SIZE,
) -> dict[str, Any]:
    """Rewrite email (and known non-email) source_keys to ``{source_id}.{message_id}``.

    Integer ``corpus_items.id`` / Qdrant point ids are not changed.
    """
    stats: dict[str, Any] = {
        "email_seen": 0,
        "email_updated": 0,
        "email_already": 0,
        "email_collision": 0,
        "email_skipped": 0,
        "sms_updated": 0,
        "memory_updated": 0,
        "messages_canonical_set": 0,
        "training_source_keys_updated": 0,
        "collisions": [],
        "dry_run": dry_run,
    }

    db = connect()
    db.execute("PRAGMA busy_timeout=60000")
    try:
        _ensure_identity_schema(db)
        db.commit()
        emails = _account_email_map(db)
        last_id = 0
        remaining = limit

        while True:
            take = batch_size
            if remaining is not None:
                if remaining <= 0:
                    break
                take = min(batch_size, remaining)

            # No body_text from corpus — only a short message body snip when needed.
            rows = db.execute(
                """
                SELECT c.id, c.source_key, c.payload, c.header_json
                FROM corpus_items c
                WHERE c.kind = 'email' AND c.id > ?
                ORDER BY c.id
                LIMIT ?
                """,
                (last_id, take),
            ).fetchall()
            if not rows:
                break

            for row in rows:
                last_id = int(row["id"])
                if remaining is not None:
                    remaining -= 1
                stats["email_seen"] += 1
                if stats["email_seen"] % 5000 == 0:
                    _progress(
                        f"migrate email_seen={stats['email_seen']} "
                        f"updated={stats['email_updated']} "
                        f"collision={stats['email_collision']} "
                        f"skipped={stats['email_skipped']}"
                    )

                old_key = row["source_key"] or ""
                if parse_canonical_id(old_key) and not old_key.startswith("imap:"):
                    stats["email_already"] += 1
                    locator = email_locator_from_payload(row["payload"])
                    if locator is None:
                        locator = parse_imap_source_key(old_key)
                    if locator and not dry_run:
                        aid, folder, uid = locator
                        cur = db.execute(
                            """
                            UPDATE messages SET canonical_id = ?
                            WHERE account_id = ? AND folder = ? AND uid = ?
                              AND (canonical_id IS NULL OR canonical_id = '')
                            """,
                            (old_key, aid, folder, uid),
                        )
                        stats["messages_canonical_set"] += cur.rowcount
                    continue

                locator = email_locator_from_payload(row["payload"])
                if locator is None:
                    locator = parse_imap_source_key(old_key)
                if locator is None:
                    stats["email_skipped"] += 1
                    continue
                account_id, folder, uid = locator
                account_email = emails.get(account_id)
                if not account_email:
                    stats["email_skipped"] += 1
                    continue

                m = db.execute(
                    """
                    SELECT message_id, gm_msgid, from_addr, date, subject
                    FROM messages
                    WHERE account_id = ? AND folder = ? AND uid = ?
                    """,
                    (account_id, folder, uid),
                ).fetchone()
                if m is None:
                    stats["email_skipped"] += 1
                    continue

                body = ""
                needs_body = not strip_rfc_message_id(m["message_id"]) and not (
                    m["gm_msgid"] or ""
                ).strip()
                if needs_body:
                    br = db.execute(
                        """
                        SELECT substr(body_text, 1, 5000) AS body_snip
                        FROM messages
                        WHERE account_id = ? AND folder = ? AND uid = ?
                        """,
                        (account_id, folder, uid),
                    ).fetchone()
                    body = (br["body_snip"] if br else "") or ""

                try:
                    new_key, synthetic = email_canonical_id(
                        account_email=account_email,
                        rfc_message_id=m["message_id"],
                        gm_msgid=m["gm_msgid"],
                        from_addr=m["from_addr"] or "",
                        date=m["date"],
                        subject=m["subject"] or "",
                        body=body,
                        allow_gmail_rfc_fallback=True,
                    )
                except ValueError:
                    stats["email_skipped"] += 1
                    continue

                if new_key == old_key:
                    stats["email_already"] += 1
                    continue

                other = get_corpus_by_source_key(db, new_key)
                if other is not None and int(other["id"]) != int(row["id"]):
                    stats["email_collision"] += 1
                    if len(stats["collisions"]) < 50:
                        stats["collisions"].append(
                            {
                                "old_id": int(row["id"]),
                                "old_key": old_key,
                                "new_key": new_key,
                                "conflict_id": int(other["id"]),
                            }
                        )
                    continue

                if dry_run:
                    stats["email_updated"] += 1
                    continue

                payload: dict[str, Any] = {}
                if row["payload"]:
                    try:
                        payload = json.loads(row["payload"])
                    except json.JSONDecodeError:
                        payload = {}
                payload["canonical_id"] = new_key
                payload["synthetic"] = synthetic
                payload["legacy_source_key"] = old_key
                payload.setdefault("account_id", account_id)
                payload.setdefault("folder", folder)
                payload.setdefault("uid", uid)
                payload.setdefault("account_email", account_email)

                header: dict[str, Any] = {}
                if row["header_json"]:
                    try:
                        header = json.loads(row["header_json"])
                    except json.JSONDecodeError:
                        header = {}
                if isinstance(header, dict):
                    header["canonical_id"] = new_key
                    header["synthetic"] = synthetic

                db.execute(
                    """
                    UPDATE corpus_items
                    SET source_key = ?, payload = ?, header_json = ?
                    WHERE id = ?
                    """,
                    (
                        new_key,
                        json.dumps(payload),
                        json.dumps(header, sort_keys=True, ensure_ascii=False)
                        if header
                        else row["header_json"],
                        int(row["id"]),
                    ),
                )
                db.execute(
                    """
                    UPDATE messages
                    SET canonical_id = ?
                    WHERE account_id = ? AND folder = ? AND uid = ?
                    """,
                    (new_key, account_id, folder, uid),
                )
                cur = db.execute(
                    "UPDATE training_samples SET source_key = ? WHERE corpus_item_id = ?",
                    (new_key, int(row["id"])),
                )
                stats["training_source_keys_updated"] += cur.rowcount
                stats["email_updated"] += 1
                stats["messages_canonical_set"] += 1

            if not dry_run:
                db.commit()
                _progress(
                    f"committed through id={last_id} "
                    f"updated={stats['email_updated']} seen={stats['email_seen']}"
                )

            if limit is not None and remaining is not None and remaining <= 0:
                break
            if len(rows) < take:
                break

        # SMS: android_sms:* → smskn.*
        for row in db.execute(
            """
            SELECT id, source_key, payload, body_text FROM corpus_items
            WHERE kind = 'sms' AND source_key LIKE 'android_sms:%'
            """
        ):
            payload = {}
            if row["payload"]:
                try:
                    payload = json.loads(row["payload"])
                except json.JSONDecodeError:
                    payload = {}
            new_key = sms_canonical_id(
                sms_id=payload.get("sms_id") or row["source_key"].split(":", 1)[-1],
                address=str(payload.get("phone") or ""),
                date=None,
                body=row["body_text"] or "",
            )
            if dry_run:
                stats["sms_updated"] += 1
                continue
            other = get_corpus_by_source_key(db, new_key)
            if other and int(other["id"]) != int(row["id"]):
                continue
            payload["canonical_id"] = new_key
            payload["legacy_source_key"] = row["source_key"]
            db.execute(
                "UPDATE corpus_items SET source_key = ?, payload = ? WHERE id = ?",
                (new_key, json.dumps(payload), int(row["id"])),
            )
            db.execute(
                "UPDATE training_samples SET source_key = ? WHERE corpus_item_id = ?",
                (new_key, int(row["id"])),
            )
            stats["sms_updated"] += 1

        # Memory: memory:* → memag.*
        for row in db.execute(
            """
            SELECT id, source_key, payload, body_text FROM corpus_items
            WHERE kind = 'memory' AND source_key LIKE 'memory:%'
            """
        ):
            payload = {}
            if row["payload"]:
                try:
                    payload = json.loads(row["payload"])
                except json.JSONDecodeError:
                    payload = {}
            fact = (payload.get("fact") or row["body_text"] or "").strip()
            if not fact:
                continue
            new_key = memory_canonical_id(fact)
            if dry_run:
                stats["memory_updated"] += 1
                continue
            other = get_corpus_by_source_key(db, new_key)
            if other and int(other["id"]) != int(row["id"]):
                continue
            payload["canonical_id"] = new_key
            payload["legacy_source_key"] = row["source_key"]
            db.execute(
                "UPDATE corpus_items SET source_key = ?, payload = ? WHERE id = ?",
                (new_key, json.dumps(payload), int(row["id"])),
            )
            db.execute(
                "UPDATE training_samples SET source_key = ? WHERE corpus_item_id = ?",
                (new_key, int(row["id"])),
            )
            stats["memory_updated"] += 1

        if not dry_run:
            db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    return stats
