from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from fish.accounts import account_by_id, load_accounts
from fish.config import ACTIONS_LOG
from fish.imap_client import delete_message, imap_session, move_messages, set_flags
from fish.search import search_messages
from fish.store import db_conn, delete_message as db_delete_message, get_message_by_id, init_db, update_message_flags, update_message_folder


def _log_action(action: str, details: dict) -> None:
    ACTIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "at": datetime.now(timezone.utc).isoformat(),
        "action": action,
        **details,
    }
    with ACTIONS_LOG.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def bulk_action(
    query: str,
    action: str,
    target_folder: str | None = None,
    dry_run: bool = True,
    limit: int = 100,
    account_email: str | None = None,
) -> dict[str, Any]:
    init_db()
    matches = search_messages(query, account_email=account_email, limit=limit)
    preview = [
        {
            "id": m["id"],
            "account_email": m.get("account_email"),
            "folder": m["folder"],
            "uid": m["uid"],
            "subject": m["subject"],
            "from_addr": m["from_addr"],
            "date": m["date"],
        }
        for m in matches
    ]

    if dry_run:
        return {
            "dry_run": True,
            "action": action,
            "match_count": len(preview),
            "matches": preview,
        }

    results: list[dict[str, Any]] = []
    by_account_folder: dict[tuple[str, str], list[tuple[int, int]]] = {}

    with db_conn() as db:
        for match in matches:
            msg = get_message_by_id(db, int(match["id"]))
            if not msg:
                continue
            account = account_by_id(int(msg["account_id"]))
            if not account:
                continue
            key = (account.email, msg["folder"])
            by_account_folder.setdefault(key, []).append((int(msg["id"]), int(msg["uid"])))

    for (email, folder), items in by_account_folder.items():
        account = next(a for a in load_accounts() if a.email == email)
        uids = [uid for _, uid in items]
        msg_ids = [mid for mid, _ in items]
        try:
            with imap_session(account) as client:
                if action == "archive":
                    dest = account.archive_folder
                    move_messages(client, folder, uids, dest)
                    with db_conn() as db:
                        for msg_id in msg_ids:
                            update_message_folder(db, msg_id, dest)
                elif action == "move":
                    if not target_folder:
                        raise ValueError("target_folder required for move action")
                    move_messages(client, folder, uids, target_folder)
                    with db_conn() as db:
                        for msg_id in msg_ids:
                            update_message_folder(db, msg_id, target_folder)
                elif action == "flag":
                    with db_conn() as db:
                        for msg_id, uid in items:
                            set_flags(client, folder, uid, [r"\Flagged"], add=True)
                            row = get_message_by_id(db, msg_id)
                            flags = json.loads(row.get("flags") or "[]") if row else []
                            if "\\Flagged" not in flags:
                                flags.append("\\Flagged")
                            update_message_flags(db, msg_id, flags)
                elif action == "delete":
                    for msg_id, uid in items:
                        delete_message(client, folder, uid)
                        with db_conn() as db:
                            db_delete_message(db, msg_id)
                else:
                    raise ValueError(f"Unknown action: {action}")

            _log_action(
                action,
                {
                    "account": email,
                    "folder": folder,
                    "uids": uids,
                    "query": query,
                },
            )
            results.append({"account": email, "folder": folder, "count": len(uids), "ok": True})
        except Exception as exc:
            results.append(
                {"account": email, "folder": folder, "count": len(uids), "ok": False, "error": str(exc)}
            )

    return {
        "dry_run": False,
        "action": action,
        "match_count": len(preview),
        "results": results,
    }
