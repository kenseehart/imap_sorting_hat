from __future__ import annotations

import json
from typing import Any

from fish.accounts import account_by_id
from fish.imap_client import imap_session, move_message, set_flags
from fish.store import (
    db_conn,
    get_message_by_id,
    init_db,
    update_message_flags,
    update_message_folder,
)


def _message_account(msg: dict[str, Any]):
    account = account_by_id(int(msg["account_id"]))
    if not account:
        raise ValueError(f"No account for message {msg['id']}")
    return account


def message_move(message_id: int, target_folder: str) -> dict[str, Any]:
    init_db()
    with db_conn() as db:
        msg = get_message_by_id(db, message_id)
        if not msg:
            raise ValueError(f"Message {message_id} not found")
        account = _message_account(msg)
        with imap_session(account) as client:
            move_message(client, msg["folder"], int(msg["uid"]), target_folder)
        update_message_folder(db, message_id, target_folder)
    return {"ok": True, "message_id": message_id, "folder": target_folder}


def message_archive(message_id: int) -> dict[str, Any]:
    init_db()
    with db_conn() as db:
        msg = get_message_by_id(db, message_id)
        if not msg:
            raise ValueError(f"Message {message_id} not found")
        archive_folder = _message_account(msg).archive_folder
    return message_move(message_id, archive_folder)


def message_flag(message_id: int, flag: str = "\\Flagged", add: bool = True) -> dict[str, Any]:
    init_db()
    with db_conn() as db:
        msg = get_message_by_id(db, message_id)
        if not msg:
            raise ValueError(f"Message {message_id} not found")
        account = _message_account(msg)
        with imap_session(account) as client:
            set_flags(client, msg["folder"], int(msg["uid"]), [flag], add=add)
        flags = json.loads(msg.get("flags") or "[]")
        if add and flag not in flags:
            flags.append(flag)
        elif not add and flag in flags:
            flags.remove(flag)
        update_message_flags(db, message_id, flags)
    return {"ok": True, "message_id": message_id, "flags": flags}
