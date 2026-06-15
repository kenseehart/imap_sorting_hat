from __future__ import annotations

import logging
from datetime import date

from openai import AuthenticationError
from tqdm import tqdm

from fish.accounts import Account, ignore_folders, load_accounts
from fish.config import DEFAULT_SYNC_DAYS, ensure_openai_api_key
from fish.embed import embed_texts, reset_client
from fish.imap_client import (
    ResilientImap,
    fetch_folder_messages,
    folder_uidvalidity,
    list_folders,
    search_uids_since,
    search_uids_since_date,
    short_imap_error,
)
from fish.parse import parse_raw_message
from fish.store import (
    count_messages_needing_embedding,
    db_conn,
    init_db,
    messages_needing_embedding,
    set_embedding,
    update_sync_state,
    upsert_account,
    upsert_message,
)

logger = logging.getLogger(__name__)

EMBED_BATCH = 50


def embed_pending(batch_size: int = EMBED_BATCH, *, auth_retry: bool = True) -> int:
    init_db()
    embedded = 0
    try:
        with db_conn() as db:
            pending = messages_needing_embedding(db, limit=batch_size)
            if not pending:
                return 0
            texts = [row["body_for_embed"] for row in pending]
            vectors = embed_texts(texts)
            for row, vector in zip(pending, vectors):
                set_embedding(db, int(row["id"]), vector)
                embedded += 1
    except AuthenticationError:
        if not auth_retry:
            raise
        tqdm.write("OpenAI API key rejected — please re-enter.")
        reset_client()
        ensure_openai_api_key(interactive=True, force=True)
        return embed_pending(batch_size=batch_size, auth_retry=False)
    return embedded


def embed_all_pending(*, show_progress: bool = True) -> int:
    init_db()
    with db_conn() as db:
        total = count_messages_needing_embedding(db)
    if total == 0:
        return 0

    embedded = 0
    bar = tqdm(
        total=total,
        desc="embedding",
        unit="msg",
        disable=not show_progress,
    )
    while True:
        count = embed_pending(batch_size=EMBED_BATCH)
        embedded += count
        bar.update(count)
        if count < EMBED_BATCH:
            break
    bar.close()
    return embedded


def _sync_one_folder(
    account: Account,
    folder: str,
    account_db_id: int,
    days: int,
    since: date | None,
    since_label: str,
    stats: dict,
    *,
    show_progress: bool,
) -> dict:
    folder_stats: dict = {"uids": 0, "stored": 0}
    imap = ResilientImap(account)
    try:
        if since:
            uids = imap.with_retry(
                lambda c: search_uids_since_date(c, folder, since), folder=folder
            )
        else:
            uids = imap.with_retry(
                lambda c: search_uids_since(c, folder, days), folder=folder
            )
        folder_stats["uids"] = len(uids)
        uidvalidity = imap.with_retry(
            lambda c: folder_uidvalidity(c, folder), folder=folder
        )

        msg_bar = tqdm(
            total=len(uids),
            desc=f"  {folder[:32]}",
            unit="msg",
            leave=False,
            disable=not show_progress or len(uids) == 0,
        )

        def on_batch(fetched: dict[int, dict]) -> None:
            with db_conn() as db:
                for uid, raw in fetched.items():
                    parsed = parse_raw_message(
                        raw["header"], raw["body"], raw.get("flags", [])
                    )
                    _msg_id, changed = upsert_message(
                        db, account_db_id, folder, int(uid), parsed
                    )
                    if changed:
                        stats["new_or_changed"] += 1
                    folder_stats["stored"] += 1

        fetch_folder_messages(
            imap,
            folder,
            uids,
            on_batch=on_batch,
            progress_cb=msg_bar.update,
        )
        msg_bar.close()

        with db_conn() as db:
            update_sync_state(
                db,
                account_db_id,
                folder,
                uidvalidity,
                max(uids) if uids else None,
                since_label,
            )
    finally:
        imap.close()

    return folder_stats


def sync_account(
    account: Account,
    days: int = DEFAULT_SYNC_DAYS,
    since: date | None = None,
    folders: list[str] | None = None,
    *,
    show_progress: bool = True,
) -> dict:
    init_db()
    stats = {"account": account.email, "folders": {}, "fetched": 0, "new_or_changed": 0, "embedded": 0}
    skip = ignore_folders()
    since_label = since.isoformat() if since else f"{days}d"

    try:
        all_folders = folders or list_folders(account)
    except Exception as exc:
        tqdm.write(f"ERROR {account.email}: cannot list folders — {short_imap_error(exc)}")
        stats["error"] = str(exc)
        return stats

    target_folders = [f for f in all_folders if f not in skip]

    with db_conn() as db:
        account_db_id = upsert_account(
            db,
            account.id or 0,
            account.email,
            account.imap_host,
            account.smtp_host,
            account.username,
            account.archive_folder,
        )

    folder_bar = tqdm(
        target_folders,
        desc=account.email,
        unit="folder",
        disable=not show_progress,
    )
    for folder in folder_bar:
        folder_bar.set_postfix_str(folder[:36], refresh=False)
        try:
            folder_stats = _sync_one_folder(
                account,
                folder,
                account_db_id,
                days,
                since,
                since_label,
                stats,
                show_progress=show_progress,
            )
            stats["fetched"] += folder_stats["uids"]
            stats["folders"][folder] = folder_stats
        except Exception as exc:
            err = short_imap_error(exc)
            logger.warning("Sync failed for %s %s: %s", account.email, folder, err)
            tqdm.write(f"WARN {account.email} / {folder}: {err}")
            stats["folders"][folder] = {"error": err}

    stats["embedded"] = embed_all_pending(show_progress=show_progress)
    return stats


def sync_all(
    days: int = DEFAULT_SYNC_DAYS,
    since: date | None = None,
    *,
    show_progress: bool = True,
) -> list[dict]:
    results = []
    accounts = load_accounts()
    account_bar = tqdm(
        accounts,
        desc="accounts",
        unit="acct",
        disable=not show_progress or not accounts,
    )
    for account in account_bar:
        account_bar.set_postfix_str(account.email, refresh=False)
        results.append(
            sync_account(account, days=days, since=since, show_progress=show_progress)
        )
    return results
