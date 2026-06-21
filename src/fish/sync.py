from __future__ import annotations

import logging
from datetime import date

from cmdline.progress import progress_bar, progress_session, progress_write
from openai import AuthenticationError

from fish.accounts import Account, ignore_folders_for_account, load_accounts
from fish.config import DEFAULT_SYNC_DAYS, ensure_openai_api_key
from fish.embed import embed_texts, reset_client
from fish.prism.inference import adapt_chunk_embedding
from fish.imap_client import (
    ResilientImap,
    fetch_folder_messages,
    folder_uidvalidity,
    list_folders,
    search_uids_since,
    search_uids_since_date,
    short_imap_error,
)
from fish.parse import parse_fetched_message, parse_raw_message
from fish.store import (
    count_corpus_needing_embedding,
    corpus_needing_embedding,
    db_conn,
    init_db,
    set_corpus_embedding,
    update_sync_state,
    upsert_account,
    upsert_message,
)

logger = logging.getLogger(__name__)

EMBED_BATCH = 100


def embed_pending(batch_size: int = EMBED_BATCH, *, auth_retry: bool = True) -> int:
    init_db()
    embedded = 0
    try:
        with db_conn() as db:
            pending = corpus_needing_embedding(db, limit=batch_size)
            if not pending:
                return 0
            texts = [row["text_for_embed"] for row in pending]
            vectors = embed_texts(texts)
            for row, vector in zip(pending, vectors):
                adapted = adapt_chunk_embedding(vector)
                set_corpus_embedding(db, int(row["id"]), adapted)
                embedded += 1
    except AuthenticationError:
        if not auth_retry:
            raise
        progress_write("OpenAI API key rejected — please re-enter.")
        reset_client()
        ensure_openai_api_key(interactive=True, force=True)
        return embed_pending(batch_size=batch_size, auth_retry=False)
    return embedded


def embed_all_pending(*, show_progress: bool = True) -> int:
    init_db()
    with db_conn() as db:
        total = count_corpus_needing_embedding(db)
    if total == 0:
        return 0

    embedded = 0
    bar = progress_bar(
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

        msg_bar = progress_bar(
            total=len(uids),
            desc=f"  {folder[:32]}",
            unit="msg",
            leave=False,
            disable=not show_progress or len(uids) == 0,
        )

        def on_batch(fetched: dict[int, dict]) -> None:
            with db_conn() as db:
                for uid, raw in fetched.items():
                    parsed = parse_fetched_message(raw)
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
            gmail=account.is_gmail,
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
    skip = ignore_folders_for_account(account)
    since_label = since.isoformat() if since else f"{days}d"

    try:
        all_folders = folders or list_folders(account)
    except Exception as exc:
        progress_write(f"ERROR {account.email}: cannot list folders — {short_imap_error(exc)}")
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

    folder_bar = progress_bar(
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
            progress_write(f"WARN {account.email} / {folder}: {err}")
            stats["folders"][folder] = {"error": err}

    stats["embedded"] = embed_all_pending(show_progress=show_progress)
    return stats


def sync_all(
    days: int = DEFAULT_SYNC_DAYS,
    since: date | None = None,
    *,
    account: str | None = None,
    show_progress: bool = True,
) -> list[dict]:
    results = []
    accounts = load_accounts()
    if account:
        accounts = [a for a in accounts if a.email.lower() == account.lower()]
        if not accounts:
            raise ValueError(f"No matching account: {account}")

    with progress_session(disable=not show_progress):
        account_bar = progress_bar(
            accounts,
            desc="accounts",
            unit="acct",
            disable=not show_progress or not accounts,
        )
        for acct in account_bar:
            account_bar.set_postfix_str(acct.email, refresh=False)
            results.append(
                sync_account(acct, days=days, since=since, show_progress=show_progress)
            )
    return results
