from __future__ import annotations

import logging
from datetime import date

from cmdline.progress import progress_bar, progress_session, progress_write
from openai import AuthenticationError

from fish.accounts import Account, ignore_folders_for_account, load_accounts
from fish.config import DEFAULT_SYNC_DAYS, ensure_openai_api_key
from fish.write_lock import fish_write_lock
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
from fish.parse import parse_fetched_message
from fish.store import (
    count_corpus_needing_embedding,
    corpus_needing_embedding,
    db_conn,
    get_sync_state,
    init_db,
    set_corpus_embedding,
    update_sync_state,
    upsert_account,
    upsert_message,
)

logger = logging.getLogger(__name__)

EMBED_BATCH = 100


def embed_pending(
    batch_size: int = EMBED_BATCH,
    *,
    auth_retry: bool = True,
    kinds: list[str] | None = None,
    like: list[str] | None = None,
    since: str | None = None,
) -> int:
    init_db()
    embedded = 0
    try:
        with db_conn() as db:
            pending = corpus_needing_embedding(
                db, limit=batch_size, kinds=kinds, like=like, since=since
            )
            if not pending:
                return 0
            # Combined + header + body: durable OpenAI vectors in SQLite.
            # Only the combined vector is also upserted to Qdrant (ANN).
            texts: list[str] = []
            slots: list[tuple[int, str]] = []  # (pending_index, kind)
            for i, row in enumerate(pending):
                texts.append(row["text_for_embed"] or "")
                slots.append((i, "combined"))
                header = (row.get("header_json") or "").strip()
                body = (row.get("body_text") or "").strip()
                texts.append(header if header else (row["text_for_embed"] or ""))
                slots.append((i, "header"))
                texts.append(body if body else (row["text_for_embed"] or ""))
                slots.append((i, "body"))
            vectors = embed_texts(texts)
            by_item: dict[int, dict[str, list[float]]] = {}
            for (i, kind), vector in zip(slots, vectors):
                by_item.setdefault(i, {})[kind] = vector
            from fish.config import active_prism_model_id
            from fish.prism.configs import LEGACY_MODEL_ID
            from fish.prism.inference import adapt_chunk_for_model

            for i, row in enumerate(pending):
                parts = by_item[i]
                combined = parts["combined"]
                model_vecs: dict[str, list[float]] = {}
                mid = active_prism_model_id()
                if mid and mid != LEGACY_MODEL_ID:
                    model_vecs[mid] = adapt_chunk_for_model(combined, mid)
                set_corpus_embedding(
                    db,
                    int(row["id"]),
                    combined,
                    header_embedding=parts.get("header"),
                    body_embedding=parts.get("body"),
                    model_embeddings=model_vecs,
                )
                embedded += 1
    except AuthenticationError:
        if not auth_retry:
            raise
        progress_write("OpenAI API key rejected — please re-enter.")
        reset_client()
        ensure_openai_api_key(interactive=True, force=True)
        return embed_pending(
            batch_size=batch_size,
            auth_retry=False,
            kinds=kinds,
            like=like,
            since=since,
        )
    return embedded


def embed_field_pending(
    batch_size: int = EMBED_BATCH, *, training_only: bool = False
) -> int:
    """Backfill header/body raw embeddings for rows that already have combined.

    Writes to SQLite only — does not touch Qdrant (field vectors are not ANN indexes).
    OpenAI calls run outside the DB connection / write lock.
    """
    from fish.store import corpus_needing_field_embeddings, set_raw_field_embeddings
    from fish.write_lock import fish_write_lock

    init_db()
    with fish_write_lock("embed-fields"):
        with db_conn() as db:
            pending = corpus_needing_field_embeddings(
                db, limit=batch_size, training_only=training_only
            )
            if not pending:
                return 0
            # Materialize row dicts before releasing the lock for OpenAI.
            pending = [dict(r) for r in pending]

    texts: list[str] = []
    slots: list[tuple[int, str]] = []
    for i, row in enumerate(pending):
        header = (row.get("header_json") or "").strip()
        body = (row.get("body_text") or "").strip()
        fallback = row.get("text_for_embed") or ""
        if row.get("header_embedding") is None:
            texts.append(header if header else fallback)
            slots.append((i, "header"))
        if row.get("body_embedding") is None:
            texts.append(body if body else fallback)
            slots.append((i, "body"))
    if not texts:
        return 0
    vectors = embed_texts(texts)
    by_item: dict[int, dict[str, list[float]]] = {}
    for (i, kind), vector in zip(slots, vectors):
        by_item.setdefault(i, {})[kind] = vector

    with fish_write_lock("embed-fields"):
        with db_conn() as db:
            n = 0
            for i, row in enumerate(pending):
                parts = by_item.get(i) or {}
                if not parts:
                    continue
                set_raw_field_embeddings(
                    db,
                    int(row["id"]),
                    header_embedding=parts.get("header"),
                    body_embedding=parts.get("body"),
                )
                n += 1
    return n


def embed_all_pending(
    *,
    show_progress: bool = True,
    max_messages: int | None = None,
    kinds: list[str] | None = None,
    like: list[str] | None = None,
    since: str | None = None,
) -> int:
    init_db()
    # Never COUNT(*) against a huge legacy vec table when scoped — that hangs.
    # Unfiltered full-corpus embeds still need a total for the progress bar.
    scoped = bool(kinds or like or since or max_messages is not None)
    if scoped:
        total = max_messages if max_messages is not None else 0
    else:
        with db_conn() as db:
            total = count_corpus_needing_embedding(db)
        if total == 0:
            return 0

    embedded = 0
    bar = progress_bar(
        total=total or None,
        desc="embedding",
        unit="msg",
        disable=not show_progress,
    )
    while True:
        remaining = None if max_messages is None else max(0, max_messages - embedded)
        if remaining == 0:
            break
        batch = EMBED_BATCH if remaining is None else min(EMBED_BATCH, remaining)
        count = embed_pending(
            batch_size=batch, kinds=kinds, like=like, since=since
        )
        embedded += count
        bar.update(count)
        if count < batch:
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
    incremental: bool = True,
) -> dict:
    folder_stats: dict = {"uids": 0, "stored": 0, "skipped_existing": 0}
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
        uidvalidity = imap.with_retry(
            lambda c: folder_uidvalidity(c, folder), folder=folder
        )

        prev_last_uid: int | None = None
        if incremental:
            with db_conn() as db:
                state = get_sync_state(db, account_db_id, folder)
            if (
                state
                and state.get("uidvalidity") is not None
                and uidvalidity is not None
                and int(state["uidvalidity"]) == int(uidvalidity)
                and state.get("last_uid") is not None
            ):
                prev_last_uid = int(state["last_uid"])
                before = len(uids)
                uids = [u for u in uids if int(u) > prev_last_uid]
                folder_stats["skipped_existing"] = before - len(uids)

        folder_stats["uids"] = len(uids)

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

        new_last_uid = max((int(u) for u in uids), default=prev_last_uid)
        with db_conn() as db:
            update_sync_state(
                db,
                account_db_id,
                folder,
                uidvalidity,
                new_last_uid,
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
    incremental: bool = True,
    embed_budget: int | None = None,
) -> dict:
    init_db()
    stats = {
        "account": account.email,
        "folders": {},
        "fetched": 0,
        "new_or_changed": 0,
        "embedded": 0,
        "skipped_existing": 0,
    }
    skip = ignore_folders_for_account(account)
    since_label = since.isoformat() if since else f"{days}d"

    try:
        # Surface missing credentials before IMAP work.
        _ = account.password
        all_folders = folders or list_folders(account)
    except Exception as exc:
        err = short_imap_error(exc) if not isinstance(exc, RuntimeError) else str(exc)
        progress_write(f"ERROR {account.email}: {err}")
        stats["error"] = err
        stats["auth_error"] = (
            "No password stored" in err
            or "authentication" in err.lower()
            or "authenticate" in err.lower()
            or "invalid credentials" in err.lower()
        )
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
                incremental=incremental,
            )
            stats["fetched"] += folder_stats["uids"]
            stats["skipped_existing"] += folder_stats.get("skipped_existing", 0)
            stats["folders"][folder] = folder_stats
        except Exception as exc:
            err = short_imap_error(exc)
            logger.warning("Sync failed for %s %s: %s", account.email, folder, err)
            progress_write(f"WARN {account.email} / {folder}: {err}")
            folder_err = {"error": err}
            if (
                "authentication" in err.lower()
                or "authenticate" in err.lower()
                or "invalid credentials" in err.lower()
            ):
                stats["auth_error"] = True
                folder_err["auth_error"] = True
            stats["folders"][folder] = folder_err

    stats["embedded"] = embed_all_pending(
        show_progress=show_progress, max_messages=embed_budget
    )
    return stats


def sync_all(
    days: int = DEFAULT_SYNC_DAYS,
    since: date | None = None,
    *,
    account: str | None = None,
    show_progress: bool = True,
    incremental: bool = True,
    embed_budget: int | None = None,
    lock_timeout_sec: float | None = None,
) -> list[dict]:
    results = []
    accounts = load_accounts()
    if account:
        accounts = [a for a in accounts if a.email.lower() == account.lower()]
        if not accounts:
            raise ValueError(f"No matching account: {account}")

    timeout = 86_400.0 if lock_timeout_sec is None else lock_timeout_sec
    with fish_write_lock("sync", timeout_sec=timeout):
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
                    sync_account(
                        acct,
                        days=days,
                        since=since,
                        show_progress=show_progress,
                        incremental=incremental,
                        embed_budget=embed_budget,
                    )
                )
    return results
