"""Fish CLI — decorator-driven commands via cmdline."""

from __future__ import annotations

import sys
from datetime import date

from cmdline import cmd, create_parser, emit_output, json_dumps, optarg, run_cli, cmds

from fish.config import DEFAULT_SYNC_DAYS, ensure_openai_api_key, load_env
from fish.write_lock import FishWriteLockError, read_lock_status
from fish.connect import connect_interactive
from fish.search import search_corpus, search_messages
from fish.store import db_conn, init_db, sync_status
from fish.sync import embed_all_pending, sync_all, sync_account


@cmd
def connect(email: str) -> int:
    """Interactively configure IMAP/SMTP credentials for an email account."""
    load_env()
    return connect_interactive(email)


@cmd(output=True)
def status(*, json_output: bool = False, md_output: bool = False) -> int:
    """Show config, connectivity, and database status."""
    from fish.accounts import auth_status, load_accounts
    from fish.imap_client import test_connection

    load_env()
    init_db()
    report = auth_status()
    connections = []
    for account in load_accounts():
        conn = test_connection(account)
        conn["email"] = account.email
        connections.append(conn)
    with db_conn() as db:
        report["sync"] = sync_status(db)
        report["message_count"] = db.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        report["corpus_count"] = db.execute("SELECT COUNT(*) FROM corpus_items").fetchone()[0]
        report["corpus_by_kind"] = {
            row["kind"]: row["n"]
            for row in db.execute(
                "SELECT kind, COUNT(*) AS n FROM corpus_items GROUP BY kind"
            ).fetchall()
        }
    report["connections"] = connections

    emit_output(report, json_output=json_output, md=md_output, title="Fish email agent status")
    return 0


@cmd
def sync(
    email: str | None = optarg(
        None,
        positional=True,
        metavar="EMAIL",
        help="Account email to sync (default: all)",
    ),
    days: int = optarg(DEFAULT_SYNC_DAYS, long_flag="--days", help="Sync window in days"),
    no_progress: bool = optarg(
        False, long_flag="--no-progress", action="store_true", help="Disable progress bars"
    ),
) -> int:
    """Sync mail from configured accounts into the local RAG database."""
    load_env()
    try:
        ensure_openai_api_key(interactive=True)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1
    init_db()
    try:
        results = sync_all(days=days, account=email, show_progress=not no_progress)
    except FishWriteLockError as exc:
        print(exc, file=sys.stderr)
        return 1
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    exit_code = 0
    for result in results:
        if result.get("folders") and any("error" in v for v in result["folders"].values() if isinstance(v, dict)):
            exit_code = 1
        print(
            f"{result['account']}: fetched={result['fetched']} "
            f"new/changed={result['new_or_changed']} embedded={result['embedded']}"
        )
    return exit_code


@cmd(output=True)
def search(
    query: str,
    account: str | None = optarg(None, long_flag="--account", help="Limit to one account email"),
    folder: str | None = optarg(None, long_flag="--folder", help="Limit to one IMAP folder"),
    unread_only: bool = optarg(
        False, long_flag="--unread", action="store_true", help="Unread messages only"
    ),
    limit: int = optarg(20, long_flag="--limit", help="Max results"),
    kinds: str | None = optarg(
        None,
        long_flag="--kinds",
        help="Comma-separated corpus kinds: email,sms,chat,memory",
    ),
    since: str | None = optarg(
        None, long_flag="--since", help="occurred_at >= ISO date (e.g. 2026-07-16)"
    ),
    until: str | None = optarg(
        None, long_flag="--until", help="occurred_at <= ISO datetime"
    ),
    from_addr: str | None = optarg(
        None,
        long_flag="--from",
        dest="from_addr",
        help="Substring match on from_addr (email)",
    ),
    context_json: str | None = optarg(
        None,
        long_flag="--context",
        dest="context_json",
        help="Session context JSON string for query augmentation",
    ),
    no_sync: bool = optarg(
        False,
        long_flag="--no-sync",
        action="store_true",
        help="Skip auto-sync-before-search",
    ),
    model_id: str | None = optarg(
        None,
        long_flag="--model-id",
        help="Retrieval model_id (default: active PRISM or legacy)",
    ),
    *,
    json_output: bool = False,
    md_output: bool = False,
) -> int:
    """ANN search (PRISM/legacy). Hard metadata filters; no keyword hybrid ranking."""
    from fish.freshness import FishAuthError, ensure_search_ready

    load_env()
    try:
        ensure_openai_api_key(interactive=False)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1
    if not no_sync:
        try:
            ensure_search_ready(account_email=account)
        except FishAuthError as exc:
            print(exc, file=sys.stderr)
            return 2
        except Exception as exc:
            print(f"Sync before search failed: {exc}", file=sys.stderr)
            return 1
    kind_list = [k.strip() for k in kinds.split(",")] if kinds else None
    try:
        payload = search_corpus(
            query,
            kinds=kind_list,
            context=context_json,
            account_email=account,
            folder=folder,
            unread_only=unread_only,
            limit=limit,
            model_id=model_id,
            since=since,
            until=until,
            from_contains=from_addr,
        )
        results = payload["results"]
    except Exception as exc:
        print(exc, file=sys.stderr)
        return 1
    if json_output:
        emit_output(payload, json_output=True)
    else:
        rows = [
            {
                "id": r["id"],
                "kind": r.get("kind"),
                "score": r.get("score"),
                "date": (r.get("occurred_at") or r.get("date") or "")[:10],
                "from": (r.get("from_addr") or "")[:40],
                "account": r.get("account_email"),
                "subject": (r.get("subject") or r.get("body_text") or "")[:100],
            }
            for r in results
        ]
        emit_output(
            rows,
            json_output=False,
            md=md_output,
            title=f'Fish search: "{query}" ({len(results)} results)',
        )
    return 0


@cmd(output=True)
def get(
    item_id: int,
    *,
    json_output: bool = False,
    md_output: bool = False,
) -> int:
    """Get a corpus item (or email message) by id."""
    from fish.parse import message_row_to_dict
    from fish.store import (
        db_conn,
        get_corpus_by_id,
        get_message_by_id,
        init_db,
    )

    init_db()
    with db_conn() as db:
        row = get_corpus_by_id(db, item_id)
        if row:
            emit_output(row, json_output=json_output, md=md_output, title=f"Corpus {item_id}")
            return 0
        msg = get_message_by_id(db, item_id)
        if not msg:
            print(f"Item {item_id} not found", file=sys.stderr)
            return 1
        acct = db.execute(
            "SELECT email FROM accounts WHERE id = ?", (msg["account_id"],)
        ).fetchone()
        item = message_row_to_dict(msg)
        item["account_email"] = acct["email"] if acct else None
        emit_output(item, json_output=json_output, md=md_output, title=f"Message {item_id}")
        return 0


@cmd(output=True)
def import_corpus(
    source: str,
    path: str,
    phone: str | None = optarg(
        None, long_flag="--phone", help="Phone filter for android-sms (default 8315352442)"
    ),
    dry_run: bool = optarg(
        False, long_flag="--dry-run", action="store_true", help="Count only, do not write"
    ),
    no_embed: bool = optarg(
        False, long_flag="--no-embed", action="store_true", help="Skip embedding after import"
    ),
    *,
    json_output: bool = False,
    md_output: bool = False,
) -> int:
    """Import SMS or chat exports into the unified corpus."""
    from pathlib import Path

    from fish.import_sources.runner import run_import

    load_env()
    if not dry_run and not no_embed:
        try:
            ensure_openai_api_key(interactive=False)
        except RuntimeError as exc:
            print(exc, file=sys.stderr)
            return 1
    try:
        result = run_import(
            source,
            Path(path),
            dry_run=dry_run,
            phone_filter=phone,
            embed=not no_embed and not dry_run,
        )
    except Exception as exc:
        print(exc, file=sys.stderr)
        return 1
    emit_output(result, json_output=json_output, md=md_output, title=f"Fish import {source}")
    return 0


@cmd(output=True)
def prism_train(
    config: str = optarg(
        "personal", long_flag="--config", help="PRISM config name from prism_models.yaml"
    ),
    epochs: int | None = optarg(None, long_flag="--epochs", help="Override config epochs"),
    output: str | None = optarg(
        None, long_flag="--output", help="Output .prz path (default models/{model_id}.prz)"
    ),
    retriever: str | None = optarg(
        None,
        long_flag="--retriever",
        help="Train on samples from this retriever only",
    ),
    collect_first: bool = optarg(
        False,
        long_flag="--collect-first",
        action="store_true",
        help="Run corpus collect+label before training",
    ),
    collect_retriever: str = optarg(
        "legacy",
        long_flag="--collect-retriever",
        help="Retriever for --collect-first",
    ),
    min_queries: int = optarg(
        50, long_flag="--min-queries", help="Min queries when --collect-first"
    ),
    top_k: int = optarg(20, long_flag="--top-k", help="Top-k when --collect-first"),
    label_limit: int = optarg(
        500, long_flag="--label-limit", help="Label limit when --collect-first"
    ),
    *,
    json_output: bool = False,
    md_output: bool = False,
) -> int:
    """Train PRISM adapters → {config}.{timestamp}.prz and register in retrieval_models."""
    from pathlib import Path

    from fish.prism.train import train_from_corpus

    load_env()
    try:
        result = train_from_corpus(
            config_name=config,
            epochs=epochs,
            output=Path(output) if output else None,
            retriever=retriever,
            collect_first=collect_first,
            collect_retriever=collect_retriever,
            min_queries=min_queries,
            top_k=top_k,
            label_limit=label_limit,
        )
    except Exception as exc:
        print(exc, file=sys.stderr)
        return 1
    emit_output(result, json_output=json_output, md=md_output, title="Fish PRISM training")
    return 0


@cmd(output=True)
def prism_reembed(
    model_id: str | None = optarg(
        None, long_flag="--model-id", help="PRISM model_id (default: active)"
    ),
    kinds: str | None = optarg(
        None,
        long_flag="--kinds",
        help="Comma-separated kinds to re-index (default: all with raw vectors)",
    ),
    limit: int | None = optarg(
        None, long_flag="--limit", help="Max items (smoke test)"
    ),
    like: str | None = optarg(
        None,
        long_flag="--like",
        help="Comma-separated SQL LIKE patterns on text (smoke-test filter)",
    ),
    since: str | None = optarg(
        None, long_flag="--since", help="Only items with occurred_at >= ISO date"
    ),
    no_progress: bool = optarg(
        False, long_flag="--no-progress", action="store_true", help="Disable progress bars"
    ),
    *,
    json_output: bool = False,
    md_output: bool = False,
) -> int:
    """Re-index a PRISM model's Qdrant collection from corpus_raw_embeddings (no OpenAI)."""
    from fish.prism.reembed import prism_reembed as run_reembed

    load_env()
    kind_list = [k.strip() for k in kinds.split(",")] if kinds else None
    like_list = [p.strip() for p in like.split(",") if p.strip()] if like else None
    try:
        result = run_reembed(
            model_id=model_id,
            kinds=kind_list,
            show_progress=not no_progress,
            limit=limit,
            like=like_list,
            since=since,
        )
    except Exception as exc:
        print(exc, file=sys.stderr)
        return 1
    emit_output(result, json_output=json_output, md=md_output, title="Fish PRISM re-index")
    return 0


@cmd(output=True)
def qdrant_migrate(
    source_table: str | None = optarg(
        None,
        long_flag="--source-table",
        help="sqlite-vec table to copy (default: corpus_vec if present)",
    ),
    limit: int | None = optarg(
        None, long_flag="--limit", help="Max vectors (smoke test)"
    ),
    batch_size: int = optarg(
        256, long_flag="--batch-size", help="Qdrant upsert batch size"
    ),
    skip_copy: bool = optarg(
        False,
        long_flag="--skip-copy",
        action="store_true",
        help="Do not copy from sqlite-vec; only upsert existing corpus_raw_embeddings",
    ),
    skip_qdrant: bool = optarg(
        False,
        long_flag="--skip-qdrant",
        action="store_true",
        help="Only copy into corpus_raw_embeddings; do not upsert Qdrant",
    ),
    no_progress: bool = optarg(
        False, long_flag="--no-progress", action="store_true", help="Disable progress bars"
    ),
    *,
    json_output: bool = False,
    md_output: bool = False,
) -> int:
    """Copy sqlite-vec → corpus_raw_embeddings and upsert legacy Qdrant collection."""
    from fish.qdrant_migrate import migrate_to_qdrant

    load_env()
    try:
        result = migrate_to_qdrant(
            copy_from_sqlite_vec=not skip_copy,
            source_table=source_table,
            limit=limit,
            batch_size=batch_size,
            show_progress=not no_progress,
            skip_qdrant=skip_qdrant,
        )
    except Exception as exc:
        print(exc, file=sys.stderr)
        return 1
    emit_output(result, json_output=json_output, md=md_output, title="Fish Qdrant migrate")
    return 0


@cmd(output=True)
def qdrant_reindex(
    limit: int | None = optarg(
        None, long_flag="--limit", help="Max items (smoke test)"
    ),
    batch_size: int = optarg(
        256, long_flag="--batch-size", help="Qdrant upsert batch size"
    ),
    kinds: str | None = optarg(
        None, long_flag="--kinds", help="Comma-separated kinds"
    ),
    no_progress: bool = optarg(
        False, long_flag="--no-progress", action="store_true", help="Disable progress bars"
    ),
    *,
    json_output: bool = False,
    md_output: bool = False,
) -> int:
    """Upsert corpus_raw_embeddings into the legacy Qdrant collection."""
    from fish.qdrant_migrate import reindex_legacy_qdrant

    load_env()
    kind_list = [k.strip() for k in kinds.split(",")] if kinds else None
    try:
        result = reindex_legacy_qdrant(
            limit=limit,
            batch_size=batch_size,
            show_progress=not no_progress,
            kinds=kind_list,
        )
    except Exception as exc:
        print(exc, file=sys.stderr)
        return 1
    emit_output(result, json_output=json_output, md=md_output, title="Fish Qdrant reindex")
    return 0


@cmd(output=True)
def index_cleanup(
    *,
    json_output: bool = False,
    md_output: bool = False,
) -> int:
    """Remove orphan ANN rows (rowid not in corpus_items) from all registered indexes."""
    from fish.store import cleanup_index_orphans, db_conn, init_db
    from fish.write_lock import fish_write_lock

    load_env()
    init_db()
    with fish_write_lock("train"):
        with db_conn() as db:
            removed = cleanup_index_orphans(db)
    emit_output(
        {"removed": removed},
        json_output=json_output,
        md=md_output,
        title="Fish index orphan cleanup",
    )
    return 0


@cmd(output=True)
def embedding_get(
    item_id: int,
    *,
    json_output: bool = False,
    md_output: bool = False,
) -> int:
    """Return the stored embedding vector for a corpus item id."""
    from fish.store import get_corpus_by_id, get_embedding

    init_db()
    with db_conn() as db:
        row = get_corpus_by_id(db, item_id)
        if not row:
            print(f"Corpus item {item_id} not found.", file=sys.stderr)
            return 1
        vec = get_embedding(db, item_id)
        if vec is None:
            print(f"Corpus item {item_id} is not embedded yet.", file=sys.stderr)
            return 1
        payload = {
            "id": item_id,
            "kind": row.get("kind"),
            "dim": len(vec),
            "embedded_at": row.get("embedded_at"),
            "embedding": vec,
        }
    emit_output(payload, json_output=json_output, md=md_output, title=f"Embedding {item_id}")
    return 0


@cmd(output=True)
def ignore(
    folder: str | None = optarg(
        None,
        positional=True,
        metavar="FOLDER",
        help="Folder name to add or remove",
    ),
    add: bool = optarg(False, long_flag="--add", action="store_true", help="Add folder to ignore list"),
    remove: bool = optarg(
        False, long_flag="--remove", action="store_true", help="Remove folder from ignore list"
    ),
    *,
    json_output: bool = False,
    md_output: bool = False,
) -> int:
    """List, add, or remove IMAP folders skipped during sync."""
    from fish.accounts import add_ignore_folder, remove_ignore_folder
    from fish.folders import list_ignore_folders

    load_env()
    if add and remove:
        print("Use only one of --add or --remove.", file=sys.stderr)
        return 1
    if folder and add:
        folders = add_ignore_folder(folder)
        emit_output(
            {"action": "add", "folder": folder, "ignore_folders": folders},
            json_output=json_output,
            md=md_output,
            title="Fish ignore folders",
        )
        return 0
    if folder and remove:
        folders = remove_ignore_folder(folder)
        emit_output(
            {"action": "remove", "folder": folder, "ignore_folders": folders},
            json_output=json_output,
            md=md_output,
            title="Fish ignore folders",
        )
        return 0
    if folder:
        print("Specify --add or --remove when providing a folder name.", file=sys.stderr)
        return 1
    folders = list_ignore_folders()
    emit_output(
        {"ignore_folders": folders},
        json_output=json_output,
        md=md_output,
        title=f"Fish ignore folders ({len(folders)})",
    )
    return 0


@cmd(output=True)
def folders(
    email: str | None = optarg(
        None,
        positional=True,
        metavar="EMAIL",
        help="Account email (default: all configured accounts)",
    ),
    *,
    json_output: bool = False,
    md_output: bool = False,
) -> int:
    """List IMAP folders for each account and whether sync skips them."""
    from fish.folders import folders_report

    load_env()
    try:
        report = folders_report(email)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    except Exception as exc:
        print(exc, file=sys.stderr)
        return 1

    if not json_output and not md_output:
        for account in report["accounts"]:
            print(f"{account['email']} ({account['sync_folder_count']} sync / {account['folder_count']} total)")
            for row in account["folders"]:
                mark = "skip" if row["ignored"] else "sync"
                print(f"  [{mark}] {row['folder']}")
        print(f"\nGlobal ignore list ({len(report['ignore_folders'])}):")
        for name in report["ignore_folders"]:
            print(f"  {name}")
        return 0

    emit_output(report, json_output=json_output, md=md_output, title="Fish IMAP folders")
    return 0


@cmd
def embed(
    limit: int | None = optarg(None, long_flag="--limit", help="Max messages to embed"),
    kinds: str | None = optarg(None, long_flag="--kinds", help="Comma-separated kinds"),
    like: str | None = optarg(
        None, long_flag="--like", help="Comma-separated SQL LIKE filters on text"
    ),
    since: str | None = optarg(None, long_flag="--since", help="occurred_at >= ISO date"),
    no_progress: bool = optarg(
        False, long_flag="--no-progress", action="store_true", help="Disable progress bars"
    ),
) -> int:
    """Embed corpus items missing from the legacy raw ANN index."""
    load_env()
    try:
        ensure_openai_api_key(interactive=False)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1
    init_db()
    kind_list = [k.strip() for k in kinds.split(",")] if kinds else None
    like_list = [p.strip() for p in like.split(",") if p.strip()] if like else None
    count = embed_all_pending(
        show_progress=not no_progress,
        max_messages=limit,
        kinds=kind_list,
        like=like_list,
        since=since,
    )
    print(f"embedded={count}")
    return 0


@cmd
def backfill(
    since: str = optarg(
        None,
        long_flag="--since",
        required=True,
        help="Start date YYYY-MM-DD",
        metavar="DATE",
    ),
    account: str | None = optarg(None, long_flag="--account", help="Limit to one account email"),
) -> int:
    """Backfill historical mail older than the default sync window."""
    from fish.accounts import load_accounts

    load_env()
    try:
        ensure_openai_api_key(interactive=True)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1
    init_db()
    since_date = date.fromisoformat(since)
    accounts = load_accounts()
    if account:
        accounts = [a for a in accounts if a.email.lower() == account.lower()]
    if not accounts:
        print("No matching accounts.", file=sys.stderr)
        return 1
    for acct in accounts:
        result = sync_account(acct, since=since_date, show_progress=True)
        print(
            f"{result['account']}: fetched={result['fetched']} "
            f"new/changed={result['new_or_changed']} embedded={result['embedded']}"
        )
    return 0


@cmd(output=True)
def write_lock_status(*, json_output: bool = False, md_output: bool = False) -> int:
    """Show whether a Fish DB write lock is held (sync, import, corpus, train)."""
    status = read_lock_status()
    payload = {
        "held": status.held,
        "path": str(status.path),
        "pid": status.pid,
        "operation": status.operation,
    }
    emit_output(payload, json_output=json_output, md=md_output, title="Fish write lock")
    return 0


def main(argv: list[str] | None = None) -> int:
    from fish.corpus_cli import corpus

    parser = create_parser(
        cmds(sys.modules[__name__]) + corpus.commands,
        prog="fish",
        description="Fish — IMAP sync, RAG, and email agent",
        groups=[corpus],
    )
    return run_cli(parser, argv, groups=[corpus])


if __name__ == "__main__":
    raise SystemExit(main())
