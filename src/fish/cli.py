"""Fish CLI — decorator-driven commands via cmdline."""

from __future__ import annotations

import sys
from datetime import date

from cmdline import cmd, create_parser, emit_output, json_dumps, optarg, run_cli, cmds

from fish.config import DEFAULT_SYNC_DAYS, ensure_openai_api_key, load_env
from fish.connect import connect_interactive
from fish.store import db_conn, init_db, sync_status
from fish.sync import sync_all, sync_account


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
    report["connections"] = connections

    emit_output(report, json_output=json_output, md=md_output, title="Fish email agent status")
    return 0


@cmd
def sync(
    days: int = optarg(DEFAULT_SYNC_DAYS, long_flag="--days", help="Sync window in days"),
    no_progress: bool = optarg(
        False, long_flag="--no-progress", action="store_true", help="Disable progress bars"
    ),
) -> int:
    """Sync mail from all configured accounts into the local RAG database."""
    load_env()
    try:
        ensure_openai_api_key(interactive=True)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1
    init_db()
    results = sync_all(days=days, show_progress=not no_progress)
    exit_code = 0
    for result in results:
        if result.get("folders") and any("error" in v for v in result["folders"].values() if isinstance(v, dict)):
            exit_code = 1
        print(
            f"{result['account']}: fetched={result['fetched']} "
            f"new/changed={result['new_or_changed']} embedded={result['embedded']}"
        )
    return exit_code


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


def main(argv: list[str] | None = None) -> int:
    parser = create_parser(
        cmds(sys.modules[__name__]),
        prog="fish",
        description="Fish — IMAP sync, RAG, and email agent",
    )
    return run_cli(parser, argv)


if __name__ == "__main__":
    raise SystemExit(main())
