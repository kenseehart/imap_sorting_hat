from __future__ import annotations

import argparse
import sys
from datetime import date

from fish.accounts import load_accounts
from fish.config import load_env
from fish.store import init_db
from fish.sync import sync_account


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill historical mail into fish DB")
    parser.add_argument("--since", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--account", help="Limit to one account email")
    args = parser.parse_args()

    load_env()
    init_db()
    since = date.fromisoformat(args.since)

    accounts = load_accounts()
    if args.account:
        accounts = [a for a in accounts if a.email == args.account]

    for account in accounts:
        result = sync_account(account, since=since)
        print(
            f"{result['account']}: fetched={result['fetched']} "
            f"new/changed={result['new_or_changed']} embedded={result['embedded']}"
        )
    sys.exit(0)


if __name__ == "__main__":
    main()
