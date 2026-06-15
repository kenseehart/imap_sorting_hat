from __future__ import annotations

import argparse
import sys

from fish.config import DEFAULT_SYNC_DAYS, load_env
from fish.store import init_db
from fish.sync import sync_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync IMAP mail into fish RAG database")
    parser.add_argument("--days", type=int, default=DEFAULT_SYNC_DAYS, help="Sync window in days")
    args = parser.parse_args()

    load_env()
    init_db()
    results = sync_all(days=args.days)
    for result in results:
        print(
            f"{result['account']}: fetched={result['fetched']} "
            f"new/changed={result['new_or_changed']} embedded={result['embedded']}"
        )
    sys.exit(0)


if __name__ == "__main__":
    main()
