from __future__ import annotations

import argparse
import json
import sys

from fish.accounts import auth_status, load_accounts
from fish.config import load_env
from fish.imap_client import test_connection
from fish.store import db_conn, init_db, sync_status


def main() -> None:
    parser = argparse.ArgumentParser(description="Fish status and connectivity check")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    load_env()
    init_db()

    status = auth_status()
    connections = []
    for account in load_accounts():
        conn = test_connection(account)
        conn["email"] = account.email
        connections.append(conn)

    with db_conn() as db:
        sync = sync_status(db)
        message_count = db.execute("SELECT COUNT(*) FROM messages").fetchone()[0]

    report = {
        **status,
        "connections": connections,
        "sync": sync,
        "message_count": message_count,
    }

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("Fish email agent status")
        print(f"  Messages in DB: {message_count}")
        print(f"  OpenAI configured: {status['openai_configured']}")
        for account in status["accounts"]:
            ok = next(
                (c["ok"] for c in connections if c["email"] == account["email"]),
                False,
            )
            print(
                f"  {account['email']}: password={'yes' if account['password_configured'] else 'NO'} "
                f"imap={account['imap_host']} connected={'yes' if ok else 'NO'}"
            )
    sys.exit(0)


if __name__ == "__main__":
    main()
