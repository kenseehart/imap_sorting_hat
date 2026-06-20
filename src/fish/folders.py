from __future__ import annotations

from typing import Any

from fish.accounts import (
    Account,
    add_ignore_folder,
    ignore_folders,
    ignore_folders_for_account,
    load_accounts,
    remove_ignore_folder,
)
from fish.imap_client import list_folders


def list_ignore_folders() -> list[str]:
    return sorted(ignore_folders())


def list_account_folders(account: Account) -> list[dict[str, Any]]:
    ignored = ignore_folders_for_account(account)
    folders = list_folders(account)
    return [
        {
            "folder": name,
            "ignored": name in ignored,
            "sync": name not in ignored,
        }
        for name in sorted(folders)
    ]


def folders_report(account_email: str | None = None) -> dict[str, Any]:
    ignored = sorted(ignore_folders())
    accounts = load_accounts()
    if account_email:
        accounts = [a for a in accounts if a.email.lower() == account_email.lower()]
        if not accounts:
            raise ValueError(f"No matching account: {account_email}")

    account_rows: list[dict[str, Any]] = []
    for account in accounts:
        rows = list_account_folders(account)
        account_rows.append(
            {
                "email": account.email,
                "folder_count": len(rows),
                "sync_folder_count": sum(1 for row in rows if row["sync"]),
                "ignored_folder_count": sum(1 for row in rows if row["ignored"]),
                "folders": rows,
            }
        )
    return {
        "ignore_folders": ignored,
        "accounts": account_rows,
    }
