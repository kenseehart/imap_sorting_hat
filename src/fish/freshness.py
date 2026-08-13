"""Ensure corpus freshness before search — sync if stale, fail loudly on auth errors."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fish.accounts import auth_status
from fish.config import (
    SEARCH_SYNC_DAYS,
    SEARCH_SYNC_EMBED_BUDGET,
    SEARCH_SYNC_LOCK_TIMEOUT_SEC,
    SEARCH_SYNC_MAX_AGE_SEC,
)
from fish.store import db_conn, init_db, newest_sync_at
from fish.sync import sync_all
from fish.write_lock import FishWriteLockError


class FishAuthError(RuntimeError):
    """IMAP credentials missing or rejected — search must not silently continue."""


def require_account_passwords(account_email: str | None = None) -> None:
    status = auth_status()
    accounts = status.get("accounts") or []
    if account_email:
        accounts = [a for a in accounts if a.get("email", "").lower() == account_email.lower()]
        if not accounts:
            raise FishAuthError(
                f"Unknown email account {account_email!r}. "
                f"Configured accounts: "
                + ", ".join(a.get("email", "?") for a in (status.get("accounts") or []))
            )
    missing = [a["email"] for a in accounts if not a.get("password_configured")]
    if missing:
        joined = ", ".join(missing)
        raise FishAuthError(
            f"Email auth not configured for: {joined}. "
            f"On the fish host run: fish connect <email> "
            f"(or set FISH_PASSWORD_* in fish.env and redeploy). "
            f"Refusing to search a stale corpus."
        )


def corpus_sync_age_sec() -> float | None:
    init_db()
    with db_conn() as db:
        ts = newest_sync_at(db)
    if not ts:
        return None
    try:
        synced = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    if synced.tzinfo is None:
        synced = synced.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - synced).total_seconds())


def _auth_failures(results: list[dict[str, Any]]) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    for result in results:
        email = str(result.get("account") or "?")
        if result.get("auth_error"):
            failures.append(
                {"account": email, "error": str(result.get("error") or "authentication failed")}
            )
            continue
        for folder, folder_stats in (result.get("folders") or {}).items():
            if isinstance(folder_stats, dict) and folder_stats.get("auth_error"):
                failures.append(
                    {
                        "account": email,
                        "error": f"{folder}: {folder_stats.get('error')}",
                    }
                )
    return failures


def ensure_search_ready(
    *,
    account_email: str | None = None,
    force: bool = False,
    max_age_sec: int = SEARCH_SYNC_MAX_AGE_SEC,
) -> dict[str, Any]:
    """Sync if corpus is stale. Raises FishAuthError on credential problems."""
    require_account_passwords(account_email)
    age = corpus_sync_age_sec()
    if not force and age is not None and age <= max_age_sec:
        return {
            "synced": False,
            "reason": "fresh",
            "age_sec": round(age),
            "max_age_sec": max_age_sec,
        }

    try:
        results = sync_all(
            days=SEARCH_SYNC_DAYS,
            account=account_email,
            show_progress=False,
            incremental=True,
            embed_budget=SEARCH_SYNC_EMBED_BUDGET,
            lock_timeout_sec=SEARCH_SYNC_LOCK_TIMEOUT_SEC,
        )
    except FishWriteLockError as exc:
        return {
            "synced": False,
            "reason": "lock_busy",
            "warning": str(exc),
            "age_sec": None if age is None else round(age),
        }

    auth_failures = _auth_failures(results)
    if auth_failures:
        detail = "; ".join(f"{f['account']}: {f['error']}" for f in auth_failures)
        raise FishAuthError(
            f"Email auth failed during sync — {detail}. "
            f"Fix credentials on the fish host (`fish connect <email>`) and retry. "
            f"Refusing to return search results."
        )

    return {
        "synced": True,
        "reason": "stale" if age is not None else "never_synced",
        "age_sec_before": None if age is None else round(age),
        "accounts": [
            {
                "account": r.get("account"),
                "fetched": r.get("fetched"),
                "new_or_changed": r.get("new_or_changed"),
                "embedded": r.get("embedded"),
                "skipped_existing": r.get("skipped_existing"),
                "error": r.get("error"),
            }
            for r in results
        ],
    }
