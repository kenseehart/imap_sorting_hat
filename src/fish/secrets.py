"""Legacy encrypted store — read-only migration into accounts.yaml."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

from fish.config import CONFIG_DIR, ensure_config_dir, load_env

SECRETS_PATH = CONFIG_DIR / "secrets.json"
KEY_PATH = CONFIG_DIR / ".master.key"


def _restrict_permissions(path: Path) -> None:
    if not path.exists():
        return
    mode = path.stat().st_mode
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def _fernet() -> Fernet:
    ensure_config_dir()
    if not KEY_PATH.exists():
        KEY_PATH.write_bytes(Fernet.generate_key())
    _restrict_permissions(KEY_PATH)
    return Fernet(KEY_PATH.read_bytes())


def _load_entries() -> dict[str, str]:
    if not SECRETS_PATH.exists():
        return {}
    try:
        data = json.loads(SECRETS_PATH.read_text())
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    entries = data.get("entries")
    return entries if isinstance(entries, dict) else {}


def _save_entries(entries: dict[str, str]) -> None:
    ensure_config_dir()
    payload = {"version": 1, "entries": entries}
    SECRETS_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    _restrict_permissions(SECRETS_PATH)


def _normalize_email(email: str) -> str:
    return email.strip().lower()


def has_password(email: str, *, legacy_env: str | None = None) -> bool:
    email = _normalize_email(email)
    if email in _load_entries():
        return True
    if legacy_env:
        load_env()
        return bool(os.getenv(legacy_env, ""))
    return False


def get_password(email: str, *, legacy_env: str | None = None, migrate: bool = True) -> str | None:
    """Return stored app password for an account, optionally migrating from legacy env."""
    email = _normalize_email(email)
    entries = _load_entries()
    token = entries.get(email)
    if token:
        try:
            return _fernet().decrypt(token.encode()).decode()
        except InvalidToken:
            raise RuntimeError(
                f"Could not decrypt password for {email}. "
                f"Master key or secrets file may be corrupt ({SECRETS_PATH})."
            )

    if legacy_env:
        load_env()
        value = os.getenv(legacy_env, "")
        if value and migrate:
            set_password(email, value)
            return value
        if value:
            return value
    return None


def set_password(email: str, password: str) -> None:
    email = _normalize_email(email)
    entries = _load_entries()
    entries[email] = _fernet().encrypt(password.encode()).decode()
    _save_entries(entries)


def delete_password(email: str) -> None:
    email = _normalize_email(email)
    entries = _load_entries()
    if email in entries:
        del entries[email]
        _save_entries(entries)


def list_configured_emails() -> list[str]:
    return sorted(_load_entries())
