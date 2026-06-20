from __future__ import annotations

import os
import stat
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from fish.config import ACCOUNTS_PATH, ensure_config_dir, load_env


@dataclass
class Account:
    id: int | None
    email: str
    imap_host: str
    smtp_host: str
    username: str
    archive_folder: str = "Archive"
    imap_port: int = 993
    smtp_port: int = 465
    password_stored: str | None = None
    password_env: str | None = None
    _password_override: str | None = field(default=None, repr=False, compare=False)

    @property
    def password(self) -> str:
        if self._password_override:
            return self._password_override
        value = resolve_password(self.email, self.password_stored, self.password_env)
        if not value:
            raise RuntimeError(
                f"No password stored for {self.email}. Run: fish connect {self.email}"
            )
        return value


def _repo_example_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "accounts.yaml.example"


def _restrict_permissions(path: Path) -> None:
    if path.exists() and path.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def ensure_accounts_file() -> Path:
    ensure_config_dir()
    if not ACCOUNTS_PATH.exists():
        example = _repo_example_path()
        if example.exists():
            ACCOUNTS_PATH.write_text(example.read_text())
        else:
            ACCOUNTS_PATH.write_text("accounts: []\nignore_folders: []\n")
    return ACCOUNTS_PATH


def load_accounts_config() -> dict:
    ensure_accounts_file()
    data = yaml.safe_load(ACCOUNTS_PATH.read_text()) or {}
    data.setdefault("accounts", [])
    data.setdefault("ignore_folders", [])
    return data


def resolve_password(
    email: str,
    password_stored: str | None,
    password_env: str | None = None,
) -> str | None:
    """Password from accounts.yaml, with legacy fallbacks from secrets.json and fish.env."""
    if password_stored:
        return password_stored
    from fish.secrets import get_password

    return get_password(email, legacy_env=password_env, migrate=False)


def has_stored_password(
    email: str,
    password_stored: str | None = None,
    password_env: str | None = None,
) -> bool:
    if password_stored:
        return True
    from fish.secrets import has_password

    return has_password(email, legacy_env=password_env)


def save_account_entry(entry: dict) -> None:
    """Write one account record to accounts.yaml (includes app password)."""
    ensure_accounts_file()
    data = load_accounts_config()
    accounts = data.get("accounts", [])
    entry = dict(entry)
    email = entry["email"].strip().lower()
    entry["email"] = email
    entry.pop("password_env", None)
    replaced = False
    for i, existing in enumerate(accounts):
        if existing.get("email", "").lower() == email:
            merged = {**existing, **entry}
            merged.pop("password_env", None)
            accounts[i] = merged
            replaced = True
            break
    if not replaced:
        accounts.append(entry)
    data["accounts"] = accounts
    ACCOUNTS_PATH.write_text(yaml.safe_dump(data, sort_keys=False, default_flow_style=False))
    if entry.get("password"):
        _restrict_permissions(ACCOUNTS_PATH)


def load_accounts() -> list[Account]:
    data = load_accounts_config()
    accounts: list[Account] = []
    for i, raw in enumerate(data.get("accounts", []), start=1):
        accounts.append(
            Account(
                id=i,
                email=raw["email"],
                imap_host=raw["imap_host"],
                smtp_host=raw.get("smtp_host", raw["imap_host"]),
                username=raw.get("username", raw["email"]),
                archive_folder=raw.get("archive_folder", "Archive"),
                imap_port=int(raw.get("imap_port", 993)),
                smtp_port=int(raw.get("smtp_port", 465)),
                password_stored=raw.get("password"),
                password_env=raw.get("password_env"),
            )
        )
    return accounts


def ignore_folders() -> set[str]:
    return set(load_accounts_config().get("ignore_folders", []))


def account_by_email(email: str) -> Account | None:
    for account in load_accounts():
        if account.email.lower() == email.lower():
            return account
    return None


def account_by_id(account_id: int) -> Account | None:
    for account in load_accounts():
        if account.id == account_id:
            return account
    return None


def auth_status() -> dict:
    load_env()
    accounts_info = []
    for account in load_accounts():
        accounts_info.append(
            {
                "email": account.email,
                "imap_host": account.imap_host,
                "smtp_host": account.smtp_host,
                "password_configured": has_stored_password(
                    account.email,
                    account.password_stored,
                    account.password_env,
                ),
            }
        )
    return {
        "accounts_path": str(ACCOUNTS_PATH),
        "env_path": str(Path.home() / ".config" / "fish" / "fish.env"),
        "accounts": accounts_info,
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
    }
