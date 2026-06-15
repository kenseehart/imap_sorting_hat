from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from fish.config import ACCOUNTS_PATH, ensure_config_dir, load_env
from fish.secrets import get_password, has_password


@dataclass
class Account:
    id: int | None
    email: str
    imap_host: str
    smtp_host: str
    username: str
    archive_folder: str = "Archive"
    imap_port: int = 993
    smtp_port: int = 587
    password_env: str | None = None
    _password_override: str | None = field(default=None, repr=False, compare=False)

    @property
    def password(self) -> str:
        if self._password_override:
            return self._password_override
        value = get_password(self.email, legacy_env=self.password_env)
        if not value:
            raise RuntimeError(
                f"No password stored for {self.email}. Run: fish connect {self.email}"
            )
        return value


def _repo_example_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "accounts.yaml.example"


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
                smtp_port=int(raw.get("smtp_port", 587)),
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
        entry = {
            "email": account.email,
            "imap_host": account.imap_host,
            "smtp_host": account.smtp_host,
            "password_configured": has_password(account.email, legacy_env=account.password_env),
        }
        if account.password_env:
            entry["password_env"] = account.password_env
        accounts_info.append(entry)
    return {
        "accounts_path": str(ACCOUNTS_PATH),
        "env_path": str(Path.home() / ".config" / "fish" / "fish.env"),
        "secrets_path": str(Path.home() / ".config" / "fish" / "secrets.json"),
        "accounts": accounts_info,
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
    }
