from __future__ import annotations

import getpass
import imaplib
import re
import smtplib
import socket
import sys

import yaml

from fish.accounts import (
    ACCOUNTS_PATH,
    Account,
    ensure_accounts_file,
    load_accounts_config,
)
from fish.config import ensure_config_dir
from fish.imap_client import test_connection
from fish.secrets import get_password, has_password, set_password

EMAIL_RE = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")

GMAIL_ARCHIVE = "[Gmail]/All Mail"
DEFAULT_ARCHIVE = "Archive"


def suggest_hosts(email: str) -> dict[str, str | int]:
    domain = email.split("@", 1)[1].lower()
    if domain == "gmail.com" or domain.endswith(".googlemail.com"):
        return {
            "imap_host": "imap.gmail.com",
            "smtp_host": "smtp.gmail.com",
            "imap_port": 993,
            "smtp_port": 587,
            "archive_folder": GMAIL_ARCHIVE,
        }
    mail_host = f"mail.{domain}"
    return {
        "imap_host": mail_host,
        "smtp_host": mail_host,
        "imap_port": 993,
        "smtp_port": 587,
        "archive_folder": DEFAULT_ARCHIVE,
    }


def _prompt(label: str, default: str = "", *, secret: bool = False) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        if secret:
            value = getpass.getpass(f"{label}{suffix}: ")
        else:
            value = input(f"{label}{suffix}: ").strip()
        if value:
            return value
        if default:
            return default
        print("  (required — press Enter only if a default is shown)")


def _prompt_int(label: str, default: int) -> int:
    while True:
        raw = _prompt(label, str(default))
        try:
            return int(raw)
        except ValueError:
            print(f"  Enter a number (default {default}).")


def _save_account_yaml(entry: dict) -> None:
    ensure_accounts_file()
    data = load_accounts_config()
    accounts = data.get("accounts", [])
    entry = dict(entry)
    entry.pop("password_env", None)
    replaced = False
    for i, existing in enumerate(accounts):
        if existing.get("email", "").lower() == entry["email"].lower():
            merged = {**existing, **entry}
            merged.pop("password_env", None)
            accounts[i] = merged
            replaced = True
            break
    if not replaced:
        accounts.append(entry)
    data["accounts"] = accounts
    ACCOUNTS_PATH.write_text(yaml.safe_dump(data, sort_keys=False, default_flow_style=False))


def _diagnose_imap_error(exc: Exception, host: str, port: int) -> str:
    if isinstance(exc, socket.gaierror):
        return (
            f"DNS lookup failed for {host!r}. Check the hostname or your network.\n"
            f"  Detail: {exc}"
        )
    if isinstance(exc, TimeoutError):
        return f"Connection to {host}:{port} timed out. Firewall or wrong host/port?"
    if isinstance(exc, imaplib.IMAP4.error):
        msg = str(exc)
        if "authentication failed" in msg.lower() or "invalid credentials" in msg.lower():
            return (
                "IMAP authentication failed.\n"
                "  • Use an app password (not your regular login password) if 2FA is on.\n"
                "  • For Gmail: Google Account → Security → App passwords.\n"
                f"  Server said: {msg}"
            )
        return f"IMAP error from {host}: {msg}"
    if isinstance(exc, OSError):
        return f"Network error reaching {host}:{port}: {exc}"
    return f"Unexpected error ({type(exc).__name__}): {exc}"


def _test_smtp(host: str, port: int, username: str, password: str) -> dict:
    try:
        with smtplib.SMTP(host, port, timeout=20) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.login(username, password)
        return {"ok": True}
    except smtplib.SMTPAuthenticationError as exc:
        return {
            "ok": False,
            "error": f"SMTP authentication failed: {exc.smtp_code} {exc.smtp_error!r}",
        }
    except Exception as exc:
        return {"ok": False, "error": _diagnose_imap_error(exc, host, port)}


def connect_interactive(email: str) -> int:
    email = email.strip().lower()
    if not EMAIL_RE.match(email):
        print(f"Invalid email address: {email!r}", file=sys.stderr)
        return 1

    print(f"Setting up fish account: {email}")
    defaults = suggest_hosts(email)
    existing = None
    for raw in load_accounts_config().get("accounts", []):
        if raw.get("email", "").lower() == email:
            existing = raw
            break

    if existing:
        print(f"  Found existing entry in {ACCOUNTS_PATH} — values below are pre-filled.")

    imap_host = _prompt("IMAP host", existing.get("imap_host", defaults["imap_host"]) if existing else str(defaults["imap_host"]))
    imap_port = _prompt_int(
        "IMAP port",
        int(existing.get("imap_port", defaults["imap_port"]) if existing else defaults["imap_port"]),
    )
    smtp_host = _prompt("SMTP host", existing.get("smtp_host", defaults["smtp_host"]) if existing else str(defaults["smtp_host"]))
    smtp_port = _prompt_int(
        "SMTP port",
        int(existing.get("smtp_port", defaults["smtp_port"]) if existing else defaults["smtp_port"]),
    )
    username = _prompt("Username", existing.get("username", email) if existing else email)
    archive_folder = _prompt(
        "Archive folder",
        existing.get("archive_folder", defaults["archive_folder"]) if existing else str(defaults["archive_folder"]),
    )

    legacy_env = existing.get("password_env") if existing else None
    existing_pwd = get_password(email, legacy_env=legacy_env, migrate=False)
    if existing_pwd:
        use_existing = input("App password already stored for this account. Replace it? [y/N]: ").strip().lower()
        if use_existing != "y":
            password = existing_pwd
        else:
            password = _prompt("App password", secret=True)
    else:
        print("Enter an app-specific password (input hidden).")
        password = _prompt("App password", secret=True)

    if not password:
        print("No password provided.", file=sys.stderr)
        return 1

    print(f"\nTesting IMAP {imap_host}:{imap_port} as {username}...")
    probe = Account(
        id=None,
        email=email,
        imap_host=imap_host,
        smtp_host=smtp_host,
        username=username,
        archive_folder=archive_folder,
        imap_port=imap_port,
        smtp_port=smtp_port,
        _password_override=password,
    )
    try:
        result = test_connection(probe)
    except Exception as exc:
        print(_diagnose_imap_error(exc, imap_host, imap_port), file=sys.stderr)
        return 1

    if not result.get("ok"):
        err = result.get("error", "unknown error")
        print(f"IMAP connection failed.\n  {err}", file=sys.stderr)
        print(
            "\nTips:\n"
            "  • Confirm IMAP is enabled in your mail provider settings.\n"
            "  • Double-check host, port (993 SSL), and app password.\n"
            "  • For custom domains, try the host from your hosting control panel.",
            file=sys.stderr,
        )
        return 1

    folder_count = result.get("folder_count", 0)
    print(f"  IMAP OK — {folder_count} folders visible.")
    sample = result.get("folders") or []
    if sample:
        print("  Sample folders:", ", ".join(sample[:5]) + ("…" if len(sample) > 5 else ""))

    print(f"\nTesting SMTP {smtp_host}:{smtp_port}...")
    smtp_result = _test_smtp(smtp_host, smtp_port, username, password)
    if smtp_result.get("ok"):
        print("  SMTP OK.")
    else:
        print(f"  SMTP warning: {smtp_result.get('error')}")
        cont = input("Save account anyway without verified SMTP? [y/N]: ").strip().lower()
        if cont != "y":
            return 1

    entry = {
        "email": email,
        "imap_host": imap_host,
        "imap_port": imap_port,
        "smtp_host": smtp_host,
        "smtp_port": smtp_port,
        "username": username,
        "archive_folder": archive_folder,
    }
    _save_account_yaml(entry)
    set_password(email, password)
    ensure_config_dir()
    print(f"\nSaved account to {ACCOUNTS_PATH}")
    print("Saved app password to ~/.config/fish/secrets.json (encrypted)")
    print("Run `fish status` to verify all accounts, then `fish sync` to pull mail.")
    return 0
