from __future__ import annotations

import imaplib
import logging
import time
from contextlib import contextmanager
from datetime import date, timedelta
from typing import Callable, Iterator, TypeVar

from imapclient import IMAPClient

from fish.accounts import Account

logger = logging.getLogger(__name__)

HEADER_KEY = b"BODY[HEADER.FIELDS (SUBJECT FROM TO CC BCC MESSAGE-ID IN-REPLY-TO DATE)]"
BODY_KEY = b"BODY[]"

FETCH_BATCH = 25
MAX_IMAP_RETRIES = 3
RETRY_PAUSE_SEC = 2.0

T = TypeVar("T")


def is_connection_error(exc: BaseException) -> bool:
    if isinstance(exc, (imaplib.IMAP4.abort, ConnectionError, TimeoutError, OSError)):
        return True
    msg = str(exc).lower()
    return "socket error" in msg or "eof" in msg or "connection reset" in msg


def short_imap_error(exc: BaseException) -> str:
    text = str(exc).strip()
    if "socket error: EOF" in text:
        return "IMAP connection dropped (server closed connection)"
    if len(text) > 120:
        return text[:117] + "..."
    return text


def connect_client(account: Account) -> IMAPClient:
    client = IMAPClient(account.imap_host, port=account.imap_port, ssl=True, timeout=120)
    client.login(account.username, account.password)
    return client


def close_client(client: IMAPClient | None) -> None:
    if client is None:
        return
    try:
        client.logout()
    except Exception:
        pass


@contextmanager
def imap_session(account: Account) -> Iterator[IMAPClient]:
    client = connect_client(account)
    try:
        yield client
    finally:
        close_client(client)


class ResilientImap:
    """IMAP session that reconnects on transient connection failures."""

    def __init__(self, account: Account) -> None:
        self.account = account
        self.client: IMAPClient | None = None

    def connect(self) -> IMAPClient:
        close_client(self.client)
        self.client = connect_client(self.account)
        return self.client

    def close(self) -> None:
        close_client(self.client)
        self.client = None

    def with_retry(self, op: Callable[[IMAPClient], T], *, folder: str | None = None) -> T:
        last_exc: BaseException | None = None
        for attempt in range(1, MAX_IMAP_RETRIES + 1):
            try:
                client = self.client or self.connect()
                if folder is not None:
                    client.select_folder(folder, readonly=True)
                return op(client)
            except Exception as exc:
                last_exc = exc
                if not is_connection_error(exc) or attempt == MAX_IMAP_RETRIES:
                    raise
                logger.warning(
                    "IMAP reconnect %s/%s for %s (attempt %d/%d): %s",
                    self.account.email,
                    folder or "?",
                    type(exc).__name__,
                    attempt,
                    MAX_IMAP_RETRIES,
                    short_imap_error(exc),
                )
                self.close()
                time.sleep(RETRY_PAUSE_SEC * attempt)
        raise last_exc  # type: ignore[misc]


def test_connection(account: Account) -> dict:
    try:
        with imap_session(account) as client:
            folders = [f[2] for f in client.list_folders()]
            return {"ok": True, "folder_count": len(folders), "folders": folders[:20]}
    except imaplib.IMAP4.error as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def list_folders(account: Account) -> list[str]:
    with imap_session(account) as client:
        return [f[2] for f in client.list_folders()]


def since_date(days: int) -> date:
    return date.today() - timedelta(days=days)


def search_uids_since(client: IMAPClient, folder: str, days: int) -> list[int]:
    client.select_folder(folder, readonly=True)
    since = since_date(days)
    return list(client.search(["SINCE", since]))


def search_uids_since_date(client: IMAPClient, folder: str, since: date) -> list[int]:
    client.select_folder(folder, readonly=True)
    return list(client.search(["SINCE", since]))


def fetch_messages(client: IMAPClient, uids: list[int]) -> dict[int, dict]:
    if not uids:
        return {}
    result: dict[int, dict] = {}
    fetched = client.fetch(uids, [HEADER_KEY, BODY_KEY, "FLAGS"])
    for uid, data in fetched.items():
        flags = [f.decode() if isinstance(f, bytes) else str(f) for f in data.get(b"FLAGS", [])]
        result[int(uid)] = {
            "header": data[HEADER_KEY],
            "body": data[BODY_KEY],
            "flags": flags,
        }
    return result


def fetch_folder_messages(
    imap: ResilientImap,
    folder: str,
    uids: list[int],
    *,
    on_batch: Callable[[dict[int, dict]], None],
    progress_cb: Callable[[int], None] | None = None,
) -> None:
    """Fetch uids in batches with reconnect + retry per batch."""
    for i in range(0, len(uids), FETCH_BATCH):
        batch = uids[i : i + FETCH_BATCH]

        def _fetch(client: IMAPClient) -> dict[int, dict]:
            return fetch_messages(client, batch)

        fetched = imap.with_retry(_fetch, folder=folder)
        on_batch(fetched)
        if progress_cb:
            progress_cb(len(batch))


def move_message(client: IMAPClient, folder: str, uid: int, dest_folder: str) -> None:
    client.select_folder(folder)
    client.move([uid], dest_folder)


def move_messages(client: IMAPClient, folder: str, uids: list[int], dest_folder: str) -> None:
    if not uids:
        return
    client.select_folder(folder)
    client.move(uids, dest_folder)


def set_flags(client: IMAPClient, folder: str, uid: int, flags: list[str], add: bool = True) -> None:
    client.select_folder(folder)
    if add:
        client.add_flags([uid], flags)
    else:
        client.remove_flags([uid], flags)


def delete_message(client: IMAPClient, folder: str, uid: int) -> None:
    client.select_folder(folder)
    client.add_flags([uid], [r"\Deleted"])
    client.expunge()


def folder_uidvalidity(client: IMAPClient, folder: str) -> int | None:
    client.select_folder(folder, readonly=True)
    status = client.folder_status(folder, ["UIDVALIDITY"])
    val = status.get(b"UIDVALIDITY")
    return int(val) if val is not None else None


def append_to_folder(
    client: IMAPClient, folder: str, message_bytes: bytes, flags: list[str] | None = None
) -> None:
    client.append(folder, message_bytes, flags=flags or [r"\Seen"])
