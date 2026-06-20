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
# Above this size, fetch only text/plain + text/html parts (skip attachment bytes).
LARGE_MESSAGE_BYTES = 256 * 1024

T = TypeVar("T")


def _atom(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("ascii", errors="replace").lower()
    return str(value).lower()


def message_size_bytes(structure: object) -> int:
    if not isinstance(structure, (tuple, list)) or not structure:
        return 0
    maintype = _atom(structure[0])
    if maintype == "multipart":
        parts = structure[-1] if len(structure) >= 3 else []
        return sum(message_size_bytes(part) for part in parts)
    if len(structure) >= 7 and isinstance(structure[6], int):
        return int(structure[6])
    return 0


def iter_text_part_sections(structure: object, prefix: str = "") -> list[tuple[str, str]]:
    """Return IMAP section ids for text/plain and text/html leaves."""
    if not isinstance(structure, (tuple, list)) or not structure:
        return []
    maintype = _atom(structure[0])
    if maintype == "multipart":
        parts = structure[-1] if len(structure) >= 3 else []
        out: list[tuple[str, str]] = []
        for i, part in enumerate(parts, start=1):
            sec = f"{prefix}.{i}" if prefix else str(i)
            out.extend(iter_text_part_sections(part, sec))
        return out
    if maintype == "text":
        subtype = _atom(structure[1])
        if subtype in ("plain", "html"):
            return [(prefix or "1", subtype)]
    return []


def _decode_flags(raw_flags: object) -> list[str]:
    if not raw_flags:
        return []
    return [f.decode() if isinstance(f, bytes) else str(f) for f in raw_flags]


def _fetch_item(data: dict, key: str) -> bytes | None:
    for candidate in (key, key.encode("ascii")):
        value = data.get(candidate)
        if value is not None:
            return value
    return None


def _extract_text_parts(fetch_data: dict, sections: list[tuple[str, str]]) -> dict[str, bytes]:
    text_parts: dict[str, bytes] = {}
    for sec, subtype in sections:
        payload = _fetch_item(fetch_data, f"BODY[{sec}]")
        if payload:
            text_parts[subtype] = payload
    return text_parts


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


def _decode_gmail_labels(raw: object) -> list[str]:
    if not raw:
        return []
    items = raw if isinstance(raw, (tuple, list)) else [raw]
    labels: list[str] = []
    for item in items:
        if isinstance(item, bytes):
            labels.append(item.decode("utf-8", errors="replace"))
        else:
            labels.append(str(item))
    return labels


def fetch_messages(client: IMAPClient, uids: list[int], *, gmail: bool = False) -> dict[int, dict]:
    if not uids:
        return {}
    meta_items: list = [HEADER_KEY, "BODYSTRUCTURE", "FLAGS", "RFC822.SIZE"]
    if gmail:
        meta_items.append("X-GM-LABELS")
    meta = client.fetch(uids, meta_items)
    result: dict[int, dict] = {}
    small_uids: list[int] = []

    for uid, data in meta.items():
        uid_int = int(uid)
        header = data[HEADER_KEY]
        flags = _decode_flags(data.get(b"FLAGS", []))
        structure = data.get(b"BODYSTRUCTURE")
        size = int(data.get(b"RFC822.SIZE") or message_size_bytes(structure))
        sections = iter_text_part_sections(structure)

        if size <= LARGE_MESSAGE_BYTES or not sections:
            small_uids.append(uid_int)
            entry: dict = {"header": header, "flags": flags, "_pending_body": True}
            if gmail:
                entry["gmail_labels"] = _decode_gmail_labels(
                    data.get(b"X-GM-LABELS") or data.get("X-GM-LABELS")
                )
            result[uid_int] = entry
            continue

        part_keys = [f"BODY[{sec}]" for sec, _ in sections]
        parts_data = client.fetch([uid_int], part_keys)
        parts_row = parts_data.get(uid_int)
        if parts_row is None and parts_data:
            parts_row = next(iter(parts_data.values()))
        text_parts = _extract_text_parts(parts_row or {}, sections)
        if text_parts:
            entry = {"header": header, "flags": flags, "text_parts": text_parts}
            if gmail:
                entry["gmail_labels"] = _decode_gmail_labels(
                    data.get(b"X-GM-LABELS") or data.get("X-GM-LABELS")
                )
            result[uid_int] = entry
        else:
            small_uids.append(uid_int)
            entry = {"header": header, "flags": flags, "_pending_body": True}
            if gmail:
                entry["gmail_labels"] = _decode_gmail_labels(
                    data.get(b"X-GM-LABELS") or data.get("X-GM-LABELS")
                )
            result[uid_int] = entry

    if small_uids:
        bodies = client.fetch(small_uids, [BODY_KEY])
        for uid, data in bodies.items():
            uid_int = int(uid)
            entry = result.get(uid_int, {})
            entry["body"] = data[BODY_KEY]
            entry.pop("_pending_body", None)
            result[uid_int] = entry

    for uid_int, entry in list(result.items()):
        entry.pop("_pending_body", None)
        if "body" not in entry and "text_parts" not in entry:
            raise RuntimeError(f"IMAP fetch returned no body for uid {uid_int}")

    return result


def fetch_folder_messages(
    imap: ResilientImap,
    folder: str,
    uids: list[int],
    *,
    on_batch: Callable[[dict[int, dict]], None],
    progress_cb: Callable[[int], None] | None = None,
    gmail: bool = False,
) -> None:
    """Fetch uids in batches with reconnect + retry per batch."""
    for i in range(0, len(uids), FETCH_BATCH):
        batch = uids[i : i + FETCH_BATCH]

        def _fetch(client: IMAPClient) -> dict[int, dict]:
            return fetch_messages(client, batch, gmail=gmail)

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
