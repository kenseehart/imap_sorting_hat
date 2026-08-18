"""Exclusive file lock for Fish DB writers (sync, import, training, corpus).

The kernel releases ``fcntl.flock`` when the holding process dies or closes the
FD — a dead process cannot leave the exclusive lock stuck. Status uses a
non-blocking flock probe as the source of truth; the PID/operation text is
metadata written only while held and cleared on unlock.
"""

from __future__ import annotations

import fcntl
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from fish.config import write_lock_path


class FishWriteLockError(RuntimeError):
    """Another Fish writer holds the lock."""


@dataclass(frozen=True)
class LockStatus:
    held: bool
    path: Path
    pid: int | None = None
    operation: str | None = None


def _parse_lock_meta(text: str) -> tuple[int | None, str | None]:
    text = text.strip()
    if not text:
        return None, None
    parts = text.split(maxsplit=1)
    pid = int(parts[0]) if parts and parts[0].isdigit() else None
    operation = parts[1] if len(parts) > 1 else None
    return pid, operation


def read_lock_status() -> LockStatus:
    """Return whether the exclusive flock is held (probe), plus holder metadata."""
    path = write_lock_path()
    if not path.is_file():
        return LockStatus(held=False, path=path)

    try:
        lock_fd = open(path, "a+")
    except OSError:
        return LockStatus(held=False, path=path)

    try:
        try:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock_fd.seek(0)
            pid, operation = _parse_lock_meta(lock_fd.read())
            return LockStatus(held=True, path=path, pid=pid, operation=operation)

        # Flock is free — clear any stale PID text left by older unlock paths.
        lock_fd.seek(0)
        stale = lock_fd.read()
        if stale.strip():
            lock_fd.seek(0)
            lock_fd.truncate()
            lock_fd.flush()
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
        # Do not report stale PID text when the flock is free.
        return LockStatus(held=False, path=path)
    finally:
        lock_fd.close()


@contextmanager
def fish_write_lock(
    operation: str,
    *,
    timeout_sec: float = 86_400.0,
    poll_sec: float = 2.0,
) -> Iterator[None]:
    """Acquire an exclusive lock before mutating fish.db or running heavy writers."""
    path = write_lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = open(path, "a+")
    deadline = time.monotonic() + timeout_sec
    try:
        while True:
            try:
                fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    status = read_lock_status()
                    if status.held:
                        holder = f"pid={status.pid} op={status.operation!r}"
                    else:
                        holder = "unknown"
                    raise FishWriteLockError(
                        f"Fish write lock busy ({holder}). "
                        f"Operation {operation!r} cannot start."
                    ) from None
                time.sleep(poll_sec)
        lock_fd.seek(0)
        lock_fd.truncate()
        lock_fd.write(f"{os.getpid()} {operation}\n")
        lock_fd.flush()
        yield
    finally:
        try:
            lock_fd.seek(0)
            lock_fd.truncate()
            lock_fd.flush()
        except OSError:
            pass
        try:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        lock_fd.close()
