"""Exclusive file lock for Fish DB writers (sync, import, training, corpus)."""

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


def read_lock_status() -> LockStatus:
    path = write_lock_path()
    if not path.is_file():
        return LockStatus(held=False, path=path)
    text = path.read_text().strip()
    if not text:
        return LockStatus(held=True, path=path)
    parts = text.split(maxsplit=1)
    pid = int(parts[0]) if parts and parts[0].isdigit() else None
    operation = parts[1] if len(parts) > 1 else None
    if pid is not None:
        try:
            os.kill(pid, 0)
        except OSError:
            return LockStatus(held=False, path=path, pid=pid, operation=operation)
    return LockStatus(held=True, path=path, pid=pid, operation=operation)


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
                    holder = f"pid={status.pid} op={status.operation!r}" if status.held else "unknown"
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
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
        lock_fd.close()
