"""Pipeline status snapshot for agents / MCP (no SSH required on the fish host).

Reads local fish.db stats, write lock, frozen corpora headers, recent ``.prz``
models, and detached compute job dirs under ``$FISH_DATA_DIR/compute/jobs``.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fish.config import data_dir, load_env, models_dir
from fish.prism.train_corpus import list_frozen_corpora_info
from fish.store import db_conn, init_db, training_corpus_stats
from fish.write_lock import read_lock_status

_JOB_KIND_MARKERS: tuple[tuple[str, str], ...] = (
    ("freeze-training", "freeze"),
    ("freeze-prep", "freeze"),
    ("corpus label", "label"),
    ("prism-train", "train"),
    ("train_nwra", "nwra"),
    ("corpus collect", "collect"),
)


def compute_jobs_dir() -> Path:
    return data_dir() / "compute" / "jobs"


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_text(path: Path, *, max_bytes: int | None = None) -> str | None:
    if not path.is_file():
        return None
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if max_bytes is not None and len(data) > max_bytes:
        data = data[-max_bytes:]
    return data.decode("utf-8", errors="replace")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _cmdline_for_pid(pid: int) -> str | None:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return None
    return raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip() or None


def _classify_command(text: str | None) -> str | None:
    if not text:
        return None
    lower = text.lower()
    for marker, kind in _JOB_KIND_MARKERS:
        if marker in lower:
            return kind
    return None


def _tail_lines(text: str | None, n: int = 12) -> list[str]:
    if not text:
        return []
    lines = text.splitlines()
    return lines[-n:] if len(lines) > n else lines


def list_local_compute_jobs(*, limit: int = 20, log_tail: int = 12) -> list[dict[str, Any]]:
    """Inspect detached job status trees on this host (no SSH)."""
    root = compute_jobs_dir()
    if not root.is_dir():
        return []
    dirs = sorted(
        (p for p in root.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:limit]
    out: list[dict[str, Any]] = []
    for job_dir in dirs:
        status = (_read_text(job_dir / "status") or "").strip() or "unknown"
        exit_raw = (_read_text(job_dir / "exit_code") or "").strip()
        pid_raw = (_read_text(job_dir / "pid") or "").strip()
        pid = int(pid_raw) if pid_raw.isdigit() else None
        alive = _pid_alive(pid) if pid is not None else False
        cmdline = _cmdline_for_pid(pid) if pid is not None and alive else None
        log_text = _read_text(job_dir / "log", max_bytes=64_000)
        kind = _classify_command(cmdline) or _classify_command(log_text)
        entry: dict[str, Any] = {
            "job_id": job_dir.name,
            "kind": kind,
            "status": status,
            "pid": pid,
            "alive": alive,
            "exit_code": int(exit_raw) if exit_raw.lstrip("-").isdigit() else None,
            "log_path": str(job_dir / "log"),
            "log_tail": _tail_lines(log_text, log_tail),
            "mtime": datetime.fromtimestamp(
                job_dir.stat().st_mtime, tz=timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        if cmdline:
            entry["cmdline"] = cmdline[:500]
        # Normalize: process gone but status still "running"
        if status == "running" and pid is not None and not alive:
            entry["status"] = "stale_running"
            entry["note"] = "status file says running but pid is dead"
        out.append(entry)
    return out


def list_recent_prz(*, limit: int = 8) -> list[dict[str, Any]]:
    root = models_dir()
    if not root.is_dir():
        return []
    files = sorted(root.glob("*.prz"), key=lambda p: p.stat().st_mtime, reverse=True)
    out: list[dict[str, Any]] = []
    for path in files[:limit]:
        out.append(
            {
                "model_id": path.stem,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "mtime": datetime.fromtimestamp(
                    path.stat().st_mtime, tz=timezone.utc
                ).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )
    return out


def pipeline_status(*, job_limit: int = 20, log_tail: int = 12) -> dict[str, Any]:
    """Full pipeline snapshot for MCP / CLI / canvas refresh."""
    load_env()
    lock = read_lock_status()
    payload: dict[str, Any] = {
        "as_of": _utcnow(),
        "data_dir": str(data_dir()),
        "write_lock": {
            "held": lock.held,
            "path": str(lock.path),
            "pid": lock.pid,
            "operation": lock.operation,
        },
        "frozen_corpora": list_frozen_corpora_info(),
        "models": list_recent_prz(),
        "compute_jobs": list_local_compute_jobs(limit=job_limit, log_tail=log_tail),
        "compute_jobs_dir": str(compute_jobs_dir()),
        "pipeline_rule": "label (ok concurrent) → freeze success → then train",
    }

    corpus: dict[str, Any]
    try:
        init_db()
        with db_conn() as db:
            corpus = training_corpus_stats(db)
        payload["corpus"] = corpus
        payload["labeling"] = {
            "labeled": int(corpus.get("samples_labeled") or 0),
            "unlabeled": int(corpus.get("samples_unlabeled") or 0),
            "samples_total": int(corpus.get("samples_total") or 0),
        }
    except Exception as exc:  # noqa: BLE001 — status must still return lock/jobs
        payload["corpus"] = None
        payload["labeling"] = None
        payload["corpus_error"] = str(exc)

    # Convenience summary for canvases
    jobs = payload["compute_jobs"]
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for job in jobs:
        kind = job.get("kind") or "other"
        by_kind.setdefault(kind, []).append(job)
    payload["tasks"] = {
        kind: [
            {
                "job_id": j["job_id"][:8],
                "status": j["status"],
                "alive": j["alive"],
                "exit_code": j.get("exit_code"),
            }
            for j in items[:5]
        ]
        for kind, items in sorted(by_kind.items())
    }
    return payload
