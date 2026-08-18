"""Frozen training corpus (``.tcz``) — pytorch-ready zip snapshot.

``fish corpus freeze-training`` reads labeled pairs from SQLite once and writes
``models/corpora/train_corpus_{timestamp}.tcz``. ``fish prism-train`` loads a
``.tcz`` and never opens fish.db for the epoch loop.
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from fish.config import models_dir
from fish.prism.model import CHUNK_REPR_COMBINED, CHUNK_REPR_HEADER_BODY
from fish.write_lock import fish_write_lock

if TYPE_CHECKING:
    from fish.prism.train import TrainingPair

TCZ_VERSION = 1
TCZ_SUFFIX = ".tcz"
VALID_CHUNK_REPRS = frozenset({CHUNK_REPR_COMBINED, CHUNK_REPR_HEADER_BODY})
# Keep at most this many train_corpus_*.tcz files under models/corpora/.
MAX_FROZEN_CORPORA = 3


def corpora_dir() -> Path:
    path = models_dir() / "corpora"
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_corpus_id(*, timestamp: str | None = None) -> str:
    ts = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"train_corpus_{ts}"


def corpus_id_from_path(path: Path) -> str:
    name = path.name
    if name.endswith(TCZ_SUFFIX):
        return name[: -len(TCZ_SUFFIX)]
    return path.stem


def tcz_path_for_id(corpus_id: str) -> Path:
    cid = corpus_id.strip()
    if cid.endswith(TCZ_SUFFIX):
        cid = cid[: -len(TCZ_SUFFIX)]
    # Absolute / relative filesystem path (not a bare id).
    if "/" in cid or "\\" in cid or cid.startswith("."):
        path = Path(cid)
        if path.suffix != TCZ_SUFFIX:
            path = path.with_suffix(TCZ_SUFFIX)
        return path
    if not cid.startswith("train_corpus_"):
        cid = f"train_corpus_{cid}"
    return corpora_dir() / f"{cid}{TCZ_SUFFIX}"


def list_frozen_corpora() -> list[Path]:
    root = corpora_dir()
    return sorted(p for p in root.glob(f"train_corpus_*{TCZ_SUFFIX}") if p.is_file())


def prune_old_frozen_corpora(
    *,
    keep: int = MAX_FROZEN_CORPORA,
) -> list[str]:
    """Delete oldest ``train_corpus_*.tcz`` files until at most ``keep`` remain.

    Files are ordered by corpus id timestamp (lexicographic on the UTC stamp).
    Returns the deleted corpus ids (stem without ``.tcz``).
    """
    if keep < 1:
        raise ValueError(f"keep must be >= 1, got {keep}")
    files = list_frozen_corpora()
    if len(files) <= keep:
        return []
    deleted: list[str] = []
    for path in files[: len(files) - keep]:
        corpus_id = corpus_id_from_path(path)
        path.unlink(missing_ok=True)
        deleted.append(corpus_id)
    return deleted


def resolve_corpus_path(corpus: str | Path) -> Path:
    """Resolve ``latest``, a corpus id, or a filesystem path to a ``.tcz`` file."""
    if isinstance(corpus, Path):
        raw = str(corpus)
    else:
        raw = str(corpus).strip()
    if not raw:
        raise ValueError("corpus must be 'latest', a train_corpus_* id, or a .tcz path")
    if raw == "latest":
        files = list_frozen_corpora()
        if not files:
            raise FileNotFoundError(
                f"No frozen corpora in {corpora_dir()}. "
                f"Run: fish corpus freeze-training --chunk-repr …"
            )
        return files[-1]

    path = Path(raw).expanduser()
    # Explicit filesystem path
    if path.suffix == TCZ_SUFFIX or "/" in raw or "\\" in raw or raw.startswith("."):
        if path.suffix != TCZ_SUFFIX and not path.is_file():
            path = path.with_suffix(TCZ_SUFFIX)
        if not path.is_file():
            raise FileNotFoundError(f"Frozen corpus not found: {path}")
        return path.resolve()

    # Bare id (with or without train_corpus_ / .tcz suffix)
    resolved = tcz_path_for_id(raw)
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Frozen corpus not found: {resolved}. "
            f"Run freeze-training or pass --corpus latest."
        )
    return resolved.resolve()


@dataclass
class FrozenCorpus:
    corpus_id: str
    path: Path
    chunk_repr: str
    retriever: str | None
    created_at: str
    pairs: list[TrainingPair]
    retrieval_similarity: list[float] | None = None

    @property
    def n_pairs(self) -> int:
        return len(self.pairs)


def _np_save_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def _np_load_bytes(data: bytes) -> np.ndarray:
    return np.load(io.BytesIO(data), allow_pickle=False)


def write_tcz(
    pairs: list[TrainingPair],
    *,
    chunk_repr: str,
    retriever: str | None = None,
    corpus_id: str | None = None,
    path: Path | None = None,
    retrieval_similarity: list[float] | None = None,
) -> Path:
    """Atomically write a frozen training corpus zip (no DB lock required)."""
    if not pairs:
        raise ValueError("Cannot freeze empty training pair list")
    if chunk_repr not in VALID_CHUNK_REPRS:
        raise ValueError(
            f"chunk_repr must be one of {sorted(VALID_CHUNK_REPRS)}, got {chunk_repr!r}"
        )
    for i, p in enumerate(pairs):
        if p.query_embedding is None or p.chunk_embedding is None:
            raise ValueError(f"pair[{i}] missing embeddings")

    cid = corpus_id or make_corpus_id()
    out = path or tcz_path_for_id(cid)
    out.parent.mkdir(parents=True, exist_ok=True)

    q = np.asarray([p.query_embedding for p in pairs], dtype=np.float32)
    c = np.asarray([p.chunk_embedding for p in pairs], dtype=np.float32)
    rel = np.asarray([p.relevance for p in pairs], dtype=np.float32)
    chunk_ids = np.asarray([p.chunk_id for p in pairs], dtype=np.int64)
    queries = [p.query for p in pairs]
    if retrieval_similarity is None:
        retrieval_similarity = [
            float(p.retrieval_similarity)
            if p.retrieval_similarity is not None
            else 0.0
            for p in pairs
        ]
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta = {
        "version": TCZ_VERSION,
        "corpus_id": cid,
        "chunk_repr": chunk_repr,
        "retriever": retriever,
        "created_at": created_at,
        "n_pairs": len(pairs),
        "q_dim": int(q.shape[1]),
        "c_dim": int(c.shape[1]),
        "has_retrieval_similarity": True,
    }

    tmp = out.with_suffix(out.suffix + ".tmp")
    try:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("meta.json", json.dumps(meta, indent=2, sort_keys=True))
            zf.writestr("queries.json", json.dumps(queries))
            zf.writestr("q.npy", _np_save_bytes(q))
            zf.writestr("c.npy", _np_save_bytes(c))
            zf.writestr("rel.npy", _np_save_bytes(rel))
            zf.writestr("chunk_ids.npy", _np_save_bytes(chunk_ids))
            rs = np.asarray(retrieval_similarity, dtype=np.float32)
            if len(rs) != len(pairs):
                raise ValueError("retrieval_similarity length must match pairs")
            zf.writestr("retrieval_similarity.npy", _np_save_bytes(rs))
        tmp.replace(out)
    except Exception:
        if tmp.is_file():
            tmp.unlink(missing_ok=True)
        raise
    return out


def load_tcz(path: Path | str) -> FrozenCorpus:
    from fish.prism.train import TrainingPair

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Frozen corpus not found: {p}")
    with zipfile.ZipFile(p, "r") as zf:
        meta = json.loads(zf.read("meta.json").decode())
        version = int(meta.get("version") or 0)
        if version != TCZ_VERSION:
            raise RuntimeError(
                f"Unsupported .tcz version {version} (expected {TCZ_VERSION})"
            )
        queries: list[str] = json.loads(zf.read("queries.json").decode())
        q = _np_load_bytes(zf.read("q.npy"))
        c = _np_load_bytes(zf.read("c.npy"))
        rel = _np_load_bytes(zf.read("rel.npy"))
        chunk_ids = _np_load_bytes(zf.read("chunk_ids.npy"))
        retrieval: list[float] | None = None
        if "retrieval_similarity.npy" in zf.namelist():
            retrieval = _np_load_bytes(zf.read("retrieval_similarity.npy")).tolist()

    n = len(queries)
    if not (len(chunk_ids) == n == len(q) == len(c) == len(rel)):
        raise RuntimeError(f"Corrupt .tcz {p}: length mismatch")
    if retrieval is not None and len(retrieval) != n:
        raise RuntimeError(f"Corrupt .tcz {p}: retrieval_similarity length mismatch")
    pairs: list[TrainingPair] = []
    for i in range(n):
        pairs.append(
            TrainingPair(
                query=queries[i],
                chunk_id=int(chunk_ids[i]),
                relevance=float(rel[i]),
                query_embedding=q[i].tolist(),
                chunk_embedding=c[i].tolist(),
                retrieval_similarity=(
                    float(retrieval[i]) if retrieval is not None else None
                ),
            )
        )
    chunk_repr = str(meta.get("chunk_repr") or CHUNK_REPR_COMBINED)
    if chunk_repr not in VALID_CHUNK_REPRS:
        raise RuntimeError(f"Corrupt .tcz {p}: bad chunk_repr {chunk_repr!r}")
    return FrozenCorpus(
        corpus_id=str(meta.get("corpus_id") or corpus_id_from_path(p)),
        path=p,
        chunk_repr=chunk_repr,
        retriever=meta.get("retriever"),
        created_at=str(meta.get("created_at") or ""),
        pairs=pairs,
        retrieval_similarity=retrieval,
    )


def freeze_training_corpus(
    *,
    chunk_repr: str,
    retriever: str | None = None,
    prep_fields: bool = True,
    lock_timeout_sec: float = 120.0,
) -> dict[str, Any]:
    """Load labeled pairs from fish.db under a short write lock and write a ``.tcz``.

    Field-embed prep (OpenAI) runs under ``freeze-prep`` first. The exclusive lock
    for ``freeze-training`` covers only the SQLite snapshot read; the ``.tcz`` is
    written after the lock is released.
    """
    from fish.prism.train import (
        ensure_training_field_embeddings,
        load_training_pairs_from_db,
    )

    if chunk_repr not in VALID_CHUNK_REPRS:
        raise ValueError(
            f"chunk_repr must be one of {sorted(VALID_CHUNK_REPRS)}, got {chunk_repr!r}"
        )

    field_prep: dict[str, int] | None = None
    if prep_fields and chunk_repr == CHUNK_REPR_HEADER_BODY:
        with fish_write_lock("freeze-prep"):
            field_prep = ensure_training_field_embeddings()

    with fish_write_lock("freeze-training", timeout_sec=lock_timeout_sec):
        pairs = load_training_pairs_from_db(
            retriever=retriever, chunk_repr=chunk_repr
        )
        if not pairs:
            raise RuntimeError(
                "No labeled training samples — run fish corpus collect and "
                "fish corpus label first"
            )
        # Snapshot lists while lock is held (embeddings already in memory).
        pairs = list(pairs)

    corpus_id = make_corpus_id()
    path = write_tcz(
        pairs,
        chunk_repr=chunk_repr,
        retriever=retriever,
        corpus_id=corpus_id,
    )
    pruned = prune_old_frozen_corpora()

    result: dict[str, Any] = {
        "corpus_id": corpus_id,
        "path": str(path),
        "n_pairs": len(pairs),
        "chunk_repr": chunk_repr,
        "retriever": retriever,
        "pruned": pruned,
    }
    if field_prep is not None:
        result["field_prep"] = field_prep
    return result
