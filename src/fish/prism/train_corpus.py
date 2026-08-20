"""Frozen training corpus (``.tcz``) — pytorch-ready snapshot.

Format: one JSON metadata line, then a zip of float32 arrays.
``fish corpus freeze-training`` writes ``models/corpora/train_corpus_{ts}.tcz``.
``fish prism-train`` loads a ``.tcz`` and never opens fish.db for the epoch loop.

v3 (preferred): both ``joint`` and ``split`` chunk matrices in one file
(``c_joint.npy``, ``c_split.npy``) so bakeoff trains share one snapshot.
Legacy member names ``c_combined.npy`` / ``c_header_body.npy`` still load.
v2: single ``c.npy`` + ``chunk_repr`` (still loadable).
v1 (bare zip) is not supported — incompatible files are deleted, not migrated.
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from fish.config import models_dir
from fish.prism.model import (
    CHUNK_REPR_JOINT,
    CHUNK_REPR_SPLIT,
    normalize_chunk_repr,
)
from fish.write_lock import fish_write_lock

if TYPE_CHECKING:
    from fish.prism.train import TrainingPair

TCZ_VERSION = 3
TCZ_VERSION_V2 = 2
SUPPORTED_TCZ_VERSIONS = frozenset({TCZ_VERSION_V2, TCZ_VERSION})
TCZ_SUFFIX = ".tcz"
CHUNK_REPR_BOTH = "both"
VALID_CHUNK_REPRS = frozenset(
    {CHUNK_REPR_JOINT, CHUNK_REPR_SPLIT, CHUNK_REPR_BOTH}
)
# Keep at most this many train_corpus_*.tcz files under models/corpora/.
MAX_FROZEN_CORPORA = 3


class IncompatibleTczError(RuntimeError):
    """``.tcz`` is not a supported version (e.g. v1 bare zip)."""


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
    """Return v2 ``.tcz`` paths only (deletes incompatible files first)."""
    delete_incompatible_frozen_corpora()
    root = corpora_dir()
    return sorted(p for p in root.glob(f"train_corpus_*{TCZ_SUFFIX}") if p.is_file())


def read_tcz_meta(path: Path | str) -> dict[str, Any]:
    """Read leading JSON metadata without loading embedding arrays."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Frozen corpus not found: {p}")
    with p.open("rb") as f:
        head = f.read(2)
        if head == b"PK":
            raise IncompatibleTczError(
                f"{p.name}: v1 zip .tcz is unsupported — delete and re-freeze"
            )
        f.seek(0)
        line = f.readline()
        if not line.strip():
            raise IncompatibleTczError(f"{p.name}: empty or missing JSON header")
        try:
            meta = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise IncompatibleTczError(
                f"{p.name}: invalid JSON header ({exc})"
            ) from exc
    if not isinstance(meta, dict):
        raise IncompatibleTczError(f"{p.name}: JSON header must be an object")
    version = int(meta.get("version") or 0)
    if version not in SUPPORTED_TCZ_VERSIONS:
        raise IncompatibleTczError(
            f"{p.name}: unsupported .tcz version {version} "
            f"(supported: {sorted(SUPPORTED_TCZ_VERSIONS)})"
        )
    return meta


def delete_incompatible_frozen_corpora() -> list[str]:
    """Delete ``.tcz`` files that are not current-version (no migration)."""
    root = corpora_dir()
    deleted: list[str] = []
    for path in sorted(p for p in root.glob(f"train_corpus_*{TCZ_SUFFIX}") if p.is_file()):
        try:
            read_tcz_meta(path)
        except (IncompatibleTczError, OSError, UnicodeDecodeError):
            cid = corpus_id_from_path(path)
            path.unlink(missing_ok=True)
            deleted.append(cid)
    return deleted


def list_frozen_corpora_info() -> list[dict[str, Any]]:
    """List frozen corpora with label counts from the JSON header."""
    delete_incompatible_frozen_corpora()
    out: list[dict[str, Any]] = []
    for path in list_frozen_corpora():
        meta = read_tcz_meta(path)
        n_labels = int(meta.get("n_labels") or meta.get("n_pairs") or 0)
        out.append(
            {
                "corpus_id": str(meta.get("corpus_id") or corpus_id_from_path(path)),
                "path": str(path),
                "n_labels": n_labels,
                "n_pairs": int(meta.get("n_pairs") or n_labels),
                "chunk_repr": meta.get("chunk_repr"),
                "chunk_reprs": meta.get("chunk_reprs"),
                "retriever": meta.get("retriever"),
                "created_at": meta.get("created_at"),
                "q_dim": meta.get("q_dim"),
                "c_dim": meta.get("c_dim"),
                "c_dims": meta.get("c_dims"),
                "size_bytes": path.stat().st_size,
                "version": int(meta.get("version") or TCZ_VERSION),
            }
        )
    return out


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


def resolve_corpus_path(
    corpus: str | Path, *, chunk_repr: str | None = None
) -> Path:
    """Resolve ``latest``, a corpus id, or a filesystem path to a ``.tcz`` file.

    When ``corpus`` is ``latest`` and ``chunk_repr`` is set, prefer the newest
    freeze that includes that representation (dual v3 files satisfy either).
    When ``chunk_repr`` is None / ``both``, prefer the newest dual freeze.
    """
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
                f"Run: fish corpus freeze-training --chunk-repr both"
            )
        want = chunk_repr
        matched: list[Path] = []
        for path in files:
            meta = read_tcz_meta(path)
            reprs = meta.get("chunk_reprs")
            if isinstance(reprs, list) and reprs:
                have = {normalize_chunk_repr(x) for x in reprs}
            else:
                cr = meta.get("chunk_repr")
                have = {normalize_chunk_repr(cr)} if cr else set()
            if want in (None, CHUNK_REPR_BOTH):
                if CHUNK_REPR_JOINT in have and CHUNK_REPR_SPLIT in have:
                    matched.append(path)
            else:
                want_n = normalize_chunk_repr(want)
                if want_n in have:
                    matched.append(path)
        if matched:
            return matched[-1]
        if want in (None, CHUNK_REPR_BOTH):
            return files[-1]
        raise FileNotFoundError(
            f"No frozen corpora with chunk_repr={want!r} in {corpora_dir()}. "
            f"Run: fish corpus freeze-training --chunk-repr both"
        )

    path = Path(raw).expanduser()
    if path.suffix == TCZ_SUFFIX or "/" in raw or "\\" in raw or raw.startswith("."):
        if path.suffix != TCZ_SUFFIX and not path.is_file():
            path = path.with_suffix(TCZ_SUFFIX)
        if not path.is_file():
            raise FileNotFoundError(f"Frozen corpus not found: {path}")
        read_tcz_meta(path)
        return path.resolve()

    resolved = tcz_path_for_id(raw)
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Frozen corpus not found: {resolved}. "
            f"Run freeze-training or pass --corpus latest."
        )
    read_tcz_meta(resolved)
    return resolved.resolve()



@dataclass
class FrozenCorpus:
    corpus_id: str
    path: Path
    chunk_reprs: tuple[str, ...]
    retriever: str | None
    created_at: str
    queries: list[str]
    chunk_ids: list[int]
    relevance: list[float]
    q: np.ndarray
    c_by_repr: dict[str, np.ndarray]
    retrieval_similarity: list[float] | None = None
    n_labels: int = 0
    _pair_cache: dict[str, list[Any]] = field(default_factory=dict, repr=False)

    @property
    def n_pairs(self) -> int:
        return len(self.queries)

    @property
    def chunk_repr(self) -> str:
        """Single-repr files return that name; dual files return ``both``."""
        if len(self.chunk_reprs) == 1:
            return self.chunk_reprs[0]
        return CHUNK_REPR_BOTH

    def has_repr(self, chunk_repr: str) -> bool:
        return chunk_repr in self.c_by_repr

    def pairs_for(self, chunk_repr: str) -> list[TrainingPair]:
        """Build TrainingPair list for one chunk representation (cached)."""
        from fish.prism.train import TrainingPair

        if chunk_repr not in self.c_by_repr:
            raise KeyError(
                f"Frozen corpus {self.corpus_id} has no chunk_repr={chunk_repr!r}; "
                f"available={sorted(self.c_by_repr)}"
            )
        cached = self._pair_cache.get(chunk_repr)
        if cached is not None:
            return cached  # type: ignore[return-value]
        c = self.c_by_repr[chunk_repr]
        pairs: list[TrainingPair] = []
        for i in range(len(self.queries)):
            pairs.append(
                TrainingPair(
                    query=self.queries[i],
                    chunk_id=int(self.chunk_ids[i]),
                    relevance=float(self.relevance[i]),
                    query_embedding=self.q[i],
                    chunk_embedding=c[i],
                    retrieval_similarity=(
                        float(self.retrieval_similarity[i])
                        if self.retrieval_similarity is not None
                        else None
                    ),
                )
            )
        self._pair_cache[chunk_repr] = pairs
        return pairs

    @property
    def pairs(self) -> list[TrainingPair]:
        if len(self.chunk_reprs) != 1:
            raise RuntimeError(
                f"Frozen corpus {self.corpus_id} is dual-repr — use pairs_for(chunk_repr)"
            )
        return self.pairs_for(self.chunk_reprs[0])


def _np_save_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def _np_load_bytes(data: bytes) -> np.ndarray:
    return np.load(io.BytesIO(data), allow_pickle=False)


def _build_meta(
    *,
    cid: str,
    chunk_reprs: list[str],
    retriever: str | None,
    created_at: str,
    n_pairs: int,
    q_dim: int,
    c_dims: dict[str, int],
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "version": TCZ_VERSION,
        "corpus_id": cid,
        "chunk_reprs": list(chunk_reprs),
        "retriever": retriever,
        "created_at": created_at,
        "n_labels": n_pairs,
        "n_pairs": n_pairs,
        "q_dim": q_dim,
        "c_dims": dict(c_dims),
        "has_retrieval_similarity": True,
    }
    # Convenience for single-repr consumers / listing UI.
    if len(chunk_reprs) == 1:
        meta["chunk_repr"] = chunk_reprs[0]
        meta["c_dim"] = c_dims[chunk_reprs[0]]
    else:
        meta["chunk_repr"] = CHUNK_REPR_BOTH
    return meta


def write_tcz(
    pairs: list[TrainingPair] | None = None,
    *,
    chunk_repr: str | None = None,
    c_by_repr: dict[str, np.ndarray] | None = None,
    queries: list[str] | None = None,
    chunk_ids: list[int] | np.ndarray | None = None,
    relevance: list[float] | np.ndarray | None = None,
    q: np.ndarray | None = None,
    retriever: str | None = None,
    corpus_id: str | None = None,
    path: Path | None = None,
    retrieval_similarity: list[float] | None = None,
) -> Path:
    """Atomically write a frozen corpus (JSON header + zip; no DB lock).

    Single-repr (legacy callers): pass ``pairs`` + ``chunk_repr``.
    Dual-repr: pass aligned ``queries``/``q``/``c_by_repr``/``relevance``/``chunk_ids``.
    """
    delete_incompatible_frozen_corpora()

    if pairs is not None:
        if not pairs:
            raise ValueError("Cannot freeze empty training pair list")
        if chunk_repr is None or chunk_repr not in (
            CHUNK_REPR_JOINT,
            CHUNK_REPR_SPLIT,
        ):
            raise ValueError(
                f"chunk_repr must be joint or split when passing pairs, "
                f"got {chunk_repr!r}"
            )
        for i, p in enumerate(pairs):
            if p.query_embedding is None or p.chunk_embedding is None:
                raise ValueError(f"pair[{i}] missing embeddings")
        queries = [p.query for p in pairs]
        chunk_ids = [p.chunk_id for p in pairs]
        relevance = [p.relevance for p in pairs]
        q = np.stack(
            [np.asarray(p.query_embedding, dtype=np.float32).reshape(-1) for p in pairs]
        ).astype(np.float32, copy=False)
        c_by_repr = {
            chunk_repr: np.stack(
                [
                    np.asarray(p.chunk_embedding, dtype=np.float32).reshape(-1)
                    for p in pairs
                ]
            ).astype(np.float32, copy=False)
        }
        if retrieval_similarity is None:
            retrieval_similarity = [
                float(p.retrieval_similarity)
                if p.retrieval_similarity is not None
                else 0.0
                for p in pairs
            ]

    if (
        queries is None
        or chunk_ids is None
        or relevance is None
        or q is None
        or not c_by_repr
    ):
        raise ValueError("write_tcz requires pairs+chunk_repr or dual arrays")

    reprs = sorted(c_by_repr.keys())
    for r in reprs:
        if r not in (CHUNK_REPR_JOINT, CHUNK_REPR_SPLIT):
            raise ValueError(f"Invalid chunk_repr in c_by_repr: {r!r}")

    n = len(queries)
    q_arr = np.asarray(q, dtype=np.float32)
    if q_arr.ndim != 2 or q_arr.shape[0] != n:
        raise ValueError(f"q shape {q_arr.shape} incompatible with n_pairs={n}")
    c_dims: dict[str, int] = {}
    c_store: dict[str, np.ndarray] = {}
    for r, mat in c_by_repr.items():
        arr = np.asarray(mat, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] != n:
            raise ValueError(f"c[{r}] shape {arr.shape} incompatible with n_pairs={n}")
        c_store[r] = arr
        c_dims[r] = int(arr.shape[1])

    cid = corpus_id or make_corpus_id()
    out = path or tcz_path_for_id(cid)
    out.parent.mkdir(parents=True, exist_ok=True)

    rel = np.asarray(relevance, dtype=np.float32)
    ids = np.asarray(chunk_ids, dtype=np.int64)
    if len(rel) != n or len(ids) != n:
        raise ValueError("relevance/chunk_ids length must match pairs")
    if retrieval_similarity is None:
        retrieval_similarity = [0.0] * n
    rs = np.asarray(retrieval_similarity, dtype=np.float32)
    if len(rs) != n:
        raise ValueError("retrieval_similarity length must match pairs")

    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta = _build_meta(
        cid=cid,
        chunk_reprs=reprs,
        retriever=retriever,
        created_at=created_at,
        n_pairs=n,
        q_dim=int(q_arr.shape[1]),
        c_dims=c_dims,
    )

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("meta.json", json.dumps(meta, indent=2, sort_keys=True))
        zf.writestr("queries.json", json.dumps(queries))
        zf.writestr("q.npy", _np_save_bytes(q_arr))
        zf.writestr("rel.npy", _np_save_bytes(rel))
        zf.writestr("chunk_ids.npy", _np_save_bytes(ids))
        zf.writestr("retrieval_similarity.npy", _np_save_bytes(rs))
        if len(reprs) == 1:
            # v2-compatible member name for single-repr freezes.
            zf.writestr("c.npy", _np_save_bytes(c_store[reprs[0]]))
        for r, arr in c_store.items():
            zf.writestr(f"c_{r}.npy", _np_save_bytes(arr))

    header = (json.dumps(meta, separators=(",", ":"), sort_keys=True) + "\n").encode(
        "utf-8"
    )
    tmp = out.with_suffix(out.suffix + ".tmp")
    try:
        with tmp.open("wb") as f:
            f.write(header)
            f.write(zip_buf.getvalue())
        tmp.replace(out)
    except Exception:
        if tmp.is_file():
            tmp.unlink(missing_ok=True)
        raise
    return out


def load_tcz(path: Path | str) -> FrozenCorpus:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Frozen corpus not found: {p}")
    with p.open("rb") as f:
        head = f.read(2)
        if head == b"PK":
            raise IncompatibleTczError(
                f"{p.name}: v1 zip .tcz is unsupported — delete and re-freeze"
            )
        f.seek(0)
        line = f.readline()
        try:
            file_meta = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise IncompatibleTczError(
                f"{p.name}: invalid JSON header ({exc})"
            ) from exc
        zip_bytes = f.read()

    version = int(file_meta.get("version") or 0)
    if version not in SUPPORTED_TCZ_VERSIONS:
        raise IncompatibleTczError(
            f"{p.name}: unsupported .tcz version {version} "
            f"(supported: {sorted(SUPPORTED_TCZ_VERSIONS)})"
        )

    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        meta = json.loads(zf.read("meta.json").decode())
        for key in (
            "n_labels",
            "n_pairs",
            "corpus_id",
            "chunk_repr",
            "chunk_reprs",
            "created_at",
        ):
            if key not in meta and key in file_meta:
                meta[key] = file_meta[key]
        queries: list[str] = json.loads(zf.read("queries.json").decode())
        q = _np_load_bytes(zf.read("q.npy"))
        rel = _np_load_bytes(zf.read("rel.npy"))
        chunk_ids = _np_load_bytes(zf.read("chunk_ids.npy"))
        retrieval: list[float] | None = None
        if "retrieval_similarity.npy" in zf.namelist():
            retrieval = _np_load_bytes(zf.read("retrieval_similarity.npy")).tolist()

        names = set(zf.namelist())
        c_by_repr: dict[str, np.ndarray] = {}
        # Canonical members + legacy combined/header_body names from earlier freezes.
        member_aliases = {
            "c_joint.npy": CHUNK_REPR_JOINT,
            "c_split.npy": CHUNK_REPR_SPLIT,
            "c_combined.npy": CHUNK_REPR_JOINT,
            "c_header_body.npy": CHUNK_REPR_SPLIT,
        }
        for member, r in member_aliases.items():
            if member in names and r not in c_by_repr:
                c_by_repr[r] = _np_load_bytes(zf.read(member))
        if not c_by_repr and "c.npy" in names:
            cr = normalize_chunk_repr(meta.get("chunk_repr"))
            c_by_repr[cr] = _np_load_bytes(zf.read("c.npy"))

    if not c_by_repr:
        raise RuntimeError(f"Corrupt .tcz {p}: no chunk matrices")

    n = len(queries)
    if not (len(chunk_ids) == n == len(q) == len(rel)):
        raise RuntimeError(f"Corrupt .tcz {p}: length mismatch")
    for r, mat in c_by_repr.items():
        if len(mat) != n:
            raise RuntimeError(f"Corrupt .tcz {p}: c_{r} length mismatch")
    if retrieval is not None and len(retrieval) != n:
        raise RuntimeError(f"Corrupt .tcz {p}: retrieval_similarity length mismatch")

    meta_reprs = meta.get("chunk_reprs")
    if isinstance(meta_reprs, list) and meta_reprs:
        chunk_reprs = tuple(normalize_chunk_repr(x) for x in meta_reprs)
    else:
        chunk_reprs = tuple(sorted(c_by_repr.keys()))

    n_labels = int(meta.get("n_labels") or meta.get("n_pairs") or n)
    return FrozenCorpus(
        corpus_id=str(meta.get("corpus_id") or corpus_id_from_path(p)),
        path=p,
        chunk_reprs=chunk_reprs,
        retriever=meta.get("retriever"),
        created_at=str(meta.get("created_at") or ""),
        queries=queries,
        chunk_ids=[int(x) for x in chunk_ids.tolist()],
        relevance=[float(x) for x in rel.tolist()],
        q=q,
        c_by_repr=c_by_repr,
        retrieval_similarity=retrieval,
        n_labels=n_labels,
    )


def freeze_training_corpus(
    *,
    chunk_repr: str = CHUNK_REPR_BOTH,
    retriever: str | None = None,
    prep_fields: bool = True,
    lock_timeout_sec: float = 120.0,
) -> dict[str, Any]:
    """Load labeled pairs from fish.db under a short write lock and write a ``.tcz``.

    Default ``chunk_repr=both`` writes joint + split matrices into one
    file (bakeoff). Field-embed prep runs under ``freeze-prep`` when split
    is included. The exclusive lock for ``freeze-training`` covers only the
    SQLite snapshot read; the ``.tcz`` is written after the lock is released.
    """
    import os

    from compute.tasks import TaskProgress
    from fish.prism.train import (
        ensure_training_field_embeddings,
        load_dual_training_arrays_from_db,
        load_training_pairs_from_db,
    )

    if chunk_repr not in VALID_CHUNK_REPRS:
        raise ValueError(
            f"chunk_repr must be one of {sorted(VALID_CHUNK_REPRS)}, got {chunk_repr!r}"
        )

    need_fields = chunk_repr in (CHUNK_REPR_SPLIT, CHUNK_REPR_BOTH)
    steps = 3 if (prep_fields and need_fields) else 2
    with TaskProgress(
        module="fish",
        task="freeze",
        n=steps,
        sec_per_unit_prior=30.0,
        detail=f"freeze chunk_repr={chunk_repr}",
        resource=os.environ.get("COMPUTE_RESOURCE"),
    ) as progress:
        field_prep: dict[str, int] | None = None
        step = 0
        if prep_fields and need_fields:
            progress.update(step, detail="freeze-prep field embeds")
            # Short locks inside ensure/embed_field_pending — not across OpenAI.
            field_prep = ensure_training_field_embeddings()
            step += 1
            progress.update(step, detail="field prep done", force=True)

        progress.update(step, detail="loading labeled pairs", force=True)
        pairs = None
        arrays = None
        with fish_write_lock("freeze-training", timeout_sec=lock_timeout_sec):
            if chunk_repr == CHUNK_REPR_BOTH:
                arrays = load_dual_training_arrays_from_db(retriever=retriever)
                if int(arrays["n_pairs"]) == 0:
                    raise RuntimeError(
                        "No labeled training samples with both joint and "
                        "split embeddings — run fish corpus label and "
                        "fish embed --fields --training-only"
                    )
            else:
                pairs = load_training_pairs_from_db(
                    retriever=retriever, chunk_repr=chunk_repr
                )
                if not pairs:
                    raise RuntimeError(
                        "No labeled training samples — run fish corpus collect and "
                        "fish corpus label first"
                    )
                pairs = list(pairs)
        step += 1

        corpus_id = make_corpus_id()
        if arrays is not None:
            n_pairs = int(arrays["n_pairs"])
            progress.update(
                step, detail=f"writing dual .tcz ({n_pairs} pairs)", force=True
            )
            path = write_tcz(
                queries=arrays["queries"],
                chunk_ids=arrays["chunk_ids"],
                relevance=arrays["relevance"],
                q=arrays["q"],
                c_by_repr=arrays["c_by_repr"],
                retrieval_similarity=arrays["retrieval_similarity"],
                retriever=retriever,
                corpus_id=corpus_id,
            )
            out_repr = CHUNK_REPR_BOTH
        else:
            assert pairs is not None
            progress.update(
                step, detail=f"writing .tcz ({len(pairs)} pairs)", force=True
            )
            path = write_tcz(
                pairs,
                chunk_repr=chunk_repr,
                retriever=retriever,
                corpus_id=corpus_id,
            )
            n_pairs = len(pairs)
            out_repr = chunk_repr

        pruned = prune_old_frozen_corpora()
        removed_old = delete_incompatible_frozen_corpora()
        step += 1
        progress.update(step, detail=f"wrote {corpus_id}", force=True)

    result: dict[str, Any] = {
        "corpus_id": corpus_id,
        "path": str(path),
        "n_labels": n_pairs,
        "n_pairs": n_pairs,
        "chunk_repr": out_repr,
        "retriever": retriever,
        "pruned": pruned,
        "deleted_incompatible": removed_old,
    }
    if field_prep is not None:
        result["field_prep"] = field_prep
    if progress.task_id:
        result["task_id"] = progress.task_id
    return result
