from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from fish.config import models_dir
from fish.prism.configs import LEGACY_MODEL_ID
from fish.prism.model import PrismModel, load_prz, new_identity_model


@lru_cache(maxsize=1)
def _loaded_model() -> PrismModel | None:
    from fish.config import active_prism_model_id

    mid = active_prism_model_id()
    if not mid or mid == LEGACY_MODEL_ID:
        return None
    return load_prism_model(mid)


@lru_cache(maxsize=8)
def _loaded_model_by_path(path: str) -> PrismModel:
    return load_prz(Path(path))


def clear_model_cache() -> None:
    _loaded_model.cache_clear()
    _loaded_model_by_path.cache_clear()


def prism_model_path_for_stem(stem: str) -> Path:
    name = stem.strip()
    if name.endswith(".prz"):
        name = name[: -len(".prz")]
    path = models_dir() / f"{name}.prz"
    if not path.exists():
        raise RuntimeError(f"PRISM model not found: {path}")
    return path


def load_prism_model(stem: str) -> PrismModel:
    """Load by model_id or prz stem (e.g. personal.20260813T120000Z)."""
    return _loaded_model_by_path(str(prism_model_path_for_stem(stem)))


def get_prism_model() -> PrismModel | None:
    return _loaded_model()


def adapt_query_embedding(vec: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    model = get_prism_model()
    if model is None:
        return arr
    return model.adapt_query(arr).astype(np.float32)


def adapt_chunk_embedding(vec: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    model = get_prism_model()
    if model is None:
        return arr
    return model.adapt_chunk(arr).astype(np.float32)


def adapt_chunk_for_model(vec: list[float] | np.ndarray, model_id: str) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if model_id == LEGACY_MODEL_ID:
        return arr
    model = load_prism_model(model_id)
    return model.adapt_chunk(arr).astype(np.float32)


def adapt_query_for_model(vec: list[float] | np.ndarray, model_id: str) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if model_id == LEGACY_MODEL_ID:
        return arr
    model = load_prism_model(model_id)
    return model.adapt_query(arr).astype(np.float32)


def cosine_similarity(
    a: list[float] | np.ndarray, b: list[float] | np.ndarray
) -> float:
    va = np.asarray(a, dtype=np.float32).reshape(-1)
    vb = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def compose_chunk_vector(
    db: Any,
    item_id: int,
    chunk_repr: str,
) -> np.ndarray | None:
    """Frozen chunk vector from SQLite raw embeds for train / re-index."""
    from fish.prism.model import CHUNK_REPR_COMBINED, CHUNK_REPR_HEADER_BODY
    from fish.store import get_raw_embedding, get_raw_field_embeddings

    if chunk_repr == CHUNK_REPR_COMBINED:
        return get_raw_embedding(db, item_id)
    if chunk_repr == CHUNK_REPR_HEADER_BODY:
        fields = get_raw_field_embeddings(db, item_id)
        h, b = fields.get("header"), fields.get("body")
        if h is None or b is None:
            return None
        return np.concatenate(
            [
                np.asarray(h, dtype=np.float32).reshape(-1),
                np.asarray(b, dtype=np.float32).reshape(-1),
            ]
        )
    raise ValueError(f"Unknown chunk_repr {chunk_repr!r}")
