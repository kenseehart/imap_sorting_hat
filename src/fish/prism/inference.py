from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from fish.config import models_dir, prism_model_path
from fish.prism.model import PrismModel, load_prz, new_identity_model


@lru_cache(maxsize=1)
def _loaded_model() -> PrismModel | None:
    path = prism_model_path()
    if path is None:
        return None
    return load_prz(path)


def clear_model_cache() -> None:
    _loaded_model.cache_clear()
    _loaded_model_by_path.cache_clear()


@lru_cache(maxsize=8)
def _loaded_model_by_path(path: str) -> PrismModel:
    return load_prz(Path(path))


def prism_model_path_for_stem(stem: str) -> Path:
    name = stem.strip()
    if name.endswith(".prz"):
        name = name[: -len(".prz")]
    path = models_dir() / f"{name}.prz"
    if not path.exists():
        raise RuntimeError(f"PRISM model not found: {path}")
    return path


def load_prism_model(stem: str) -> PrismModel:
    return _loaded_model_by_path(str(prism_model_path_for_stem(stem)))


def get_prism_model() -> PrismModel | None:
    return _loaded_model()


def adapt_query_embedding(vec: list[float]) -> list[float]:
    model = get_prism_model()
    if model is None:
        return vec
    adapted = model.adapt_query(vec)
    return adapted.astype(np.float32).tolist()


def adapt_chunk_embedding(vec: list[float]) -> list[float]:
    model = get_prism_model()
    if model is None:
        return vec
    adapted = model.adapt_chunk(vec)
    return adapted.astype(np.float32).tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)
