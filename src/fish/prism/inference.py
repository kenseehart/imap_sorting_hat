from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from fish.config import MODELS_DIR, prism_model_path
from fish.prism.model import PrismModel, load_prz, new_identity_model


@lru_cache(maxsize=1)
def _loaded_model() -> PrismModel | None:
    path = prism_model_path()
    if path is None:
        return None
    return load_prz(path)


def clear_model_cache() -> None:
    _loaded_model.cache_clear()


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
