from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class PrismAdapter:
    w1: np.ndarray
    b1: np.ndarray
    ln_gamma: np.ndarray
    ln_beta: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    alpha: float

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = x @ self.w1.T + self.b1
        mean = h.mean(axis=-1, keepdims=True)
        var = h.var(axis=-1, keepdims=True)
        h = self.ln_gamma * (h - mean) / np.sqrt(var + 1e-5) + self.ln_beta
        h = h * self._gelu(h)
        out = h @ self.w2.T + self.b2
        alpha = float(np.clip(self.alpha, 0.0, 1.0))
        return alpha * x + (1.0 - alpha) * out

    @staticmethod
    def _gelu(x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


@dataclass
class PrismModel:
    query_adapter: PrismAdapter
    chunk_adapter: PrismAdapter
    embed_dim: int = 1536
    embed_model: str = "text-embedding-3-small"

    def adapt_query(self, vec: list[float] | np.ndarray) -> np.ndarray:
        x = np.asarray(vec, dtype=np.float32).reshape(1, -1)
        return self.query_adapter.forward(x)[0]

    def adapt_chunk(self, vec: list[float] | np.ndarray) -> np.ndarray:
        x = np.asarray(vec, dtype=np.float32).reshape(1, -1)
        return self.chunk_adapter.forward(x)[0]

    def to_dict(self) -> dict[str, Any]:
        def pack(adapter: PrismAdapter) -> dict[str, Any]:
            return {
                "w1": adapter.w1.tolist(),
                "b1": adapter.b1.tolist(),
                "ln_gamma": adapter.ln_gamma.tolist(),
                "ln_beta": adapter.ln_beta.tolist(),
                "w2": adapter.w2.tolist(),
                "b2": adapter.b2.tolist(),
                "alpha": adapter.alpha,
            }

        return {
            "embed_dim": self.embed_dim,
            "embed_model": self.embed_model,
            "query_adapter": pack(self.query_adapter),
            "chunk_adapter": pack(self.chunk_adapter),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PrismModel:
        def unpack(raw: dict[str, Any]) -> PrismAdapter:
            return PrismAdapter(
                w1=np.asarray(raw["w1"], dtype=np.float32),
                b1=np.asarray(raw["b1"], dtype=np.float32),
                ln_gamma=np.asarray(raw["ln_gamma"], dtype=np.float32),
                ln_beta=np.asarray(raw["ln_beta"], dtype=np.float32),
                w2=np.asarray(raw["w2"], dtype=np.float32),
                b2=np.asarray(raw["b2"], dtype=np.float32),
                alpha=float(raw["alpha"]),
            )

        return cls(
            query_adapter=unpack(data["query_adapter"]),
            chunk_adapter=unpack(data["chunk_adapter"]),
            embed_dim=int(data.get("embed_dim", 1536)),
            embed_model=str(data.get("embed_model", "text-embedding-3-small")),
        )


def new_identity_model(dim: int = 1536) -> PrismModel:
    def identity_adapter() -> PrismAdapter:
        return PrismAdapter(
            w1=np.eye(dim, dtype=np.float32),
            b1=np.zeros(dim, dtype=np.float32),
            ln_gamma=np.ones(dim, dtype=np.float32),
            ln_beta=np.zeros(dim, dtype=np.float32),
            w2=np.eye(dim, dtype=np.float32),
            b2=np.zeros(dim, dtype=np.float32),
            alpha=1.0,
        )

    return PrismModel(
        query_adapter=identity_adapter(),
        chunk_adapter=identity_adapter(),
        embed_dim=dim,
    )


def save_prz(model: PrismModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model.to_dict()))


def load_prz(path: Path) -> PrismModel:
    if not path.exists():
        raise FileNotFoundError(f"PRISM model not found: {path}")
    return PrismModel.from_dict(json.loads(path.read_text()))
