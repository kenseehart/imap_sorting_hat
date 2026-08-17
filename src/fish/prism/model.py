"""PRISM dual adapters + binary zip .prz serialization (no residual alpha)."""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PRZ_FORMAT_VERSION = 4

CHUNK_REPR_COMBINED = "combined"
CHUNK_REPR_HEADER_BODY = "header_body"
VALID_CHUNK_REPR = frozenset({CHUNK_REPR_COMBINED, CHUNK_REPR_HEADER_BODY})

ADAPTER_SHARING_DUAL = "dual"
ADAPTER_SHARING_SIAMESE = "siamese"
VALID_ADAPTER_SHARING = frozenset({ADAPTER_SHARING_DUAL, ADAPTER_SHARING_SIAMESE})

SCORING_COSINE = "cosine"
SCORING_MLP_HEAD = "mlp_head"
VALID_SCORING = frozenset({SCORING_COSINE, SCORING_MLP_HEAD})


@dataclass
class PrismAdapter:
    w1: np.ndarray
    b1: np.ndarray
    ln_gamma: np.ndarray
    ln_beta: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    identity: bool = False

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.identity:
            return x
        h = x @ self.w1.T + self.b1
        mean = h.mean(axis=-1, keepdims=True)
        var = h.var(axis=-1, keepdims=True)
        h = self.ln_gamma * (h - mean) / np.sqrt(var + 1e-5) + self.ln_beta
        h = h * self._gelu(h)
        return h @ self.w2.T + self.b2

    @staticmethod
    def _gelu(x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


@dataclass
class RerankHead:
    """MLP: concat(Aq, Ac) → hidden → scalar relevance in [0, 1]."""

    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray

    def forward(self, concat: np.ndarray) -> float:
        x = np.asarray(concat, dtype=np.float32).reshape(1, -1)
        h = x @ self.w1.T + self.b1
        h = 0.5 * h * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (h + 0.044715 * h**3)))
        logit = (h @ self.w2.T + self.b2).reshape(-1)[0]
        return float(1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0))))


@dataclass
class PrismModel:
    query_adapter: PrismAdapter
    chunk_adapter: PrismAdapter
    embed_dim: int = 1536
    embed_model: str = "text-embedding-3-small"
    model_id: str | None = None
    config_name: str | None = None
    chunk_repr: str = CHUNK_REPR_COMBINED
    adapter_sharing: str = ADAPTER_SHARING_DUAL
    scoring: str = SCORING_COSINE
    rerank_head: RerankHead | None = None

    @property
    def chunk_input_dim(self) -> int:
        if self.chunk_repr == CHUNK_REPR_HEADER_BODY:
            return int(self.embed_dim) * 2
        return int(self.embed_dim)

    def adapt_query(self, vec: list[float] | np.ndarray) -> np.ndarray:
        x = np.asarray(vec, dtype=np.float32).reshape(1, -1)
        if x.shape[-1] != self.embed_dim:
            raise ValueError(
                f"query dim {x.shape[-1]} != embed_dim {self.embed_dim}"
            )
        return self.query_adapter.forward(x)[0]

    def adapt_chunk(self, vec: list[float] | np.ndarray) -> np.ndarray:
        x = np.asarray(vec, dtype=np.float32).reshape(1, -1)
        expect = self.chunk_input_dim
        if x.shape[-1] != expect:
            raise ValueError(
                f"chunk dim {x.shape[-1]} != chunk_input_dim {expect} "
                f"(chunk_repr={self.chunk_repr!r})"
            )
        # Identity baseline for header_body: mean of E(h)|E(b) halves → embed_dim
        # (raw identity would leave 2d and break cosine vs A_q).
        if self.chunk_adapter.identity and expect == 2 * self.embed_dim:
            half = int(self.embed_dim)
            return 0.5 * (x[0, :half] + x[0, half:])
        return self.chunk_adapter.forward(x)[0]

    def score_pair(
        self,
        query_vec: list[float] | np.ndarray,
        chunk_vec: list[float] | np.ndarray,
    ) -> float:
        """Relevance score: adapted cosine, or MLP head if scoring=mlp_head."""
        aq = self.adapt_query(query_vec)
        ac = self.adapt_chunk(chunk_vec)
        if self.scoring == SCORING_MLP_HEAD:
            if self.rerank_head is None:
                raise ValueError("scoring=mlp_head requires rerank_head weights")
            return self.rerank_head.forward(np.concatenate([aq, ac], axis=-1))
        denom = float(np.linalg.norm(aq) * np.linalg.norm(ac))
        if denom == 0:
            return 0.0
        return float(np.dot(aq, ac) / denom)

    def to_dict(self) -> dict[str, Any]:
        def pack(adapter: PrismAdapter) -> dict[str, Any]:
            return {
                "w1": adapter.w1.tolist(),
                "b1": adapter.b1.tolist(),
                "ln_gamma": adapter.ln_gamma.tolist(),
                "ln_beta": adapter.ln_beta.tolist(),
                "w2": adapter.w2.tolist(),
                "b2": adapter.b2.tolist(),
            }

        out: dict[str, Any] = {
            "embed_dim": self.embed_dim,
            "embed_model": self.embed_model,
            "model_id": self.model_id,
            "config_name": self.config_name,
            "chunk_repr": self.chunk_repr,
            "adapter_sharing": self.adapter_sharing,
            "scoring": self.scoring,
            "query_adapter": pack(self.query_adapter),
            "chunk_adapter": pack(self.chunk_adapter),
        }
        if self.rerank_head is not None:
            out["rerank_head"] = {
                "w1": self.rerank_head.w1.tolist(),
                "b1": self.rerank_head.b1.tolist(),
                "w2": self.rerank_head.w2.tolist(),
                "b2": self.rerank_head.b2.tolist(),
            }
        return out

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
            )

        chunk_repr = str(data.get("chunk_repr") or CHUNK_REPR_COMBINED)
        if chunk_repr not in VALID_CHUNK_REPR:
            raise ValueError(f"Invalid chunk_repr {chunk_repr!r}")
        sharing = str(data.get("adapter_sharing") or ADAPTER_SHARING_DUAL)
        if sharing not in VALID_ADAPTER_SHARING:
            raise ValueError(f"Invalid adapter_sharing {sharing!r}")
        scoring = str(data.get("scoring") or SCORING_COSINE)
        if scoring not in VALID_SCORING:
            raise ValueError(f"Invalid scoring {scoring!r}")
        head = None
        raw_head = data.get("rerank_head")
        if isinstance(raw_head, dict):
            head = RerankHead(
                w1=np.asarray(raw_head["w1"], dtype=np.float32),
                b1=np.asarray(raw_head["b1"], dtype=np.float32),
                w2=np.asarray(raw_head["w2"], dtype=np.float32),
                b2=np.asarray(raw_head["b2"], dtype=np.float32),
            )
        return cls(
            query_adapter=unpack(data["query_adapter"]),
            chunk_adapter=unpack(data["chunk_adapter"]),
            embed_dim=int(data.get("embed_dim", 1536)),
            embed_model=str(data.get("embed_model", "text-embedding-3-small")),
            model_id=data.get("model_id"),
            config_name=data.get("config_name"),
            chunk_repr=chunk_repr,
            adapter_sharing=sharing,
            scoring=scoring,
            rerank_head=head,
        )


def new_identity_model(
    dim: int = 1536, *, chunk_repr: str = CHUNK_REPR_COMBINED
) -> PrismModel:
    """Pass-through adapters (baseline ≈ raw cosine for combined repr)."""
    if chunk_repr not in VALID_CHUNK_REPR:
        raise ValueError(f"Invalid chunk_repr {chunk_repr!r}")
    chunk_in = dim * 2 if chunk_repr == CHUNK_REPR_HEADER_BODY else dim

    def identity_adapter(in_dim: int, out_dim: int) -> PrismAdapter:
        # Unused weights; forward short-circuits via identity=True.
        return PrismAdapter(
            w1=np.zeros((out_dim, in_dim), dtype=np.float32),
            b1=np.zeros(out_dim, dtype=np.float32),
            ln_gamma=np.ones(out_dim, dtype=np.float32),
            ln_beta=np.zeros(out_dim, dtype=np.float32),
            w2=np.eye(out_dim, dtype=np.float32),
            b2=np.zeros(out_dim, dtype=np.float32),
            identity=True,
        )

    return PrismModel(
        query_adapter=identity_adapter(dim, dim),
        chunk_adapter=identity_adapter(chunk_in, dim),
        embed_dim=dim,
        chunk_repr=chunk_repr,
    )


def _adapter_arrays(prefix: str, adapter: PrismAdapter) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_w1": np.asarray(adapter.w1, dtype=np.float32),
        f"{prefix}_b1": np.asarray(adapter.b1, dtype=np.float32),
        f"{prefix}_ln_gamma": np.asarray(adapter.ln_gamma, dtype=np.float32),
        f"{prefix}_ln_beta": np.asarray(adapter.ln_beta, dtype=np.float32),
        f"{prefix}_w2": np.asarray(adapter.w2, dtype=np.float32),
        f"{prefix}_b2": np.asarray(adapter.b2, dtype=np.float32),
    }


def _adapter_from_npz(prefix: str, data: np.lib.npyio.NpzFile) -> PrismAdapter:
    return PrismAdapter(
        w1=np.asarray(data[f"{prefix}_w1"], dtype=np.float32),
        b1=np.asarray(data[f"{prefix}_b1"], dtype=np.float32),
        ln_gamma=np.asarray(data[f"{prefix}_ln_gamma"], dtype=np.float32),
        ln_beta=np.asarray(data[f"{prefix}_ln_beta"], dtype=np.float32),
        w2=np.asarray(data[f"{prefix}_w2"], dtype=np.float32),
        b2=np.asarray(data[f"{prefix}_b2"], dtype=np.float32),
    )


def save_prz(model: PrismModel, path: Path) -> None:
    """Write binary zip .prz (npz weights + JSON manifest)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": PRZ_FORMAT_VERSION,
        "embed_dim": model.embed_dim,
        "embed_model": model.embed_model,
        "model_id": model.model_id,
        "config_name": model.config_name,
        "chunk_repr": model.chunk_repr,
        "adapter_sharing": model.adapter_sharing,
        "scoring": model.scoring,
    }
    arrays = {
        **_adapter_arrays("q", model.query_adapter),
        **_adapter_arrays("c", model.chunk_adapter),
    }
    if model.rerank_head is not None:
        arrays["head_w1"] = np.asarray(model.rerank_head.w1, dtype=np.float32)
        arrays["head_b1"] = np.asarray(model.rerank_head.b1, dtype=np.float32)
        arrays["head_w2"] = np.asarray(model.rerank_head.w2, dtype=np.float32)
        arrays["head_b2"] = np.asarray(model.rerank_head.b2, dtype=np.float32)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    weights_bytes = buf.getvalue()
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        zf.writestr("weights.npz", weights_bytes)


def load_prz(path: Path) -> PrismModel:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PRISM model not found: {path}")
    raw = path.read_bytes()
    # Legacy format: plain JSON text (huge). New format: zip.
    if raw[:1] == b"{" or raw[:1] == b"[":
        return PrismModel.from_dict(json.loads(raw.decode("utf-8")))
    if not zipfile.is_zipfile(path):
        raise RuntimeError(f"Unrecognized .prz format: {path}")
    with zipfile.ZipFile(path, "r") as zf:
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        weights_bytes = zf.read("weights.npz")
    data = np.load(io.BytesIO(weights_bytes))
    fmt = int(manifest.get("format") or 1)
    chunk_repr = str(manifest.get("chunk_repr") or CHUNK_REPR_COMBINED)
    if chunk_repr not in VALID_CHUNK_REPR:
        raise ValueError(f"Invalid chunk_repr in {path}: {chunk_repr!r}")
    sharing = str(manifest.get("adapter_sharing") or ADAPTER_SHARING_DUAL)
    if sharing not in VALID_ADAPTER_SHARING:
        raise ValueError(f"Invalid adapter_sharing in {path}: {sharing!r}")
    scoring = str(manifest.get("scoring") or SCORING_COSINE)
    if scoring not in VALID_SCORING:
        raise ValueError(f"Invalid scoring in {path}: {scoring!r}")
    if fmt < 3 and f"q_alpha" in data.files:
        # Old residual-alpha checkpoints: load weights, ignore alpha dilution.
        pass
    head = None
    if "head_w1" in data.files:
        head = RerankHead(
            w1=np.asarray(data["head_w1"], dtype=np.float32),
            b1=np.asarray(data["head_b1"], dtype=np.float32),
            w2=np.asarray(data["head_w2"], dtype=np.float32),
            b2=np.asarray(data["head_b2"], dtype=np.float32),
        )
    if scoring == SCORING_MLP_HEAD and head is None:
        raise RuntimeError(f"scoring=mlp_head but no head weights in {path}")
    return PrismModel(
        query_adapter=_adapter_from_npz("q", data),
        chunk_adapter=_adapter_from_npz("c", data),
        embed_dim=int(manifest.get("embed_dim", 1536)),
        embed_model=str(manifest.get("embed_model", "text-embedding-3-small")),
        model_id=manifest.get("model_id"),
        config_name=manifest.get("config_name"),
        chunk_repr=chunk_repr,
        adapter_sharing=sharing,
        scoring=scoring,
        rerank_head=head,
    )
