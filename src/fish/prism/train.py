from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from fish.config import EMBED_DIM, embedding_model, models_dir
from fish.prism.inference import cosine_similarity, compose_chunk_vector
from fish.prism.model import (
    CHUNK_REPR_JOINT,
    CHUNK_REPR_SPLIT,
    SCORING_COSINE,
    SCORING_MLP_HEAD,
    PrismAdapter,
    PrismModel,
    RerankHead,
    new_identity_model,
    save_prz,
)
from fish.store import db_conn, get_raw_embedding, init_db, load_labeled_training_pairs
from fish.write_lock import fish_write_lock


def resolve_train_device(device: str | None = None) -> Any:
    """Resolve ``cpu`` / ``cuda`` / ``auto`` (default) to a ``torch.device``."""
    import torch

    raw = (device or "auto").strip().lower()
    if raw in ("auto", ""):
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if raw == "cpu":
        return torch.device("cpu")
    if raw in ("cuda", "gpu"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Requested CUDA but torch.cuda.is_available() is False. "
                "Install a CUDA build of torch or pass --device cpu."
            )
        return torch.device("cuda")
    if raw.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested {raw!r} but CUDA is unavailable")
        return torch.device(raw)
    raise ValueError(f"Unknown train device {device!r} (use auto, cpu, cuda, or cuda:N)")


def train_progress_path(config_name: str) -> Path:
    """Live progress JSON for agents/canvas: models/checkpoints/{config}.progress.json"""
    return models_dir() / "checkpoints" / f"{config_name}.progress.json"


def _optimizer_to_device(opt: Any, device: Any) -> None:
    for state in opt.state.values():
        for key, val in state.items():
            if hasattr(val, "to"):
                state[key] = val.to(device)


@dataclass
class TrainingPair:
    query: str
    chunk_id: int
    relevance: float
    query_embedding: np.ndarray | None = None
    chunk_embedding: np.ndarray | None = None
    retrieval_similarity: float | None = None


def _pair_hash(query: str, chunk_id: int) -> str:
    return hashlib.sha256(f"{query}\0{chunk_id}".encode()).hexdigest()


def load_training_pairs_from_db(
    *,
    exclude_superseded: bool = True,
    retriever: str | None = None,
    chunk_repr: str = CHUNK_REPR_JOINT,
) -> list[TrainingPair]:
    init_db()
    with db_conn() as db:
        rows = load_labeled_training_pairs(
            db,
            exclude_superseded=exclude_superseded,
            retriever=retriever,
        )
        pairs: list[TrainingPair] = []
        missing_fields = 0
        for row in rows:
            q_emb = row.get("query_embedding")
            if not isinstance(q_emb, np.ndarray) or q_emb.size == 0:
                continue
            q_emb = np.asarray(q_emb, dtype=np.float32).reshape(-1)
            item_id = int(row["corpus_item_id"])
            if chunk_repr == CHUNK_REPR_JOINT:
                c_emb = row.get("message_embedding")
                if not isinstance(c_emb, np.ndarray) or c_emb.size == 0:
                    c_emb = get_raw_embedding(db, item_id)
            else:
                c_emb = compose_chunk_vector(db, item_id, chunk_repr)
                if c_emb is None:
                    missing_fields += 1
                    continue
            if not isinstance(c_emb, np.ndarray) or c_emb.size == 0:
                continue
            c_emb = np.asarray(c_emb, dtype=np.float32).reshape(-1)
            pairs.append(
                TrainingPair(
                    query=row["query_text"],
                    chunk_id=item_id,
                    relevance=float(row["target_relevance"]),
                    query_embedding=q_emb,
                    chunk_embedding=c_emb,
                    retrieval_similarity=(
                        float(row["retrieval_similarity"])
                        if row.get("retrieval_similarity") is not None
                        else None
                    ),
                )
            )
    if chunk_repr == CHUNK_REPR_SPLIT and not pairs and missing_fields:
        raise RuntimeError(
            f"chunk_repr=split but no pairs have split (header+body) embeddings "
            f"({missing_fields} labeled rows missing fields). "
            f"Run: fish embed --fields --training-only"
        )
    return pairs


def load_dual_training_arrays_from_db(
    *,
    exclude_superseded: bool = True,
    retriever: str | None = None,
) -> dict[str, Any]:
    """Aligned arrays for a dual-repr freeze (joint + split).

    Only includes labeled pairs that have both chunk representations so bakeoff
    models train on the exact same (query, doc, relevance) rows.
    """
    init_db()
    queries: list[str] = []
    chunk_ids: list[int] = []
    relevance: list[float] = []
    retrieval: list[float] = []
    q_rows: list[np.ndarray] = []
    c_joint: list[np.ndarray] = []
    c_split_rows: list[np.ndarray] = []
    missing_joint = 0
    missing_split = 0
    with db_conn() as db:
        rows = load_labeled_training_pairs(
            db,
            exclude_superseded=exclude_superseded,
            retriever=retriever,
        )
        for row in rows:
            q_emb = row.get("query_embedding")
            if not isinstance(q_emb, np.ndarray) or q_emb.size == 0:
                continue
            q_emb = np.asarray(q_emb, dtype=np.float32).reshape(-1)
            item_id = int(row["corpus_item_id"])
            c_comb = row.get("message_embedding")
            if not isinstance(c_comb, np.ndarray) or c_comb.size == 0:
                c_comb = get_raw_embedding(db, item_id)
            if not isinstance(c_comb, np.ndarray) or c_comb.size == 0:
                missing_joint += 1
                continue
            c_comb = np.asarray(c_comb, dtype=np.float32).reshape(-1)
            c_split = compose_chunk_vector(db, item_id, CHUNK_REPR_SPLIT)
            if c_split is None:
                missing_split += 1
                continue
            c_split = np.asarray(c_split, dtype=np.float32).reshape(-1)
            queries.append(str(row["query_text"]))
            chunk_ids.append(item_id)
            relevance.append(float(row["target_relevance"]))
            retrieval.append(
                float(row["retrieval_similarity"])
                if row.get("retrieval_similarity") is not None
                else 0.0
            )
            q_rows.append(q_emb)
            c_joint.append(c_comb)
            c_split_rows.append(c_split)

    n = len(queries)
    if n == 0:
        return {
            "n_pairs": 0,
            "missing_joint": missing_joint,
            "missing_split": missing_split,
            "queries": [],
            "chunk_ids": [],
            "relevance": [],
            "retrieval_similarity": [],
            "q": np.zeros((0, 0), dtype=np.float32),
            "c_by_repr": {},
        }
    return {
        "n_pairs": n,
        "missing_joint": missing_joint,
        "missing_split": missing_split,
        "queries": queries,
        "chunk_ids": chunk_ids,
        "relevance": relevance,
        "retrieval_similarity": retrieval,
        "q": np.stack(q_rows).astype(np.float32, copy=False),
        "c_by_repr": {
            CHUNK_REPR_JOINT: np.stack(c_joint).astype(np.float32, copy=False),
            CHUNK_REPR_SPLIT: np.stack(c_split_rows).astype(np.float32, copy=False),
        },
    }


def split_pairs(
    pairs: list[TrainingPair], test_fraction: float = 0.2
) -> tuple[list[TrainingPair], list[TrainingPair]]:
    train: list[TrainingPair] = []
    test: list[TrainingPair] = []
    for pair in pairs:
        h = int(_pair_hash(pair.query, pair.chunk_id), 16)
        if (h % 1000) / 1000.0 < test_fraction:
            test.append(pair)
        else:
            train.append(pair)
    return train, test


def _spearman(scores: list[float], labels: list[float]) -> float:
    if len(scores) < 2:
        return 0.0
    xs = np.asarray(scores, dtype=np.float64)
    ys = np.asarray(labels, dtype=np.float64)
    xs = xs.argsort().argsort().astype(np.float64)
    ys = ys.argsort().argsort().astype(np.float64)
    if xs.std() == 0 or ys.std() == 0:
        return 0.0
    return float(np.corrcoef(xs, ys)[0, 1])


def evaluate_model(model: PrismModel, pairs: list[TrainingPair]) -> dict[str, float]:
    if not pairs:
        return {"spearman_raw": 0.0, "spearman_prism": 0.0, "count": 0.0}

    raw_scores: list[float] = []
    adapted_scores: list[float] = []
    labels: list[float] = []
    for pair in pairs:
        q = pair.query_embedding
        c = pair.chunk_embedding
        if q is None or c is None:
            continue
        labels.append(pair.relevance)
        # Raw cosine only defined when dims match (joint). For split,
        # compare query to mean of h/b halves as a crude baseline.
        q_arr = np.asarray(q, dtype=np.float32).reshape(-1)
        c_arr = np.asarray(c, dtype=np.float32).reshape(-1)
        if c_arr.shape[0] == q_arr.shape[0]:
            raw_scores.append(cosine_similarity(q_arr, c_arr))
        elif c_arr.shape[0] == 2 * q_arr.shape[0]:
            half = int(q_arr.shape[0])
            mean_c = 0.5 * (c_arr[:half] + c_arr[half:])
            raw_scores.append(cosine_similarity(q_arr, mean_c))
        else:
            raw_scores.append(0.0)
        adapted_scores.append(model.score_pair(q_arr, c_arr))

    return {
        "spearman_raw": _spearman(raw_scores, labels),
        "spearman_prism": _spearman(adapted_scores, labels),
        "count": float(len(labels)),
    }


def evaluate_retrieval_similarity(pairs_with_retrieval: list[dict[str, Any]]) -> dict[str, float]:
    scores = [float(r["retrieval_similarity"]) for r in pairs_with_retrieval]
    labels = [float(r["target_relevance"]) for r in pairs_with_retrieval]
    return {
        "spearman_retrieval": _spearman(scores, labels),
        "count": float(len(labels)),
    }


def train_checkpoint_path(config_name: str) -> Path:
    """In-progress train state: models/checkpoints/{config_name}.pt"""
    return models_dir() / "checkpoints" / f"{config_name}.pt"


def _train_fingerprint(
    *,
    corpus_id: str,
    config_name: str,
    chunk_repr: str,
    adapter_sharing: str,
    scoring: str = SCORING_COSINE,
    hidden_dim: int = 1536,
) -> str:
    """Stable id so resume fails loud if the frozen corpus or config changed."""
    h = hashlib.sha256()
    for part in (
        corpus_id,
        config_name,
        chunk_repr,
        adapter_sharing,
        scoring,
        str(int(hidden_dim)),
    ):
        h.update(part.encode())
        h.update(b"\0")
    return h.hexdigest()


def ensure_training_field_embeddings() -> dict[str, int]:
    """header_json + OpenAI field embeds for labeled training items only.

    Used before split training so split configs do not trigger a
    full-corpus field backfill. Takes short write locks per batch — does not
    hold the exclusive lock across OpenAI round-trips.
    """
    from fish.store import (
        backfill_corpus_header_json,
        count_corpus_needing_field_embeddings,
        db_conn,
    )
    from fish.sync import embed_field_pending
    from fish.write_lock import fish_write_lock

    init_db()
    with fish_write_lock("freeze-prep"):
        with db_conn() as db:
            headers = backfill_corpus_header_json(db, training_only=True)
            need = count_corpus_needing_field_embeddings(db, training_only=True)
    done = 0
    while True:
        # embed_field_pending does its own short lock around DB write; OpenAI
        # waits outside the exclusive flock (see sync.embed_field_pending).
        n = embed_field_pending(batch_size=50, training_only=True)
        if n == 0:
            break
        done += n
    return {
        "header_json_backfilled": headers,
        "field_need_before": need,
        "field_embedded": done,
    }


def train_prism_model(
    pairs: list[TrainingPair],
    *,
    config_name: str = "smoke_joint",
    corpus_id: str,
    epochs: int | None = None,
    lr: float | None = None,
    batch_size: int | None = None,
    weight_decay: float | None = None,
    output: Path | None = None,
    eval_pairs: list[TrainingPair] | None = None,
    register: bool = True,
    activate: bool = True,
    resume: bool = True,
    fresh: bool = False,
    checkpoint_every: int = 1,
    early_stop_patience: int | None = None,
    early_stop_min_delta: float | None = None,
    device: str | None = None,
) -> tuple[PrismModel, dict[str, Any]]:
    import copy

    import torch
    import torch.nn as nn

    from fish.prism.configs import get_prism_config, make_model_id
    from fish.prism.inference import clear_model_cache
    from fish.prism.registry import register_prism_model
    from fish.store import db_conn

    if not pairs:
        raise ValueError("No training pairs")
    if not corpus_id or not str(corpus_id).strip():
        raise ValueError("corpus_id is required (frozen .tcz id)")
    if fresh and resume:
        # --fresh wins: start a new run even if a checkpoint exists.
        resume = False

    cfg = get_prism_config(config_name)
    epochs = int(epochs if epochs is not None else cfg["epochs"])
    lr = float(lr if lr is not None else cfg["lr"])
    batch_size = int(batch_size if batch_size is not None else cfg["batch_size"])
    weight_decay = float(
        weight_decay if weight_decay is not None else cfg["weight_decay"]
    )
    patience = int(
        early_stop_patience
        if early_stop_patience is not None
        else cfg["early_stop_patience"]
    )
    min_delta = float(
        early_stop_min_delta
        if early_stop_min_delta is not None
        else cfg["early_stop_min_delta"]
    )
    dim = int(cfg.get("embed_dim") or EMBED_DIM)
    hidden_dim = int(cfg.get("hidden_dim") or dim)
    chunk_repr = str(cfg["chunk_repr"])
    adapter_sharing = str(cfg["adapter_sharing"])
    scoring = str(cfg.get("scoring") or SCORING_COSINE)
    head_hidden = int(cfg.get("head_hidden") or hidden_dim)
    chunk_in = dim * 2 if chunk_repr == CHUNK_REPR_SPLIT else dim
    if adapter_sharing == "siamese" and chunk_in != dim:
        raise ValueError(
            "siamese adapter_sharing requires chunk_repr=joint "
            f"(chunk_in={chunk_in} != dim={dim})"
        )
    # Early stop needs a holdout set; overfit (eval==train) still works but
    # monitors train Spearman — prefer real holdout for personal configs.
    holdout = eval_pairs if eval_pairs is not None else pairs
    use_early_stop = patience > 0
    ckpt_path = train_checkpoint_path(config_name)
    progress_path = train_progress_path(config_name)
    if fresh and ckpt_path.is_file():
        ckpt_path.unlink()
        progress_path.unlink(missing_ok=True)
    torch_device = resolve_train_device(device if device is not None else cfg.get("device"))
    pairs_fp = _train_fingerprint(
        corpus_id=corpus_id,
        config_name=config_name,
        chunk_repr=chunk_repr,
        adapter_sharing=adapter_sharing,
        scoring=scoring,
        hidden_dim=hidden_dim,
    )

    class Adapter(nn.Module):
        def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
            super().__init__()
            self.w1 = nn.Linear(in_dim, hidden)
            self.ln = nn.LayerNorm(hidden)
            self.w2 = nn.Linear(hidden, out_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = torch.nn.functional.gelu(self.ln(self.w1(x)))
            return self.w2(h)

    class Head(nn.Module):
        def __init__(self, in_dim: int, hidden: int) -> None:
            super().__init__()
            self.w1 = nn.Linear(in_dim, hidden)
            self.w2 = nn.Linear(hidden, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = torch.nn.functional.gelu(self.w1(x))
            return torch.sigmoid(self.w2(h)).squeeze(-1)

    q_adapter = Adapter(dim, hidden_dim, dim)
    if adapter_sharing == "siamese":
        c_adapter = q_adapter
        opt_params = list(q_adapter.parameters())
    else:
        c_adapter = Adapter(chunk_in, hidden_dim, dim)
        opt_params = list(q_adapter.parameters()) + list(c_adapter.parameters())
    head: Head | None = None
    if scoring == SCORING_MLP_HEAD:
        head = Head(2 * dim, head_hidden)
        opt_params = opt_params + list(head.parameters())
    opt = torch.optim.AdamW(opt_params, lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    start_epoch = 0
    model_id = make_model_id(config_name)
    resumed = False
    best_holdout = float("-inf")
    stall_epochs = 0
    best_q_state: dict[str, Any] | None = None
    best_c_state: dict[str, Any] | None = None
    best_head_state: dict[str, Any] | None = None
    best_epoch = -1
    if resume and ckpt_path.is_file():
        try:
            blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            blob = torch.load(ckpt_path, map_location="cpu")
        if blob.get("pairs_fingerprint") != pairs_fp:
            raise RuntimeError(
                f"Checkpoint {ckpt_path} is for a different frozen corpus/config "
                f"(fingerprint mismatch; checkpoint corpus_id="
                f"{blob.get('corpus_id')!r}, requested {corpus_id!r}). "
                f"Pass --fresh to discard it, or resume with the same --corpus."
            )
        if int(blob.get("epochs_total") or 0) != epochs:
            raise RuntimeError(
                f"Checkpoint epochs_total={blob.get('epochs_total')} != "
                f"requested epochs={epochs}. Pass --fresh or match --epochs."
            )
        if str(blob.get("chunk_repr")) != chunk_repr:
            raise RuntimeError(
                f"Checkpoint chunk_repr={blob.get('chunk_repr')!r} != {chunk_repr!r}"
            )
        if str(blob.get("adapter_sharing") or "dual") != adapter_sharing:
            raise RuntimeError(
                f"Checkpoint adapter_sharing={blob.get('adapter_sharing')!r} != "
                f"{adapter_sharing!r}"
            )
        if str(blob.get("scoring") or SCORING_COSINE) != scoring:
            raise RuntimeError(
                f"Checkpoint scoring={blob.get('scoring')!r} != {scoring!r}"
            )
        q_adapter.load_state_dict(blob["q_adapter"])
        if adapter_sharing != "siamese":
            c_adapter.load_state_dict(blob["c_adapter"])
        if head is not None:
            if blob.get("head") is None:
                raise RuntimeError(
                    f"Checkpoint {ckpt_path} missing mlp head for scoring=mlp_head"
                )
            head.load_state_dict(blob["head"])
        opt.load_state_dict(blob["optimizer"])
        start_epoch = int(blob["epoch"]) + 1
        model_id = str(blob["model_id"])
        best_holdout = float(blob.get("best_holdout", float("-inf")))
        stall_epochs = int(blob.get("stall_epochs", 0))
        best_epoch = int(blob.get("best_epoch", -1))
        if blob.get("best_q_adapter") is not None:
            best_q_state = blob["best_q_adapter"]
            best_c_state = blob.get("best_c_adapter")
            best_head_state = blob.get("best_head")
        resumed = True
        if start_epoch >= epochs:
            raise RuntimeError(
                f"Checkpoint already completed {epochs} epochs at {ckpt_path}. "
                f"Pass --fresh to start a new run."
            )

    q_adapter = q_adapter.to(torch_device)
    if adapter_sharing != "siamese":
        c_adapter = c_adapter.to(torch_device)
    if head is not None:
        head = head.to(torch_device)
    _optimizer_to_device(opt, torch_device)

    q_mats: list[np.ndarray] = []
    c_mats: list[np.ndarray] = []
    rel_rows: list[float] = []
    for pair in pairs:
        if pair.query_embedding is None or pair.chunk_embedding is None:
            continue
        q_vec = np.asarray(pair.query_embedding, dtype=np.float32).reshape(-1)
        c_vec = np.asarray(pair.chunk_embedding, dtype=np.float32).reshape(-1)
        if q_vec.shape[0] != dim:
            raise ValueError(f"query embed dim {q_vec.shape[0]} != {dim}")
        if c_vec.shape[0] != chunk_in:
            raise ValueError(
                f"chunk embed dim {c_vec.shape[0]} != {chunk_in} "
                f"(chunk_repr={chunk_repr})"
            )
        q_mats.append(q_vec)
        c_mats.append(c_vec)
        rel_rows.append(float(pair.relevance))

    if not q_mats:
        raise ValueError("No training pairs with embeddings")

    q_np = np.stack(q_mats).astype(np.float32, copy=False)
    c_np = np.stack(c_mats).astype(np.float32, copy=False)
    rel_np = np.asarray(rel_rows, dtype=np.float32)
    q_all = torch.from_numpy(q_np).to(torch_device)
    c_all = torch.from_numpy(c_np).to(torch_device)
    rel_all = torch.from_numpy(rel_np).to(torch_device)
    n = q_all.shape[0]
    indices = list(range(n))
    every = max(1, int(checkpoint_every))

    # Holdout tensors on device (full set fits easily; ~MB for personal corpora).
    hq_mats: list[np.ndarray] = []
    hc_mats: list[np.ndarray] = []
    hrel_rows: list[float] = []
    for pair in holdout:
        if pair.query_embedding is None or pair.chunk_embedding is None:
            continue
        hq_mats.append(np.asarray(pair.query_embedding, dtype=np.float32).reshape(-1))
        hc_mats.append(np.asarray(pair.chunk_embedding, dtype=np.float32).reshape(-1))
        hrel_rows.append(float(pair.relevance))
    if hq_mats:
        hq_all = torch.from_numpy(np.stack(hq_mats).astype(np.float32, copy=False)).to(
            torch_device
        )
        hc_all = torch.from_numpy(np.stack(hc_mats).astype(np.float32, copy=False)).to(
            torch_device
        )
        hrel_np = np.asarray(hrel_rows, dtype=np.float32)
    else:
        hq_all = hc_all = None
        hrel_np = np.asarray([], dtype=np.float32)

    def export_adapter(module: Adapter) -> PrismAdapter:
        return PrismAdapter(
            w1=module.w1.weight.detach().cpu().numpy(),
            b1=module.w1.bias.detach().cpu().numpy(),
            ln_gamma=module.ln.weight.detach().cpu().numpy(),
            ln_beta=module.ln.bias.detach().cpu().numpy(),
            w2=module.w2.weight.detach().cpu().numpy(),
            b2=module.w2.bias.detach().cpu().numpy(),
        )

    def export_head(module: Head) -> RerankHead:
        return RerankHead(
            w1=module.w1.weight.detach().cpu().numpy(),
            b1=module.w1.bias.detach().cpu().numpy(),
            w2=module.w2.weight.detach().cpu().numpy(),
            b2=module.w2.bias.detach().cpu().numpy(),
        )

    def snapshot_best() -> None:
        nonlocal best_q_state, best_c_state, best_head_state
        best_q_state = copy.deepcopy({k: v.detach().cpu() for k, v in q_adapter.state_dict().items()})
        if adapter_sharing != "siamese":
            best_c_state = copy.deepcopy(
                {k: v.detach().cpu() for k, v in c_adapter.state_dict().items()}
            )
        else:
            best_c_state = None
        best_head_state = (
            copy.deepcopy({k: v.detach().cpu() for k, v in head.state_dict().items()})
            if head is not None
            else None
        )

    def holdout_spearman() -> float:
        """Batched GPU eval — avoids per-epoch CPU export_snap + Python pair loop."""
        if hq_all is None or hc_all is None or hrel_np.size == 0:
            return 0.0
        was_training = q_adapter.training
        q_adapter.eval()
        if adapter_sharing != "siamese":
            c_adapter.eval()
        if head is not None:
            head.eval()
        scores: list[float] = []
        with torch.no_grad():
            hn = int(hq_all.shape[0])
            for start in range(0, hn, batch_size):
                q = hq_all[start : start + batch_size]
                c = hc_all[start : start + batch_size]
                q_out = q_adapter(q)
                c_out = c_adapter(c)
                if scoring == SCORING_MLP_HEAD:
                    assert head is not None
                    score = head(torch.cat([q_out, c_out], dim=-1))
                else:
                    q_norm = torch.nn.functional.normalize(q_out, dim=-1)
                    c_norm = torch.nn.functional.normalize(c_out, dim=-1)
                    score = (q_norm * c_norm).sum(dim=-1)
                scores.extend(float(x) for x in score.detach().cpu().tolist())
        if was_training:
            q_adapter.train()
            if adapter_sharing != "siamese":
                c_adapter.train()
            if head is not None:
                head.train()
        return float(_spearman(scores, [float(x) for x in hrel_np.tolist()]))

    def write_checkpoint(epoch_done: int) -> None:
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = ckpt_path.with_suffix(".pt.tmp")
        # Persist CPU tensors so resume is device-agnostic.
        q_cpu = {k: v.detach().cpu() for k, v in q_adapter.state_dict().items()}
        opt_cpu = copy.deepcopy(opt.state_dict())
        for state in opt_cpu.get("state", {}).values():
            for key, val in list(state.items()):
                if hasattr(val, "cpu"):
                    state[key] = val.cpu()
        payload: dict[str, Any] = {
            "epoch": epoch_done,
            "epochs_total": epochs,
            "model_id": model_id,
            "config_name": config_name,
            "chunk_repr": chunk_repr,
            "adapter_sharing": adapter_sharing,
            "scoring": scoring,
            "embed_dim": dim,
            "hidden_dim": hidden_dim,
            "pairs_fingerprint": pairs_fp,
            "corpus_id": corpus_id,
            "q_adapter": q_cpu,
            "optimizer": opt_cpu,
            "best_holdout": best_holdout,
            "stall_epochs": stall_epochs,
            "best_epoch": best_epoch,
            "best_q_adapter": best_q_state,
            "best_head": best_head_state,
            "early_stop_patience": patience,
            "early_stop_min_delta": min_delta,
            "device": str(torch_device),
        }
        if adapter_sharing != "siamese":
            payload["c_adapter"] = {
                k: v.detach().cpu() for k, v in c_adapter.state_dict().items()
            }
            payload["best_c_adapter"] = best_c_state
        if head is not None:
            payload["head"] = {
                k: v.detach().cpu() for k, v in head.state_dict().items()
            }
        torch.save(payload, tmp)
        tmp.replace(ckpt_path)

    def write_progress(
        *,
        epoch_done: int,
        elapsed_sec: float,
        holdout: float,
        status: str = "running",
        task_progress: Any | None = None,
    ) -> dict[str, Any]:
        epochs_done = max(0, epoch_done - start_epoch)
        ep_per_sec = (epochs_done / elapsed_sec) if elapsed_sec > 0 else 0.0
        best_val = best_holdout if best_holdout > float("-inf") else holdout
        payload = {
            "status": status,
            "config_name": config_name,
            "model_id": model_id,
            "device": str(torch_device),
            "epoch": epoch_done,
            "epochs_total": epochs,
            "start_epoch": start_epoch,
            "elapsed_sec": round(elapsed_sec, 3),
            "epochs_per_sec": round(ep_per_sec, 6),
            "holdout_spearman": holdout,
            "best_holdout_spearman": (
                best_holdout if best_holdout > float("-inf") else None
            ),
            "best_epoch": best_epoch if best_epoch >= 0 else None,
            "stall_epochs": stall_epochs,
            "pairs": n,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = progress_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(progress_path)
        # Human + agent log line (pipeline captures stdout/stderr).
        line = (
            f"epoch {epoch_done}/{epochs}  elapsed={elapsed_sec:.1f}s  "
            f"ep/s={ep_per_sec:.4f}  holdout={holdout:.4f}  "
            f"best={best_val:.4f}@{best_epoch if best_epoch >= 0 else epoch_done}  "
            f"device={torch_device}"
        )
        print(line, flush=True)
        print(line, file=sys.stderr, flush=True)
        print(f"@compute progress {epoch_done} {epochs}", file=sys.stderr, flush=True)
        if task_progress is not None:
            from compute.tasks import TaskCancelled

            try:
                task_progress.update(
                    epoch_done,
                    n=epochs,
                    detail=(
                        f"{config_name} {torch_device} "
                        f"epoch {epoch_done}/{epochs} holdout={holdout:.4f}"
                    ),
                    force=True,
                )
            except TaskCancelled:
                raise
        return payload

    print(
        f"prism-train {config_name} corpus={corpus_id} device={torch_device} "
        f"pairs={n} epochs={start_epoch}→{epochs} "
        f"scoring={scoring} sharing={adapter_sharing}",
        flush=True,
    )

    import os as _os

    from compute.tasks import TaskCancelled, TaskProgress

    stopped_early = False
    cancelled = False
    epochs_run = start_epoch
    history: list[dict[str, float | int]] = []
    train_t0 = time.monotonic()
    device_tag = "gpu" if torch_device.type == "cuda" else "cpu"
    with TaskProgress(
        module="fish",
        task=f"train:{config_name}:{device_tag}",
        n=epochs,
        sec_per_unit_prior=None,
        detail=f"{config_name} {torch_device} starting ({n} pairs)",
        resource=_os.environ.get("COMPUTE_RESOURCE"),
        meta={
            "config_name": config_name,
            "corpus_id": corpus_id,
            "device": str(torch_device),
            "gpu": torch_device.type == "cuda",
            "n_pairs": n,
        },
        emit_stderr_progress=False,
    ) as task_progress:
        try:
            for epoch in range(start_epoch, epochs):
                random.shuffle(indices)
                for start in range(0, n, batch_size):
                    batch_idx = indices[start : start + batch_size]
                    q = q_all[batch_idx]
                    c = c_all[batch_idx]
                    rel = rel_all[batch_idx]
                    opt.zero_grad()
                    q_out = q_adapter(q)
                    c_out = c_adapter(c)
                    if scoring == SCORING_MLP_HEAD:
                        assert head is not None
                        score = head(torch.cat([q_out, c_out], dim=-1))
                    else:
                        q_norm = torch.nn.functional.normalize(q_out, dim=-1)
                        c_norm = torch.nn.functional.normalize(c_out, dim=-1)
                        score = (q_norm * c_norm).sum(dim=-1)
                    loss = loss_fn(score, rel)
                    loss.backward()
                    opt.step()
                epochs_run = epoch + 1
                spear = holdout_spearman()
                history.append({"epoch": epochs_run, "spearman_holdout": spear})
                improved = spear > best_holdout + min_delta
                if improved:
                    best_holdout = spear
                    best_epoch = epochs_run
                    stall_epochs = 0
                    snapshot_best()
                else:
                    stall_epochs += 1
                elapsed = time.monotonic() - train_t0
                write_progress(
                    epoch_done=epochs_run,
                    elapsed_sec=elapsed,
                    holdout=spear,
                    task_progress=task_progress,
                )
                if (epoch + 1) % every == 0 or epoch + 1 == epochs:
                    write_checkpoint(epoch)
                if use_early_stop and stall_epochs >= patience:
                    stopped_early = True
                    break
        except TaskCancelled:
            cancelled = True
            stopped_early = True

    if cancelled:
        raise TaskCancelled(
            f"prism-train {config_name} cancelled after epoch {epochs_run}/{epochs}"
        )

    if best_q_state is not None:
        q_adapter.load_state_dict(best_q_state)
        q_adapter = q_adapter.to(torch_device)
        if adapter_sharing != "siamese" and best_c_state is not None:
            c_adapter.load_state_dict(best_c_state)
            c_adapter = c_adapter.to(torch_device)
        if head is not None and best_head_state is not None:
            head.load_state_dict(best_head_state)
            head = head.to(torch_device)

    print(f"prism-train {config_name}: exporting adapters…", flush=True)
    shared = export_adapter(q_adapter)
    model = PrismModel(
        query_adapter=shared,
        # Siamese: same weights serialized under both slots for .prz loaders.
        chunk_adapter=shared if adapter_sharing == "siamese" else export_adapter(c_adapter),
        embed_dim=dim,
        embed_model=embedding_model(),
        model_id=model_id,
        config_name=config_name,
        chunk_repr=chunk_repr,
        adapter_sharing=adapter_sharing,
        scoring=scoring,
        rerank_head=export_head(head) if head is not None else None,
    )
    # Prefer batched GPU holdout already tracked during train — avoid slow
    # per-pair numpy evaluate_model (hangs for minutes on large adapters).
    print(f"prism-train {config_name}: final holdout via GPU batch…", flush=True)
    final_holdout = float(holdout_spearman())
    metrics = {
        "spearman_raw": 0.0,
        "spearman_prism": final_holdout,
        "count": float(len(holdout)),
    }
    out = output or models_dir() / f"{model_id}.prz"
    tmp_out = Path("/tmp") / f"{model_id}.prz"
    print(f"prism-train {config_name}: saving {tmp_out}…", flush=True)
    save_prz(model, tmp_out)
    print(f"prism-train {config_name}: copying to {out}…", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(tmp_out.read_bytes())
    print(f"prism-train {config_name}: saved {out} ({out.stat().st_size} bytes)", flush=True)
    if ckpt_path.is_file():
        ckpt_path.unlink()
    final_elapsed = time.monotonic() - train_t0
    write_progress(
        epoch_done=epochs_run,
        elapsed_sec=final_elapsed,
        holdout=float(metrics["spearman_prism"]),
        status="done",
    )
    metrics["output"] = str(out)
    metrics["model_id"] = model_id
    metrics["config_name"] = config_name
    metrics["chunk_repr"] = chunk_repr
    metrics["adapter_sharing"] = adapter_sharing
    metrics["scoring"] = scoring
    metrics["hidden_dim"] = hidden_dim
    metrics["head_hidden"] = head_hidden
    metrics["device"] = str(torch_device)
    metrics["epochs"] = epochs
    metrics["epochs_run"] = epochs_run
    metrics["best_epoch"] = best_epoch
    metrics["best_holdout_spearman"] = best_holdout if best_holdout > float("-inf") else metrics["spearman_prism"]
    metrics["stopped_early"] = stopped_early
    metrics["early_stop_patience"] = patience
    metrics["early_stop_min_delta"] = min_delta
    metrics["holdout_history"] = history
    metrics["resumed"] = resumed
    metrics["start_epoch"] = start_epoch
    metrics["elapsed_sec"] = round(final_elapsed, 3)
    metrics["epochs_per_sec"] = round(
        (max(0, epochs_run - start_epoch) / final_elapsed) if final_elapsed > 0 else 0.0,
        6,
    )
    metrics["progress_path"] = str(progress_path)
    metrics["corpus_id"] = corpus_id

    if register:
        from fish.write_lock import fish_write_lock

        init_db()
        with fish_write_lock("train"):
            with db_conn() as db:
                register_prism_model(
                    db,
                    config_name=config_name,
                    meta_json=json.dumps(cfg),
                    # Rerank is not an ANN index — keep inactive.
                    activate=activate and scoring != SCORING_MLP_HEAD,
                    timestamp=model_id.split(".", 1)[1],
                )
        clear_model_cache()

    return model, metrics


def train_from_corpus(
    *,
    config_name: str | list[str] = "smoke_joint",
    epochs: int | None = None,
    output: Path | None = None,
    retriever: str | None = None,
    collect_first: bool = False,
    collect_retriever: str = "legacy",
    min_queries: int = 50,
    top_k: int = 20,
    label_limit: int = 500,
    overfit: bool = False,
    resume: bool = True,
    fresh: bool = False,
    device: str | None = None,
    corpus: str = "latest",
    from_db: bool = False,
    register: bool = True,
) -> dict[str, Any]:
    """Train one or more adapter configs from frozen ``.tcz`` file(s).

    Multiple configs (list or comma-expanded via CLI) share one loaded corpus per
    ``chunk_repr`` so embeddings stay resident while iterating models.
    Epochs never open fish.db. Use ``--from-db`` / ``--collect-first`` to freeze.
    """
    from fish.prism.configs import get_prism_config, parse_config_list
    from fish.prism.train_corpus import (
        freeze_training_corpus,
        load_tcz,
        resolve_corpus_path,
    )

    if isinstance(config_name, str):
        config_names = parse_config_list(config_name)
    else:
        config_names = list(config_name)
        if not config_names:
            raise ValueError("config_name list is empty")

    if output is not None and len(config_names) > 1:
        raise ValueError("--output is only valid when training a single --config")

    labeling: dict[str, Any] | None = None
    if collect_first:
        from fish.prism.collect import collect_samples
        from fish.prism.relevance import label_batch

        with fish_write_lock("train"):
            collect_samples(
                retriever=collect_retriever,
                min_queries=min_queries,
                top_k=top_k,
                label=False,
                label_limit=label_limit,
            )
        labeling = label_batch(limit=label_limit)
        from_db = True

    # Group by chunk_repr so embeddings stay resident while iterating models.
    # joint and split are separate resident sets (different c matrices);
    # a dual .tcz still loads once from disk.
    by_repr: dict[str, list[str]] = {}
    for name in config_names:
        repr_key = str(get_prism_config(name)["chunk_repr"])
        by_repr.setdefault(repr_key, []).append(name)

    freeze_info: dict[str, Any] | None = None
    if from_db:
        # Dual freeze is the bakeoff default (joint + split in one file).
        freeze_info = freeze_training_corpus(
            chunk_repr="both",
            retriever=retriever,
            prep_fields=True,
        )
        corpus_ref = freeze_info["corpus_id"]
    else:
        corpus_ref = corpus

    # Prefer dual freeze when training across both reprs.
    want = "both" if len(by_repr) > 1 else next(iter(by_repr))
    frozen_path = resolve_corpus_path(corpus_ref, chunk_repr=want)
    frozen = load_tcz(frozen_path)
    for chunk_repr in by_repr:
        if not frozen.has_repr(chunk_repr):
            raise RuntimeError(
                f"Frozen corpus {frozen.corpus_id} missing chunk_repr={chunk_repr!r} "
                f"(has {sorted(frozen.c_by_repr)}). Freeze with --chunk-repr both "
                f"or pass a matching --corpus."
            )

    runs: list[dict[str, Any]] = []
    for chunk_repr, names in by_repr.items():
        pairs = frozen.pairs_for(chunk_repr)
        if overfit:
            train, test = pairs, pairs
        else:
            train, test = split_pairs(pairs)
            if not test:
                test = train

        # Resident pairs shared across configs in this repr group (no reload).
        baseline = new_identity_model(chunk_repr=chunk_repr)
        baseline_metrics = evaluate_model(baseline, test)

        if frozen.retrieval_similarity is not None and len(
            frozen.retrieval_similarity
        ) == len(pairs):
            retrieval_rows = [
                {
                    "retrieval_similarity": frozen.retrieval_similarity[i],
                    "target_relevance": pairs[i].relevance,
                }
                for i in range(len(pairs))
            ]
            retrieval_eval = evaluate_retrieval_similarity(retrieval_rows)
        else:
            rows = [
                {
                    "retrieval_similarity": float(p.retrieval_similarity or 0.0),
                    "target_relevance": p.relevance,
                }
                for p in pairs
                if p.retrieval_similarity is not None
            ]
            retrieval_eval = (
                evaluate_retrieval_similarity(rows)
                if rows
                else {"spearman_retrieval": 0.0, "count": 0.0, "skipped": True}
            )

        print(
            f"prism-train group chunk_repr={chunk_repr} corpus={frozen.corpus_id} "
            f"configs={names} pairs={len(pairs)} train={len(train)} test={len(test)}",
            flush=True,
        )

        for name in names:
            cfg = get_prism_config(name)
            _, metrics = train_prism_model(
                train,
                config_name=name,
                corpus_id=frozen.corpus_id,
                epochs=epochs,
                output=output if len(config_names) == 1 else None,
                eval_pairs=test,
                resume=resume,
                fresh=fresh,
                device=device,
                register=register,
                # Multi-config: do not steal active from the production model.
                activate=False if len(config_names) > 1 else True,
            )
            runs.append(
                {
                    "config_name": name,
                    "pairs": len(pairs),
                    "train": len(train),
                    "test": len(test),
                    "overfit": overfit,
                    "chunk_repr": chunk_repr,
                    "hidden_dim": int(cfg["hidden_dim"]),
                    "adapter_sharing": str(cfg["adapter_sharing"]),
                    "scoring": str(cfg.get("scoring") or "cosine"),
                    "device": metrics.get("device"),
                    "baseline": baseline_metrics,
                    "retrieval_eval": retrieval_eval,
                    "trained": metrics,
                    "corpus_id": frozen.corpus_id,
                    "corpus_path": str(frozen.path),
                }
            )

    result: dict[str, Any] = {
        "configs": config_names,
        "runs": runs,
        "corpus_id": frozen.corpus_id,
        "corpus_path": str(frozen.path),
        "chunk_reprs": list(frozen.chunk_reprs),
    }
    if len(runs) == 1:
        # Backward-compatible flat shape for single-config callers.
        result.update(runs[0])
    if freeze_info is not None:
        result["freeze"] = freeze_info
    if labeling is not None:
        result["labeling"] = labeling
    return result
