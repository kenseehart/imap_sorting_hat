from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fish.config import EMBED_DIM, embedding_model, models_dir
from fish.prism.inference import cosine_similarity, compose_chunk_vector
from fish.prism.model import (
    CHUNK_REPR_COMBINED,
    CHUNK_REPR_HEADER_BODY,
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


@dataclass
class TrainingPair:
    query: str
    chunk_id: int
    relevance: float
    query_embedding: list[float] | None = None
    chunk_embedding: list[float] | None = None


def _pair_hash(query: str, chunk_id: int) -> str:
    return hashlib.sha256(f"{query}\0{chunk_id}".encode()).hexdigest()


def load_training_pairs_from_db(
    *,
    exclude_superseded: bool = True,
    retriever: str | None = None,
    chunk_repr: str = CHUNK_REPR_COMBINED,
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
            if not isinstance(q_emb, list):
                continue
            item_id = int(row["corpus_item_id"])
            if chunk_repr == CHUNK_REPR_COMBINED:
                c_emb = row.get("message_embedding")
                if not isinstance(c_emb, list):
                    c_emb = get_raw_embedding(db, item_id)
            else:
                c_emb = compose_chunk_vector(db, item_id, chunk_repr)
                if c_emb is None:
                    missing_fields += 1
                    continue
            if not isinstance(c_emb, list):
                continue
            pairs.append(
                TrainingPair(
                    query=row["query_text"],
                    chunk_id=item_id,
                    relevance=float(row["target_relevance"]),
                    query_embedding=q_emb,
                    chunk_embedding=c_emb,
                )
            )
    if chunk_repr == CHUNK_REPR_HEADER_BODY and not pairs and missing_fields:
        raise RuntimeError(
            f"chunk_repr=header_body but no pairs have header+body embeddings "
            f"({missing_fields} labeled rows missing fields). "
            f"Run: fish embed --fields --training-only"
        )
    return pairs


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
        # Raw cosine only defined when dims match (combined). For header_body,
        # compare query to mean of h/b halves as a crude baseline.
        if len(c) == len(q):
            raw_scores.append(cosine_similarity(q, c))
        elif len(c) == 2 * len(q):
            half = len(q)
            mean_c = [(c[i] + c[half + i]) * 0.5 for i in range(half)]
            raw_scores.append(cosine_similarity(q, mean_c))
        else:
            raw_scores.append(0.0)
        adapted_scores.append(model.score_pair(q, c))

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


def _pairs_fingerprint(
    pairs: list[TrainingPair],
    *,
    config_name: str,
    chunk_repr: str,
    adapter_sharing: str,
    scoring: str = SCORING_COSINE,
) -> str:
    """Stable id so resume fails loud if the labeled set changed."""
    h = hashlib.sha256()
    h.update(config_name.encode())
    h.update(b"\0")
    h.update(chunk_repr.encode())
    h.update(b"\0")
    h.update(adapter_sharing.encode())
    h.update(b"\0")
    h.update(scoring.encode())
    h.update(b"\0")
    for pair in sorted(pairs, key=lambda p: (p.chunk_id, p.query)):
        h.update(f"{pair.chunk_id}\0{pair.query}\0{pair.relevance:.6f}\n".encode())
    return h.hexdigest()


def ensure_training_field_embeddings() -> dict[str, int]:
    """header_json + OpenAI field embeds for labeled training items only.

    Used before header_body training so smoke/personal_fields do not trigger a
    full-corpus field backfill.
    """
    from fish.store import (
        backfill_corpus_header_json,
        count_corpus_needing_field_embeddings,
        db_conn,
    )
    from fish.sync import embed_field_pending

    init_db()
    with db_conn() as db:
        headers = backfill_corpus_header_json(db, training_only=True)
        need = count_corpus_needing_field_embeddings(db, training_only=True)
    done = 0
    while True:
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
    config_name: str = "smoke_combined",
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
) -> tuple[PrismModel, dict[str, Any]]:
    import copy
    import json

    import torch
    import torch.nn as nn

    from fish.prism.configs import get_prism_config, make_model_id
    from fish.prism.inference import clear_model_cache
    from fish.prism.registry import register_prism_model
    from fish.store import db_conn

    if not pairs:
        raise ValueError("No training pairs")
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
    chunk_repr = str(cfg["chunk_repr"])
    adapter_sharing = str(cfg["adapter_sharing"])
    scoring = str(cfg.get("scoring") or SCORING_COSINE)
    head_hidden = int(cfg.get("head_hidden") or 512)
    chunk_in = dim * 2 if chunk_repr == CHUNK_REPR_HEADER_BODY else dim
    if adapter_sharing == "siamese" and chunk_in != dim:
        raise ValueError(
            "siamese adapter_sharing requires chunk_repr=combined "
            f"(chunk_in={chunk_in} != dim={dim})"
        )
    # Early stop needs a holdout set; overfit (eval==train) still works but
    # monitors train Spearman — prefer real holdout for personal configs.
    holdout = eval_pairs if eval_pairs is not None else pairs
    use_early_stop = patience > 0
    ckpt_path = train_checkpoint_path(config_name)
    pairs_fp = _pairs_fingerprint(
        pairs,
        config_name=config_name,
        chunk_repr=chunk_repr,
        adapter_sharing=adapter_sharing,
        scoring=scoring,
    )

    class Adapter(nn.Module):
        def __init__(self, in_dim: int, out_dim: int) -> None:
            super().__init__()
            self.w1 = nn.Linear(in_dim, out_dim)
            self.ln = nn.LayerNorm(out_dim)
            self.w2 = nn.Linear(out_dim, out_dim)

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

    q_adapter = Adapter(dim, dim)
    if adapter_sharing == "siamese":
        c_adapter = q_adapter
        opt_params = list(q_adapter.parameters())
    else:
        c_adapter = Adapter(chunk_in, dim)
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
                f"Checkpoint {ckpt_path} is for different training pairs "
                f"(fingerprint mismatch). Pass --fresh to discard it, or restore "
                f"the same labeled set."
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

    q_rows: list[list[float]] = []
    c_rows: list[list[float]] = []
    rel_rows: list[float] = []
    for pair in pairs:
        if pair.query_embedding is None or pair.chunk_embedding is None:
            continue
        if len(pair.query_embedding) != dim:
            raise ValueError(
                f"query embed dim {len(pair.query_embedding)} != {dim}"
            )
        if len(pair.chunk_embedding) != chunk_in:
            raise ValueError(
                f"chunk embed dim {len(pair.chunk_embedding)} != {chunk_in} "
                f"(chunk_repr={chunk_repr})"
            )
        q_rows.append(pair.query_embedding)
        c_rows.append(pair.chunk_embedding)
        rel_rows.append(float(pair.relevance))

    if not q_rows:
        raise ValueError("No training pairs with embeddings")

    q_all = torch.tensor(q_rows, dtype=torch.float32)
    c_all = torch.tensor(c_rows, dtype=torch.float32)
    rel_all = torch.tensor(rel_rows, dtype=torch.float32)
    n = q_all.shape[0]
    indices = list(range(n))
    every = max(1, int(checkpoint_every))

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
        best_q_state = copy.deepcopy(q_adapter.state_dict())
        if adapter_sharing != "siamese":
            best_c_state = copy.deepcopy(c_adapter.state_dict())
        else:
            best_c_state = None
        best_head_state = (
            copy.deepcopy(head.state_dict()) if head is not None else None
        )

    def export_snap() -> PrismModel:
        shared = export_adapter(q_adapter)
        return PrismModel(
            query_adapter=shared,
            chunk_adapter=(
                shared if adapter_sharing == "siamese" else export_adapter(c_adapter)
            ),
            embed_dim=dim,
            embed_model=embedding_model(),
            chunk_repr=chunk_repr,
            adapter_sharing=adapter_sharing,
            scoring=scoring,
            rerank_head=export_head(head) if head is not None else None,
        )

    def holdout_spearman() -> float:
        return float(evaluate_model(export_snap(), holdout)["spearman_prism"])

    def write_checkpoint(epoch_done: int) -> None:
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = ckpt_path.with_suffix(".pt.tmp")
        payload: dict[str, Any] = {
            "epoch": epoch_done,
            "epochs_total": epochs,
            "model_id": model_id,
            "config_name": config_name,
            "chunk_repr": chunk_repr,
            "adapter_sharing": adapter_sharing,
            "scoring": scoring,
            "embed_dim": dim,
            "pairs_fingerprint": pairs_fp,
            "q_adapter": q_adapter.state_dict(),
            "optimizer": opt.state_dict(),
            "best_holdout": best_holdout,
            "stall_epochs": stall_epochs,
            "best_epoch": best_epoch,
            "best_q_adapter": best_q_state,
            "best_head": best_head_state,
            "early_stop_patience": patience,
            "early_stop_min_delta": min_delta,
        }
        if adapter_sharing != "siamese":
            payload["c_adapter"] = c_adapter.state_dict()
            payload["best_c_adapter"] = best_c_state
        if head is not None:
            payload["head"] = head.state_dict()
        torch.save(payload, tmp)
        tmp.replace(ckpt_path)

    if fresh and ckpt_path.is_file():
        ckpt_path.unlink()

    stopped_early = False
    epochs_run = start_epoch
    history: list[dict[str, float | int]] = []
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
        if (epoch + 1) % every == 0 or epoch + 1 == epochs:
            write_checkpoint(epoch)
        if use_early_stop and stall_epochs >= patience:
            stopped_early = True
            break

    if best_q_state is not None:
        q_adapter.load_state_dict(best_q_state)
        if adapter_sharing != "siamese" and best_c_state is not None:
            c_adapter.load_state_dict(best_c_state)
        if head is not None and best_head_state is not None:
            head.load_state_dict(best_head_state)

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
    metrics = evaluate_model(model, holdout)
    out = output or models_dir() / f"{model_id}.prz"
    save_prz(model, out)
    if ckpt_path.is_file():
        ckpt_path.unlink()
    metrics["output"] = str(out)
    metrics["model_id"] = model_id
    metrics["config_name"] = config_name
    metrics["chunk_repr"] = chunk_repr
    metrics["adapter_sharing"] = adapter_sharing
    metrics["scoring"] = scoring
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

    if register:
        init_db()
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
    config_name: str = "smoke_combined",
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
) -> dict[str, Any]:
    """Train adapters. ``overfit=True``: no holdout — eval on the full train set."""
    from fish.prism.configs import get_prism_config

    cfg = get_prism_config(config_name)
    chunk_repr = str(cfg["chunk_repr"])

    with fish_write_lock("train"):
        if collect_first:
            from fish.prism.collect import collect_samples

            collect_samples(
                retriever=collect_retriever,
                min_queries=min_queries,
                top_k=top_k,
                label=True,
                label_limit=label_limit,
            )

        field_prep: dict[str, int] | None = None
        if chunk_repr == CHUNK_REPR_HEADER_BODY:
            # Smoke and personal_fields only need labeled items for training —
            # not a full-corpus OpenAI field backfill.
            field_prep = ensure_training_field_embeddings()

        pairs = load_training_pairs_from_db(
            retriever=retriever, chunk_repr=chunk_repr
        )
        if not pairs:
            raise RuntimeError(
                "No labeled training samples — run fish corpus collect and "
                "fish corpus label first"
            )

        if overfit:
            train, test = pairs, pairs
        else:
            train, test = split_pairs(pairs)
            if not test:
                test = train

        baseline = new_identity_model(chunk_repr=chunk_repr)
        baseline_metrics = evaluate_model(baseline, test)

        init_db()
        with db_conn() as db:
            labeled_rows = load_labeled_training_pairs(
                db, exclude_superseded=True, retriever=retriever
            )
        retrieval_eval = evaluate_retrieval_similarity(labeled_rows)

        _, metrics = train_prism_model(
            train,
            config_name=config_name,
            epochs=epochs,
            output=output,
            eval_pairs=test,
            resume=resume,
            fresh=fresh,
        )
        result = {
            "pairs": len(pairs),
            "train": len(train),
            "test": len(test),
            "overfit": overfit,
            "chunk_repr": chunk_repr,
            "adapter_sharing": str(cfg["adapter_sharing"]),
            "baseline": baseline_metrics,
            "retrieval_eval": retrieval_eval,
            "trained": metrics,
        }
        if field_prep is not None:
            result["field_prep"] = field_prep
        return result
