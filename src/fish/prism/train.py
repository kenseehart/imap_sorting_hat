from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fish.config import embedding_model, models_dir
from fish.prism.inference import cosine_similarity
from fish.prism.model import PrismAdapter, PrismModel, new_identity_model, save_prz
from fish.store import db_conn, init_db, load_labeled_training_pairs
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
) -> list[TrainingPair]:
    init_db()
    with db_conn() as db:
        rows = load_labeled_training_pairs(
            db,
            exclude_superseded=exclude_superseded,
            retriever=retriever,
        )
    pairs: list[TrainingPair] = []
    for row in rows:
        q_emb = row.get("query_embedding")
        c_emb = row.get("message_embedding")
        if not isinstance(q_emb, list) or not isinstance(c_emb, list):
            continue
        pairs.append(
            TrainingPair(
                query=row["query_text"],
                chunk_id=int(row["corpus_item_id"]),
                relevance=float(row["target_relevance"]),
                query_embedding=q_emb,
                chunk_embedding=c_emb,
            )
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
        return {"spearman": 0.0, "count": 0.0}

    raw_scores: list[float] = []
    adapted_scores: list[float] = []
    retrieval_scores: list[float] = []
    labels: list[float] = []
    for pair in pairs:
        q = pair.query_embedding
        c = pair.chunk_embedding
        if q is None or c is None:
            continue
        labels.append(pair.relevance)
        raw_scores.append(cosine_similarity(q, c))
        aq = model.adapt_query(q).tolist()
        ac = model.adapt_chunk(c).tolist()
        adapted_scores.append(cosine_similarity(aq, ac))

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


def train_prism_model(
    pairs: list[TrainingPair],
    *,
    epochs: int = 5,
    lr: float = 2e-5,
    output: Path | None = None,
) -> tuple[PrismModel, dict[str, float]]:
    import torch
    import torch.nn as nn

    if not pairs:
        raise ValueError("No training pairs")

    dim = EMBED_DIM

    class Adapter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w1 = nn.Linear(dim, dim)
            self.ln = nn.LayerNorm(dim)
            self.w2 = nn.Linear(dim, dim)
            self.alpha = nn.Parameter(torch.tensor(0.9))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = torch.nn.functional.gelu(self.ln(self.w1(x)))
            out = self.w2(h)
            alpha = torch.clamp(self.alpha, 0.0, 1.0)
            return alpha * x + (1.0 - alpha) * out

    q_adapter = Adapter()
    c_adapter = Adapter()
    opt = torch.optim.AdamW(
        list(q_adapter.parameters()) + list(c_adapter.parameters()),
        lr=lr,
        weight_decay=0.01,
    )
    loss_fn = nn.MSELoss()

    tensors = []
    for pair in pairs:
        if pair.query_embedding is None or pair.chunk_embedding is None:
            continue
        tensors.append(
            (
                torch.tensor(pair.query_embedding, dtype=torch.float32),
                torch.tensor(pair.chunk_embedding, dtype=torch.float32),
                torch.tensor([pair.relevance], dtype=torch.float32),
            )
        )

    if not tensors:
        raise ValueError("No training pairs with embeddings")

    for _epoch in range(epochs):
        random.shuffle(tensors)
        for q, c, rel in tensors:
            opt.zero_grad()
            q_out = q_adapter(q.unsqueeze(0))
            c_out = c_adapter(c.unsqueeze(0))
            q_norm = torch.nn.functional.normalize(q_out, dim=-1)
            c_norm = torch.nn.functional.normalize(c_out, dim=-1)
            score = (q_norm * c_norm).sum(dim=-1, keepdim=True)
            loss = loss_fn(score, rel)
            loss.backward()
            opt.step()

    def export_adapter(module: Adapter) -> dict[str, Any]:
        return {
            "w1": module.w1.weight.detach().cpu().numpy(),
            "b1": module.w1.bias.detach().cpu().numpy(),
            "ln_gamma": module.ln.weight.detach().cpu().numpy(),
            "ln_beta": module.ln.bias.detach().cpu().numpy(),
            "w2": module.w2.weight.detach().cpu().numpy(),
            "b2": module.w2.bias.detach().cpu().numpy(),
            "alpha": float(module.alpha.detach().cpu().item()),
        }

    model = PrismModel(
        query_adapter=PrismAdapter(**{k: v for k, v in export_adapter(q_adapter).items()}),
        chunk_adapter=PrismAdapter(**{k: v for k, v in export_adapter(c_adapter).items()}),
        embed_dim=dim,
        embed_model=embedding_model(),
    )
    _, test = split_pairs(pairs)
    metrics = evaluate_model(model, test)
    out = output or models_dir() / "personal.prz"
    save_prz(model, out)
    metrics["output"] = str(out)
    return model, metrics


def train_from_corpus(
    *,
    epochs: int = 5,
    output: Path | None = None,
    retriever: str | None = None,
    collect_first: bool = False,
    collect_retriever: str = "legacy",
    min_queries: int = 50,
    top_k: int = 20,
    label_limit: int = 500,
) -> dict[str, Any]:
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

        pairs = load_training_pairs_from_db(retriever=retriever)
        if not pairs:
            raise RuntimeError(
                "No labeled training samples — run fish corpus collect and fish corpus label first"
            )

        train, test = split_pairs(pairs)
        baseline = new_identity_model()
        baseline_metrics = evaluate_model(baseline, test)

        init_db()
        with db_conn() as db:
            labeled_rows = load_labeled_training_pairs(
                db, exclude_superseded=True, retriever=retriever
            )
        retrieval_eval = evaluate_retrieval_similarity(labeled_rows)

        _, metrics = train_prism_model(train, epochs=epochs, output=output)
        return {
            "pairs": len(pairs),
            "train": len(train),
            "test": len(test),
            "baseline": baseline_metrics,
            "retrieval_eval": retrieval_eval,
            "trained": metrics,
        }
