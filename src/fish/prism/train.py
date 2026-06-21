from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI

from fish.config import EMBED_DIM, MODELS_DIR, embedding_model, openai_api_key
from fish.embed import embed_text, embed_texts
from fish.prism.inference import cosine_similarity
from fish.prism.model import PrismModel, new_identity_model, save_prz
from fish.store import db_conn, get_corpus_by_id, init_db


@dataclass
class TrainingPair:
    query: str
    chunk_id: int
    relevance: float


def _pair_hash(query: str, chunk_id: int) -> str:
    return hashlib.sha256(f"{query}\0{chunk_id}".encode()).hexdigest()


def sample_corpus_queries(limit: int = 200) -> list[str]:
    init_db()
    with db_conn() as db:
        rows = db.execute(
            """
            SELECT text_for_embed, payload, kind FROM corpus_items
            WHERE embedded_at IS NOT NULL
            ORDER BY RANDOM()
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    queries: list[str] = []
    for row in rows:
        kind = row["kind"]
        text = row["text_for_embed"] or ""
        if kind == "email":
            payload = json.loads(row["payload"] or "{}")
            subject = payload.get("subject") or ""
            if subject:
                queries.append(subject)
            if text:
                queries.append(text[:200])
        elif kind in ("sms", "chat", "memory"):
            queries.append(text[:200])
    return queries[:limit]


def generate_training_pairs(
    *,
    sample_size: int = 500,
    label_model: str = "gpt-4o-mini",
    seed: int = 0,
) -> list[TrainingPair]:
    init_db()
    rng = random.Random(seed)
    queries = sample_corpus_queries(limit=max(50, sample_size // 5))
    if not queries:
        raise RuntimeError("No embedded corpus items — sync or import data first")

    with db_conn() as db:
        items = db.execute(
            """
            SELECT id, text_for_embed, kind FROM corpus_items
            WHERE embedded_at IS NOT NULL
            ORDER BY RANDOM()
            LIMIT ?
            """,
            (sample_size * 2,),
        ).fetchall()

    if not items:
        raise RuntimeError("No embedded corpus items for training")

    client = OpenAI(api_key=openai_api_key())
    pairs: list[TrainingPair] = []
    for query in queries:
        candidates = rng.sample(list(items), min(8, len(items)))
        for row in candidates:
            chunk_id = int(row["id"])
            chunk_text = (row["text_for_embed"] or "")[:1500]
            prompt = (
                "Rate relevance of this document to the query on a 0.0-1.0 scale.\n"
                f"Query: {query}\n\nDocument:\n{chunk_text}\n\n"
                "Reply with only a number."
            )
            response = client.chat.completions.create(
                model=label_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8,
            )
            raw = (response.choices[0].message.content or "0").strip()
            try:
                relevance = float(raw.split()[0])
            except (ValueError, IndexError):
                relevance = 0.0
            relevance = max(0.0, min(1.0, relevance))
            pairs.append(TrainingPair(query=query, chunk_id=chunk_id, relevance=relevance))
            if len(pairs) >= sample_size:
                return pairs
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
    init_db()
    query_texts = list({p.query for p in pairs})
    q_map = {q: embed_text(q) for q in query_texts}
    chunk_ids = list({p.chunk_id for p in pairs})
    with db_conn() as db:
        from fish.store import get_embedding

        chunks = {cid: get_embedding(db, cid) for cid in chunk_ids}

    raw_scores: list[float] = []
    adapted_scores: list[float] = []
    labels: list[float] = []
    for pair in pairs:
        q = q_map[pair.query]
        c = chunks.get(pair.chunk_id)
        if c is None:
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

    query_texts = list({p.query for p in pairs})
    q_embeds = {q: embed_text(q) for q in query_texts}
    chunk_ids = list({p.chunk_id for p in pairs})
    init_db()
    with db_conn() as db:
        from fish.store import get_embedding

        c_embeds = {cid: get_embedding(db, cid) for cid in chunk_ids}

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
        c = c_embeds.get(pair.chunk_id)
        if c is None:
            continue
        tensors.append(
            (
                torch.tensor(q_embeds[pair.query], dtype=torch.float32),
                torch.tensor(c, dtype=torch.float32),
                torch.tensor([pair.relevance], dtype=torch.float32),
            )
        )

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

    from fish.prism.model import PrismAdapter, PrismModel

    model = PrismModel(
        query_adapter=PrismAdapter(**{k: v for k, v in export_adapter(q_adapter).items()}),
        chunk_adapter=PrismAdapter(**{k: v for k, v in export_adapter(c_adapter).items()}),
        embed_dim=dim,
        embed_model=embedding_model(),
    )
    _, test = split_pairs(pairs)
    metrics = evaluate_model(model, test)
    out = output or MODELS_DIR / "personal.prz"
    save_prz(model, out)
    metrics["output"] = str(out)
    return model, metrics


def train_from_corpus(
    *,
    sample_size: int = 500,
    epochs: int = 5,
    output: Path | None = None,
) -> dict[str, Any]:
    pairs = generate_training_pairs(sample_size=sample_size)
    train, test = split_pairs(pairs)
    baseline = new_identity_model()
    baseline_metrics = evaluate_model(baseline, test)
    _, metrics = train_prism_model(train, epochs=epochs, output=output)
    return {
        "pairs": len(pairs),
        "train": len(train),
        "test": len(test),
        "baseline": baseline_metrics,
        "trained": metrics,
    }
