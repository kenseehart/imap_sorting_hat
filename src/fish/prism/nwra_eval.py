"""NWRA report over a frozen training corpus (same-candidate ranking).

NWRA = WRA(model order) / WRA(RA-optimal order). Null when perfect WRA is 0.

Scores from a dual ``.tcz`` (``q`` / ``c_joint`` / ``c_split`` / ``rel``) —
same artifact as ``fish prism-train --corpus``. Does **not** open fish.db or
re-fetch embeddings for the hot path.

Also reports Spearman(model_score, RA) over the same pairs.
``rerank_*`` (scoring=mlp_head) scores via MLP(Aq‖Ac); other models use
adapted cosine (or identity cosine for legacy).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from fish.prism.inference import load_prism_model
from fish.prism.model import (
    CHUNK_REPR_JOINT,
    CHUNK_REPR_SPLIT,
    SCORING_MLP_HEAD,
    new_identity_model,
    normalize_chunk_repr,
)
from fish.prism.train import _spearman
from fish.prism.train_corpus import load_tcz, resolve_corpus_path


def rank_weight(i: int) -> float:
    return 1.0 / (2.0 + i)


def weighted_ra(rels: list[float]) -> float:
    if not rels:
        return 0.0
    ws = [rank_weight(i) for i in range(len(rels))]
    return float(sum(w * r for w, r in zip(ws, rels)) / sum(ws))


def nwra(rels_in_model_order: list[float]) -> float | None:
    perfect = weighted_ra(sorted(rels_in_model_order, reverse=True))
    if perfect <= 1e-12:
        return None
    return weighted_ra(rels_in_model_order) / perfect


BURN_QUERIES = {
    "Burning Man",
    "messages about Burn CREW without saying Burning Man",
    "Interaction Café",
    "Burn CREW",
    "camp belonging Interaction Café",
    "packing list Burning Man",
    "Axis Mundi tickets",
    "who is running the café",
}


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def _adapt_query_batch(model: Any, q: np.ndarray) -> np.ndarray:
    x = np.asarray(q, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[-1] != model.embed_dim:
        raise ValueError(f"query dim {x.shape[-1]} != embed_dim {model.embed_dim}")
    return model.query_adapter.forward(x)


def _adapt_chunk_batch(model: Any, c: np.ndarray) -> np.ndarray:
    x = np.asarray(c, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    expect = model.chunk_input_dim
    if x.shape[-1] != expect:
        raise ValueError(
            f"chunk dim {x.shape[-1]} != chunk_input_dim {expect} "
            f"(chunk_repr={model.chunk_repr!r})"
        )
    if model.chunk_adapter.identity and expect == 2 * model.embed_dim:
        half = int(model.embed_dim)
        return 0.5 * (x[:, :half] + x[:, half:])
    return model.chunk_adapter.forward(x)


def score_pairs_batch(
    model: Any,
    q: np.ndarray,
    c: np.ndarray,
    *,
    batch_size: int = 2048,
) -> np.ndarray:
    """Batched scores for aligned (q[i], c[i]) rows."""
    n = len(q)
    out = np.empty(n, dtype=np.float32)
    for start in range(0, n, batch_size):
        sl = slice(start, min(start + batch_size, n))
        aq = _adapt_query_batch(model, q[sl])
        ac = _adapt_chunk_batch(model, c[sl])
        if getattr(model, "scoring", None) == SCORING_MLP_HEAD:
            head = model.rerank_head
            if head is None:
                raise ValueError("scoring=mlp_head requires rerank_head weights")
            concat = np.concatenate([aq, ac], axis=-1)
            h = _gelu(concat @ head.w1.T + head.b1)
            logit = (h @ head.w2.T + head.b2).reshape(-1)
            out[sl] = 1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0)))
        else:
            nq = np.linalg.norm(aq, axis=-1)
            nc = np.linalg.norm(ac, axis=-1)
            denom = nq * nc
            dots = np.sum(aq * ac, axis=-1)
            scores = np.zeros_like(dots)
            ok = denom > 0
            scores[ok] = dots[ok] / denom[ok]
            out[sl] = scores
    return out


def _group_indices(queries: list[str]) -> dict[str, list[int]]:
    by_query: dict[str, list[int]] = defaultdict(list)
    for i, qtext in enumerate(queries):
        by_query[str(qtext)].append(i)
    return by_query


def metrics_from_scores(
    scores: np.ndarray,
    rel: np.ndarray,
    by_query: dict[str, list[int]],
    *,
    query_filter: set[str] | None = None,
) -> dict[str, Any]:
    """NWRA + Spearman from precomputed per-pair scores (index-aligned with rel)."""
    per_query_nwra: list[float] = []
    all_scores: list[float] = []
    all_labels: list[float] = []
    skipped = 0
    for qtext, idxs in by_query.items():
        if query_filter is not None and qtext not in query_filter:
            continue
        if len(idxs) < 2:
            skipped += 1
            continue
        s = scores[idxs]
        r = rel[idxs]
        all_scores.extend(float(x) for x in s)
        all_labels.extend(float(x) for x in r)
        order = np.argsort(-s)
        rels = [float(r[j]) for j in order]
        val = nwra(rels)
        if val is not None:
            per_query_nwra.append(val)
    return {
        "nwra_mean": float(np.mean(per_query_nwra)) if per_query_nwra else None,
        "spearman": _spearman(all_scores, all_labels) if len(all_scores) >= 2 else None,
        "n_pairs": len(all_scores),
        "n_queries": len(per_query_nwra),
        "skipped_queries": skipped,
    }


def metrics_for_model_tcz(
    model: Any,
    q: np.ndarray,
    c: np.ndarray,
    rel: np.ndarray,
    by_query: dict[str, list[int]],
) -> dict[str, Any]:
    """Score once; report all-queries + burn subset (no double scoring)."""
    scores = score_pairs_batch(model, q, c)
    return {
        "all": metrics_from_scores(scores, rel, by_query),
        "burn": metrics_from_scores(
            scores, rel, by_query, query_filter=BURN_QUERIES
        ),
    }


def build_nwra_report(
    model_ids: list[str],
    *,
    corpus: str = "latest",
    out_path: Path | None = None,
) -> dict[str, Any]:
    """Build report from a frozen dual ``.tcz`` (joint + split preferred)."""
    from compute.tasks import TaskProgress

    n_systems = 1 + len(model_ids)
    with TaskProgress(
        module="fish",
        task="nwra",
        n=n_systems + 1,
        detail="loading .tcz",
        meta={"n_models": len(model_ids), "corpus": corpus, "out": str(out_path) if out_path else None},
    ) as task:
        path = resolve_corpus_path(corpus, chunk_repr="both")
        frozen = load_tcz(path)
        if CHUNK_REPR_JOINT not in frozen.c_by_repr:
            raise RuntimeError(
                f"{frozen.corpus_id} missing c_joint — re-freeze with --chunk-repr both"
            )
        if CHUNK_REPR_SPLIT not in frozen.c_by_repr:
            raise RuntimeError(
                f"{frozen.corpus_id} missing c_split — re-freeze with --chunk-repr both"
            )

        q = np.asarray(frozen.q, dtype=np.float32)
        rel = np.asarray(frozen.relevance, dtype=np.float32)
        c_joint = np.asarray(frozen.c_by_repr[CHUNK_REPR_JOINT], dtype=np.float32)
        c_split = np.asarray(frozen.c_by_repr[CHUNK_REPR_SPLIT], dtype=np.float32)
        by_query = _group_indices(frozen.queries)

        task.update(
            1,
            detail=(
                f"loaded {frozen.corpus_id} "
                f"({frozen.n_pairs} pairs / {len(by_query)} queries)"
            ),
            force=True,
        )

        systems: dict[str, Any] = {}
        legacy = new_identity_model()
        legacy.chunk_repr = CHUNK_REPR_JOINT
        task.update(1, detail="scoring legacy", force=True)
        block = metrics_for_model_tcz(legacy, q, c_joint, rel, by_query)
        systems["legacy"] = block
        task.update(2, detail="scored legacy", force=True)

        for i, mid in enumerate(model_ids, start=1):
            task.update(1 + i, detail=f"scoring {mid}", force=True)
            try:
                model = load_prism_model(mid)
            except Exception as exc:
                systems[mid] = {"error": str(exc)}
                task.update(2 + i, detail=f"error {mid}", force=True)
                continue
            chunk_repr = normalize_chunk_repr(getattr(model, "chunk_repr", None))
            c = c_split if chunk_repr == CHUNK_REPR_SPLIT else c_joint
            block = metrics_for_model_tcz(model, q, c, rel, by_query)
            systems[mid] = {
                **block,
                "chunk_repr": chunk_repr,
                "adapter_sharing": getattr(model, "adapter_sharing", None),
                "scoring": getattr(model, "scoring", None),
            }
            task.update(2 + i, detail=f"scored {mid}", force=True)

        report = {
            "metric": "NWRA = WRA(model)/WRA(RA-optimal); Spearman(model_score, RA)",
            "corpus_id": frozen.corpus_id,
            "corpus_path": str(frozen.path),
            "n_labeled_rows": frozen.n_pairs,
            "n_queries": len(by_query),
            "systems": systems,
        }
        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(report, indent=2))
            task.update(n_systems + 1, detail=f"wrote {out_path.name}", force=True)
        return report


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", action="append", default=[], help="PRISM model_id")
    p.add_argument(
        "--corpus",
        default="latest",
        help="Frozen .tcz: latest (default), train_corpus_* id, or path",
    )
    p.add_argument("--out", default="/data/fish/nwra_report.json")
    args = p.parse_args()
    report = build_nwra_report(
        args.model_id,
        corpus=args.corpus,
        out_path=Path(args.out),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
