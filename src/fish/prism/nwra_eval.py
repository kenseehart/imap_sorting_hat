"""NWRA report over labeled training samples (same-candidate ranking).

NWRA = WRA(model order) / WRA(RA-optimal order). Null when perfect WRA is 0.

Also reports Spearman(model_score, RA) over the same pairs.
``personal_rerank`` (scoring=mlp_head) scores via MLP(Aq‖Ac); other models use
adapted cosine (or identity cosine for legacy).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from fish.prism.inference import (
    compose_chunk_vector,
    load_prism_model,
)
from fish.prism.model import (
    CHUNK_REPR_COMBINED,
    CHUNK_REPR_HEADER_BODY,
    new_identity_model,
)
from fish.prism.train import _spearman
from fish.store import db_conn, get_raw_embedding, init_db, load_labeled_training_pairs


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


def _score_pairs(
    db: Any,
    model: Any,
    pairs: list[dict[str, Any]],
) -> list[tuple[float, float]]:
    """Return (model_score, target_relevance) for each pair."""
    chunk_repr = getattr(model, "chunk_repr", CHUNK_REPR_COMBINED)
    out: list[tuple[float, float]] = []
    for row in pairs:
        q = row.get("query_embedding")
        if not isinstance(q, list):
            continue
        item_id = int(row["corpus_item_id"])
        if chunk_repr == CHUNK_REPR_HEADER_BODY:
            c = compose_chunk_vector(db, item_id, CHUNK_REPR_HEADER_BODY)
        else:
            c = row.get("message_embedding")
            if not isinstance(c, list):
                c = get_raw_embedding(db, item_id)
        if not isinstance(c, list):
            continue
        try:
            s = float(model.score_pair(q, c))
        except Exception:
            continue
        out.append((s, float(row["target_relevance"])))
    return out


def metrics_for_model(
    db: Any,
    model: Any,
    by_query: dict[str, list[dict[str, Any]]],
    *,
    query_filter: set[str] | None = None,
) -> dict[str, Any]:
    per_query_nwra: list[float] = []
    all_scores: list[float] = []
    all_labels: list[float] = []
    skipped = 0
    for qtext, rows in by_query.items():
        if query_filter is not None and qtext not in query_filter:
            continue
        scored = _score_pairs(db, model, rows)
        if len(scored) < 2:
            skipped += 1
            continue
        for s, r in scored:
            all_scores.append(s)
            all_labels.append(r)
        scored.sort(key=lambda t: t[0], reverse=True)
        rels = [r for _, r in scored]
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


def nwra_for_model(
    db: Any,
    model: Any,
    by_query: dict[str, list[dict[str, Any]]],
    *,
    query_filter: set[str] | None = None,
) -> dict[str, Any]:
    """Backward-compatible alias — returns NWRA block (plus spearman)."""
    return metrics_for_model(db, model, by_query, query_filter=query_filter)


def build_nwra_report(
    model_ids: list[str],
    *,
    agent_version: str | None = None,
    out_path: Path | None = None,
) -> dict[str, Any]:
    """Build report. ``agent_version=None`` → all labeled rows (recommended)."""
    init_db()
    with db_conn() as db:
        rows = load_labeled_training_pairs(db, exclude_superseded=True)
    if agent_version:
        rows = [
            r
            for r in rows
            if (r.get("relevance_agent_version") or "") == agent_version
        ]
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_query[str(r["query_text"])].append(r)

    systems: dict[str, Any] = {}
    with db_conn() as db:
        systems["legacy"] = {
            "all": metrics_for_model(db, new_identity_model(), by_query),
            "burn": metrics_for_model(
                db, new_identity_model(), by_query, query_filter=BURN_QUERIES
            ),
        }
        for mid in model_ids:
            try:
                model = load_prism_model(mid)
            except Exception as exc:
                systems[mid] = {"error": str(exc)}
                continue
            systems[mid] = {
                "all": metrics_for_model(db, model, by_query),
                "burn": metrics_for_model(
                    db, model, by_query, query_filter=BURN_QUERIES
                ),
                "chunk_repr": getattr(model, "chunk_repr", None),
                "adapter_sharing": getattr(model, "adapter_sharing", None),
                "scoring": getattr(model, "scoring", None),
            }

    report = {
        "metric": "NWRA = WRA(model)/WRA(RA-optimal); Spearman(model_score, RA)",
        "agent_version": agent_version,
        "n_labeled_rows": len(rows),
        "n_queries": len(by_query),
        "systems": systems,
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", action="append", default=[], help="PRISM model_id")
    p.add_argument(
        "--agent-version",
        default="",
        help="Filter to this RA version (default: all labeled rows)",
    )
    p.add_argument("--out", default="/data/fish/nwra_report.json")
    args = p.parse_args()
    ver = args.agent_version.strip() or None
    report = build_nwra_report(
        args.model_id,
        agent_version=ver,
        out_path=Path(args.out),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
