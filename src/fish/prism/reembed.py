"""Re-index a registered PRISM model's Qdrant collection from corpus_raw_embeddings."""

from __future__ import annotations

from typing import Any

from cmdline.progress import progress_bar

from fish.prism.configs import LEGACY_MODEL_ID
from fish.prism.inference import (
    adapt_chunk_for_model,
    clear_model_cache,
    compose_chunk_vector,
    load_prism_model,
)
from fish.qdrant_store import all_point_ids, build_payload, upsert_points_batch
from fish.store import db_conn, init_db
from fish.write_lock import fish_write_lock


def prism_reembed(
    *,
    model_id: str | None = None,
    kinds: list[str] | None = None,
    show_progress: bool = True,
    limit: int | None = None,
    like: list[str] | None = None,
    since: str | None = None,
    batch_size: int = 128,
    force: bool = False,
) -> dict[str, Any]:
    """Rewrite one PRISM model's Qdrant collection from stored raw embeddings.

    Streams raw embeddings from SQLite in id-ordered pages and upserts to
    Qdrant in batches. Resumable: existing point ids are skipped unless
    ``force=True``.
    """
    init_db()
    clear_model_cache()
    with fish_write_lock("train"):
        with db_conn() as db:
            from fish.prism.registry import active_prism_model, get_retrieval_model

            if model_id is None:
                active = active_prism_model(db)
                if active is None:
                    raise RuntimeError(
                        "No active PRISM model. Pass --model-id or activate one "
                        "after fish prism-train."
                    )
                model_id = active["model_id"]
            if model_id == LEGACY_MODEL_ID:
                raise ValueError("Cannot re-index legacy/raw — it is the source index")
            model = get_retrieval_model(db, model_id)
            if model is None:
                raise KeyError(f"Unknown model_id {model_id!r}")
            collection = model["vec_table"]
            prism = load_prism_model(model_id)
            chunk_repr = prism.chunk_repr

            already: set[int] = set()
            if not force:
                already = all_point_ids(collection)

            count_sql = """
                SELECT COUNT(*)
                FROM corpus_items c
                JOIN corpus_raw_embeddings r ON r.item_id = c.id
                WHERE 1=1
            """
            count_params: list[Any] = []
            if kinds:
                placeholders = ",".join("?" for _ in kinds)
                count_sql += f" AND c.kind IN ({placeholders})"
                count_params.extend(kinds)
            if since:
                count_sql += " AND c.occurred_at >= ?"
                count_params.append(since)
            if like:
                like_clauses = []
                for pattern in like:
                    like_clauses.append(
                        "(ifnull(c.text_for_embed,'') LIKE ? OR ifnull(c.body_text,'') LIKE ?)"
                    )
                    count_params.extend([pattern, pattern])
                count_sql += " AND (" + " OR ".join(like_clauses) + ")"
            total = int(db.execute(count_sql, count_params).fetchone()[0])
            if limit is not None:
                total = min(total, int(limit))
            if total == 0:
                return {
                    "model_id": model_id,
                    "collection": collection,
                    "adapted": 0,
                    "openai_calls": 0,
                    "note": (
                        "No raw vectors in corpus_raw_embeddings — "
                        "run fish embed / qdrant-migrate first"
                    ),
                }

            bar = progress_bar(
                total=total,
                desc=f"index {model_id}",
                unit="msg",
                disable=not show_progress,
            )
            adapted = 0
            skipped_existing = 0
            scanned = 0
            last_id = 0
            while scanned < total:
                chunk = min(batch_size, total - scanned)
                id_sql = """
                    SELECT c.id
                    FROM corpus_items c
                    JOIN corpus_raw_embeddings r ON r.item_id = c.id
                    WHERE c.id > ?
                """
                id_params: list[Any] = [last_id]
                if kinds:
                    placeholders = ",".join("?" for _ in kinds)
                    id_sql += f" AND c.kind IN ({placeholders})"
                    id_params.extend(kinds)
                if since:
                    id_sql += " AND c.occurred_at >= ?"
                    id_params.append(since)
                if like:
                    like_clauses = []
                    for pattern in like:
                        like_clauses.append(
                            "(ifnull(c.text_for_embed,'') LIKE ? OR ifnull(c.body_text,'') LIKE ?)"
                        )
                        id_params.extend([pattern, pattern])
                    id_sql += " AND (" + " OR ".join(like_clauses) + ")"
                id_sql += " ORDER BY c.id ASC LIMIT ?"
                id_params.append(chunk)
                id_rows = db.execute(id_sql, id_params).fetchall()
                if not id_rows:
                    break
                page_ids = [int(r[0]) for r in id_rows]
                last_id = page_ids[-1]
                scanned += len(page_ids)

                to_load = [i for i in page_ids if i not in already]
                skipped = len(page_ids) - len(to_load)
                if skipped:
                    skipped_existing += skipped
                    bar.update(skipped)
                if not to_load:
                    continue

                placeholders = ",".join("?" for _ in to_load)
                row_sql = f"""
                    SELECT c.id, c.kind, c.source, c.occurred_at, c.payload
                    FROM corpus_items c
                    JOIN corpus_raw_embeddings r ON r.item_id = c.id
                    WHERE c.id IN ({placeholders})
                    ORDER BY c.id ASC
                """
                batch: list[tuple[int, list[float], dict[str, Any]]] = []
                for row in db.execute(row_sql, to_load).fetchall():
                    raw = compose_chunk_vector(db, int(row["id"]), chunk_repr)
                    if not raw:
                        bar.update(1)
                        continue
                    ac = adapt_chunk_for_model(raw, model_id)
                    corpus = {
                        "id": int(row["id"]),
                        "kind": row["kind"],
                        "source": row["source"],
                        "occurred_at": row["occurred_at"],
                        "payload": row["payload"],
                    }
                    batch.append((int(row["id"]), ac, build_payload(corpus)))
                    bar.update(1)
                if batch:
                    upsert_points_batch(collection, batch)
                    adapted += len(batch)
                    already.update(item_id for item_id, _, _ in batch)
            bar.close()

    return {
        "model_id": model_id,
        "collection": collection,
        "adapted": adapted,
        "skipped_existing": skipped_existing,
        "scanned": scanned,
        "limit": limit,
        "like": like,
        "since": since,
        "force": force,
        "chunk_repr": chunk_repr,
        "openai_calls": 0,
    }
