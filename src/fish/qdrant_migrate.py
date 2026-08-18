"""Migrate raw embeddings from sqlite-vec → SQLite blobs + Qdrant ANN."""

from __future__ import annotations

from typing import Any

from cmdline.progress import progress_bar

from fish.prism.configs import LEGACY_MODEL_ID
from fish.qdrant_store import (
    all_point_ids,
    build_payload,
    collection_point_count,
    ensure_collection,
    upsert_points_batch,
)
from fish.store import (
    blob_to_embedding,
    db_conn,
    init_db,
)
from fish.write_lock import fish_write_lock


def _utcnow() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _sqlite_vec_table_names(db) -> list[str]:
    rows = db.execute(
        """
        SELECT name FROM sqlite_master
        WHERE type='table' AND (name='corpus_vec' OR name LIKE 'corpus_vec%'
              OR name='message_vec')
        ORDER BY name
        """
    ).fetchall()
    return [r[0] for r in rows]


def _load_sqlite_vec(db) -> None:
    try:
        import sqlite_vec
    except ImportError as exc:
        raise RuntimeError(
            "sqlite-vec is required to copy legacy ANN tables into "
            "corpus_raw_embeddings. Install sqlite-vec, then re-run migrate."
        ) from exc
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)


def copy_raw_from_sqlite_vec(
    *,
    source_table: str | None = None,
    limit: int | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Copy float32 vectors from a sqlite-vec table into corpus_raw_embeddings."""
    init_db()
    with fish_write_lock("train"):
        with db_conn() as db:
            _load_sqlite_vec(db)
            tables = _sqlite_vec_table_names(db)
            if not tables:
                return {
                    "copied": 0,
                    "note": "No sqlite-vec tables found",
                    "tables": [],
                }
            table = source_table or (
                "corpus_vec" if "corpus_vec" in tables else tables[0]
            )
            if table not in tables:
                raise ValueError(
                    f"Unknown sqlite-vec table {table!r}. Found: {tables}"
                )
            sql = f"SELECT rowid, embedding FROM {table}"
            if limit is not None:
                sql += f" LIMIT {int(limit)}"
            rows = db.execute(sql).fetchall()
            bar = progress_bar(
                total=len(rows),
                desc=f"copy {table}",
                unit="vec",
                disable=not show_progress,
            )
            copied = 0
            skipped = 0

            for row in rows:
                item_id = int(row[0])
                exists = db.execute(
                    "SELECT 1 FROM corpus_items WHERE id = ?", (item_id,)
                ).fetchone()
                if not exists:
                    skipped += 1
                    bar.update(1)
                    continue
                blob = row[1]
                if isinstance(blob, memoryview):
                    blob = blob.tobytes()
                db.execute(
                    """
                    INSERT INTO corpus_raw_embeddings (item_id, embedding, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(item_id) DO UPDATE SET
                        embedding = excluded.embedding,
                        updated_at = excluded.updated_at
                    """,
                    (item_id, blob, _utcnow()),
                )
                copied += 1
                bar.update(1)
            bar.close()
    return {
        "source_table": table,
        "tables_seen": tables,
        "copied": copied,
        "skipped_orphans": skipped,
        "limit": limit,
    }


def reindex_legacy_qdrant(
    *,
    limit: int | None = None,
    batch_size: int = 256,
    show_progress: bool = True,
    kinds: list[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Upsert corpus_raw_embeddings into the legacy Qdrant collection (streamed).

    Resumable: by default, point ids already in the collection are skipped
    (no re-decode, no re-upsert). Pass ``force=True`` to rewrite every point.
    """
    init_db()
    with fish_write_lock("train"):
        with db_conn() as db:
            from fish.prism.registry import get_retrieval_model

            model = get_retrieval_model(db, LEGACY_MODEL_ID)
            if model is None:
                raise RuntimeError("legacy model missing")
            collection = model["vec_table"]
            ensure_collection(collection)

            already: set[int] = set()
            if not force:
                already = all_point_ids(collection)

            count_sql = """
                SELECT COUNT(*) FROM corpus_raw_embeddings r
                JOIN corpus_items c ON c.id = r.item_id
                WHERE 1=1
            """
            count_params: list[Any] = []
            if kinds:
                placeholders = ",".join("?" for _ in kinds)
                count_sql += f" AND c.kind IN ({placeholders})"
                count_params.extend(kinds)
            total = int(db.execute(count_sql, count_params).fetchone()[0])
            if limit is not None:
                total = min(total, int(limit))

            bar = progress_bar(
                total=total,
                desc=f"qdrant {collection}",
                unit="pt",
                disable=not show_progress,
            )
            upserted = 0
            skipped_existing = 0
            scanned = 0
            last_id = 0
            while scanned < total:
                chunk = min(batch_size, total - scanned)
                id_sql = """
                    SELECT c.id
                    FROM corpus_raw_embeddings r
                    JOIN corpus_items c ON c.id = r.item_id
                    WHERE c.id > ?
                """
                id_params: list[Any] = [last_id]
                if kinds:
                    placeholders = ",".join("?" for _ in kinds)
                    id_sql += f" AND c.kind IN ({placeholders})"
                    id_params.extend(kinds)
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
                    SELECT c.id, c.kind, c.source, c.occurred_at, c.payload, r.embedding
                    FROM corpus_raw_embeddings r
                    JOIN corpus_items c ON c.id = r.item_id
                    WHERE c.id IN ({placeholders})
                    ORDER BY c.id ASC
                """
                rows = db.execute(row_sql, to_load).fetchall()
                batch: list[tuple[int, Any, dict[str, Any]]] = []
                for row in rows:
                    item_id = int(row["id"])
                    emb = blob_to_embedding(row["embedding"])
                    if emb is None:
                        bar.update(1)
                        continue
                    corpus = {
                        "id": item_id,
                        "kind": row["kind"],
                        "source": row["source"],
                        "occurred_at": row["occurred_at"],
                        "payload": row["payload"],
                    }
                    batch.append((item_id, emb, build_payload(corpus)))
                if batch:
                    upsert_points_batch(collection, batch)
                    upserted += len(batch)
                    already.update(item_id for item_id, _, _ in batch)
                    bar.update(len(batch))
            bar.close()
            count = collection_point_count(collection)
    return {
        "collection": collection,
        "upserted": upserted,
        "skipped_existing": skipped_existing,
        "scanned": scanned,
        "points_in_collection": count,
        "force": force,
        "limit": limit,
    }


def migrate_to_qdrant(
    *,
    copy_from_sqlite_vec: bool = True,
    source_table: str | None = None,
    limit: int | None = None,
    batch_size: int = 256,
    show_progress: bool = True,
    skip_qdrant: bool = False,
) -> dict[str, Any]:
    """One-shot: optional sqlite-vec copy → upsert legacy collection."""
    result: dict[str, Any] = {}
    if copy_from_sqlite_vec:
        result["copy"] = copy_raw_from_sqlite_vec(
            source_table=source_table,
            limit=limit,
            show_progress=show_progress,
        )
    if not skip_qdrant:
        result["qdrant"] = reindex_legacy_qdrant(
            limit=limit,
            batch_size=batch_size,
            show_progress=show_progress,
            force=False,
        )
    return result
