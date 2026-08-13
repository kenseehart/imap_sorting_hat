"""Retrieval model registry: legacy (raw cosine) + PRISM model_id → vec table + .prz."""

from __future__ import annotations

import sqlite3
from typing import Any

from fish.config import models_dir
from fish.prism.configs import (
    LEGACY_MODEL_ID,
    LEGACY_VEC_TABLE,
    make_model_id,
    parse_model_id,
    vec_table_for_model_id,
)
from fish.store import _ensure_vec_table, _utcnow


def ensure_legacy_model(db: sqlite3.Connection) -> None:
    row = db.execute(
        "SELECT model_id FROM retrieval_models WHERE model_id = ?",
        (LEGACY_MODEL_ID,),
    ).fetchone()
    if row:
        return
    db.execute(
        """
        INSERT INTO retrieval_models (
            model_id, config_name, vec_table, prz_name, created_at, active, meta_json
        ) VALUES (?, ?, ?, NULL, ?, 1, NULL)
        """,
        (LEGACY_MODEL_ID, LEGACY_MODEL_ID, LEGACY_VEC_TABLE, _utcnow()),
    )
    _ensure_vec_table(db, LEGACY_VEC_TABLE)


def list_retrieval_models(db: sqlite3.Connection) -> list[dict[str, Any]]:
    ensure_legacy_model(db)
    rows = db.execute(
        "SELECT * FROM retrieval_models ORDER BY created_at, model_id"
    ).fetchall()
    return [dict(r) for r in rows]


def get_retrieval_model(db: sqlite3.Connection, model_id: str) -> dict[str, Any] | None:
    ensure_legacy_model(db)
    row = db.execute(
        "SELECT * FROM retrieval_models WHERE model_id = ?", (model_id,)
    ).fetchone()
    return dict(row) if row else None


def active_prism_model(db: sqlite3.Connection) -> dict[str, Any] | None:
    """Non-legacy model marked active (at most one)."""
    ensure_legacy_model(db)
    row = db.execute(
        """
        SELECT * FROM retrieval_models
        WHERE active = 1 AND model_id != ?
        ORDER BY created_at DESC LIMIT 1
        """,
        (LEGACY_MODEL_ID,),
    ).fetchone()
    return dict(row) if row else None


def register_prism_model(
    db: sqlite3.Connection,
    *,
    config_name: str,
    meta_json: str | None = None,
    activate: bool = True,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Create model_id, vec table, registry row. prz file written separately by train."""
    ensure_legacy_model(db)
    model_id = make_model_id(config_name, timestamp=timestamp)
    parse_model_id(model_id)  # validate
    vec_table = vec_table_for_model_id(model_id)
    prz_name = f"{model_id}.prz"
    if activate:
        db.execute(
            "UPDATE retrieval_models SET active = 0 WHERE model_id != ?",
            (LEGACY_MODEL_ID,),
        )
    db.execute(
        """
        INSERT INTO retrieval_models (
            model_id, config_name, vec_table, prz_name, created_at, active, meta_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            model_id,
            config_name,
            vec_table,
            prz_name,
            _utcnow(),
            1 if activate else 0,
            meta_json,
        ),
    )
    _ensure_vec_table(db, vec_table)
    row = get_retrieval_model(db, model_id)
    assert row is not None
    return row


def set_active_prism_model(db: sqlite3.Connection, model_id: str) -> None:
    if model_id == LEGACY_MODEL_ID:
        db.execute(
            "UPDATE retrieval_models SET active = 0 WHERE model_id != ?",
            (LEGACY_MODEL_ID,),
        )
        return
    row = get_retrieval_model(db, model_id)
    if row is None:
        raise KeyError(f"Unknown model_id {model_id!r}")
    db.execute(
        "UPDATE retrieval_models SET active = 0 WHERE model_id != ?",
        (LEGACY_MODEL_ID,),
    )
    db.execute(
        "UPDATE retrieval_models SET active = 1 WHERE model_id = ?",
        (model_id,),
    )


def prz_path_for_model(model: dict[str, Any]) -> Any:
    name = model.get("prz_name")
    if not name:
        return None
    return models_dir() / name


def ensure_model_vec_tables(db: sqlite3.Connection) -> None:
    for model in list_retrieval_models(db):
        _ensure_vec_table(db, model["vec_table"])
