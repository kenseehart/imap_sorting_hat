from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from fish.config import MODELS_DIR
from fish.store import db_conn, init_db


def _model_path(name: str = "folder_classifier.pkl") -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR / name


def train_classifier(
    labels: dict[int, str],
    name: str = "folder_classifier",
) -> dict[str, Any]:
    init_db()
    ids = list(labels.keys())
    if len(ids) < 2:
        raise ValueError("Need at least 2 labeled messages to train")

    with db_conn() as db:
        vectors = []
        y = []
        for msg_id, label in labels.items():
            row = db.execute(
                "SELECT embedding FROM message_vec WHERE rowid = ?", (msg_id,)
            ).fetchone()
            if not row:
                continue
            vec = np.frombuffer(row["embedding"], dtype=np.float32)
            vectors.append(vec)
            y.append(label)

    if len(vectors) < 2:
        raise ValueError("Not enough embedded messages among labeled ids")

    X = np.array(vectors)
    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(X, y)

    path = _model_path(f"{name}.pkl")
    path.write_bytes(pickle.dumps({"model": clf, "labels": labels}))
    return {"model": name, "samples": len(y), "classes": list(clf.classes_)}


def predict_labels(message_ids: list[int], name: str = "folder_classifier") -> list[dict[str, Any]]:
    path = _model_path(f"{name}.pkl")
    if not path.exists():
        raise FileNotFoundError(f"Classifier {name} not trained")
    data = pickle.loads(path.read_bytes())
    clf: RandomForestClassifier = data["model"]

    init_db()
    results = []
    with db_conn() as db:
        for msg_id in message_ids:
            row = db.execute(
                "SELECT embedding FROM message_vec WHERE rowid = ?", (msg_id,)
            ).fetchone()
            if not row:
                continue
            vec = np.frombuffer(row["embedding"], dtype=np.float32).reshape(1, -1)
            pred = clf.predict(vec)[0]
            proba = clf.predict_proba(vec)[0]
            ranked = sorted(zip(clf.classes_, proba), key=lambda x: x[1], reverse=True)
            results.append(
                {
                    "message_id": msg_id,
                    "prediction": pred,
                    "probabilities": {label: float(p) for label, p in ranked},
                }
            )
    return results
