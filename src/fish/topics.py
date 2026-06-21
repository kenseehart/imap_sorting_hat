from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

import numpy as np
from sklearn.cluster import KMeans

from fish.config import openai_api_key
from fish.store import db_conn, init_db
from openai import OpenAI


def _fetch_recent_embeddings(limit: int = 200) -> list[tuple[int, list[float], str]]:
    init_db()
    with db_conn() as db:
        rows = db.execute(
            """
            SELECT c.id, json_extract(c.payload, '$.subject') AS subject, v.embedding
            FROM corpus_items c
            JOIN corpus_vec v ON v.rowid = c.id
            ORDER BY c.occurred_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    result = []
    for row in rows:
        blob = row["embedding"]
        vec = np.frombuffer(blob, dtype=np.float32).tolist()
        result.append((int(row["id"]), vec, row["subject"] or ""))
    return result


def _label_cluster(subjects: list[str]) -> str:
    words: list[str] = []
    for subject in subjects:
        words.extend(re.findall(r"[A-Za-z]{4,}", subject.lower()))
    if not words:
        return "misc"
    common = Counter(words).most_common(3)
    return " / ".join(w for w, _ in common)


def extract_topics(k: int = 8, limit: int = 200) -> list[dict[str, Any]]:
    data = _fetch_recent_embeddings(limit=limit)
    if len(data) < k:
        k = max(1, len(data))
    if not data:
        return []

    ids = [d[0] for d in data]
    vectors = np.array([d[1] for d in data])
    subjects = [d[2] for d in data]

    model = KMeans(n_clusters=k, random_state=0, n_init=10)
    labels = model.fit_predict(vectors)

    now = datetime.now(timezone.utc).isoformat()
    topics_out: list[dict[str, Any]] = []

    with db_conn() as db:
        db.execute("DELETE FROM message_topics")
        db.execute("DELETE FROM topics")

        for cluster_id in range(k):
            member_idx = [i for i, label in enumerate(labels) if label == cluster_id]
            if not member_idx:
                continue
            cluster_subjects = [subjects[i] for i in member_idx]
            label = _label_cluster(cluster_subjects)
            cur = db.execute(
                "INSERT INTO topics (label, created_at) VALUES (?, ?)",
                (label, now),
            )
            topic_id = int(cur.lastrowid)
            for i in member_idx:
                db.execute(
                    "INSERT INTO message_topics (message_id, topic_id, score) VALUES (?, ?, ?)",
                    (ids[i], topic_id, 1.0),
                )
            topics_out.append(
                {
                    "id": topic_id,
                    "label": label,
                    "message_count": len(member_idx),
                }
            )

    return topics_out


def list_topics() -> list[dict[str, Any]]:
    init_db()
    with db_conn() as db:
        rows = db.execute(
            """
            SELECT t.id, t.label, COUNT(mt.message_id) AS message_count
            FROM topics t
            LEFT JOIN message_topics mt ON mt.topic_id = t.id
            GROUP BY t.id
            ORDER BY message_count DESC
            """
        ).fetchall()
        return [dict(r) for r in rows]


def topic_messages(topic_id: int, limit: int = 50) -> list[dict[str, Any]]:
    init_db()
    with db_conn() as db:
        rows = db.execute(
            """
            SELECT m.*, a.email AS account_email
            FROM message_topics mt
            JOIN messages m ON m.id = mt.message_id
            JOIN accounts a ON a.id = m.account_id
            WHERE mt.topic_id = ?
            ORDER BY m.date DESC
            LIMIT ?
            """,
            (topic_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def topic_graph() -> dict[str, Any]:
    init_db()
    with db_conn() as db:
        topic_rows = db.execute("SELECT id, label FROM topics").fetchall()
        nodes = [{"id": f"topic:{r['id']}", "type": "topic", "label": r["label"]} for r in topic_rows]

        people: set[str] = set()
        topic_people: dict[int, set[str]] = defaultdict(set)
        rows = db.execute(
            """
            SELECT mt.topic_id, m.from_addr
            FROM message_topics mt
            JOIN messages m ON m.id = mt.message_id
            """
        ).fetchall()
        for row in rows:
            sender = row["from_addr"]
            if sender:
                people.add(sender)
                topic_people[int(row["topic_id"])].add(sender)

        for person in people:
            nodes.append({"id": f"person:{person}", "type": "person", "label": person})

        edges = []
        for topic_id, senders in topic_people.items():
            for sender in senders:
                edges.append(
                    {
                        "source": f"topic:{topic_id}",
                        "target": f"person:{sender}",
                        "weight": 1,
                    }
                )

        co_rows = db.execute(
            """
            SELECT m1.from_addr AS a, m2.from_addr AS b, COUNT(*) AS weight
            FROM messages m1
            JOIN messages m2 ON m1.in_reply_to = m2.message_id
            WHERE m1.from_addr IS NOT NULL AND m2.from_addr IS NOT NULL
              AND m1.from_addr != m2.from_addr
            GROUP BY m1.from_addr, m2.from_addr
            HAVING weight >= 1
            LIMIT 100
            """
        ).fetchall()
        for row in co_rows:
            edges.append(
                {
                    "source": f"person:{row['a']}",
                    "target": f"person:{row['b']}",
                    "weight": int(row["weight"]),
                }
            )

        return {"nodes": nodes, "edges": edges}


def label_topics_with_llm() -> list[dict[str, Any]]:
    topics = list_topics()
    if not topics:
        extract_topics()
        topics = list_topics()
    client = OpenAI(api_key=openai_api_key())
    updated = []
    for topic in topics:
        messages = topic_messages(int(topic["id"]), limit=5)
        subjects = [m.get("subject", "") for m in messages]
        prompt = "Label this email topic cluster in 3-6 words:\n" + "\n".join(subjects)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
        )
        label = (response.choices[0].message.content or topic["label"]).strip()
        with db_conn() as db:
            db.execute("UPDATE topics SET label = ? WHERE id = ?", (label, topic["id"]))
        updated.append({"id": topic["id"], "label": label})
    return updated
