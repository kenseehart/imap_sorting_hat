"""Hybrid search: keyword index (FTS5 + triggers), RRF fusion, rerank ordering.

Runs on a temporary fish.db; Qdrant, OpenAI embeddings and the relevance
LLM are replaced at the module boundaries search.py calls through.
"""

from __future__ import annotations

import json

import pytest


@pytest.fixture()
def fishdb(tmp_path, monkeypatch):
    monkeypatch.setenv("FISH_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("FISH_DB_PATH", str(tmp_path / "fish.db"))
    import fish.prism.registry as registry
    import fish.qdrant_store as qs

    # No Qdrant in tests: init_db only needs the collections to "exist".
    monkeypatch.setattr(registry, "ensure_collection", lambda name: None)
    monkeypatch.setattr(qs, "ensure_collection", lambda name: None)
    from fish.store import db_conn, init_db

    init_db()
    return db_conn


def _add(db, sk, text, *, subject=None, sender=None, body=None, kind="email"):
    payload = {"subject": subject, "from_addr": sender, "to_addrs": ["ken@example.com"]}
    cur = db.execute(
        "INSERT INTO corpus_items (kind, source, source_key, text_for_embed, body_text,"
        " occurred_at, payload) VALUES (?, 'test', ?, ?, ?, '2026-11-01', ?)",
        (kind, sk, text, body, json.dumps(payload)))
    return int(cur.lastrowid)


def test_keyword_index_follows_corpus_and_refuses_until_built(fishdb):
    from fish import keyword_index as ki

    with fishdb() as db:
        a = _add(db, "a", "From: Ana. Subject: Invoice 4471.", subject="Invoice 4471",
                 sender="ana@shop.example", body="Your order shipped; invoice 4471 attached.")
        b = _add(db, "b", "From: Bo. Subject: lunch", subject="lunch", sender="bo@example.com",
                 body="tacos on friday?")
        with pytest.raises(RuntimeError, match="fts-reindex"):
            ki.search(db, "invoice", limit=10)
        # Rows from before the index existed: simulate by emptying it.
        db.execute("DELETE FROM corpus_fts")
        out = ki.backfill(db, progress=False)
        assert out == {**out, "corpus_items": 2, "indexed": 2, "added": 2}
        assert ki.search(db, "invoice 4471", limit=10) == [a]
        # People column; "example" also matches bo@example.com, but BM25 ranks Ana first.
        assert ki.search(db, "ana@shop.example", limit=10) == [a, b]
        assert ki.search(db, "shipping", limit=10) == [a]          # porter stem of "shipped"
        db.execute("UPDATE corpus_items SET body_text = 'burrito invoice' WHERE id = ?", (b,))
        assert set(ki.search(db, "burrito", limit=10)) == {b}
        db.execute("DELETE FROM corpus_items WHERE id = ?", (a,))
        assert ki.search(db, "4471", limit=10) == []
        assert ki.backfill(db, progress=False)["indexed"] == 1


def test_match_expression_quotes_operators():
    from fish.keyword_index import match_expression

    assert match_expression('NEAR("a" b) OR -c*') == '"near" OR "a" OR "b" OR "or" OR "c"'
    assert match_expression("?!") is None


def test_rrf_fuse_rewards_agreement():
    from fish.search import rrf_fuse

    fused = dict(rrf_fuse([[1, 2, 3], [3, 4]], k=60))
    assert max(fused, key=fused.get) == 3          # in both lists
    assert fused[1] > fused[2] == fused[4]   # ranks 2 and 2 tie


def _patch_arms(monkeypatch, vector_ids, passing=None):
    import fish.qdrant_store as qs
    import fish.prism.queries as queries
    import fish.search as search

    monkeypatch.setattr(search, "embed_text", lambda text: [0.0] * 4)
    monkeypatch.setattr(queries, "log_real_query", lambda *a, **k: None)
    monkeypatch.setattr(search, "corpus_vector_search",
                        lambda db, emb, limit, model_id, **f: [(i, 0.1) for i in vector_ids])
    monkeypatch.setattr(qs, "filter_ids",
                        lambda coll, ids, **f: set(ids) if passing is None else passing)


def test_search_fuses_keyword_only_hits_and_reranks(fishdb, monkeypatch):
    from fish import keyword_index as ki
    from fish.search import search_corpus

    with fishdb() as db:
        a = _add(db, "a", "Subject: Invoice 4471", subject="Invoice 4471", body="invoice 4471")
        b = _add(db, "b", "Subject: payments overview", subject="payments", body="billing notes")
        c = _add(db, "c", "Subject: garden", subject="garden", body="tomatoes")
        ki.backfill(db, progress=False)
    # The vector arm misses the exact invoice number; keyword finds it.
    _patch_arms(monkeypatch, [b, c])
    out = search_corpus("invoice 4471", limit=3, rerank=False)
    ids = [r["id"] for r in out["results"]]
    assert set(ids) == {a, b, c}
    hit = next(r for r in out["results"] if r["id"] == a)
    assert hit["keyword_rank"] == 1 and hit["vector_rank"] is None and hit["relevance"] is None

    rel = {"invoice 4471": 0.9, "payments overview": 0.4, "garden": 0.0}
    scorer = lambda q, doc, context_json=None: next(v for k, v in rel.items() if k in doc.lower())
    out = search_corpus("invoice 4471", limit=3, rerank=True, _scorer=scorer)
    assert [r["id"] for r in out["results"]] == [a, b, c]
    assert out["results"][0]["score"] == 0.9 and out["ranking"]["rerank"] is True

    # Metadata filters bind the keyword arm too (through qdrant_store.filter_ids).
    _patch_arms(monkeypatch, [b], passing={b})
    out = search_corpus("invoice 4471", limit=3, rerank=False)
    assert [r["id"] for r in out["results"]] == [b]

    with pytest.raises(TypeError, match="vector_weight"):
        search_corpus("x", vector_weight=0.7)


def test_rerank_settings_fail_loudly(monkeypatch):
    from fish import rerank

    monkeypatch.setenv("FISH_RERANK", "maybe")
    with pytest.raises(ValueError, match="FISH_RERANK"):
        rerank.rerank_enabled()
    monkeypatch.setenv("FISH_RERANK_K", "0")
    with pytest.raises(ValueError, match="FISH_RERANK_K"):
        rerank.rerank_k()


def test_filter_ids_applies_search_filter_in_qdrant(monkeypatch):
    """filter_ids against a real (in-memory) Qdrant: same filter as search."""
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qm

    import fish.qdrant_store as qs

    client = QdrantClient(":memory:")
    monkeypatch.setattr(qs, "get_qdrant_client", lambda: client)
    qs.ensure_collection("t", dim=4)
    pts = [
        (1, {"kind": "email", "account_email": "a@x", "occurred_at_ts": 100, "unread": True}),
        (2, {"kind": "email", "account_email": "b@x", "occurred_at_ts": 200, "unread": False}),
        (3, {"kind": "sms", "occurred_at_ts": 300, "unread": False}),
    ]
    client.upsert("t", [qm.PointStruct(id=i, vector=[1.0, 0, 0, 0], payload=p) for i, p in pts])
    assert qs.filter_ids("t", [1, 2, 3]) == {1, 2, 3}
    assert qs.filter_ids("t", [1, 2, 3], kinds=["email"]) == {1, 2}
    assert qs.filter_ids("t", [1, 2, 3], account_email="b@x") == {2}
    assert qs.filter_ids("t", [1, 2], unread_only=True) == {1}
    assert qs.filter_ids("t", [2, 3], kinds=["email"]) == {2}   # only ids asked for
    assert qs.filter_ids("t", [], kinds=["email"]) == set()
