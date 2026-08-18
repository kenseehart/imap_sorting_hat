"""Qdrant ANN index for fish — one collection per retrieval model_id."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from fish.config import EMBED_DIM, load_env
from fish.prism.configs import collection_for_model_id


def _vector_for_qdrant(embedding: Any) -> list[float]:
    """Qdrant PointStruct requires a Python list; convert at this edge only."""
    import numpy as np

    if isinstance(embedding, np.ndarray):
        return embedding.astype(np.float32, copy=False).reshape(-1).tolist()
    return [float(x) for x in embedding]


def qdrant_url() -> str:
    load_env()
    return os.getenv("FISH_QDRANT_URL", "http://127.0.0.1:6333").strip()


def qdrant_api_key() -> str | None:
    load_env()
    key = os.getenv("FISH_QDRANT_API_KEY", "").strip()
    return key or None


def occurred_at_to_ts(occurred_at: str | None) -> int | None:
    if not occurred_at:
        return None
    raw = occurred_at.strip()
    if not raw:
        return None
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except ValueError:
        # date-only YYYY-MM-DD
        try:
            dt = datetime.fromisoformat(raw[:10]).replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except ValueError:
            return None


def qdrant_path() -> str | None:
    """Optional local embedded Qdrant storage (mutually exclusive with URL)."""
    load_env()
    path = os.getenv("FISH_QDRANT_PATH", "").strip()
    return path or None


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """Shared client. Fails fast if neither URL nor path works."""
    path = qdrant_path()
    url = qdrant_url()
    # Explicit timeout: default httpx timeout is too short under VM memory pressure
    # (we saw ResponseHandlingException: timed out on fish_legacy searches).
    load_env()
    timeout = float(os.getenv("FISH_QDRANT_TIMEOUT_SEC", "120").strip() or "60")
    if timeout <= 0:
        raise RuntimeError(
            f"FISH_QDRANT_TIMEOUT_SEC must be positive, got {timeout!r}"
        )
    if path:
        client = QdrantClient(path=path)
    elif url in (":memory:", "memory"):
        client = QdrantClient(":memory:")
    else:
        if not url:
            raise RuntimeError(
                "Set FISH_QDRANT_URL (e.g. http://127.0.0.1:6333) "
                "or FISH_QDRANT_PATH for embedded storage"
            )
        client = QdrantClient(
            url=url,
            api_key=qdrant_api_key(),
            prefer_grpc=False,
            check_compatibility=False,
            timeout=timeout,
        )
    # Probe connectivity
    client.get_collections()
    return client


def reset_qdrant_client() -> None:
    get_qdrant_client.cache_clear()


def ensure_collection(collection: str, *, dim: int = EMBED_DIM) -> None:
    client = get_qdrant_client()
    names = {c.name for c in client.get_collections().collections}
    if collection in names:
        return
    client.create_collection(
        collection_name=collection,
        vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
    )
    # Payload indexes for filtered ANN
    for field, schema in (
        ("kind", qm.PayloadSchemaType.KEYWORD),
        ("account_email", qm.PayloadSchemaType.KEYWORD),
        ("folder", qm.PayloadSchemaType.KEYWORD),
        ("from_addr", qm.PayloadSchemaType.KEYWORD),
        ("from_addr_lower", qm.PayloadSchemaType.TEXT),
        ("source", qm.PayloadSchemaType.KEYWORD),
        ("occurred_at_ts", qm.PayloadSchemaType.INTEGER),
        ("occurred_at", qm.PayloadSchemaType.KEYWORD),
        ("unread", qm.PayloadSchemaType.BOOL),
    ):
        try:
            client.create_payload_index(
                collection_name=collection,
                field_name=field,
                field_schema=schema,
            )
        except Exception:
            # Index may already exist on recreate races
            pass


def build_payload(row: dict[str, Any]) -> dict[str, Any]:
    """Build Qdrant payload from a corpus_items-like dict (+ optional email fields)."""
    import json

    payload_obj = row.get("payload") or {}
    if isinstance(payload_obj, str):
        try:
            payload_obj = json.loads(payload_obj)
        except json.JSONDecodeError:
            payload_obj = {}
    if not isinstance(payload_obj, dict):
        payload_obj = {}

    from_addr = (
        row.get("from_addr")
        or payload_obj.get("from_addr")
        or ""
    )
    flags = payload_obj.get("flags") or row.get("flags") or []
    if isinstance(flags, str):
        try:
            flags = json.loads(flags)
        except json.JSONDecodeError:
            flags = []
    if not isinstance(flags, list):
        flags = []
    kind = row.get("kind") or ""
    # Non-email rows are never "unread" so unread_only filters exclude them
    unread = kind == "email" and "\\Seen" not in flags
    occurred = row.get("occurred_at")
    out: dict[str, Any] = {
        "kind": kind,
        "source": row.get("source") or "",
        "occurred_at": occurred or "",
        "account_email": payload_obj.get("account_email") or row.get("account_email") or "",
        "folder": payload_obj.get("folder") or row.get("folder") or "",
        "from_addr": from_addr,
        "from_addr_lower": from_addr.lower(),
        "unread": unread,
    }
    ts = occurred_at_to_ts(occurred if isinstance(occurred, str) else None)
    if ts is not None:
        out["occurred_at_ts"] = ts
    return out


def upsert_point(
    collection: str,
    item_id: int,
    embedding: list[float] | Any,
    payload: dict[str, Any],
) -> None:
    ensure_collection(collection)
    get_qdrant_client().upsert(
        collection_name=collection,
        points=[
            qm.PointStruct(
                id=int(item_id),
                vector=_vector_for_qdrant(embedding),
                payload=payload,
            )
        ],
        wait=True,
    )


def upsert_points_batch(
    collection: str,
    points: list[tuple[int, Any, dict[str, Any]]],
) -> None:
    if not points:
        return
    ensure_collection(collection)
    get_qdrant_client().upsert(
        collection_name=collection,
        points=[
            qm.PointStruct(
                id=int(item_id),
                vector=_vector_for_qdrant(emb),
                payload=payload,
            )
            for item_id, emb, payload in points
        ],
        wait=True,
    )


def delete_point(collection: str, item_id: int) -> None:
    client = get_qdrant_client()
    names = {c.name for c in client.get_collections().collections}
    if collection not in names:
        return
    client.delete(
        collection_name=collection,
        points_selector=qm.PointIdsList(points=[int(item_id)]),
        wait=True,
    )


def delete_collection(collection: str) -> None:
    client = get_qdrant_client()
    names = {c.name for c in client.get_collections().collections}
    if collection in names:
        client.delete_collection(collection_name=collection)


def get_point_vector(collection: str, item_id: int) -> list[float] | None:
    client = get_qdrant_client()
    names = {c.name for c in client.get_collections().collections}
    if collection not in names:
        return None
    points = client.retrieve(
        collection_name=collection,
        ids=[int(item_id)],
        with_vectors=True,
        with_payload=False,
    )
    if not points:
        return None
    vec = points[0].vector
    if isinstance(vec, dict):
        # named vectors — we use unnamed
        vec = next(iter(vec.values()), None)
    if vec is None:
        return None
    return [float(x) for x in vec]


def point_exists(collection: str, item_id: int) -> bool:
    return int(item_id) in existing_point_ids(collection, [int(item_id)])


def existing_point_ids(collection: str, ids: list[int]) -> set[int]:
    """Return the subset of ``ids`` that already exist in the collection."""
    if not ids:
        return set()
    client = get_qdrant_client()
    names = {c.name for c in client.get_collections().collections}
    if collection not in names:
        return set()
    found: set[int] = set()
    # retrieve accepts large id lists; chunk to keep requests bounded
    chunk = 256
    for i in range(0, len(ids), chunk):
        batch = [int(x) for x in ids[i : i + chunk]]
        points = client.retrieve(
            collection_name=collection,
            ids=batch,
            with_vectors=False,
            with_payload=False,
        )
        for p in points:
            found.add(int(p.id))
    return found


def all_point_ids(collection: str) -> set[int]:
    """Scroll every point id in the collection (vectors/payload omitted)."""
    return set(scroll_ids(collection, limit=None))


def _build_filter(
    *,
    kinds: list[str] | None = None,
    since: str | None = None,
    until: str | None = None,
    from_contains: str | None = None,
    account_email: str | None = None,
    folder: str | None = None,
    unread_only: bool = False,
) -> qm.Filter | None:
    must: list[qm.Condition] = []
    if kinds:
        if len(kinds) == 1:
            must.append(
                qm.FieldCondition(key="kind", match=qm.MatchValue(value=kinds[0]))
            )
        else:
            must.append(
                qm.FieldCondition(key="kind", match=qm.MatchAny(any=list(kinds)))
            )
    if account_email:
        must.append(
            qm.FieldCondition(
                key="account_email", match=qm.MatchValue(value=account_email)
            )
        )
    if folder:
        must.append(
            qm.FieldCondition(key="folder", match=qm.MatchValue(value=folder))
        )
    if from_contains:
        needle = from_contains.strip().lower()
        if needle:
            # Text match on lowercased from_addr (substring / token)
            must.append(
                qm.FieldCondition(
                    key="from_addr_lower", match=qm.MatchText(text=needle)
                )
            )
    if unread_only:
        must.append(
            qm.FieldCondition(key="unread", match=qm.MatchValue(value=True))
        )
    range_args: dict[str, int] = {}
    since_ts = occurred_at_to_ts(since)
    until_ts = occurred_at_to_ts(until)
    if since_ts is not None:
        range_args["gte"] = since_ts
    if until_ts is not None:
        range_args["lte"] = until_ts
    if range_args:
        must.append(
            qm.FieldCondition(key="occurred_at_ts", range=qm.Range(**range_args))
        )
    if not must:
        return None
    return qm.Filter(must=must)


def search(
    collection: str,
    query_embedding: list[float] | Any,
    *,
    limit: int = 20,
    kinds: list[str] | None = None,
    since: str | None = None,
    until: str | None = None,
    from_contains: str | None = None,
    account_email: str | None = None,
    folder: str | None = None,
    unread_only: bool = False,
) -> list[tuple[int, float]]:
    """Return (item_id, distance) with distance = 1 - cosine_similarity."""
    ensure_collection(collection)
    qfilter = _build_filter(
        kinds=kinds,
        since=since,
        until=until,
        from_contains=from_contains,
        account_email=account_email,
        folder=folder,
        unread_only=unread_only,
    )
    hits = get_qdrant_client().query_points(
        collection_name=collection,
        query=_vector_for_qdrant(query_embedding),
        query_filter=qfilter,
        limit=int(limit),
        with_payload=False,
    )
    out: list[tuple[int, float]] = []
    for h in hits.points:
        # Qdrant cosine score is similarity; convert to distance-like for callers.
        score = float(h.score)
        dist = 1.0 - score
        out.append((int(h.id), dist))
    return out


def collection_point_count(collection: str) -> int:
    client = get_qdrant_client()
    names = {c.name for c in client.get_collections().collections}
    if collection not in names:
        return 0
    info = client.get_collection(collection)
    return int(info.points_count or 0)


def scroll_ids(collection: str, *, limit: int | None = None) -> list[int]:
    """Scroll point ids. ``limit=None`` means the entire collection."""
    client = get_qdrant_client()
    names = {c.name for c in client.get_collections().collections}
    if collection not in names:
        return []
    ids: list[int] = []
    offset = None
    page = 256
    while True:
        if limit is not None:
            remaining = limit - len(ids)
            if remaining <= 0:
                break
            page_limit = min(page, remaining)
        else:
            page_limit = page
        points, offset = client.scroll(
            collection_name=collection,
            limit=page_limit,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        for p in points:
            ids.append(int(p.id))
        if offset is None:
            break
        if limit is not None and len(ids) >= limit:
            break
    return ids
