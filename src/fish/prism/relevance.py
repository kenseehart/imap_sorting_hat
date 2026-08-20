from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from openai import OpenAI

from fish.config import openai_api_key
from fish.prism.queries import query_text_for_search
from fish.store import (
    db_conn,
    get_corpus_by_id,
    get_training_query,
    init_db,
    list_unlabeled_samples,
    update_sample_relevance_with_retry,
)

RELEVANCE_AGENT_VERSION = "2.0.0"
DEFAULT_RELEVANCE_MODEL = "gpt-4o-mini"
DEFAULT_LABEL_CONCURRENCY = 16

# Serialize SQLite writes from worker threads (connections are not shared).
_db_write_lock = threading.Lock()


def relevance_model() -> str:
    import os

    from fish.config import load_env

    load_env()
    return os.getenv("FISH_RELEVANCE_MODEL", DEFAULT_RELEVANCE_MODEL)


def default_label_concurrency() -> int:
    import os

    from fish.config import load_env

    load_env()
    raw = os.getenv("FISH_LABEL_CONCURRENCY", str(DEFAULT_LABEL_CONCURRENCY))
    try:
        n = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"FISH_LABEL_CONCURRENCY must be an int, got {raw!r}"
        ) from exc
    if n < 1:
        raise ValueError(f"FISH_LABEL_CONCURRENCY must be >= 1, got {n}")
    return n


def score(
    query_text: str,
    document_text: str,
    *,
    context_json: str | None = None,
    client: OpenAI | None = None,
) -> float:
    augmented = query_text_for_search(query_text, context_json)
    doc = (document_text or "")[:1500]
    api = client or OpenAI(api_key=openai_api_key())
    prompt = (
        "The query is an explicit information need: a request for information "
        "that retrieval should help answer.\n\n"
        "Score how useful this document is for answering that need, on a scale "
        "from 0.0 to 1.0. Score for utility, not topical or semantic similarity.\n"
        "Use the scale as follows:\n"
        "- 0.0 — entirely off-target; wrong subject matter for the need.\n"
        "- ~0.2 — related topic, entity, or keywords, but the document does not "
        "supply information that helps answer the need.\n"
        "- Mid range — partially useful (some relevant facts, incomplete or only "
        "adjacent to what was asked).\n"
        "- High (toward 1.0) — the document actually supplies the information "
        "asked for (concrete facts, specifics, lists, decisions, or other content "
        "that would help form an answer).\n"
        "Shared theme alone is never enough for a high score. A different entity, "
        "group, event, or context that fails the need should stay near ~0.2, not "
        "high.\n\n"
        f"Query: {augmented}\n\nDocument:\n{doc}\n\n"
        'Respond with JSON only: {"relevance": <number>}'
    )
    response = api.chat.completions.create(
        model=relevance_model(),
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_tokens=32,
    )
    raw = (response.choices[0].message.content or "").strip()
    data = json.loads(raw)
    value = float(data.get("relevance", 0.0))
    return max(0.0, min(1.0, value))


def _load_label_job(sample: dict[str, Any], *, force: bool) -> dict[str, Any] | None:
    """Resolve query/doc text for a sample. None = already labeled (skip)."""
    sample_id = int(sample["id"])
    with db_conn() as db:
        row = db.execute(
            "SELECT * FROM training_samples WHERE id = ?", (sample_id,)
        ).fetchone()
        if not row:
            raise ValueError(f"Sample {sample_id} not found")
        sample = dict(row)
        if not force and sample.get("target_relevance") is not None:
            return None
        query = get_training_query(db, int(sample["query_id"]))
        if not query:
            raise ValueError(f"Query {sample['query_id']} not found")
        corpus = get_corpus_by_id(db, int(sample["corpus_item_id"]))
        if not corpus:
            raise ValueError(f"Corpus item {sample['corpus_item_id']} not found")
        doc_text = corpus.get("text_for_embed") or corpus.get("body_text") or ""
        return {
            "sample_id": sample_id,
            "query_text": query["text"],
            "doc_text": doc_text,
            "context_json": query.get("context_json"),
        }


def _score_and_store(job: dict[str, Any], *, client: OpenAI) -> float:
    rel = score(
        job["query_text"],
        job["doc_text"],
        context_json=job.get("context_json"),
        client=client,
    )
    with _db_write_lock:
        update_sample_relevance_with_retry(
            int(job["sample_id"]),
            target_relevance=rel,
            agent_version=RELEVANCE_AGENT_VERSION,
            relevance_model=relevance_model(),
        )
    return rel


def label_sample(sample_id: int, *, force: bool = False) -> float | None:
    init_db()
    with db_conn() as db:
        row = db.execute(
            "SELECT * FROM training_samples WHERE id = ?", (sample_id,)
        ).fetchone()
        if not row:
            raise ValueError(f"Sample {sample_id} not found")
        sample = dict(row)
    job = _load_label_job(sample, force=force)
    if job is None:
        return float(sample["target_relevance"])
    return _score_and_store(job, client=OpenAI(api_key=openai_api_key()))


def label_batch(
    *,
    limit: int = 500,
    force: bool = False,
    concurrency: int | None = None,
) -> dict[str, Any]:
    """Label unlabeled samples. OpenAI calls run concurrently; DB writes are serialized.

    Does not hold the Fish write lock across API waits so train/sync can proceed.
    """
    from compute.tasks import TaskCancelled, TaskProgress

    init_db()
    workers = concurrency if concurrency is not None else default_label_concurrency()
    if workers < 1:
        raise ValueError(f"concurrency must be >= 1, got {workers}")

    with db_conn() as db:
        samples = list_unlabeled_samples(
            db,
            limit=limit,
            agent_version=RELEVANCE_AGENT_VERSION,
            force=force,
        )

    jobs: list[dict[str, Any]] = []
    errors: list[str] = []
    skipped = 0
    for sample in samples:
        try:
            job = _load_label_job(sample, force=force)
            if job is None:
                skipped += 1
                continue
            jobs.append(job)
        except Exception as exc:
            errors.append(f"sample {sample['id']}: {exc}")

    labeled = 0
    if not jobs:
        return {
            "labeled": labeled,
            "skipped": skipped,
            "errors": errors,
            "concurrency": workers,
        }

    # One client per worker thread (OpenAI client is not guaranteed thread-safe).
    thread_local = threading.local()

    def _client() -> OpenAI:
        c = getattr(thread_local, "client", None)
        if c is None:
            c = OpenAI(api_key=openai_api_key())
            thread_local.client = c
        return c

    def _work(job: dict[str, Any]) -> tuple[int, float]:
        rel = _score_and_store(job, client=_client())
        return int(job["sample_id"]), rel

    cancelled = False
    task_id: str | None = None
    try:
        with TaskProgress(
            module="fish",
            task="label",
            n=len(jobs),
            sec_per_unit_prior=1.5,
            detail=f"labeling {len(jobs)} samples",
            resource=os.environ.get("COMPUTE_RESOURCE"),
        ) as progress:
            task_id = progress.task_id
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_work, job): job for job in jobs}
                for fut in as_completed(futures):
                    job = futures[fut]
                    try:
                        fut.result()
                        labeled += 1
                    except Exception as exc:
                        errors.append(f"sample {job['sample_id']}: {exc}")
                    progress.update(
                        labeled,
                        detail=f"labeled {labeled}/{len(jobs)}",
                    )
    except TaskCancelled:
        cancelled = True

    result = {
        "labeled": labeled,
        "skipped": skipped,
        "errors": errors,
        "concurrency": workers,
    }
    if task_id:
        result["task_id"] = task_id
    if cancelled:
        result["cancelled"] = True
    return result
