"""Training corpus CLI — fish corpus collect|label|stats|purge."""

from __future__ import annotations

import sys

from cmdline import cmd_group, emit_output, optarg

from fish.config import ensure_openai_api_key, load_env
from fish.prism.collect import collect_samples
from fish.prism.relevance import label_batch
from fish.store import db_conn, init_db, mark_stale_samples, purge_training_samples, training_corpus_stats
from fish.write_lock import fish_write_lock

corpus = cmd_group("corpus", help="PRISM training corpus (queries, samples, labeling)")


@corpus.cmd(output=True)
def collect(
    retriever: str = optarg(
        ...,
        long_flag="--retriever",
        help="Retriever for this run: legacy or model stem (e.g. personal)",
    ),
    min_queries: int = optarg(
        50, long_flag="--min-queries", help="Minimum queries before collecting samples"
    ),
    synthesis_batch: int = optarg(
        5, long_flag="--synthesis-batch", help="Synthetic queries per synthesis round"
    ),
    top_k: int = optarg(20, long_flag="--top-k", help="Top-k hits per query"),
    label: bool = optarg(
        False, long_flag="--label", action="store_true", help="Label new samples after collect"
    ),
    label_limit: int = optarg(
        500, long_flag="--label-limit", help="Max samples to label when --label"
    ),
    *,
    json_output: bool = False,
    md_output: bool = False,
) -> int:
    """Extend query set if needed, top-k retrieve, and insert training samples."""
    load_env()
    try:
        ensure_openai_api_key(interactive=False)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1
    try:
        with fish_write_lock("corpus"):
            result = collect_samples(
                retriever=retriever,
                min_queries=min_queries,
                synthesis_batch=synthesis_batch,
                top_k=top_k,
                label=label,
                label_limit=label_limit,
            )
    except Exception as exc:
        print(exc, file=sys.stderr)
        return 1
    emit_output(result, json_output=json_output, md=md_output, title="Fish corpus collect")
    return 0


@corpus.cmd(output=True)
def label(
    limit: int = optarg(500, long_flag="--limit", help="Max samples to label"),
    force: bool = optarg(
        False, long_flag="--force", action="store_true", help="Re-label even if agent version matches"
    ),
    *,
    json_output: bool = False,
    md_output: bool = False,
) -> int:
    """Run RelevanceAgent on unlabeled training samples."""
    load_env()
    try:
        ensure_openai_api_key(interactive=False)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1
    try:
        with fish_write_lock("corpus"):
            result = label_batch(limit=limit, force=force)
    except Exception as exc:
        print(exc, file=sys.stderr)
        return 1
    emit_output(result, json_output=json_output, md=md_output, title="Fish corpus label")
    return 0


@corpus.cmd(output=True)
def stats(
    *,
    json_output: bool = False,
    md_output: bool = False,
) -> int:
    """Show training query and sample counts."""
    init_db()
    with db_conn() as db:
        report = training_corpus_stats(db)
    emit_output(report, json_output=json_output, md=md_output, title="Fish corpus stats")
    return 0


@corpus.cmd(output=True)
def purge(
    stale: bool = optarg(
        False, long_flag="--stale", action="store_true", help="Delete samples with content_hash mismatch"
    ),
    superseded: bool = optarg(
        False,
        long_flag="--superseded",
        action="store_true",
        help="Delete superseded samples only",
    ),
    kind: str | None = optarg(None, long_flag="--kind", help="Filter by corpus kind"),
    before: str | None = optarg(
        None, long_flag="--before", help="Delete samples with occurred_at before ISO date"
    ),
    retriever: str | None = optarg(None, long_flag="--retriever", help="Filter by retriever"),
    mark_stale: bool = optarg(
        False,
        long_flag="--mark-stale",
        action="store_true",
        help="Mark stale samples superseded instead of deleting",
    ),
    *,
    json_output: bool = False,
    md_output: bool = False,
) -> int:
    """Remove or mark stale/superseded training samples."""
    init_db()
    with fish_write_lock("corpus"):
        with db_conn() as db:
            if mark_stale:
                count = mark_stale_samples(db)
                result = {"marked_stale": count}
            else:
                count = purge_training_samples(
                    db,
                    stale=stale,
                    kind=kind,
                    before=before,
                    retriever=retriever,
                    superseded_only=superseded,
                )
                result = {"deleted": count}
    emit_output(result, json_output=json_output, md=md_output, title="Fish corpus purge")
    return 0


@corpus.cmd
def browse(
    port: int = optarg(8001, long_flag="--port", help="HTTP port"),
    host: str = optarg(
        "127.0.0.1",
        long_flag="--host",
        help="Bind address (127.0.0.1 = local only)",
    ),
) -> int:
    """Browse fish.db in a local web UI (same as: dbserv fish)."""
    from util.dbserv import serve

    try:
        return serve("fish", host=host, port=port)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1
