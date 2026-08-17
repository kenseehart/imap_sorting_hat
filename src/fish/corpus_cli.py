"""Training corpus CLI — fish corpus collect|label|stats|queries|add-gold|purge."""

from __future__ import annotations

import sys

from cmdline import cmd_group, emit_output, optarg

from fish.config import ensure_openai_api_key, load_env
from fish.prism.collect import collect_samples
from fish.prism.relevance import label_batch
from fish.store import (
    db_conn,
    dedupe_training_sample_pairs,
    init_db,
    mark_stale_samples,
    purge_training_samples,
    training_corpus_stats,
)
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
        False,
        long_flag="--force",
        action="store_true",
        help="Re-label even when target_relevance is already set",
    ),
    concurrency: int | None = optarg(
        None,
        long_flag="--concurrency",
        help=(
            "Parallel OpenAI RelevanceAgent calls "
            "(default: FISH_LABEL_CONCURRENCY or 16). "
            "Does not hold the Fish write lock across API waits."
        ),
    ),
    *,
    json_output: bool = False,
    md_output: bool = False,
) -> int:
    """Run RelevanceAgent on unlabeled training samples (null target_relevance only)."""
    load_env()
    try:
        ensure_openai_api_key(interactive=False)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1
    try:
        # Short SQLite writes only — do not hold fish_write_lock across OpenAI
        # waits so prism-train / sync can run concurrently.
        result = label_batch(limit=limit, force=force, concurrency=concurrency)
    except Exception as exc:
        print(exc, file=sys.stderr)
        return 1
    emit_output(result, json_output=json_output, md=md_output, title="Fish corpus label")
    return 0


@corpus.cmd(output=True)
def inject_positives(
    query: str = optarg(..., long_flag="--query", help="Training query text (created if missing)"),
    like: str = optarg(
        ...,
        long_flag="--like",
        help="Comma-separated SQL LIKE patterns against text/subject",
    ),
    since: str | None = optarg(
        None, long_flag="--since", help="Only corpus items with occurred_at >= ISO date"
    ),
    kinds: str = optarg("email", long_flag="--kinds", help="Comma-separated kinds"),
    limit: int = optarg(200, long_flag="--limit", help="Max matching corpus items"),
    retriever: str = optarg(
        "legacy", long_flag="--retriever", help="Retriever tag on inserted samples"
    ),
    *,
    json_output: bool = False,
    md_output: bool = False,
) -> int:
    """Force (query, corpus) training pairs for cold-start (e.g. Burning Man → camp mail)."""
    from fish.prism.inject import inject_positives as run_inject

    load_env()
    try:
        ensure_openai_api_key(interactive=False)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1
    likes = [p.strip() for p in like.split(",") if p.strip()]
    if not likes:
        print("At least one --like pattern is required", file=sys.stderr)
        return 1
    kind_list = [k.strip() for k in kinds.split(",") if k.strip()]
    try:
        with fish_write_lock("corpus"):
            result = run_inject(
                query=query,
                likes=likes,
                since=since,
                kinds=kind_list,
                limit=limit,
                retriever=retriever,
            )
    except Exception as exc:
        print(exc, file=sys.stderr)
        return 1
    emit_output(result, json_output=json_output, md=md_output, title="Fish corpus inject-positives")
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
def queries(
    origin: str | None = optarg(
        None,
        long_flag="--origin",
        help="Filter: gold | curated | synth (default: all)",
    ),
    source: str | None = optarg(
        None, long_flag="--source", help="Filter by source string"
    ),
    limit: int | None = optarg(None, long_flag="--limit", help="Max rows"),
    embeddings: bool = optarg(
        False,
        long_flag="--embeddings",
        action="store_true",
        help="Include query_embedding vectors (large)",
    ),
    *,
    json_output: bool = False,
    md_output: bool = False,
) -> int:
    """Dump training queries (gold/curated/synth) with source, created_at, metadata."""
    from fish.prism.gold import dump_queries

    init_db()
    rows = dump_queries(
        origin=origin,
        source=source,
        limit=limit,
        include_embeddings=embeddings,
    )
    # Compact table fields for non-JSON
    if not json_output:
        rows = [
            {
                "id": r["id"],
                "origin": r["origin"],
                "source": r.get("source"),
                "created_at": r.get("created_at"),
                "text": r.get("text"),
                "meta": r.get("meta") or r.get("meta_json"),
                "has_embedding": bool(r.get("embed_model")),
            }
            for r in rows
        ]
    emit_output(
        rows,
        json_output=json_output,
        md=md_output,
        title=f"Fish training queries ({len(rows)})",
    )
    return 0


@corpus.cmd(output=True)
def add_curated(
    file: str | None = optarg(
        None,
        long_flag="--file",
        help="JSONL path (default: fish/config/gold_queries.jsonl)",
    ),
    no_embed: bool = optarg(
        False,
        long_flag="--no-embed",
        action="store_true",
        help="Skip OpenAI embeddings (collect will embed later)",
    ),
    source: str | None = optarg(
        None,
        long_flag="--source",
        help="Default source if a line omits source",
    ),
    replace: bool = optarg(
        False,
        long_flag="--replace",
        action="store_true",
        help="Permanently delete all curated queries (and their samples), then load file",
    ),
    *,
    json_output: bool = False,
    md_output: bool = False,
) -> int:
    """Load curated queries (origin=curated) from JSONL into training_queries."""
    from pathlib import Path

    from fish.prism.gold import (
        DEFAULT_SOURCE,
        add_gold_queries,
        load_gold_file,
        replace_gold_queries,
    )

    load_env()
    if not no_embed:
        try:
            ensure_openai_api_key(interactive=False)
        except RuntimeError as exc:
            print(exc, file=sys.stderr)
            return 1
    try:
        entries = load_gold_file(Path(file) if file else None)
        with fish_write_lock("corpus"):
            if replace:
                result = replace_gold_queries(
                    entries,
                    embed=not no_embed,
                    default_source=source or DEFAULT_SOURCE,
                )
            else:
                result = add_gold_queries(
                    entries,
                    embed=not no_embed,
                    default_source=source or DEFAULT_SOURCE,
                )
                result["file_count"] = len(entries)
    except Exception as exc:
        print(exc, file=sys.stderr)
        return 1
    emit_output(
        result, json_output=json_output, md=md_output, title="Fish corpus add-curated"
    )
    return 0


@corpus.cmd(output=True)
def add_gold(
    file: str | None = optarg(
        None,
        long_flag="--file",
        help="JSONL path (default: fish/config/gold_queries.jsonl)",
    ),
    no_embed: bool = optarg(
        False,
        long_flag="--no-embed",
        action="store_true",
        help="Skip OpenAI embeddings (collect will embed later)",
    ),
    source: str | None = optarg(
        None,
        long_flag="--source",
        help="Default source if a line omits source",
    ),
    replace: bool = optarg(
        False,
        long_flag="--replace",
        action="store_true",
        help="Permanently delete all curated queries (and their samples), then load file",
    ),
    *,
    json_output: bool = False,
    md_output: bool = False,
) -> int:
    """Deprecated alias for add-curated (origin=curated)."""
    return add_curated(
        file=file,
        no_embed=no_embed,
        source=source,
        replace=replace,
        json_output=json_output,
        md_output=md_output,
    )


@corpus.cmd(output=True)
def dedupe_pairs(
    dry_run: bool = optarg(
        False,
        long_flag="--dry-run",
        action="store_true",
        help="Report actions without writing",
    ),
    *,
    json_output: bool = False,
    md_output: bool = False,
) -> int:
    """Collapse duplicate (query, item) samples; pair identity ignores retriever.

    Keeps the best labeled row (prefer RA 2.0.0, else any label, else newest).
    Backfills null relevance_agent_version to 1.0.0 on labeled rows. Losers are
    superseded (not deleted).
    """
    init_db()
    with fish_write_lock("corpus"):
        with db_conn() as db:
            result = dedupe_training_sample_pairs(db, dry_run=dry_run)
    emit_output(
        result, json_output=json_output, md=md_output, title="Fish corpus dedupe-pairs"
    )
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
