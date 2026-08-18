# Agent onboarding — fish (personal PRISM corpus)

> **Status**: Active at `/home/ken/ws/fish`. GitHub: [kenseehart/imap_sorting_hat](https://github.com/kenseehart/imap_sorting_hat).

## Shared resources

Cross-project assets: **`/home/ken/ws/shared`**. Workspace index: **`/home/ken/ws/AGENTS.md`**.

## What this project is

Personal knowledge corpus with optional **PRISM** retrieval:

- **Email** — multi-account IMAP sync (1 message = 1 chunk)
- **SMS** — Android Backup & Restore XML (default filter `(831)535-2442`)
- **Chat** — ChatGPT / Claude.ai official export ZIPs (turn-level chunks)
- **Memory** — agent-written facts (`fish_memory_upsert`); similar facts are reconciled by LLM (duplicate / merge / distinct)

**PRISM** trains dual adapters so \(\cos(A_q(q), A_c(c))\) predicts **RelevanceAgent** scores — not raw semantic similarity. See [`docs/prism.md`](docs/prism.md) and workspace `prism_whitepaper.md`.

Storage: SQLite **`fish.db`** (documents + durable raw embeds: combined / header / body)
+ **Qdrant** ANN for combined/adapted vectors only
(`retrieval_models`: **`legacy` → `fish_legacy`**; **`{config}.{timestamp}` → `fish_{…}`**).
Binary zip `.prz`. See [`docs/prism.md`](docs/prism.md).

**Production corpus:** canonical db on GCP `mcp-services` at `/data/fish/fish.db` (PD `fish-data`). See [`docs/cloud.md`](docs/cloud.md).

## Architecture

- **Identity**: `{source_id}.{message_id}` as `corpus_items.source_key` (UNIQUE); integer PK is surrogate for Qdrant/training only — see [`docs/identity.md`](docs/identity.md)
- **Sync**: `imapclient` → `messages` + mirrored `corpus_items` (kind=email) — **cloud cron on mcp-services**
- **Import**: `fish import-corpus` — SMS, ChatGPT, Claude — see [`docs/import-runbook.md`](docs/import-runbook.md)
- **Search**: Qdrant ANN (raw or PRISM-adapted); metadata filters applied **in** the vector query (`--since`, `--until`, `--from`, `--account`, …). No keyword hybrid ranking (deferred).
- **MCP (remote)**: `https://mcp.seehart.com/fish/mcp` (Claude.ai connector)
- **MCP (local dev)**: `python -m fish.mcp_server` — optional; reads local db unless `FISH_DB_PATH` set
- **Training**: RelevanceAgent labels → `fish prism-train` (MSE) → `.prz`; then `fish prism-reembed` (from stored raw, no OpenAI) — see [`docs/prism.md`](docs/prism.md), [`docs/cloud.md`](docs/cloud.md)
- **Write lock**: exclusive flock for sync / import / corpus collect / freeze / train register (not epochs or `corpus label`) — `fish write-lock-status`

## Training corpus

Real queries are logged automatically on every `fish search` / `fish_search` call into `training_queries`.

| Table | Purpose |
|-------|---------|
| `training_queries` | **gold** (logged searches), **curated** (JSONL seeds), **synth** (LLM expansions of gold) |

| `origin` | Meaning | Timestamp |
|----------|---------|-----------|
| `gold` | Actual logged searches | `created_at` at log time; `source=logged` |
| `curated` | Hand-authored seeds from `config/gold_queries.jsonl` | `created_at` at load; `source` e.g. `curated:email-kb:2026-08-13` |
| `synth` | LLM “like these but different” from gold | `created_at` at synth; `source=synth:…` |

Load curated: `fish corpus add-curated` (alias: `add-gold`). Synth fills until `gold+synth ≥ --min-queries` (curated excluded from that count).
| `training_samples` | (query, corpus item) pairs with metadata |

Key sample fields:

| Field | Meaning |
|-------|---------|
| `retrieval_similarity` | Eval only — cosine from retriever at collect time |
| `target_relevance` | RelevanceAgent label — **training target** |
| `retriever` | `legacy` or model stem without `.prz` (e.g. `personal`) |

Workflow (full detail: [`docs/prism.md`](docs/prism.md)):

```bash
fish search "some query" --since 2026-07-16   # logs real query; adapted cos if .prz active
fish corpus collect --retriever legacy --min-queries 50 --top-k 20
fish corpus inject-positives --query "…" --like "%pattern%"   # cold-start hard positives
fish corpus label --limit 500           # RelevanceAgent → target_relevance
fish corpus freeze-training --chunk-repr combined  # → models/corpora/train_corpus_*.tcz
fish corpus stats
fish prism-train                        # MSE from --corpus latest; checkpoints under models/checkpoints/
fish prism-reembed --limit 200           # re-index ANN from raw (smoke); no OpenAI
fish prism-reembed                       # full re-index from raw_embedding
```

`header_body` configs auto-prep field embeds for **labeled training items only** before train (not full corpus). Full corpus: `fish embed --fields`.

Compare retrievers by collecting with `--retriever legacy` vs `--retriever personal` (separate runs).

## Setup

```bash
cd /home/ken/ws/fish
uv sync
mkdo fish -d ~/.local/bin -t global   # or project .venv

mkdir -p ~/.config/fish
cp config/accounts.yaml.example ~/.config/fish/accounts.yaml
cp .env.example ~/.config/fish/fish.env
fish connect <email>
```

## Commands

| Command | Purpose |
|---------|---------|
| `fish connect <email>` | Interactive IMAP/SMTP setup |
| `fish search <query>` | ANN search (`--since`, `--until`, `--from`, `--kinds`, `--account`, `--folder`, `--model-id`) |
| `fish get <id>` | Get corpus item / message by id |
| `fish import-corpus <source> <path>` | Import `android-sms`, `chatgpt`, or `claude` |
| `fish memory` / MCP | `fish_memory_upsert` for agent memories |
| `fish embedding-get <id>` | Stored embedding vector for a corpus item |
| `fish embed` | Embed pending combined vectors (SQLite + Qdrant); also embeds header/body into SQLite |
| `fish embed --fields` | Backfill `header_json` + header/body raw embeddings in SQLite only (no Qdrant) |
| `fish prism-train` | Train from frozen `.tcz` (`--corpus latest`) → `{config}.{timestamp}.prz`; `--from-db` freezes first; `--gpu` / `--overfit` |
| `fish prism-reembed` | Rewrite PRISM Qdrant collection from raw (streamed, skips existing; `--force` rewrites); `--limit` / `--like` / `--since` for smoke |
| `fish qdrant-migrate` | One-shot: copy sqlite-vec → `corpus_raw_embeddings` + upsert legacy Qdrant collection |
| `fish qdrant-reindex` | Upsert `corpus_raw_embeddings` into legacy Qdrant (skips existing ids; `--force` rewrites) |
| `fish corpus collect` | `--retriever legacy\|personal`, synthesize queries, top-k samples |
| `fish corpus inject-positives` | Force (query, doc) pairs into training set (cold-start) |
| `fish corpus label` | RelevanceAgent labels (`target_relevance`) |
| `fish corpus freeze-training` | Snapshot labeled embeds → `models/corpora/train_corpus_{ts}.tcz` |
| `fish corpus stats` | Query/sample counts |
| `fish corpus queries` | Dump training queries (`--origin gold\|curated\|synth`, `--source`, `--json`) |
| `fish corpus add-curated` | Load `config/gold_queries.jsonl` as `origin=curated` (`add-gold` alias) |
| `fish corpus purge` | Remove stale or superseded samples |
| `fish corpus browse` | Local Datasette UI for `fish.db` (alias: `dbserv fish`) |
| `fish sync` | IMAP sync + embed |
| `fish repair-headers` | Rebuild email `header_json` via source_key→message (not id); neutralize `test:sms:1` |
| `fish migrate-canonical-ids` | Rewrite `source_key` → `{source_id}.{message_id}` (keeps integer PKs); `--dry-run` |
| `fish status` | Config, connectivity, corpus counts by kind |

## MCP tools

Read: `fish_search` (auto-syncs when stale; fails loudly on auth errors; `context_json`, `kinds`), `fish_corpus_get`, `fish_embedding_get`, `fish_message_get`, `fish_thread_get`, `fish_sync_status`, `fish_priority_inbox`, `fish_digest`, `fish_topics_*`, `fish_import`, `fish_memory_upsert`

Write: `fish_sync_run`, `fish_message_move`, `fish_message_archive`, `fish_bulk_action`, `fish_compose`, `fish_send`

## Config paths

| File | Purpose |
|------|---------|
| `~/.config/fish/accounts.yaml` | IMAP/SMTP accounts |
| `~/.config/fish/fish.env` | `OPENAI_API_KEY`, `FISH_QDRANT_URL`, optional `FISH_PRISM_MODEL`, `FISH_DATA_DIR`, `FISH_DB_PATH` |
| `fish.db` | Corpus + IMAP state + raw embeddings — **cloud:** `/data/fish/fish.db`; **local dev:** `~/.config/fish/fish.db` |
| Qdrant | ANN indexes — **cloud:** Docker on `mcp-services` (`FISH_QDRANT_URL=http://127.0.0.1:6333`) |
| `models/` | PRISM `.prz` files — **cloud:** `/data/fish/models/` |
| `~/.config/fish/context_rules.yaml` | Context-based retrieval boosts |
| `imports/` | Drop zone for export files — **cloud:** `/data/fish/imports/` |

## PRISM

Objective: \(\mathrm{MSE}(\cos(A_q(q), A_c(c)), r)\) with \(r\) = RelevanceAgent — see [`docs/prism.md`](docs/prism.md).

```bash
# fish.env — then prism-reembed so the ANN index stores Ac(c)
FISH_PRISM_MODEL=personal.prz
```

Heavy training: RunPod per [`compute.yaml`](compute.yaml), [`docs/cloud.md`](docs/cloud.md), and [`docs/deploy.md`](docs/deploy.md).
