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

Storage: SQLite **`fish.db`** (documents + `corpus_raw_embeddings`) + **Qdrant** ANN
(`retrieval_models`: **`legacy` → `fish_legacy`**; **`{config}.{timestamp}` → `fish_{…}`**).
Binary zip `.prz`. See [`docs/prism.md`](docs/prism.md).

**Production corpus:** canonical db on GCP `mcp-services` at `/data/fish/fish.db` (PD `fish-data`). See [`docs/cloud.md`](docs/cloud.md).

## Architecture

- **Sync**: `imapclient` → `messages` + mirrored `corpus_items` (kind=email) — **cloud cron on mcp-services**
- **Import**: `fish import-corpus` — SMS, ChatGPT, Claude — see [`docs/import-runbook.md`](docs/import-runbook.md)
- **Search**: Qdrant ANN (raw or PRISM-adapted); metadata filters applied **in** the vector query (`--since`, `--until`, `--from`, `--account`, …). No keyword hybrid ranking (deferred).
- **MCP (remote)**: `https://mcp.seehart.com/fish/mcp` (Claude.ai connector)
- **MCP (local dev)**: `python -m fish.mcp_server` — optional; reads local db unless `FISH_DB_PATH` set
- **Training**: RelevanceAgent labels → `fish prism-train` (MSE) → `.prz`; then `fish prism-reembed` (from stored raw, no OpenAI) — see [`docs/prism.md`](docs/prism.md), [`docs/cloud.md`](docs/cloud.md)
- **Write lock**: exclusive lock for sync / import / corpus / train — `fish write-lock-status`

## Training corpus

Real queries are logged automatically on every `fish search` / `fish_search` call into `training_queries`.

| Table | Purpose |
|-------|---------|
| `training_queries` | Real (logged searches), **gold** (curated), and synthetic queries |

Gold queries: `origin=gold`, plus `source` (e.g. `curated:email-kb:2026-08-13`) and `meta_json` (topics/intent). Seed file: `config/gold_queries.jsonl` — **email/knowledge search only** (what you'd send to `fish_search`, not Tesla/compute/CLI ops). Replace with `fish corpus add-gold --replace`.
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
fish corpus stats
fish prism-train                        # MSE(cos(Aq(q),Ac(c)), target_relevance)
fish prism-reembed --limit 200           # re-index ANN from raw (smoke); no OpenAI
fish prism-reembed                       # full re-index from raw_embedding
```

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
| `fish prism-train` | Train PRISM adapters (MSE vs RelevanceAgent) → `personal.prz` |
| `fish prism-reembed` | Rewrite PRISM Qdrant collection from raw (streamed, skips existing; `--force` rewrites); `--limit` / `--like` / `--since` for smoke |
| `fish qdrant-migrate` | One-shot: copy sqlite-vec → `corpus_raw_embeddings` + upsert legacy Qdrant collection |
| `fish qdrant-reindex` | Upsert `corpus_raw_embeddings` into legacy Qdrant (skips existing ids; `--force` rewrites) |
| `fish corpus collect` | `--retriever legacy\|personal`, synthesize queries, top-k samples |
| `fish corpus inject-positives` | Force (query, doc) pairs into training set (cold-start) |
| `fish corpus label` | RelevanceAgent labels (`target_relevance`) |
| `fish corpus stats` | Query/sample counts |
| `fish corpus queries` | Dump training queries (`--origin gold\|real\|synthetic`, `--source`, `--json`) |
| `fish corpus add-gold` | Load curated gold JSONL (`config/gold_queries.jsonl`) into `training_queries` |
| `fish corpus purge` | Remove stale or superseded samples |
| `fish corpus browse` | Local Datasette UI for `fish.db` (alias: `dbserv fish`) |
| `fish sync` | IMAP sync + embed |
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
