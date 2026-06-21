# Agent onboarding — fish (personal PRISM corpus)

> **Status**: Active at `/home/ken/ws/fish`. GitHub: [kenseehart/imap_sorting_hat](https://github.com/kenseehart/imap_sorting_hat).

## Shared resources

Cross-project assets: **`/home/ken/ws/shared`**. Workspace index: **`/home/ken/ws/AGENTS.md`**.

## What this project is

Personal knowledge corpus with PRISM retrieval:

- **Email** — multi-account IMAP sync (1 message = 1 chunk)
- **SMS** — Android Backup & Restore XML (default filter `(831)535-2442`)
- **Chat** — ChatGPT / Claude.ai official export ZIPs (turn-level chunks)
- **Memory** — agent-written facts (`fish_memory_upsert`); similar facts are reconciled by LLM (duplicate / merge / distinct)

Storage: SQLite **`fish.db`** + **sqlite-vec** (`corpus_items` / `corpus_vec`). Optional PRISM adapters (`.prz` in `models/`). Session **context JSON** augments search and prompts — see [`docs/context.md`](docs/context.md).

**Production corpus:** canonical db on GCP `mcp-services` at `/data/fish/fish.db` (PD `fish-data`). See [`docs/cloud.md`](docs/cloud.md).

## Architecture

- **Sync**: `imapclient` → `messages` + mirrored `corpus_items` (kind=email) — **cloud cron on mcp-services**
- **Import**: `fish import-corpus` — SMS, ChatGPT, Claude — see [`docs/import-runbook.md`](docs/import-runbook.md)
- **Search**: hybrid semantic + keyword + context boosts; PRISM query/chunk adapters when `FISH_PRISM_MODEL` set
- **MCP (remote)**: `https://mcp.seehart.com/fish/mcp` (Claude.ai connector)
- **MCP (local dev)**: `python -m fish.mcp_server` — optional; reads local db unless `FISH_DB_PATH` set
- **Training**: `fish prism-train` on RunPod after `compute sync mcp-services pull fish.db` — see [`docs/cloud.md`](docs/cloud.md)
- **Write lock**: exclusive lock for sync / import / corpus / train — `fish write-lock-status`

## Training corpus

Real queries are logged automatically on every `fish search` / `fish_search` call into `training_queries`.

| Table | Purpose |
|-------|---------|
| `training_queries` | Real (logged searches) and synthetic (QuerySynthesisAgent) queries |
| `training_samples` | (query, corpus item) pairs with metadata |

Key sample fields:

| Field | Meaning |
|-------|---------|
| `retrieval_similarity` | Eval only — cosine from retriever at collect time |
| `target_relevance` | RelevanceAgent label — **training target** |
| `retriever` | `legacy` or model stem without `.prz` (e.g. `personal`) |

Workflow:

```bash
fish search "some query"              # logs real query
fish corpus collect --retriever legacy --min-queries 50 --top-k 20
fish corpus label --limit 500
fish corpus stats
fish prism-train                      # MSE on target_relevance
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
| `fish search <query>` | Corpus search (`--kinds`, `--context`, `--account`, `--json`) |
| `fish import-corpus <source> <path>` | Import `android-sms`, `chatgpt`, or `claude` |
| `fish memory` / MCP | `fish_memory_upsert` for agent memories |
| `fish embedding-get <id>` | Stored embedding vector for a corpus item |
| `fish prism-train` | Train PRISM adapters from labeled samples → `personal.prz` |
| `fish corpus collect` | `--retriever legacy\|personal`, synthesize queries, top-k samples |
| `fish corpus label` | RelevanceAgent labels (`target_relevance`) |
| `fish corpus stats` | Query/sample counts |
| `fish corpus purge` | Remove stale or superseded samples |
| `fish corpus browse` | Local Datasette UI for `fish.db` (alias: `dbserv fish`) |
| `fish sync` | IMAP sync + embed |
| `fish status` | Config, connectivity, corpus counts by kind |

## MCP tools

Read: `fish_search` (with `context_json`, `kinds`), `fish_corpus_get`, `fish_embedding_get`, `fish_message_get`, `fish_thread_get`, `fish_sync_status`, `fish_priority_inbox`, `fish_digest`, `fish_topics_*`, `fish_import`, `fish_memory_upsert`

Write: `fish_sync_run`, `fish_message_move`, `fish_message_archive`, `fish_bulk_action`, `fish_compose`, `fish_send`

## Config paths

| File | Purpose |
|------|---------|
| `~/.config/fish/accounts.yaml` | IMAP/SMTP accounts |
| `~/.config/fish/fish.env` | `OPENAI_API_KEY`, optional `FISH_PRISM_MODEL`, `FISH_DATA_DIR`, `FISH_DB_PATH` |
| `fish.db` | Corpus + IMAP state — **cloud:** `/data/fish/fish.db`; **local dev:** `~/.config/fish/fish.db` |
| `models/` | PRISM `.prz` files — **cloud:** `/data/fish/models/` |
| `~/.config/fish/context_rules.yaml` | Context-based retrieval boosts |
| `imports/` | Drop zone for export files — **cloud:** `/data/fish/imports/` |

## PRISM

Activate after training:

```bash
# fish.env
FISH_PRISM_MODEL=personal.prz
```

Heavy training: RunPod `prism-train` per [`compute.yaml`](compute.yaml), [`docs/cloud.md`](docs/cloud.md), and [`docs/deploy.md`](docs/deploy.md).
