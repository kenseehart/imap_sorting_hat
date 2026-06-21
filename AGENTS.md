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

Storage: SQLite `~/.config/fish/fish.db` + **sqlite-vec** (`corpus_items` / `corpus_vec`). Optional PRISM adapters (`.prz` in `~/.config/fish/models/`). Session **context JSON** augments search and prompts — see [`docs/context.md`](docs/context.md).

## Architecture

- **Sync**: `imapclient` → `messages` + mirrored `corpus_items` (kind=email)
- **Import**: `fish import-corpus` — SMS, ChatGPT, Claude — see [`docs/import-runbook.md`](docs/import-runbook.md)
- **Search**: hybrid semantic + keyword + context boosts; PRISM query/chunk adapters when `FISH_PRISM_MODEL` set
- **MCP (local)**: `python -m fish.mcp_server` — registered as `fish` in `.cursor/mcp.json`
- **Training**: `fish prism-train` (labels pairs via OpenAI, trains adapters; needs `uv sync --extra prism` for torch)

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
| `fish prism-train` | Train PRISM adapters → `~/.config/fish/models/personal.prz` |
| `fish sync` | IMAP sync + embed |
| `fish status` | Config, connectivity, corpus counts by kind |

## MCP tools

Read: `fish_search` (with `context_json`, `kinds`), `fish_corpus_get`, `fish_embedding_get`, `fish_message_get`, `fish_thread_get`, `fish_sync_status`, `fish_priority_inbox`, `fish_digest`, `fish_topics_*`, `fish_import`, `fish_memory_upsert`

Write: `fish_sync_run`, `fish_message_move`, `fish_message_archive`, `fish_bulk_action`, `fish_compose`, `fish_send`

## Config paths

| File | Purpose |
|------|---------|
| `~/.config/fish/accounts.yaml` | IMAP/SMTP accounts |
| `~/.config/fish/fish.env` | `OPENAI_API_KEY`, optional `FISH_PRISM_MODEL` |
| `~/.config/fish/fish.db` | Corpus + IMAP state |
| `~/.config/fish/models/` | PRISM `.prz` files, classifiers |
| `~/.config/fish/context_rules.yaml` | Context-based retrieval boosts |
| `~/.config/fish/imports/` | Drop zone for export files |

## PRISM

Activate after training:

```bash
# fish.env
FISH_PRISM_MODEL=personal.prz
```

Heavy training: RunPod `prism-train` per [`compute.yaml`](compute.yaml) and [`docs/deploy.md`](docs/deploy.md).
