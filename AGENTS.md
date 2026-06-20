# Agent onboarding — fish (IMAP email AI)

> **Status**: Active at `/home/ken/ws/fish`. GitHub: [kenseehart/imap_sorting_hat](https://github.com/kenseehart/imap_sorting_hat).

## Shared resources

Cross-project assets: **`/home/ken/ws/shared`**. Workspace index: **`/home/ken/ws/AGENTS.md`**.

## What this project is

Multi-account IMAP email sync with RAG (1 message = 1 chunk), hybrid search, importance ranking, topic graphs, and a FastMCP server with full read/write mail operations.

## Repo

- Path: **`/home/ken/ws/fish`**
- GitHub: [kenseehart/imap_sorting_hat](https://github.com/kenseehart/imap_sorting_hat)

## Architecture

- **Sync**: `imapclient` → SQLite (`~/.config/fish/fish.db`) + **sqlite-vec** embeddings
- **Accounts**: `~/.config/fish/accounts.yaml` (hosts, usernames, app passwords)
- **MCP (local)**: `python -m fish.mcp_server` — registered as `fish` in `.cursor/mcp.json`
- **MCP (remote)**: `python -m fish.http_server` — OAuth pattern from `tesla/` for Claude mobile
- **Legacy**: `ish.py` retired; parsing logic lives in `src/fish/parse.py`

## Setup

```bash
cd /home/ken/ws/fish
uv sync
uv run python -m util.mkdo_setup
mkdo fish -d .venv/bin

mkdir -p ~/.config/fish
cp config/accounts.yaml.example ~/.config/fish/accounts.yaml
cp .env.example ~/.config/fish/fish.env   # OpenAI key only
fish connect <email>                      # stores app password encrypted per account
```

## Commands

| Command | Purpose |
|---------|---------|
| `fish connect <email>` | Interactive IMAP/SMTP setup for one account |
| `fish search <query>` | Hybrid semantic + keyword search (`--limit`, `--account`, `--json`) |
| `fish status` | Check config, IMAP connectivity, DB counts |
| `fish sync` | Sync last 90 days from all accounts |
| `fish backfill --since 2020-01-01` | Historical mail backfill |

## MCP tools

Read: `fish_search`, `fish_message_get`, `fish_thread_get`, `fish_sync_status`, `fish_priority_inbox`, `fish_digest`, `fish_topics_*`, `fish_topic_graph`

Write: `fish_sync_run`, `fish_message_move`, `fish_message_archive`, `fish_bulk_action`, `fish_compose`, `fish_send`

All tools are `autoApprove` in Cursor — agent can archive/move/send without prompts.

## Config paths

| File | Purpose |
|------|---------|
| `~/.config/fish/accounts.yaml` | IMAP/SMTP hosts and app passwords per mailbox |
| `~/.config/fish/fish.env` | OpenAI API key |
| `~/.config/fish/fish.db` | Messages + vectors |
| `~/.config/fish/actions.log` | Bulk action audit log |
