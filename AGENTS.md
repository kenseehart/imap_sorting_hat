# Agent onboarding — fish (IMAP email AI)

> **Status**: Cloned at `/home/ken/fish`. GitHub: [kenseehart/imap_sorting_hat](https://github.com/kenseehart/imap_sorting_hat).

## Shared resources

Cross-project assets: **`/home/ken/shared`**. Workspace index: **`/home/ken/AGENTS.md`**.

## What this project is

AI-powered email management across multiple IMAP accounts. Likely needs rewrite for FastMCP + modern Claude workflow.

## Repo

- Path: **`/home/ken/fish`**
- GitHub: [kenseehart/imap_sorting_hat](https://github.com/kenseehart/imap_sorting_hat)

## Planned architecture

- FastMCP tools: list mailboxes, search, read, summarize, label/move
- OAuth MCP deploy for Claude mobile (pattern: `tesla/`)
- Optional chat UI: `shared/web/chat/` + backend handler

## Quick start

```bash
cd /home/ken/fish
uv sync   # once pyproject exists
```

## MCP target

Register as `fish` in `.cursor/mcp.json` when `fish/mcp_server.py` exists.
