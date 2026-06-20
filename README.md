# fish (imap_sorting_hat)

IMAP email sync, RAG, and MCP agent for multi-account mail.

## Features

- Sync 3 mailboxes (seehart, agi.green, gmail) via IMAP
- One message = one RAG chunk in SQLite + sqlite-vec
- Hybrid semantic + keyword search
- Bulk archive/move/flag by natural-language query
- Importance ranking, topic clusters, mind-map graph export
- Compose and send via SMTP
- FastMCP for Cursor; optional HTTP OAuth deploy for Claude mobile

## Quick start

See [AGENTS.md](AGENTS.md).

## Status

Replaces the 2022 `ish.py` prototype with a modern `uv` package under `src/fish/`.
