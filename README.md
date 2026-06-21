# fish (imap_sorting_hat)

Personal PRISM corpus: IMAP email, SMS, chat exports, agent memories.

## Features

- Unified corpus in SQLite + sqlite-vec (`corpus_items` / `corpus_vec`)
- IMAP sync (multi-account), hybrid search, PRISM dual-adapter retrieval
- Import Android SMS, ChatGPT, Claude exports
- Agent memory upsert with dedup
- Session context JSON for query augmentation and prompt assembly
- FastMCP for Cursor

See [AGENTS.md](AGENTS.md), [docs/import-runbook.md](docs/import-runbook.md), [docs/context.md](docs/context.md).
