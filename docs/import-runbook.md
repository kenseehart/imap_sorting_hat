# Import runbook — SMS and chat history

Batch imports land in the unified corpus (`corpus_items` in `~/.config/fish/fish.db`). Re-imports are idempotent (matched by `source_key`).

## Android SMS

1. On phone: [SMS Backup & Restore](https://synctech.com.au/sms-backup-restore/) → backup SMS to XML.
2. Copy XML to this machine (e.g. `~/.config/fish/imports/`).
3. Import (filters to `(831)535-2442` by default):

```bash
fish import-corpus android-sms ~/path/to/sms-backup.xml
```

Re-export from the phone periodically; re-run import to merge new messages.

## ChatGPT

1. [chatgpt.com](https://chatgpt.com) → Settings → Data controls → Export.
2. Wait for email (often 15–30 minutes). Download ZIP within 24 hours.
3. Import:

```bash
fish import-corpus chatgpt ~/Downloads/chatgpt-export.zip
```

If the ZIP includes `memory.json`, those entries import as `kind=memory` with source `chatgpt_memory`.

## Claude.ai

1. [claude.ai](https://claude.ai) → Settings → Privacy → Export data.
2. Download ZIP from email within 24 hours.
3. Import:

```bash
fish import-corpus claude ~/Downloads/claude-export.zip
```

**Not included:** Claude Projects and Claude Memory (standard export). Use `fish_memory_upsert` for portable agent memories.

## Options

| Flag | Purpose |
|------|---------|
| `--dry-run` | Parse and count without writing |
| `--no-embed` | Skip OpenAI embedding after import |
| `--phone` | Override SMS phone filter |

## MCP

```
fish_import(source="chatgpt", path="/path/to/export.zip")
```

After import, embeddings run automatically unless `dry_run=true`.
