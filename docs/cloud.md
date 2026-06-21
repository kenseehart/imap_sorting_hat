# Fish cloud corpus (mcp-services)

Canonical **`fish.db`** lives on GCP **`mcp-services`**, not on the laptop. Phone Claude, scheduled IMAP sync, and MCP search all use the same cloud database.

## Layout

| Path (VM) | Purpose |
|-----------|---------|
| `/data/fish/fish.db` | Corpus + training samples (GCP PD `fish-data`, 100 GB) |
| `/data/fish/models/` | PRISM `.prz` adapters |
| `/data/fish/imports/` | Drop zone for SMS/chat export uploads |
| `/home/mcp/.config/fish/fish.env` | Secrets + `FISH_DATA_DIR` / `FISH_DB_PATH` |
| `/home/mcp/.config/fish/accounts.yaml` | IMAP accounts |

Env on the VM:

```bash
FISH_DATA_DIR=/data/fish
FISH_DB_PATH=/data/fish/fish.db
```

## One-time setup

```bash
compute up mcp-services
sitehost setup-fish-cloud --dry-run
sitehost setup-fish-cloud              # creates PD, mounts, uploads laptop fish.db
sitehost deploy-mcp-gateway
```

Options:

- `--no-migrate` — skip uploading laptop `fish.db`
- `--force-migrate` — overwrite remote db
- `--skip-disk` — PD already attached/mounted

## Write lock

Heavy writers acquire an exclusive lock at `{FISH_DB_PATH}.write.lock`:

| Operation | Lock name |
|-----------|-----------|
| `fish sync` | `sync` |
| `fish import-corpus` | `import` |
| `fish corpus collect/label/purge` | `corpus` |
| `fish prism-train` | `train` |

IMAP sync and PRISM training never run concurrently. MCP read tools (search) do not take the lock.

```bash
fish write-lock-status
```

## Scheduled sync

`fish-sync.timer` on the VM runs `fish sync --no-progress` every 6 hours (00:00, 06:00, 12:00, 18:00 UTC).

```bash
compute ssh mcp-services -- sudo systemctl status fish-sync.timer
compute ssh mcp-services -- sudo systemctl start fish-sync.service   # manual run
```

## Laptop / RunPod

The laptop is for **code**, **uploading imports**, and **GPU training** — not day-to-day corpus sync.

**Pull db for PRISM training:**

```bash
compute sync mcp-services pull fish.db
FISH_DB_PATH=~/.config/fish/fish.db FISH_DATA_DIR=~/.config/fish fish prism-train
compute sync mcp-services push models/personal.prz
```

On RunPod (`daime-prism`), sync the pulled db into the pod workspace first, then train with `FISH_DB_PATH` pointing at the snapshot.

**Upload an import from laptop:**

```bash
gcloud compute scp export.zip mcp-services:/data/fish/imports/ --zone=us-central1-a --tunnel-through-iap
compute ssh mcp-services -- sudo -u mcp fish import-corpus chatgpt /data/fish/imports/export.zip
```

## SQLite

Single writer, WAL mode, local PD on the VM — see architecture discussion in [`../AGENTS.md`](../AGENTS.md).
