# Fish cloud corpus (`gcp-e2-mcp`)

Canonical **`fish.db`** lives on GCP compute resource **`gcp-e2-mcp`** (VM name
`mcp-services`), not on the laptop. Phone Claude, scheduled IMAP sync, and MCP
search all use the same cloud database.

**Split:** always-on MCP + corpus on GCP; sparse **label / freeze / train** on
RunPod (`runpod-cpu32` / `runpod-l4`) per [`../compute.yaml`](../compute.yaml).
Do **not** run heavy pipeline jobs on the MCP host (OOM risk). Nodes do not
share a filesystem — sync snapshots / `.tcz` / `.prz` with `compute sync`.

Legacy alias: `mcp-services` still resolves to `gcp-e2-mcp` for one release.

## Layout

| Path (VM) | Purpose |
|-----------|---------|
| `/data/fish/fish.db` | Corpus + training samples + `corpus_raw_embeddings` (GCP PD `fish-data`, 100 GB) |
| `/data/fish/models/` | PRISM `.prz` adapters |
| `/data/fish/imports/` | Drop zone for SMS/chat export uploads |
| `/data/fish/qdrant/` | Qdrant storage (Docker volume / bind mount) |
| `/data/fish/compute/jobs/` | Detached job status trees (MCP-local jobs only) |
| `/home/mcp/.config/fish/fish.env` | Secrets + `FISH_DATA_DIR` / `FISH_DB_PATH` / `FISH_QDRANT_URL` |
| `/home/mcp/.config/fish/accounts.yaml` | IMAP accounts |

Env on the VM:

```bash
FISH_DATA_DIR=/data/fish
FISH_DB_PATH=/data/fish/fish.db
FISH_QDRANT_URL=http://127.0.0.1:6333
# Optional: HTTP client timeout seconds (default 60). Raise if searches time out under memory pressure.
# FISH_QDRANT_TIMEOUT_SEC=60
```

Status without SSH thrash: MCP tool **`fish_pipeline_status`** / CLI `fish pipeline-status`.

## Qdrant (ANN)

Run Qdrant on the same VM as the corpus (localhost only). Prefer Docker with
`--restart unless-stopped` and storage on `/data/fish/qdrant` (already how
`fish-qdrant` runs on this host).

```bash
sudo mkdir -p /data/fish/qdrant
docker run -d --name fish-qdrant --restart unless-stopped \
  -p 127.0.0.1:6333:6333 -p 127.0.0.1:6334:6334 \
  -v /data/fish/qdrant:/qdrant/storage \
  qdrant/qdrant:v1.15.0
```

Keep HNSW on disk so ANN does not pin ~113k×1536 vectors in RAM:

```bash
curl -X PATCH http://127.0.0.1:6333/collections/fish_legacy \
  -H 'Content-Type: application/json' \
  -d '{"hnsw_config":{"on_disk":true}}'
# wait until collection status is green before trusting latency
```

Client timeout: `FISH_QDRANT_TIMEOUT_SEC` (default 60). Timeouts under memory pressure
are a capacity bug — fix RAM / on_disk indexing; do not paper over with endless timeout growth.

One-time migrate from legacy sqlite-vec tables:

```bash
fish qdrant-migrate --limit 1000   # smoke
fish qdrant-migrate                # full (~113k)
fish prism-reembed                 # active PRISM collection from raw
```

## One-time setup

```bash
compute up gcp-e2-mcp
sitehost setup-fish-cloud --dry-run
sitehost setup-fish-cloud              # creates PD, mounts, uploads laptop fish.db
sitehost deploy-mcp-gateway
```

Options:

- `--no-migrate` — skip uploading laptop `fish.db`
- `--force-migrate` — overwrite remote db
- `--skip-disk` — PD already attached/mounted

## Write lock

Heavy writers acquire an exclusive `fcntl.flock` at `{FISH_DB_PATH}.write.lock`.
The kernel releases the flock when the holder exits — a dead process cannot leave
it stuck. `fish write-lock-status` probes the flock (PID text is metadata only;
cleared on unlock).

| Operation | Lock name | Duration |
|-----------|-----------|----------|
| `fish sync` | `sync` | Whole sync |
| `fish import-corpus` | `import` | Whole import |
| `fish corpus collect` (sample insert) | `corpus` | Collect only — **not** labeling |
| `fish corpus label` | *(none)* | Short SQLite UPDATEs; concurrent with train |
| `fish corpus freeze-training` | `freeze-prep` / `freeze-training` | Field prep (if needed), then short snapshot lock |
| `fish corpus purge` / inject / add-curated | `corpus` | Whole op |
| `fish prism-train` | `train` only for post-train model register | Epochs load a `.tcz` — **no fish.db** |
| `fish prism-reembed` / qdrant-* | `train` | Whole op |
| `fish repair-headers` | `repair-headers` | Whole op |

IMAP sync and corpus freeze never run concurrently. Epoch training does not hold
the Fish lock and does not open fish.db (unless `--from-db` / `--collect-first`
freezes first). SQLite uses WAL + 30s `busy_timeout`; label UPDATEs retry a few
times on lock contention.

```bash
fish write-lock-status
```

## Scheduled sync

`fish-sync.timer` on the VM runs `python -m fish sync --no-progress` every 6 hours (00:00, 06:00, 12:00, 18:00 UTC).

```bash
compute iap-ensure gcp-e2-mcp          # persistent IAP tunnel (avoids pile-up)
compute run gcp-e2-mcp 'sudo systemctl status fish-sync.timer'
compute ssh gcp-e2-mcp                 # interactive
# then: sudo systemctl start fish-sync.service
```

Do **not** use parallel `gcloud compute ssh --tunnel-through-iap` — see compute `iap-ensure` / workspace rule `iap-tunnel`.

`fish_search` (MCP) also auto-syncs when the corpus is older than ~5 minutes and **fails loudly** if any account is missing IMAP credentials.

## Auth on the VM

IMAP passwords live in `/home/mcp/.config/fish/` (`accounts.yaml` and/or `FISH_PASSWORD_*` in `fish.env`). After changing local secrets:

```bash
sitehost setup-fish-cloud --skip-disk --no-migrate   # pushes accounts.yaml + fish.env (incl. FISH_PASSWORD_*)
sitehost deploy-mcp-gateway                          # restarts mcp-fish + fish-sync unit
```

Do not paste app passwords into Claude/Cursor chat.

## Header integrity repair

Email `corpus_items.header_json` must be resolved via
`source_key = imap:{account_id}:{folder}:{uid}` → `messages`, **not**
`messages.id = corpus_items.id`. A non-email row that occupies a message PK
(e.g. `test:sms:1`) forces email upserts onto skewed ids; id-equality backfill
then stamps neighbor message headers onto the correct body.

In-place repair on `gcp-e2-mcp` (does not wipe/rebuild the corpus).
Default scans email corpus ids `>= 110920` (after the `test:sms:1` PK
collision) and rewrites disagreeing `header_json` via source_key → message:

```bash
compute iap-ensure gcp-e2-mcp
compute run gcp-e2-mcp --cwd /tmp -- \
  sudo -u mcp env FISH_DB_PATH=/data/fish/fish.db FISH_DATA_DIR=/data/fish \
  /home/mcp/mcp-gateway/.venv/bin/python -m fish repair-headers --json
# optional: fish embed --fields   # refresh cleared header_embedding rows
```

`--dry-run` reports counts without writing. `--skip-neutralize` leaves
`test:sms:1` in place. `--missing-only` only fills empty headers (full corpus).

## Laptop / RunPod (sparse compute)

The laptop is for **code** and **uploading imports**. Heavy PRISM work runs on
RunPod workers (`fish/compute.yaml` workloads), not on `gcp-e2-mcp`.

```bash
# Example: GPU train on runpod-l4
compute up runpod-l4
compute sync gcp-e2-mcp pull fish.db          # or sync .tcz / models as needed
# … run fish prism-train on the pod …
compute sync gcp-e2-mcp push models/personal.prz
compute down runpod-l4
```

CPU label/freeze: prefer `runpod-cpu32` (see `workloads` in `compute.yaml`).
RunPod cold start is often **1–3 minutes** (longer if the image must pull).

**Upload an import from laptop:**

```bash
compute sync gcp-e2-mcp push imports/export.zip   # prefer sync over raw IAP scp
compute run gcp-e2-mcp --cwd /tmp -- \
  sudo -u mcp env FISH_DATA_DIR=/data/fish FISH_DB_PATH=/data/fish/fish.db \
  /home/mcp/mcp-gateway/.venv/bin/python -m fish import-corpus chatgpt /data/fish/imports/export.zip
```

## Ops: recover dead guest / deploy

If IAP fails with SSL/`RECORD_LAYER` errors after OOM, the guest NIC may be dead
while the VM still shows RUNNING. Soft-reset preserves PD `fish-data`:

```bash
gcloud compute instances reset mcp-services --zone=us-central1-a --project=agi-green
# wait ~1–2 min
compute iap-stop gcp-e2-mcp || true
compute iap-ensure gcp-e2-mcp
compute run gcp-e2-mcp 'hostname'
sitehost deploy-mcp-gateway
```

Then run label → freeze → train on RunPod (`runpod-cpu32` / `runpod-l4`), not on MCP.

## SQLite

Single writer, WAL mode, local PD on the VM — see architecture discussion in [`../AGENTS.md`](../AGENTS.md).
