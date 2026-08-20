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

The laptop is for **code** and **uploading imports**. GPU train runs on
RunPod (`runpod-l4`). Do **not** copy canonical `fish.db` to a worker for
epochs — freeze a `.tcz` on the MCP host (or sync that snapshot) and train
from it. MCP, Qdrant, and `fish.db` stay on `gcp-e2-mcp`.

There is **no** cross-cloud POSIX filesystem (GCS FUSE / Filestore / rclone
mount). SQLite + `fcntl.flock` require the GCP PD. The RunPod network volume
at `/workspace` is the worker persistence domain (venv, checkpoints, `.tcz`,
`.prz`). Do not move the MCP server onto that volume.

```bash
# GPU train on runpod-l4 (after bootstrap, below)
compute up runpod-l4
# sync a frozen .tcz onto the volume, or freeze on the pod from a snapshot
compute sync runpod-l4 --push models/corpora/
# … source /workspace/fish/env.sh && fish prism-train …
compute sync gcp-e2-mcp --push models/   # finished .prz back to canonical models/
compute down runpod-l4
```

CPU label/freeze: still routed to `runpod-cpu32` in `compute.yaml` (they need
the live DB). Prefer not to round-trip `fish.db`; a later change may run
those on `gcp-e2-mcp` now that it has 16 GB RAM. RunPod cold start is often
**1–3 minutes** (longer if the image must pull).

### RunPod volume bootstrap

Only `/workspace` survives a pod stop. `fish` **fail-fasts** on RunPod unless
`FISH_DATA_DIR` (and `FISH_DB_PATH`, if set) resolve under `/workspace`.
First session on a volume (from `~/ws`, so resource `local_root` is the
workspace — not from `fish/` which would nest paths):

```bash
compute up runpod-l4
# re-bind if the pod recycled its SSH endpoint
cd ~/ws
compute sync runpod-l4 --push fish
compute sync runpod-l4 --push compute
compute sync runpod-l4 --push shared/cmdline
# Do not rsync a laptop fish.db onto the volume (epochs use a .tcz).
compute run runpod-l4 --cwd /workspace/fish -- python3 src/fish/runpod_setup.py
# later sessions, venv already on the volume:
#   source /workspace/fish/env.sh && fish prism-train --config bakeoff --gpu
```

`runpod_setup.py` is stdlib-only so it runs on the stock torch image. It
writes `FISH_DATA_DIR=/workspace/fish` into `~/.config/fish/fish.env`,
creates `/workspace/fish/.venv` with `--system-site-packages` (reuses image
CUDA torch — does not pip-install torch), installs fish's train deps,
editable `cmdline` / `compute` / fish, and `apt-get install rsync`.
Idempotent. Do not trust a leftover `/workspace/fish/fish/.venv` from older
attempts — this path is `/workspace/fish/.venv`. If that venv is broken,
`rm -rf /workspace/fish/.venv` and re-run setup.

### RunPod L4 pitfalls (learned running an 8-model bakeoff)

- **Ephemeral container disk.** A pod can stop mid-job (observed: RunPod
  marked it "exited" during an unattended bakeoff). Checkpoints live at
  `/workspace/fish/models/checkpoints/` only if `FISH_DATA_DIR=/workspace/fish`.
  Writing under `/root` loses the run. Resume: same `fish prism-train` command.
- **Re-bind after every restart.** A restarted pod gets a new public IP/port;
  `compute run runpod-l4 …` fails with a stale-endpoint error until you
  `compute bind runpod-l4 --ssh root@<new-ip>:<new-port>` (copy the exposed
  TCP line from the RunPod console, or the API pod detail) again.
- **`pgrep -f` self-match when polling over SSH.** A liveness check like
  `ssh pod "echo RUN=$(pgrep -f 'python3 -m fish prism-train' | head -1)"`
  matches its own remote shell process, because that literal string appears
  in the command line passed to `pgrep -f`. Result: `running` stays true
  after the job finished. Use a completion marker in the log, expected
  output files, or exclude the poller's pid.

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

## MCP gateway code layout (`gcp-e2-mcp`)

`sitehost deploy-mcp-gateway` is the supported way to ship code changes to
the gateway. If you ever need to hand-patch one file for a quick fix,
the fish package lives **nested one level deeper than the repo layout**:
`/home/mcp/mcp-gateway/src/fish/src/fish/prism/nwra_eval.py` (i.e.
`mcp-gateway/src/fish/` is a checkout of this repo, which itself has
`src/fish/...`) — not `/home/mcp/mcp-gateway/src/fish/prism/...`. Prefer the
real deploy path; verify with `compute run gcp-e2-mcp 'find /home/mcp/mcp-gateway -name nwra_eval.py'`
if unsure before copying.

## SQLite

Single writer, WAL mode, local PD on the VM — see architecture discussion in [`../AGENTS.md`](../AGENTS.md).
