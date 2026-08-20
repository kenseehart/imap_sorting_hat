# Remote MCP deploy (Claude mobile)

**Production corpus** lives on GCP `gcp-e2-mcp` — see [`docs/cloud.md`](cloud.md).

```bash
sitehost setup-fish-cloud
sitehost deploy-mcp-gateway
```

## Claude.ai connector

Registered at [claude.ai/customize/connectors](https://claude.ai/customize/connectors):

- URL: `https://mcp.seehart.com/fish/mcp`
- Client ID: `fish-mcp`
- Secret: `FISH_MCP_CLIENT_SECRET` from `~/.config/fish/fish.env`

## PRISM training (RunPod)

**Training** uses `compute` — see `fish/compute.yaml` (`runpod-l4`: L4, 86 GB RAM, ~$0.39/hr; label/freeze on `runpod-cpu32`). Train from a frozen `.tcz`, not a copy of canonical `fish.db`. On RunPod, `FISH_DATA_DIR` must be `/workspace/fish` (see [`cloud.md`](cloud.md) bootstrap).

```bash
compute up runpod-l4
# first session on the volume:
#   python3 src/fish/runpod_setup.py
source /workspace/fish/env.sh
fish prism-train --config bakeoff --gpu
compute sync gcp-e2-mcp --push models/
```

Bind the live pod after IP/port changes (RunPod console → **SSH over exposed TCP**):

```bash
compute bind runpod-l4 --ssh root@HOST:PORT \
  --proxy-user lutibaqqa6gnbi-64411dc8 --identity ~/.ssh/id_ed25519_personal
compute ssh runpod-l4
```
