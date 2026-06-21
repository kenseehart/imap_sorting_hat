# Remote MCP deploy (Claude mobile)

**Production corpus** lives on GCP `mcp-services` — see [`docs/cloud.md`](cloud.md).

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

**Training** uses `compute` — see `fish/compute.yaml` (`daime-prism` pod: L4, 86 GB RAM, ~$0.39/hr).

Pull a snapshot of the cloud db before training:

```bash
compute sync mcp-services pull fish.db
FISH_DB_PATH=~/.config/fish/fish.db fish prism-train
compute sync mcp-services push models/personal.prz
```

Bind the live pod after IP/port changes (RunPod console → **SSH over exposed TCP**):

```bash
compute bind daime-prism --ssh root@66.92.198.138:11405 \
  --proxy-user lutibaqqa6gnbi-64411dc8 --identity ~/.ssh/id_ed25519_personal
compute ssh daime-prism
```
