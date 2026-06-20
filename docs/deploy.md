# Remote MCP deploy (Claude mobile)

**Training** (heavy) uses `compute` — see `fish/compute.yaml` (`daime-prism` RunPod pod: L4, 86 GB RAM, `/workspace` on volume `daime_prism_volume`, ~$0.39/hr).

Bind the live pod after IP/port changes (RunPod console → **SSH over exposed TCP**):

```bash
compute bind daime-prism --ssh root@66.92.198.138:11405 \
  --proxy-user lutibaqqa6gnbi-64411dc8 --identity ~/.ssh/id_ed25519_personal
compute ssh daime-prism
compute tunnel daime-prism --port 8888   # Jupyter Lab
```

## Claude.ai connector

**Deferred** until `host/docs/hosting-python.md` experiment confirms URL + OAuth.

Use Advanced Settings with `FISH_MCP_CLIENT_ID` and `FISH_MCP_CLIENT_SECRET`.
