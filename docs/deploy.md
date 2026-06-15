# Remote MCP deploy (Claude mobile)

Pattern matches `tesla/` — OAuth + streamable-http on my.hosting.com.

## Env vars (`~/.config/fish/fish.env`)

```
FISH_MCP_BASE_URL=https://your-domain.com/fish
FISH_MCP_CLIENT_ID=fish-mcp
FISH_MCP_CLIENT_SECRET=change-me
FISH_MCP_HOST=0.0.0.0
FISH_MCP_PORT=8753
```

## Run

```bash
cd /home/ken/fish
source .venv/bin/activate
python -m fish.http_server
```

## systemd (on hosting)

```ini
[Unit]
Description=Fish MCP HTTP
After=network.target

[Service]
WorkingDirectory=/home/ken/fish
EnvironmentFile=/home/ken/.config/fish/fish.env
ExecStart=/home/ken/fish/.venv/bin/python -m fish.http_server
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

nginx: reverse-proxy `/fish` → `localhost:8753`.

## Claude.ai connector

Use Advanced Settings with `FISH_MCP_CLIENT_ID` and `FISH_MCP_CLIENT_SECRET`.
