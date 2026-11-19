from __future__ import annotations

import os

from dotenv import load_dotenv
from fastmcp import FastMCP

from fish.config import load_env
from fish.store import init_db
from fish.tools_registry import FISH_INSTRUCTIONS, register_tools
from ken_mcp import add_health_route, audited_tool, gateway_auth, mcp_bind_host

load_env()

# One hostname per service on the always-on box (host repo, docs/box.md).
MCP_BASE_URL = os.getenv("FISH_MCP_BASE_URL", "https://mcp-fish.seehart.com")
HOST = mcp_bind_host("FISH_MCP_HOST")
PORT = int(os.getenv("FISH_MCP_PORT", "8753"))
OAUTH_STATE_DIR = os.path.expanduser("~/.config/fish/oauth-state")


def build_http_mcp() -> FastMCP:
    """Fish over streamable HTTP, signed in through the gateway's GitHub sign-in."""
    from pathlib import Path

    from fish.config import data_dir

    auth = gateway_auth(service="fish", base_url=MCP_BASE_URL, state_dir=OAUTH_STATE_DIR)
    mcp = FastMCP("fish", instructions=FISH_INSTRUCTIONS, auth=auth)
    register_tools(mcp, as_json=False, audit_decorator=audited_tool("fish"))
    add_health_route(mcp, service="fish", repo=Path(__file__).resolve().parents[2],
                     watch=[OAUTH_STATE_DIR, data_dir()])
    return mcp


def main() -> None:
    init_db()
    mcp = build_http_mcp()
    mcp.run(transport="streamable-http", host=HOST, port=PORT)


if __name__ == "__main__":
    main()
