from __future__ import annotations

import os

from dotenv import load_dotenv
from fastmcp import FastMCP

from fish.config import load_env
from fish.store import init_db
from fish.tools_registry import FISH_INSTRUCTIONS, register_tools
from ken_mcp import audited_tool, gateway_mode, mcp_bind_host, require_mcp_auth

load_env()

MCP_BASE_URL = os.getenv("FISH_MCP_BASE_URL", "https://mcp.seehart.com/fish")
MCP_CLIENT_ID = os.getenv("FISH_MCP_CLIENT_ID", "fish-mcp")
MCP_CLIENT_SECRET = os.getenv("FISH_MCP_CLIENT_SECRET", "")
HOST = mcp_bind_host("FISH_MCP_HOST")
PORT = int(os.getenv("FISH_MCP_PORT", "8753"))


def build_http_mcp(*, require_auth: bool = True) -> FastMCP:
    auth = None
    if require_auth or gateway_mode():
        auth = require_mcp_auth(
            base_url=MCP_BASE_URL,
            client_id=MCP_CLIENT_ID,
            client_secret=MCP_CLIENT_SECRET,
            service="fish",
            state_dir=str(os.path.expanduser("~/.config/fish/oauth-state")),
        )
    elif MCP_CLIENT_SECRET:
        from ken_mcp import PersonalAuthProvider

        auth = PersonalAuthProvider(
            base_url=MCP_BASE_URL,
            client_id=MCP_CLIENT_ID,
            client_secret=MCP_CLIENT_SECRET,
            state_dir=str(os.path.expanduser("~/.config/fish/oauth-state")),
        )
    mcp = FastMCP("fish", instructions=FISH_INSTRUCTIONS, auth=auth)
    register_tools(mcp, as_json=False, audit_decorator=audited_tool("fish"))
    return mcp


def main() -> None:
    init_db()
    mcp = build_http_mcp(require_auth=True)
    mcp.run(transport="streamable-http", host=HOST, port=PORT)


if __name__ == "__main__":
    main()
