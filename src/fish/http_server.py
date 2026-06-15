from __future__ import annotations

import os

from dotenv import load_dotenv
from fastmcp import FastMCP

from fish.config import load_env
from fish.personal_auth import PersonalAuthProvider
from fish.store import init_db
from fish.tools_registry import FISH_INSTRUCTIONS, register_tools

load_env()

MCP_BASE_URL = os.getenv("FISH_MCP_BASE_URL", "https://your-domain.com/fish")
MCP_CLIENT_ID = os.getenv("FISH_MCP_CLIENT_ID", "fish-mcp")
MCP_CLIENT_SECRET = os.getenv("FISH_MCP_CLIENT_SECRET", "")
HOST = os.getenv("FISH_MCP_HOST", "0.0.0.0")
PORT = int(os.getenv("FISH_MCP_PORT", "8753"))


def build_http_mcp() -> FastMCP:
    auth = None
    if MCP_CLIENT_SECRET:
        auth = PersonalAuthProvider(
            base_url=MCP_BASE_URL,
            client_id=MCP_CLIENT_ID,
            client_secret=MCP_CLIENT_SECRET,
            state_dir=str(os.path.expanduser("~/.config/fish/oauth-state")),
        )
    mcp = FastMCP("fish", instructions=FISH_INSTRUCTIONS, auth=auth)
    register_tools(mcp, as_json=False)
    return mcp


def main() -> None:
    init_db()
    mcp = build_http_mcp()
    mcp.run(transport="streamable-http", host=HOST, port=PORT)


if __name__ == "__main__":
    main()
