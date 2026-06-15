from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from fish.store import init_db
from fish.tools_registry import FISH_INSTRUCTIONS, register_tools

mcp = FastMCP("fish", instructions=FISH_INSTRUCTIONS)
register_tools(mcp, as_json=True)


def main() -> None:
    init_db()
    mcp.run()


if __name__ == "__main__":
    main()
