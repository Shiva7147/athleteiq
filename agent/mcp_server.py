"""
agent/mcp_server.py
Model Context Protocol (FastMCP) server.
Exposes AthleteIQ's core tools directly to Claude Desktop, Cursor, or any external MCP client.
"""
from typing import Dict, Any, List
import json
try:
    from mcp.server.mcpserver import MCPServer
    mcp_server = MCPServer("AthleteIQ-Decision-Core")
except (ImportError, ModuleNotFoundError):
    try:
        from mcp.server.fastmcp import FastMCP
        mcp_server = FastMCP("AthleteIQ-Decision-Core")
    except Exception:
        # Fallback decorator mock for unit tests if MCP server module differs
        class MockMCP:
            def __init__(self, name): self.name = name
            def tool(self):
                def decorator(f): return f
                return decorator
            def run(self): pass
        mcp_server = MockMCP("AthleteIQ-Decision-Core")

from agent.tools import get_player_load, get_risk_ranking, simulate_what_if_trajectory, search_literature


@mcp_server.tool()
def mcp_get_player_load(player_id: str) -> str:
    """Retrieves point-in-time verified EWMA, ACWR, and strain features for an athlete."""
    res = get_player_load(player_id)
    return json.dumps(res, indent=2)


@mcp_server.tool()
def mcp_get_risk_ranking(top_n: int = 3) -> str:
    """Returns top squad members ranked by 7-day injury hazard with SHAP reasons."""
    res = get_risk_ranking(top_n=top_n)
    return json.dumps(res, indent=2)


@mcp_server.tool()
def mcp_simulate_what_if(player_id: str, planned_minutes: int, is_tier_1_final: bool = False) -> str:
    """Simulates forward physiological trajectory for an athlete under a proposed match duration."""
    res = simulate_what_if_trajectory(player_id, planned_minutes, is_tier_1_final)
    return json.dumps(res, indent=2)


@mcp_server.tool()
def mcp_search_literature(query: str) -> str:
    """Searches peer-reviewed sports-science literature with verified page citations."""
    res = search_literature(query)
    return json.dumps(res, indent=2)


if __name__ == "__main__":
    mcp_server.run()
