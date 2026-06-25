"""Model Context Protocol (MCP) server — exposes the harness's tools to MCP clients."""

from agent.mcp.server import MCPServer, PROTOCOL_VERSION, SERVER_NAME, SERVER_VERSION

__all__ = ["MCPServer", "PROTOCOL_VERSION", "SERVER_NAME", "SERVER_VERSION"]
