"""Tests for the MCP server."""

from __future__ import annotations

import json

import pytest

from agent.mcp.server import MCPServer, PROTOCOL_VERSION, SERVER_NAME
from agent.tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_initialize_returns_capabilities() -> None:
    server = MCPServer(registry=ToolRegistry())
    resp = await server.handle({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": PROTOCOL_VERSION, "capabilities": {}},
    })
    assert resp["id"] == 1
    result = resp["result"]
    assert result["protocolVersion"] == PROTOCOL_VERSION
    assert result["serverInfo"]["name"] == SERVER_NAME
    assert "tools" in result["capabilities"]
    assert "resources" in result["capabilities"]


@pytest.mark.asyncio
async def test_initialized_notification_returns_none() -> None:
    server = MCPServer(registry=ToolRegistry())
    out = await server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert out is None
    assert server._initialized is True


@pytest.mark.asyncio
async def test_tools_list_includes_registered_tools() -> None:
    reg = ToolRegistry()

    @reg.tool(description="echo a message", side_effect="read_only", timeout=2.0)
    async def echo(message: str) -> str:
        return message

    server = MCPServer(registry=reg)
    resp = await server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    names = [t["name"] for t in resp["result"]["tools"]]
    assert "echo" in names
    tool = next(t for t in resp["result"]["tools"] if t["name"] == "echo")
    assert tool["inputSchema"]["type"] == "object"
    assert tool["inputSchema"]["properties"]["message"]["type"] == "string"


@pytest.mark.asyncio
async def test_tools_call_runs_and_returns_content() -> None:
    reg = ToolRegistry()

    @reg.tool(description="add", side_effect="read_only", timeout=2.0)
    async def add(a: int, b: int) -> int:
        return a + b

    server = MCPServer(registry=reg)
    resp = await server.handle({
        "jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": {"name": "add", "arguments": {"a": 2, "b": 3}},
    })
    result = resp["result"]
    assert result["isError"] is False
    text = result["content"][0]["text"]
    assert "5" in text


@pytest.mark.asyncio
async def test_tools_call_unknown_name_returns_iserror() -> None:
    server = MCPServer(registry=ToolRegistry())
    resp = await server.handle({
        "jsonrpc": "2.0", "id": 4, "method": "tools/call",
        "params": {"name": "nonexistent", "arguments": {}},
    })
    result = resp["result"]
    assert result["isError"] is True
    assert "unknown tool" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_unknown_method_returns_jsonrpc_error() -> None:
    server = MCPServer(registry=ToolRegistry())
    resp = await server.handle({"jsonrpc": "2.0", "id": 5, "method": "totally-fake"})
    assert "error" in resp
    assert resp["error"]["code"] == -32601


@pytest.mark.asyncio
async def test_ping_returns_empty_result() -> None:
    server = MCPServer(registry=ToolRegistry())
    resp = await server.handle({"jsonrpc": "2.0", "id": 6, "method": "ping"})
    assert resp["result"] == {}


@pytest.mark.asyncio
async def test_resources_list_exposes_index_and_session() -> None:
    server = MCPServer(registry=ToolRegistry())
    resp = await server.handle({"jsonrpc": "2.0", "id": 7, "method": "resources/list"})
    uris = [r["uri"] for r in resp["result"]["resources"]]
    assert "tantalus://index/stats" in uris
    assert "tantalus://session/last" in uris


@pytest.mark.asyncio
async def test_default_registry_tools_are_exposed_through_mcp() -> None:
    from agent.tools import default_registry

    server = MCPServer(registry=default_registry())
    resp = await server.handle({"jsonrpc": "2.0", "id": 8, "method": "tools/list"})
    names = {t["name"] for t in resp["result"]["tools"]}
    # Spot-check that core built-ins are surfaced to Cursor.
    assert {"bash_exec", "file_read", "file_write", "multi_edit", "index_search"}.issubset(names)


@pytest.mark.asyncio
async def test_streams_round_trip_via_handle() -> None:
    """Exercise the dispatch path end-to-end via JSON-encoded messages."""
    server = MCPServer(registry=ToolRegistry())
    line = json.dumps({"jsonrpc": "2.0", "id": 9, "method": "initialize", "params": {}})
    msg = json.loads(line)
    resp = await server.handle(msg)
    encoded = json.dumps(resp)
    decoded = json.loads(encoded)
    assert decoded["id"] == 9
    assert "result" in decoded
