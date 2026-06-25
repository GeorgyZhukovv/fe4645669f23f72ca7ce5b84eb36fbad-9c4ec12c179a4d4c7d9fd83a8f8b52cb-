"""MCP server: exposes the harness's :class:`ToolRegistry` over newline-delimited JSON-RPC.

The stdio transport is what Cursor / Claude Desktop / Cody use. Each line on stdin
is one JSON-RPC 2.0 message; each line on stdout is one response.

This server registers every built-in tool from :func:`agent.tools.default_registry`
plus a few read-only ``resources/`` style endpoints for the codebase index.
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass, field
from typing import Any, BinaryIO, TextIO

from agent.tools import default_registry
from agent.tools.registry import ToolRegistry


PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "tantalus"
SERVER_VERSION = "0.1.0"


@dataclass
class MCPServer:
    """Newline-delimited JSON-RPC 2.0 MCP server bound to a :class:`ToolRegistry`."""

    registry: ToolRegistry = field(default_factory=default_registry)
    server_name: str = SERVER_NAME
    server_version: str = SERVER_VERSION
    protocol_version: str = PROTOCOL_VERSION
    _initialized: bool = False

    async def handle(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """Dispatch one JSON-RPC message. Notifications (no ``id``) return ``None``."""
        method = message.get("method")
        req_id = message.get("id")
        params = message.get("params") or {}
        is_notification = "id" not in message

        if method == "initialize":
            return self._wrap(req_id, self._initialize(params))
        if method in {"notifications/initialized", "initialized"}:
            self._initialized = True
            return None
        if method == "ping":
            return self._wrap(req_id, {})
        if method == "tools/list":
            return self._wrap(req_id, await self._tools_list())
        if method == "tools/call":
            return self._wrap(req_id, await self._tools_call(params))
        if method == "resources/list":
            return self._wrap(req_id, await self._resources_list())
        if method == "resources/read":
            return self._wrap(req_id, await self._resources_read(params))
        if method == "shutdown":
            return self._wrap(req_id, {})

        if is_notification:
            return None
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }

    def _wrap(self, req_id: Any, result: Any) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    def _initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "protocolVersion": self.protocol_version,
            "capabilities": {
                "tools": {"listChanged": False},
                "resources": {"listChanged": False, "subscribe": False},
            },
            "serverInfo": {"name": self.server_name, "version": self.server_version},
        }

    async def _tools_list(self) -> dict[str, Any]:
        tools: list[dict[str, Any]] = []
        for spec in self.registry.specs():
            schema = spec.schema.get("input_schema", {"type": "object", "properties": {}})
            tools.append({
                "name": spec.name,
                "description": spec.description,
                "inputSchema": schema,
            })
        return {"tools": tools}

    async def _tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        args = params.get("arguments") or {}
        if not isinstance(name, str):
            return {"isError": True, "content": [{"type": "text", "text": "missing tool name"}]}
        if name not in set(self.registry.names()):
            return {"isError": True, "content": [{"type": "text", "text": f"unknown tool: {name}"}]}
        result = await self.registry.invoke(name, args)
        if result.ok:
            body = json.dumps(result.output, default=str, indent=2)
            return {
                "isError": False,
                "content": [{"type": "text", "text": body[:80_000]}],
            }
        return {
            "isError": True,
            "content": [{"type": "text", "text": result.error or "tool execution failed"}],
        }

    async def _resources_list(self) -> dict[str, Any]:
        # Codebase index summary is exposed as a single read-only resource.
        return {
            "resources": [
                {
                    "uri": "tantalus://index/stats",
                    "name": "Codebase index stats",
                    "description": "Symbol count, file count, languages for the current project.",
                    "mimeType": "application/json",
                },
                {
                    "uri": "tantalus://session/last",
                    "name": "Last session summary",
                    "description": "Latest session JSON if one exists.",
                    "mimeType": "application/json",
                },
            ]
        }

    async def _resources_read(self, params: dict[str, Any]) -> dict[str, Any]:
        uri = params.get("uri", "")
        if uri == "tantalus://index/stats":
            from agent.tools.indexer import get_or_build_index

            idx = await asyncio.to_thread(get_or_build_index, ".")
            stats = idx.stats()
            body = json.dumps({
                "total_files": stats.total_files,
                "total_symbols": stats.total_symbols,
                "by_language": stats.by_language,
                "last_updated": stats.last_updated,
            })
            return {"contents": [{"uri": uri, "mimeType": "application/json", "text": body}]}
        if uri == "tantalus://session/last":
            from pathlib import Path

            sess_dir = Path(".agent_state") / "sessions"
            if sess_dir.exists():
                files = sorted(sess_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
                if files:
                    body = files[0].read_text(encoding="utf-8")
                    return {"contents": [{"uri": uri, "mimeType": "application/json", "text": body}]}
            return {"contents": [{"uri": uri, "mimeType": "application/json", "text": "{}"}]}
        return {"contents": [{"uri": uri, "mimeType": "text/plain", "text": f"unknown resource: {uri}"}]}

    async def serve_streams(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Run the read → dispatch → write loop until stdin closes."""
        while True:
            line = await reader.readline()
            if not line:
                return
            text = line.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            try:
                message = json.loads(text)
            except json.JSONDecodeError:
                self._write_line(writer, {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": "parse error"},
                })
                continue
            try:
                response = await self.handle(message)
            except Exception as exc:  # noqa: BLE001
                response = {
                    "jsonrpc": "2.0",
                    "id": message.get("id"),
                    "error": {"code": -32603, "message": f"{type(exc).__name__}: {exc}"},
                }
            if response is not None:
                self._write_line(writer, response)

    def _write_line(self, writer: asyncio.StreamWriter, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, default=str) + "\n"
        writer.write(line.encode("utf-8"))


async def serve_stdio(registry: ToolRegistry | None = None) -> None:
    """Wire stdin/stdout to an :class:`MCPServer` instance and run it."""
    server = MCPServer(registry=registry or default_registry())
    loop = asyncio.get_event_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)
    transport, _ = await loop.connect_write_pipe(asyncio.streams.FlowControlMixin, sys.stdout)
    writer = asyncio.StreamWriter(transport, protocol, reader, loop)
    await server.serve_streams(reader, writer)
