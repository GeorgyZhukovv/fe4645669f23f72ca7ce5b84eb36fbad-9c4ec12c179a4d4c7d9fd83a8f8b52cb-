"""Language Server Protocol client + server pool + lsp_* tools."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.tools.registry import GLOBAL_REGISTRY


@dataclass
class LSPServerConfig:
    """Launch + capability config for a single language server."""

    name: str
    command: list[str]
    languages: list[str]
    initialization_options: dict[str, Any] = field(default_factory=dict)
    file_extensions: list[str] = field(default_factory=list)


@dataclass
class Diagnostic:
    """A single LSP diagnostic."""

    file: str
    severity: int
    message: str
    code: str | None
    line: int
    column: int


@dataclass
class HoverResult:
    """LSP hover response."""

    contents: str
    range: dict[str, Any] | None = None


@dataclass
class Location:
    """LSP Location (file + range)."""

    uri: str
    line: int
    column: int


def detect_language_servers(project_root: str | Path) -> dict[str, LSPServerConfig]:
    """Detect installed language servers on the host."""
    configs: dict[str, LSPServerConfig] = {}
    if shutil.which("pyright-langserver"):
        configs["python"] = LSPServerConfig(
            name="pyright",
            command=["pyright-langserver", "--stdio"],
            languages=["python"],
            file_extensions=[".py"],
        )
    elif shutil.which("pylsp"):
        configs["python"] = LSPServerConfig(
            name="pylsp",
            command=["pylsp"],
            languages=["python"],
            file_extensions=[".py"],
        )
    if shutil.which("typescript-language-server"):
        configs["typescript"] = LSPServerConfig(
            name="typescript-language-server",
            command=["typescript-language-server", "--stdio"],
            languages=["typescript", "javascript"],
            file_extensions=[".ts", ".tsx", ".js", ".jsx"],
        )
    if shutil.which("rust-analyzer"):
        configs["rust"] = LSPServerConfig(
            name="rust-analyzer",
            command=["rust-analyzer"],
            languages=["rust"],
            file_extensions=[".rs"],
        )
    if shutil.which("gopls"):
        configs["go"] = LSPServerConfig(
            name="gopls",
            command=["gopls", "serve"],
            languages=["go"],
            file_extensions=[".go"],
        )
    if shutil.which("clangd"):
        configs["c"] = LSPServerConfig(
            name="clangd",
            command=["clangd"],
            languages=["c", "cpp"],
            file_extensions=[".c", ".h", ".cpp", ".cc", ".hpp"],
        )
    return configs


class LSPClient:
    """Minimal JSON-RPC LSP client over stdio."""

    def __init__(self, config: LSPServerConfig) -> None:
        """Create a client for ``config`` (server not started until :meth:`start`)."""
        self.config = config
        self._proc: asyncio.subprocess.Process | None = None
        self._id = 0
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._reader_task: asyncio.Task[None] | None = None
        self._diagnostics: dict[str, list[Diagnostic]] = {}
        self._init_done = False
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Launch the server subprocess + reader task."""
        if self._proc is not None:
            return
        self._proc = await asyncio.create_subprocess_exec(
            *self.config.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        self._reader_task = asyncio.create_task(self._reader())

    async def _reader(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        while True:
            try:
                line = await self._proc.stdout.readline()
            except Exception:
                return
            if not line:
                return
            if not line.lower().startswith(b"content-length"):
                continue
            try:
                length = int(line.split(b":")[1].strip())
            except (IndexError, ValueError):
                continue
            # consume blank line
            await self._proc.stdout.readline()
            body = await self._proc.stdout.readexactly(length)
            try:
                msg = json.loads(body)
            except json.JSONDecodeError:
                continue
            await self._dispatch(msg)

    async def _dispatch(self, msg: dict[str, Any]) -> None:
        if "id" in msg and msg["id"] in self._pending:
            fut = self._pending.pop(msg["id"])
            if "error" in msg:
                fut.set_exception(RuntimeError(str(msg["error"])))
            else:
                fut.set_result(msg.get("result"))
            return
        if msg.get("method") == "textDocument/publishDiagnostics":
            params = msg.get("params") or {}
            uri = params.get("uri", "")
            self._diagnostics[uri] = [
                Diagnostic(
                    file=uri,
                    severity=int(d.get("severity", 1)),
                    message=d.get("message", ""),
                    code=str(d.get("code")) if d.get("code") is not None else None,
                    line=int(d.get("range", {}).get("start", {}).get("line", 0)),
                    column=int(d.get("range", {}).get("start", {}).get("character", 0)),
                )
                for d in (params.get("diagnostics") or [])
            ]

    async def _write(self, payload: dict[str, Any]) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        body = json.dumps(payload).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode()
        self._proc.stdin.write(header + body)
        await self._proc.stdin.drain()

    async def _request(self, method: str, params: Any) -> Any:
        async with self._lock:
            self._id += 1
            req_id = self._id
            fut: asyncio.Future[Any] = asyncio.get_event_loop().create_future()
            self._pending[req_id] = fut
            await self._write({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
        return await asyncio.wait_for(fut, timeout=30)

    async def _notify(self, method: str, params: Any) -> None:
        await self._write({"jsonrpc": "2.0", "method": method, "params": params})

    async def initialize(self, root_uri: str, capabilities: dict[str, Any] | None = None) -> dict[str, Any]:
        await self.start()
        result = await self._request("initialize", {
            "processId": os.getpid(),
            "rootUri": root_uri,
            "capabilities": capabilities or {},
            "initializationOptions": self.config.initialization_options,
        })
        await self._notify("initialized", {})
        self._init_done = True
        return result

    async def did_open(self, uri: str, language_id: str, text: str) -> None:
        await self._notify("textDocument/didOpen", {
            "textDocument": {"uri": uri, "languageId": language_id, "version": 1, "text": text},
        })

    async def did_change(self, uri: str, text: str) -> None:
        await self._notify("textDocument/didChange", {
            "textDocument": {"uri": uri, "version": 2},
            "contentChanges": [{"text": text}],
        })

    async def get_diagnostics(self, uri: str) -> list[Diagnostic]:
        # Diagnostics arrive via push notification; give the server a moment.
        for _ in range(20):
            if uri in self._diagnostics:
                break
            await asyncio.sleep(0.1)
        return list(self._diagnostics.get(uri, []))

    async def get_hover(self, uri: str, line: int, character: int) -> HoverResult | None:
        res = await self._request("textDocument/hover", {
            "textDocument": {"uri": uri},
            "position": {"line": line, "character": character},
        })
        if not res:
            return None
        contents = res.get("contents")
        if isinstance(contents, dict):
            contents = contents.get("value", "")
        if isinstance(contents, list):
            parts = []
            for c in contents:
                parts.append(c.get("value", "") if isinstance(c, dict) else str(c))
            contents = "\n".join(parts)
        return HoverResult(contents=str(contents or ""), range=res.get("range"))

    async def get_definition(self, uri: str, line: int, character: int) -> list[Location]:
        res = await self._request("textDocument/definition", {
            "textDocument": {"uri": uri},
            "position": {"line": line, "character": character},
        })
        return _locations(res)

    async def get_references(self, uri: str, line: int, character: int) -> list[Location]:
        res = await self._request("textDocument/references", {
            "textDocument": {"uri": uri},
            "position": {"line": line, "character": character},
            "context": {"includeDeclaration": True},
        })
        return _locations(res)

    async def get_completion(self, uri: str, line: int, character: int) -> list[dict[str, Any]]:
        res = await self._request("textDocument/completion", {
            "textDocument": {"uri": uri},
            "position": {"line": line, "character": character},
        })
        if isinstance(res, dict):
            return list(res.get("items", []))
        return list(res or [])

    async def get_code_actions(self, uri: str, start: tuple[int, int], end: tuple[int, int], diagnostics: list[Diagnostic] | None = None) -> list[dict[str, Any]]:
        res = await self._request("textDocument/codeAction", {
            "textDocument": {"uri": uri},
            "range": {
                "start": {"line": start[0], "character": start[1]},
                "end": {"line": end[0], "character": end[1]},
            },
            "context": {"diagnostics": [d.__dict__ for d in (diagnostics or [])]},
        })
        return list(res or [])

    async def shutdown(self) -> None:
        if self._proc is None:
            return
        try:
            await asyncio.wait_for(self._request("shutdown", None), timeout=5)
            await self._notify("exit", None)
        except Exception:
            pass
        if self._reader_task:
            self._reader_task.cancel()
        self._proc.terminate()
        try:
            await asyncio.wait_for(self._proc.wait(), timeout=5)
        except (asyncio.TimeoutError, TimeoutError):
            self._proc.kill()
        self._proc = None


def _locations(value: Any) -> list[Location]:
    """Normalise LSP location or location[] responses."""
    if not value:
        return []
    items = value if isinstance(value, list) else [value]
    out: list[Location] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        uri = item.get("uri") or item.get("targetUri") or ""
        rng = item.get("range") or item.get("targetRange") or {}
        start = rng.get("start", {})
        out.append(Location(uri=uri, line=int(start.get("line", 0)), column=int(start.get("character", 0))))
    return out


class LSPServerPool:
    """Process-local pool of long-lived :class:`LSPClient` instances."""

    def __init__(self, project_root: str | Path = ".") -> None:
        self.project_root = str(Path(project_root).resolve())
        self.root_uri = Path(self.project_root).as_uri()
        self._configs: dict[str, LSPServerConfig] = detect_language_servers(self.project_root)
        self._clients: dict[str, LSPClient] = {}
        self._lock = asyncio.Lock()

    def _language_for(self, path: str) -> str | None:
        ext = Path(path).suffix.lower()
        for lang, cfg in self._configs.items():
            if ext in cfg.file_extensions:
                return lang
        return None

    async def client_for(self, path: str) -> LSPClient | None:
        """Return (lazily creating) the client appropriate for ``path``."""
        lang = self._language_for(path)
        if lang is None:
            return None
        async with self._lock:
            client = self._clients.get(lang)
            if client is not None:
                if client._proc and client._proc.returncode is None:
                    return client
                # Health check failed; drop and rebuild.
                await client.shutdown()
            client = LSPClient(self._configs[lang])
            self._clients[lang] = client
        await client.initialize(self.root_uri)
        return client

    async def shutdown_all(self) -> None:
        for client in list(self._clients.values()):
            await client.shutdown()
        self._clients.clear()


_DEFAULT_POOL = LSPServerPool()


def _path_to_uri(path: str) -> str:
    """Convert an OS path to a ``file://`` URI."""
    p = Path(path).resolve()
    return p.as_uri()


@GLOBAL_REGISTRY.tool(
    description="LSP diagnostics for one file (errors + warnings). Falls back to no-op when no server is available.",
    side_effect="read_only",
    timeout=60.0,
)
async def lsp_diagnostics(path: str) -> dict:
    """Open ``path`` in the appropriate LSP server and collect diagnostics."""
    client = await _DEFAULT_POOL.client_for(path)
    if client is None:
        return {"path": path, "available": False, "diagnostics": []}
    uri = _path_to_uri(path)
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    await client.did_open(uri, _language_of(path), text)
    diags = await client.get_diagnostics(uri)
    return {
        "path": path,
        "available": True,
        "diagnostics": [d.__dict__ for d in diags],
    }


@GLOBAL_REGISTRY.tool(
    description="LSP hover info at a (line, column).",
    side_effect="read_only",
    timeout=30.0,
)
async def lsp_hover(path: str, line: int, col: int) -> dict:
    """Hover information at the cursor."""
    client = await _DEFAULT_POOL.client_for(path)
    if client is None:
        return {"path": path, "available": False}
    uri = _path_to_uri(path)
    await client.did_open(uri, _language_of(path), Path(path).read_text(encoding="utf-8", errors="replace"))
    hover = await client.get_hover(uri, line, col)
    return {"path": path, "available": True, "contents": hover.contents if hover else ""}


@GLOBAL_REGISTRY.tool(
    description="Go-to-definition via LSP.",
    side_effect="read_only",
    timeout=30.0,
)
async def lsp_definition(path: str, line: int, col: int) -> dict:
    """Return the location(s) defining the symbol under the cursor."""
    client = await _DEFAULT_POOL.client_for(path)
    if client is None:
        return {"path": path, "available": False, "locations": []}
    uri = _path_to_uri(path)
    await client.did_open(uri, _language_of(path), Path(path).read_text(encoding="utf-8", errors="replace"))
    locs = await client.get_definition(uri, line, col)
    return {"path": path, "available": True, "locations": [l.__dict__ for l in locs]}


@GLOBAL_REGISTRY.tool(
    description="LSP find-references.",
    side_effect="read_only",
    timeout=30.0,
)
async def lsp_references(path: str, line: int, col: int) -> dict:
    """Find every reference to the symbol under the cursor."""
    client = await _DEFAULT_POOL.client_for(path)
    if client is None:
        return {"path": path, "available": False, "locations": []}
    uri = _path_to_uri(path)
    await client.did_open(uri, _language_of(path), Path(path).read_text(encoding="utf-8", errors="replace"))
    locs = await client.get_references(uri, line, col)
    return {"path": path, "available": True, "locations": [l.__dict__ for l in locs]}


@GLOBAL_REGISTRY.tool(
    description="LSP code actions at a (line, col).",
    side_effect="read_only",
    timeout=30.0,
)
async def lsp_code_action(path: str, line: int, col: int) -> dict:
    """Return available code actions."""
    client = await _DEFAULT_POOL.client_for(path)
    if client is None:
        return {"path": path, "available": False, "actions": []}
    uri = _path_to_uri(path)
    await client.did_open(uri, _language_of(path), Path(path).read_text(encoding="utf-8", errors="replace"))
    actions = await client.get_code_actions(uri, (line, col), (line, col))
    return {"path": path, "available": True, "actions": actions}


def _language_of(path: str) -> str:
    """Map an extension to an LSP languageId."""
    ext = Path(path).suffix.lower()
    return {
        ".py": "python", ".ts": "typescript", ".tsx": "typescriptreact",
        ".js": "javascript", ".jsx": "javascriptreact",
        ".rs": "rust", ".go": "go", ".c": "c", ".h": "c", ".cpp": "cpp", ".cc": "cpp", ".hpp": "cpp",
    }.get(ext, "plaintext")
