"""Tests for the LSP client + tools (using a fake transport — no real LSP server)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from agent.tools import lsp


def test_detect_language_servers_returns_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lsp.shutil, "which", lambda _name: None)
    out = lsp.detect_language_servers(".")
    assert out == {}


def test_path_to_uri() -> None:
    assert lsp._path_to_uri(__file__).startswith("file://")


def test_language_of_known_extensions() -> None:
    assert lsp._language_of("/tmp/x.py") == "python"
    assert lsp._language_of("a.ts") == "typescript"
    assert lsp._language_of("a.go") == "go"
    assert lsp._language_of("readme.md") == "plaintext"


def test_locations_normaliser_handles_dict_and_list() -> None:
    single = {"uri": "file:///x", "range": {"start": {"line": 3, "character": 4}}}
    out = lsp._locations(single)
    assert out[0].line == 3 and out[0].column == 4
    multi = [{"targetUri": "file:///y", "targetRange": {"start": {"line": 1, "character": 0}}}]
    out2 = lsp._locations(multi)
    assert out2[0].uri == "file:///y"


@pytest.mark.asyncio
async def test_lsp_diagnostics_unavailable_when_no_server(workdir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Re-init the pool without any servers detected.
    monkeypatch.setattr(lsp.shutil, "which", lambda _name: None)
    lsp._DEFAULT_POOL = lsp.LSPServerPool(workdir)
    target = workdir / "x.py"
    target.write_text("x = 1\n")
    result = await lsp.lsp_diagnostics(str(target))
    assert result["available"] is False


@pytest.mark.asyncio
async def test_lsp_client_request_response_with_fake_server(workdir: Path) -> None:
    """Drive the LSPClient against a tiny fake JSON-RPC echo server (cat-based)."""

    class _FakeServer:
        """A minimal server that responds to every request with a canned reply."""

        def __init__(self) -> None:
            self.stdin_w = asyncio.StreamReader()
            self.stdout_r = asyncio.StreamReader()

    # Instead of building a fake subprocess, exercise _dispatch directly.
    cfg = lsp.LSPServerConfig(name="fake", command=["true"], languages=["python"], file_extensions=[".py"])
    client = lsp.LSPClient(cfg)
    # Inject a result by hand: _pending future + dispatch
    fut: asyncio.Future = asyncio.get_event_loop().create_future()
    client._pending[7] = fut
    await client._dispatch({"id": 7, "result": {"ok": True}})
    assert (await fut) == {"ok": True}

    # Diagnostics push notification
    await client._dispatch({
        "method": "textDocument/publishDiagnostics",
        "params": {
            "uri": "file:///x.py",
            "diagnostics": [
                {"severity": 1, "message": "boom",
                 "code": "E001", "range": {"start": {"line": 2, "character": 0}}},
            ],
        },
    })
    diags = client._diagnostics["file:///x.py"]
    assert diags[0].message == "boom"
    assert diags[0].line == 2
