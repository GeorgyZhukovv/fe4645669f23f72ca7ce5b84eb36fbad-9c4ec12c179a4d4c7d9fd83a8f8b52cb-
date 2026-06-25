"""Tests for the tool registry and the built-in tool implementations."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent.tools import default_registry
from agent.tools.bash import is_destructive
from agent.tools.files import _apply_unified_diff, diff_strings
from agent.tools.registry import ToolRegistry


async def test_registry_lists_default_tools() -> None:
    reg = default_registry()
    names = set(reg.names())
    expected = {
        "bash_exec", "file_read", "file_write", "file_search",
        "web_fetch", "code_lint", "test_runner", "git_ops",
    }
    assert expected.issubset(names), names

    schemas = reg.schemas()
    for s in schemas:
        assert "name" in s and "input_schema" in s
        assert s["input_schema"]["type"] == "object"


async def test_registry_decorator_records_metadata() -> None:
    reg = ToolRegistry()

    @reg.tool(description="echo back", side_effect="read_only", timeout=5.0)
    async def echo(message: str) -> str:
        return message

    spec = reg.get("echo")
    assert spec.side_effect == "read_only"
    assert spec.timeout == pytest.approx(5.0)
    schema = spec.schema
    assert schema["input_schema"]["properties"]["message"]["type"] == "string"
    assert schema["input_schema"]["required"] == ["message"]


async def test_registry_invoke_retries_then_succeeds() -> None:
    reg = ToolRegistry()
    state = {"n": 0}

    @reg.tool(description="flaky", side_effect="read_only", timeout=2.0)
    async def flaky() -> str:
        state["n"] += 1
        if state["n"] < 2:
            raise RuntimeError("boom")
        return "ok"

    result = await reg.invoke("flaky", retries=2, backoff=0.0)
    assert result.ok is True
    assert result.output == "ok"
    assert state["n"] == 2


async def test_registry_invoke_timeout_returns_failure() -> None:
    import asyncio as _asyncio

    reg = ToolRegistry()

    @reg.tool(description="slow", timeout=0.05)
    async def slow() -> str:
        await _asyncio.sleep(1.0)
        return "done"

    result = await reg.invoke("slow", retries=0)
    assert result.ok is False
    assert "TimeoutError" in (result.error or "")


async def test_bash_exec_captures_stdout(workdir: Path) -> None:
    reg = default_registry()
    res = await reg.invoke(
        "bash_exec",
        {"command": "echo hello && echo world 1>&2", "timeout": 5, "working_dir": str(workdir)},
    )
    assert res.ok
    assert "hello" in res.output["stdout"]
    assert "world" in res.output["stderr"]
    assert res.output["exit_code"] == 0


async def test_bash_exec_timeout(workdir: Path) -> None:
    reg = default_registry()
    res = await reg.invoke(
        "bash_exec",
        {"command": "sleep 5", "timeout": 1, "working_dir": str(workdir)},
    )
    assert res.ok is True
    assert res.output["timed_out"] is True


def test_is_destructive_patterns() -> None:
    assert is_destructive("rm -rf /tmp/x") is True
    assert is_destructive("git push origin main") is True
    assert is_destructive("echo hi > out.txt") is True
    assert is_destructive("ls -la") is False
    assert is_destructive("cat README.md") is False


async def test_file_read_and_write_round_trip(workdir: Path) -> None:
    reg = default_registry()
    target = workdir / "hello.txt"
    res = await reg.invoke(
        "file_write",
        {"path": str(target), "content": "first\nsecond\n", "mode": "overwrite"},
    )
    assert res.ok and res.output["applied"] is True
    read = await reg.invoke("file_read", {"path": str(target), "encoding": "utf-8"})
    assert read.ok
    assert read.output["content"] == "first\nsecond\n"
    assert read.output["is_binary"] is False
    append = await reg.invoke(
        "file_write", {"path": str(target), "content": "third\n", "mode": "append"}
    )
    assert append.ok
    after = await reg.invoke("file_read", {"path": str(target)})
    assert "third" in after.output["content"]


async def test_file_write_atomic_overwrite(workdir: Path) -> None:
    reg = default_registry()
    target = workdir / "subdir" / "atomic.txt"
    await reg.invoke(
        "file_write",
        {"path": str(target), "content": "A", "mode": "overwrite"},
    )
    await reg.invoke(
        "file_write",
        {"path": str(target), "content": "B", "mode": "overwrite"},
    )
    assert target.read_text() == "B"
    tmps = list((workdir / "subdir").glob("*.tmp"))
    assert tmps == []


def test_apply_unified_diff_replaces_line() -> None:
    original = "alpha\nbeta\ngamma\n"
    patch = diff_strings(original, "alpha\nBETA\ngamma\n", a_label="a", b_label="b")
    patched = _apply_unified_diff(original, patch)
    assert patched == "alpha\nBETA\ngamma\n"


async def test_file_write_patch_mode(workdir: Path) -> None:
    reg = default_registry()
    target = workdir / "patch.txt"
    target.write_text("alpha\nbeta\ngamma\n")
    diff = diff_strings("alpha\nbeta\ngamma\n", "alpha\nBETA\ngamma\n")
    res = await reg.invoke(
        "file_write", {"path": str(target), "content": diff, "mode": "patch"}
    )
    assert res.ok
    assert target.read_text() == "alpha\nBETA\ngamma\n"


async def test_file_search_python_fallback(workdir: Path) -> None:
    reg = default_registry()
    (workdir / "a.py").write_text("def foo():\n    return 1\n")
    (workdir / "b.py").write_text("def bar():\n    return foo()\n")
    res = await reg.invoke(
        "file_search",
        {"pattern": "foo", "path": str(workdir), "case_sensitive": True, "max_results": 10},
    )
    assert res.ok
    assert res.output["count"] >= 2
    matches = res.output["matches"]
    assert all("foo" in m["text"] for m in matches)


async def test_audit_log_records_calls(tmp_path: Path) -> None:
    from agent.audit import AuditLog

    reg = ToolRegistry(audit=AuditLog(tmp_path / "audit.jsonl"))

    @reg.tool(description="noop", side_effect="read_only", timeout=2.0)
    async def noop(x: int = 0) -> int:
        return x + 1

    await reg.invoke("noop", {"x": 41})
    tail = reg.audit.tail()
    assert tail[-1]["event"] == "tool_call"
    assert tail[-1]["payload"]["ok"] is True


async def test_git_ops_status_in_empty_repo(workdir: Path) -> None:
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=workdir, check=False)
    subprocess.run(["git", "config", "user.email", "x@x"], cwd=workdir, check=False)
    subprocess.run(["git", "config", "user.name", "x"], cwd=workdir, check=False)
    reg = default_registry()
    res = await reg.invoke("git_ops", {"operation": "status", "args": {}, "working_dir": str(workdir)})
    assert res.ok
    assert "data" in res.output


async def test_test_runner_returns_note_when_framework_missing(workdir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reg = default_registry()
    monkeypatch.setenv("PATH", "")  # hide all binaries
    res = await reg.invoke(
        "test_runner", {"path": str(workdir), "framework": "jest"}
    )
    assert res.ok
    assert "note" in res.output


async def test_code_lint_returns_note_when_unavailable(workdir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reg = default_registry()
    monkeypatch.setenv("PATH", "")
    target = workdir / "x.rs"
    target.write_text("fn main() {}\n")
    res = await reg.invoke("code_lint", {"path": str(target), "language": "rust"})
    assert res.ok
    assert res.output["ok"] is True
