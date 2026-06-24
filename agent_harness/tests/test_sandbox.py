"""Tests for sandbox backend selection + subprocess backend."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.sandbox.base import SandboxConfig, detect_project_image, select_sandbox_backend
from agent.sandbox.subprocess_backend import SubprocessSandbox, _parse_mem


def test_select_backend_default_is_subprocess(monkeypatch) -> None:
    cfg = SandboxConfig(backend="subprocess")
    backend = select_sandbox_backend(cfg)
    assert backend.name == "subprocess"


def test_detect_project_image(workdir: Path) -> None:
    assert detect_project_image(str(workdir)) == "ubuntu:24.04"
    (workdir / "pyproject.toml").write_text("[project]\nname='x'\n")
    assert detect_project_image(str(workdir)) == "python:3.12-slim"
    (workdir / "pyproject.toml").unlink()
    (workdir / "package.json").write_text("{}")
    assert detect_project_image(str(workdir)) == "node:20-slim"


def test_parse_mem() -> None:
    assert _parse_mem("1g") == 1024 * 1024 * 1024
    assert _parse_mem("256m") == 256 * 1024 * 1024
    assert _parse_mem("4096k") == 4096 * 1024
    assert _parse_mem("1024") == 1024
    assert _parse_mem("") == 0


@pytest.mark.asyncio
async def test_subprocess_exec_captures(workdir: Path) -> None:
    cfg = SandboxConfig(backend="subprocess")
    sandbox = SubprocessSandbox()
    await sandbox.setup(str(workdir), cfg)
    result = await sandbox.exec("echo hello && echo bye 1>&2")
    assert result.exit_code == 0
    assert "hello" in result.stdout
    assert "bye" in result.stderr
    await sandbox.teardown()


@pytest.mark.asyncio
async def test_subprocess_blocks_sensitive_env(workdir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "should-not-leak")
    monkeypatch.setenv("SOMETHING_BENIGN", "ok")
    cfg = SandboxConfig(backend="subprocess")
    sandbox = SubprocessSandbox()
    await sandbox.setup(str(workdir), cfg)
    res = await sandbox.exec("env | grep -E 'ANTHROPIC_API_KEY|SOMETHING_BENIGN' || true")
    assert "ANTHROPIC_API_KEY" not in res.stdout
    assert "SOMETHING_BENIGN=ok" in res.stdout
    await sandbox.teardown()


@pytest.mark.asyncio
async def test_subprocess_timeout(workdir: Path) -> None:
    cfg = SandboxConfig(backend="subprocess")
    sandbox = SubprocessSandbox()
    await sandbox.setup(str(workdir), cfg)
    res = await sandbox.exec("sleep 5", timeout=1)
    assert res.timed_out is True
    await sandbox.teardown()


def test_health_check_subprocess() -> None:
    import asyncio
    sandbox = SubprocessSandbox()
    assert asyncio.run(sandbox.health_check()) is True
