"""Bash execution tool with timeout enforcement and structured output."""

from __future__ import annotations

import asyncio
import os
import re
import shlex
from dataclasses import dataclass

from agent.tools.registry import GLOBAL_REGISTRY


DESTRUCTIVE_RE = re.compile(
    r"(\brm\b\s+-[rR]?f|\bmv\b|\bchmod\b|\bchown\b|\b(dd|mkfs|fdisk)\b|"
    r"\bpip(3)?\s+install|\bnpm\s+install|\bcurl\s+[^\|]*\|\s*(sh|bash)|"
    r">\s*[\w./]+|>>\s*[\w./]+|\bgit\s+push\b|\bgit\s+reset\s+--hard\b)"
)


@dataclass
class BashResult:
    """Structured outcome of a shell command invocation."""

    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: float
    timed_out: bool = False

    def is_destructive(self) -> bool:
        """Return ``True`` if this command matched the destructive-pattern regex."""
        return is_destructive(self.command)


def is_destructive(command: str) -> bool:
    """Cheap regex check for destructive shell commands."""
    return bool(DESTRUCTIVE_RE.search(command))


_ACTIVE_SANDBOX = None  # type: ignore[var-annotated]


def set_active_sandbox(sandbox) -> None:  # type: ignore[no-untyped-def]
    """Install a :class:`agent.sandbox.base.SandboxBackend` for all ``bash_exec`` calls."""
    global _ACTIVE_SANDBOX
    _ACTIVE_SANDBOX = sandbox


def get_active_sandbox():  # type: ignore[no-untyped-def]
    """Return the currently installed sandbox (or ``None`` for direct subprocess)."""
    return _ACTIVE_SANDBOX


@GLOBAL_REGISTRY.tool(
    description=(
        "Execute a shell command in a subprocess and capture stdout, stderr, and exit code. "
        "Supports a hard timeout (seconds). Use working_dir to set the CWD. If a sandbox "
        "backend is active for the session, execution is routed through it."
    ),
    side_effect="destructive",
    timeout=180.0,
)
async def bash_exec(command: str, timeout: int = 60, working_dir: str = ".") -> dict:
    """Execute ``command`` with a hard timeout, returning a structured result dict.

    Args:
        command: The shell command to execute. Will be run via ``/bin/sh -c``.
        timeout: Hard timeout in seconds.
        working_dir: Working directory passed to the subprocess (ignored when a
            sandbox backend is active).

    Returns:
        A serialisable dict with ``exit_code``, ``stdout``, ``stderr``,
        ``duration_ms``, and ``timed_out``.
    """
    import time

    if _ACTIVE_SANDBOX is not None:
        res = await _ACTIVE_SANDBOX.exec(command, timeout=timeout)
        return {
            "command": res.command,
            "exit_code": res.exit_code,
            "stdout": res.stdout[-8000:],
            "stderr": res.stderr[-4000:],
            "duration_ms": res.duration_ms,
            "timed_out": res.timed_out,
            "destructive": is_destructive(command),
            "backend": res.backend,
        }

    cwd = os.path.abspath(working_dir or ".")
    start = time.perf_counter()
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
    )
    timed_out = False
    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        timed_out = True
        proc.kill()
        try:
            stdout_b, stderr_b = await proc.communicate()
        except Exception:
            stdout_b, stderr_b = b"", b""
    duration_ms = (time.perf_counter() - start) * 1000.0
    result = BashResult(
        command=command,
        exit_code=proc.returncode if proc.returncode is not None else -1,
        stdout=stdout_b.decode("utf-8", errors="replace"),
        stderr=stderr_b.decode("utf-8", errors="replace"),
        duration_ms=duration_ms,
        timed_out=timed_out,
    )
    return {
        "command": result.command,
        "exit_code": result.exit_code,
        "stdout": result.stdout[-8000:],
        "stderr": result.stderr[-4000:],
        "duration_ms": result.duration_ms,
        "timed_out": result.timed_out,
        "destructive": result.is_destructive(),
        "backend": "subprocess",
    }


def safe_split(command: str) -> list[str]:
    """Best-effort argv split for display, falling back to a one-element list."""
    try:
        return shlex.split(command)
    except ValueError:
        return [command]
