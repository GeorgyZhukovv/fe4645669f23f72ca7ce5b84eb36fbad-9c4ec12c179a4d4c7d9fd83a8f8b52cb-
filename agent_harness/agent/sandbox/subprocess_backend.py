"""Subprocess sandbox backend (default, no isolation beyond env masking)."""

from __future__ import annotations

import asyncio
import os
import resource
import shutil
import signal
import time
from pathlib import Path

from agent.sandbox.base import ExecResult, SandboxBackend, SandboxConfig


class SubprocessSandbox(SandboxBackend):
    """Direct subprocess execution with ulimit + env masking + process-group kill."""

    name = "subprocess"

    def __init__(self) -> None:
        """No persistent state required for this backend."""
        self._config: SandboxConfig | None = None
        self._project_root: str = "."

    async def setup(self, project_root: str, config: SandboxConfig) -> None:
        self._config = config
        self._project_root = project_root

    async def exec(self, command: str, timeout: int = 60, env: dict[str, str] | None = None) -> ExecResult:
        cfg = self._config or SandboxConfig()
        merged_env = dict(os.environ)
        if env:
            merged_env.update(env)
        for key in cfg.blocked_env:
            merged_env.pop(key, None)

        def _preexec() -> None:  # pragma: no cover - platform-specific
            os.setpgrp()
            mem_bytes = _parse_mem(cfg.memory_limit)
            if mem_bytes:
                try:
                    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
                except (ValueError, OSError):
                    pass
            try:
                resource.setrlimit(resource.RLIMIT_CPU, (int(timeout) + 5, int(timeout) + 5))
            except (ValueError, OSError):
                pass
            if cfg.pids_limit:
                try:
                    resource.setrlimit(resource.RLIMIT_NPROC, (cfg.pids_limit, cfg.pids_limit))
                except (ValueError, OSError):
                    pass

        start = time.perf_counter()
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self._project_root,
            env=merged_env,
            preexec_fn=_preexec if hasattr(os, "setpgrp") else None,
        )
        timed_out = False
        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            timed_out = True
            if proc.pid is not None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except OSError:
                    proc.kill()
            try:
                stdout_b, stderr_b = await proc.communicate()
            except Exception:
                stdout_b, stderr_b = b"", b""
        duration = (time.perf_counter() - start) * 1000.0
        return ExecResult(
            command=command,
            exit_code=proc.returncode if proc.returncode is not None else -1,
            stdout=stdout_b.decode("utf-8", "replace"),
            stderr=stderr_b.decode("utf-8", "replace"),
            duration_ms=duration,
            timed_out=timed_out,
            backend=self.name,
        )

    async def copy_to(self, local_path: str, sandbox_path: str) -> None:
        Path(sandbox_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, sandbox_path)

    async def copy_from(self, sandbox_path: str, local_path: str) -> None:
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sandbox_path, local_path)

    async def teardown(self) -> None:
        return None

    async def health_check(self) -> bool:
        return True


def _parse_mem(spec: str) -> int:
    """Parse ``2g`` / ``512m`` / ``1024k`` / ``1048576`` into bytes."""
    if not spec:
        return 0
    spec = spec.strip().lower()
    if spec.endswith("k"):
        return int(spec[:-1]) * 1024
    if spec.endswith("m"):
        return int(spec[:-1]) * 1024 * 1024
    if spec.endswith("g"):
        return int(spec[:-1]) * 1024 * 1024 * 1024
    try:
        return int(spec)
    except ValueError:
        return 0
