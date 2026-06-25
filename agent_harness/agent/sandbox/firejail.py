"""firejail sandbox backend (Linux)."""

from __future__ import annotations

import asyncio
import os
import shutil
import time
from pathlib import Path

from agent.sandbox.base import ExecResult, SandboxBackend, SandboxConfig


class FirejailSandbox(SandboxBackend):
    """Wrap each command in ``firejail --noprofile --noroot --private-tmp ...``."""

    name = "firejail"

    def __init__(self) -> None:
        self._project_root: str = "."
        self._config: SandboxConfig | None = None

    async def setup(self, project_root: str, config: SandboxConfig) -> None:
        if not shutil.which("firejail"):
            raise RuntimeError("firejail not available on PATH")
        self._project_root = str(Path(project_root).resolve())
        self._config = config

    async def exec(self, command: str, timeout: int = 60, env: dict[str, str] | None = None) -> ExecResult:
        cfg = self._config or SandboxConfig()
        merged_env = dict(os.environ)
        if env:
            merged_env.update(env)
        for key in cfg.blocked_env:
            merged_env.pop(key, None)
        net_flag = {"none": "--net=none", "bridge": "--net=br0", "host": ""}.get(cfg.network_mode, "--net=none")
        args = ["firejail", "--quiet", "--noprofile", "--noroot", "--private-tmp", "--nosound"]
        if net_flag:
            args.append(net_flag)
        args.extend(cfg.extra_flags)
        args.extend(["/bin/sh", "-c", command])
        start = time.perf_counter()
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self._project_root,
            env=merged_env,
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
        return shutil.which("firejail") is not None
