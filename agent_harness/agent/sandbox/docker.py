"""Docker sandbox backend."""

from __future__ import annotations

import asyncio
import os
import shutil
import time
import uuid
from pathlib import Path

from agent.sandbox.base import ExecResult, SandboxBackend, SandboxConfig, detect_project_image


async def _run(args: list[str], timeout: int = 60) -> tuple[int, str, str]:
    """Run a host command and capture (rc, stdout, stderr)."""
    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        out, err = await proc.communicate()
    return proc.returncode or 0, out.decode("utf-8", "replace"), err.decode("utf-8", "replace")


class DockerSandbox(SandboxBackend):
    """Long-lived Docker container reused across calls within a session."""

    name = "docker"

    def __init__(self) -> None:
        self._container_id: str | None = None
        self._image: str = "ubuntu:24.04"
        self._project_root: str = "."
        self._config: SandboxConfig | None = None

    async def setup(self, project_root: str, config: SandboxConfig) -> None:
        if not shutil.which("docker"):
            raise RuntimeError("docker not available on PATH")
        self._project_root = str(Path(project_root).resolve())
        self._config = config
        self._image = config.image or detect_project_image(self._project_root)
        await _run(["docker", "pull", self._image], timeout=600)
        name = f"agent-sandbox-{uuid.uuid4().hex[:8]}"
        args = [
            "docker", "run", "-d", "--rm", "--name", name,
            "--cpus", str(config.cpu_quota),
            "--memory", config.memory_limit,
            "--pids-limit", str(config.pids_limit),
            "--network", config.network_mode,
            "-v", f"{self._project_root}:/workspace/src:ro",
            "-w", "/workspace/src",
        ]
        if config.read_only_root:
            args.append("--read-only")
        args.extend(config.extra_flags)
        args.extend([self._image, "sleep", "infinity"])
        rc, out, err = await _run(args, timeout=60)
        if rc != 0:
            raise RuntimeError(f"docker run failed: {err.strip()}")
        self._container_id = out.strip().splitlines()[-1].strip()
        await _run(["docker", "exec", self._container_id, "mkdir", "-p", "/workspace/out"])

    async def exec(self, command: str, timeout: int = 60, env: dict[str, str] | None = None) -> ExecResult:
        if self._container_id is None:
            raise RuntimeError("sandbox not set up")
        cfg = self._config or SandboxConfig()
        args = ["docker", "exec"]
        merged_env = dict(env or {})
        for key in cfg.blocked_env:
            merged_env.pop(key, None)
        for k, v in merged_env.items():
            args.extend(["-e", f"{k}={v}"])
        args.extend([self._container_id, "/bin/sh", "-c", command])
        start = time.perf_counter()
        rc, out, err = await _run(args, timeout=timeout + 5)
        duration = (time.perf_counter() - start) * 1000.0
        timed_out = duration / 1000.0 > timeout + 1
        return ExecResult(
            command=command,
            exit_code=rc,
            stdout=out,
            stderr=err,
            duration_ms=duration,
            timed_out=timed_out,
            backend=self.name,
        )

    async def copy_to(self, local_path: str, sandbox_path: str) -> None:
        if self._container_id is None:
            raise RuntimeError("sandbox not set up")
        await _run(["docker", "cp", local_path, f"{self._container_id}:{sandbox_path}"])

    async def copy_from(self, sandbox_path: str, local_path: str) -> None:
        if self._container_id is None:
            raise RuntimeError("sandbox not set up")
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        await _run(["docker", "cp", f"{self._container_id}:{sandbox_path}", local_path])

    async def teardown(self) -> None:
        if self._container_id is None:
            return
        # Sync the writable overlay back to the host, respecting .gitignore.
        out_host = Path(self._project_root) / ".agent_state" / "sandbox_out"
        out_host.mkdir(parents=True, exist_ok=True)
        await _run(["docker", "cp", f"{self._container_id}:/workspace/out/.", str(out_host)])
        await _run(["docker", "stop", "-t", "1", self._container_id])
        self._container_id = None

    async def health_check(self) -> bool:
        if self._container_id is None:
            return False
        rc, _out, _err = await _run(["docker", "inspect", "-f", "{{.State.Running}}", self._container_id])
        return rc == 0
