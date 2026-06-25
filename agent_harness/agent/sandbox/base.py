"""Sandbox backend abstract interface + selector."""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class SandboxConfig:
    """Configuration shared by every backend."""

    backend: Literal["auto", "subprocess", "docker", "firejail"] = "auto"
    project_root: str = "."
    image: str | None = None
    cpu_quota: float = 1.0
    memory_limit: str = "2g"
    pids_limit: int = 256
    network_mode: Literal["none", "bridge", "host"] = "bridge"
    read_only_root: bool = False
    blocked_env: list[str] = field(default_factory=lambda: [
        "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "GITHUB_TOKEN",
        "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GITLAB_TOKEN", "NPM_TOKEN",
    ])
    extra_flags: list[str] = field(default_factory=list)


@dataclass
class ExecResult:
    """Outcome of a sandboxed command execution."""

    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: float
    timed_out: bool = False
    backend: str = ""


class SandboxBackend(ABC):
    """Abstract interface every sandbox backend implements."""

    name: str = "base"

    @abstractmethod
    async def setup(self, project_root: str, config: SandboxConfig) -> None:
        """One-time setup before any exec."""

    @abstractmethod
    async def exec(self, command: str, timeout: int = 60, env: dict[str, str] | None = None) -> ExecResult:
        """Run a single command inside the sandbox."""

    @abstractmethod
    async def copy_to(self, local_path: str, sandbox_path: str) -> None:
        """Copy a file from host into the sandbox."""

    @abstractmethod
    async def copy_from(self, sandbox_path: str, local_path: str) -> None:
        """Copy a file from sandbox back to host."""

    @abstractmethod
    async def teardown(self) -> None:
        """Shut down resources owned by the sandbox."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Return ``True`` if the sandbox is responsive."""


def select_sandbox_backend(config: SandboxConfig) -> SandboxBackend:
    """Pick the most appropriate available backend for the given config.

    Args:
        config: The :class:`SandboxConfig` whose ``backend`` field drives selection.

    Returns:
        A concrete :class:`SandboxBackend` instance. Falls back to subprocess if
        the requested backend isn't available on the host.
    """
    from agent.sandbox.docker import DockerSandbox
    from agent.sandbox.firejail import FirejailSandbox
    from agent.sandbox.subprocess_backend import SubprocessSandbox

    requested = config.backend
    if requested == "docker" and shutil.which("docker"):
        return DockerSandbox()
    if requested == "firejail" and shutil.which("firejail"):
        return FirejailSandbox()
    if requested == "auto":
        if shutil.which("docker"):
            return DockerSandbox()
        if shutil.which("firejail"):
            return FirejailSandbox()
    return SubprocessSandbox()


def detect_project_image(project_root: str) -> str:
    """Guess a sensible base Docker image from project marker files."""
    from pathlib import Path

    root = Path(project_root)
    if (root / "pyproject.toml").exists() or list(root.glob("requirements*.txt")):
        return "python:3.12-slim"
    if (root / "package.json").exists():
        return "node:20-slim"
    if (root / "Cargo.toml").exists():
        return "rust:1.78-slim"
    if (root / "go.mod").exists():
        return "golang:1.22-bookworm"
    return "ubuntu:24.04"
