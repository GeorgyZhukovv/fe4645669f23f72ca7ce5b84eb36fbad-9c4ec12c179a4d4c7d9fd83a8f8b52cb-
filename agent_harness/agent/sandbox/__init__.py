"""Pluggable sandboxed execution backends."""

from agent.sandbox.base import ExecResult, SandboxBackend, SandboxConfig, select_sandbox_backend
from agent.sandbox.docker import DockerSandbox
from agent.sandbox.firejail import FirejailSandbox
from agent.sandbox.subprocess_backend import SubprocessSandbox

__all__ = [
    "DockerSandbox",
    "ExecResult",
    "FirejailSandbox",
    "SandboxBackend",
    "SandboxConfig",
    "SubprocessSandbox",
    "select_sandbox_backend",
]
