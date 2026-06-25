"""Git operations tool with structured output."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Literal

from agent.tools.registry import GLOBAL_REGISTRY


@dataclass
class GitResult:
    """Structured response for a git operation."""

    operation: str
    ok: bool
    data: dict[str, Any]
    stderr: str = ""


async def _git(args: list[str], cwd: str | None = None) -> tuple[int, str, str]:
    """Run a ``git`` command and capture (exit_code, stdout, stderr)."""
    proc = await asyncio.create_subprocess_exec(
        "git",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=120)
    except TimeoutError:
        proc.kill()
        out, err = await proc.communicate()
    return proc.returncode or 0, out.decode("utf-8", "replace"), err.decode("utf-8", "replace")


def _parse_status(stdout: str) -> dict[str, Any]:
    """Parse ``git status --porcelain=v1`` output."""
    entries: list[dict[str, str]] = []
    for line in stdout.splitlines():
        if len(line) < 3:
            continue
        xy, _, path = line.partition(" ")
        entries.append({"status": xy.strip(), "path": path.strip()})
    return {"entries": entries, "clean": not entries}


def _parse_log(stdout: str) -> dict[str, Any]:
    """Parse ``git log`` with a fixed pretty format."""
    commits = []
    for chunk in stdout.split("\x1e"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split("\x1f")
        if len(parts) < 4:
            continue
        commits.append(
            {"hash": parts[0], "author": parts[1], "date": parts[2], "subject": parts[3]}
        )
    return {"commits": commits}


@GLOBAL_REGISTRY.tool(
    description=(
        "Wrap common git operations. Supported: status, diff, commit, branch, log. "
        "`args` is a dict of operation-specific parameters."
    ),
    side_effect="destructive",
    timeout=120.0,
)
async def git_ops(
    operation: Literal["status", "diff", "commit", "branch", "log"],
    args: dict | None = None,
    working_dir: str = ".",
) -> dict:
    """Run a single git operation.

    Args:
        operation: One of ``status``, ``diff``, ``commit``, ``branch``, ``log``.
        args: Operation-specific arguments (see code for keys).
        working_dir: CWD passed to git.
    """
    args = args or {}
    cwd = working_dir
    if operation == "status":
        code, out, err = await _git(["status", "--porcelain=v1"], cwd=cwd)
        return GitResult(
            operation="status",
            ok=code == 0,
            data=_parse_status(out),
            stderr=err,
        ).__dict__
    if operation == "diff":
        cmd = ["diff"]
        if args.get("cached"):
            cmd.append("--cached")
        if "paths" in args:
            cmd += ["--", *args["paths"]]
        code, out, err = await _git(cmd, cwd=cwd)
        return GitResult(
            operation="diff",
            ok=code == 0,
            data={"patch": out[:200_000]},
            stderr=err,
        ).__dict__
    if operation == "commit":
        message = args.get("message")
        if not message:
            return GitResult(operation="commit", ok=False, data={}, stderr="message required").__dict__
        files = args.get("files")
        if files:
            await _git(["add", *files], cwd=cwd)
        elif args.get("all"):
            await _git(["add", "-A"], cwd=cwd)
        code, out, err = await _git(["commit", "-m", message], cwd=cwd)
        return GitResult(operation="commit", ok=code == 0, data={"stdout": out}, stderr=err).__dict__
    if operation == "branch":
        sub = args.get("sub", "list")
        if sub == "list":
            code, out, err = await _git(["branch", "--all"], cwd=cwd)
            branches = [ln.strip().lstrip("* ") for ln in out.splitlines() if ln.strip()]
            return GitResult(operation="branch", ok=code == 0, data={"branches": branches}, stderr=err).__dict__
        if sub == "create":
            code, out, err = await _git(["checkout", "-b", args["name"]], cwd=cwd)
            return GitResult(operation="branch", ok=code == 0, data={"created": args["name"]}, stderr=err).__dict__
        if sub == "switch":
            code, out, err = await _git(["checkout", args["name"]], cwd=cwd)
            return GitResult(operation="branch", ok=code == 0, data={"switched": args["name"]}, stderr=err).__dict__
        return GitResult(operation="branch", ok=False, data={}, stderr=f"unknown sub {sub}").__dict__
    if operation == "log":
        limit = args.get("limit", 10)
        fmt = "--pretty=format:%H%x1f%an%x1f%ad%x1f%s%x1e"
        code, out, err = await _git(["log", f"-n{limit}", fmt], cwd=cwd)
        return GitResult(operation="log", ok=code == 0, data=_parse_log(out), stderr=err).__dict__
    return GitResult(operation=operation, ok=False, data={}, stderr=f"unknown op {operation}").__dict__
