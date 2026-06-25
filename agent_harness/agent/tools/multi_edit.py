"""Multi-file edit engine: atomic, rollback-safe, batched filesystem changes."""

from __future__ import annotations

import asyncio
import difflib
import hashlib
import os
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from agent.tools.bash import bash_exec
from agent.tools.files import _apply_unified_diff
from agent.tools.registry import GLOBAL_REGISTRY


OpKind = Literal["overwrite", "patch", "create", "delete", "rename", "mkdir"]


class EditOperation(BaseModel):
    """One filesystem operation inside an :class:`EditPlan`."""

    kind: OpKind
    path: str
    content: str | None = None
    dst: str | None = None


class EditPlan(BaseModel):
    """A coordinated multi-file change with snapshot-based atomic rollback."""

    edit_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    description: str
    operations: list[EditOperation]
    rollback_snapshot: dict[str, str] = Field(default_factory=dict)
    verification_commands: list[str] = Field(default_factory=list)
    status: Literal["pending", "applied", "verified", "rolled_back", "failed"] = "pending"

    def affected_paths(self) -> set[str]:
        """Return every path the plan reads or writes."""
        out: set[str] = set()
        for op in self.operations:
            out.add(op.path)
            if op.dst:
                out.add(op.dst)
        return out

    def is_high_risk(self) -> bool:
        """``True`` if any operation is a delete or rename."""
        return any(op.kind in {"delete", "rename"} for op in self.operations)


@dataclass
class EditResult:
    """Outcome of applying an :class:`EditPlan`."""

    edit_id: str
    status: str
    applied_ops: list[EditOperation] = field(default_factory=list)
    verification_outputs: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


def _snapshot_path(path: Path) -> str:
    """Read the file at ``path`` as base64 (utf-8 storage of bytes hex), or empty if absent."""
    if not path.exists():
        return ""
    return path.read_bytes().hex()


def _restore_path(path: Path, snapshot_hex: str) -> None:
    """Restore ``path`` from a hex-encoded snapshot (empty string → delete)."""
    if not snapshot_hex:
        if path.exists():
            path.unlink()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes.fromhex(snapshot_hex))


def _trash_dir() -> Path:
    """Per-session trash directory for soft-deletes."""
    base = Path(".agent_trash") / time.strftime("%Y%m%d-%H%M%S")
    base.mkdir(parents=True, exist_ok=True)
    return base


def _atomic_write(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` via temp-file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _take_snapshots(plan: EditPlan) -> None:
    """Populate ``plan.rollback_snapshot`` with hex bytes for every affected file."""
    snap: dict[str, str] = {}
    for p in plan.affected_paths():
        snap[p] = _snapshot_path(Path(p))
    plan.rollback_snapshot = snap


def _apply_one(op: EditOperation) -> None:
    """Execute a single operation. Raises on failure (caller rolls back)."""
    path = Path(op.path)
    if op.kind == "overwrite":
        if op.content is None:
            raise ValueError(f"overwrite requires content for {op.path}")
        _atomic_write(path, op.content)
    elif op.kind == "create":
        if path.exists():
            raise FileExistsError(f"create target already exists: {op.path}")
        if op.content is None:
            raise ValueError(f"create requires content for {op.path}")
        _atomic_write(path, op.content)
    elif op.kind == "patch":
        if op.content is None:
            raise ValueError(f"patch requires content (diff) for {op.path}")
        original = path.read_text(encoding="utf-8") if path.exists() else ""
        patched = _apply_unified_diff(original, op.content)
        _atomic_write(path, patched)
    elif op.kind == "delete":
        if not path.exists():
            return
        trash = _trash_dir() / path.name
        # Avoid name collisions by appending the hash of the original path.
        trash = trash.with_name(trash.name + "." + hashlib.sha1(op.path.encode()).hexdigest()[:8])
        shutil.move(str(path), str(trash))
    elif op.kind == "rename":
        if op.dst is None:
            raise ValueError(f"rename requires dst for {op.path}")
        dst = Path(op.dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(path, dst)
    elif op.kind == "mkdir":
        path.mkdir(parents=True, exist_ok=True)
    else:
        raise ValueError(f"unknown op kind: {op.kind}")


def rollback_edit_plan(plan: EditPlan) -> None:
    """Restore every snapshotted path. Safe to call multiple times."""
    for path_str, snap in plan.rollback_snapshot.items():
        _restore_path(Path(path_str), snap)
    plan.status = "rolled_back"


async def apply_edit_plan(plan: EditPlan) -> EditResult:
    """Snapshot, apply, verify; rollback on any failure.

    Args:
        plan: The plan to execute.

    Returns:
        An :class:`EditResult` capturing applied ops and verification output.
    """
    _take_snapshots(plan)
    applied: list[EditOperation] = []
    try:
        for op in plan.operations:
            _apply_one(op)
            applied.append(op)
        plan.status = "applied"
    except Exception as exc:
        rollback_edit_plan(plan)
        return EditResult(
            edit_id=plan.edit_id,
            status="rolled_back",
            applied_ops=applied,
            error=f"{type(exc).__name__}: {exc}",
        )

    verification_outputs: list[dict[str, Any]] = []
    for cmd in plan.verification_commands:
        out = await bash_exec(command=cmd, timeout=180, working_dir=".")
        verification_outputs.append(out)
        if out["exit_code"] != 0:
            rollback_edit_plan(plan)
            return EditResult(
                edit_id=plan.edit_id,
                status="rolled_back",
                applied_ops=applied,
                verification_outputs=verification_outputs,
                error=f"verification failed: {cmd!r}",
            )
    plan.status = "verified"
    return EditResult(
        edit_id=plan.edit_id,
        status="verified",
        applied_ops=applied,
        verification_outputs=verification_outputs,
    )


def render_plan_diff(plan: EditPlan) -> str:
    """Render the plan as a single unified diff suitable for terminal preview."""
    chunks: list[str] = []
    for op in plan.operations:
        path = Path(op.path)
        if op.kind == "overwrite":
            before = path.read_text(encoding="utf-8") if path.exists() else ""
            chunks.append("".join(difflib.unified_diff(
                before.splitlines(keepends=True),
                (op.content or "").splitlines(keepends=True),
                fromfile=f"a/{op.path}",
                tofile=f"b/{op.path}",
            )))
        elif op.kind == "create":
            chunks.append("".join(difflib.unified_diff(
                [],
                (op.content or "").splitlines(keepends=True),
                fromfile="/dev/null",
                tofile=f"b/{op.path}",
            )))
        elif op.kind == "patch":
            chunks.append(op.content or "")
        elif op.kind == "delete":
            before = path.read_text(encoding="utf-8") if path.exists() else ""
            chunks.append("".join(difflib.unified_diff(
                before.splitlines(keepends=True),
                [],
                fromfile=f"a/{op.path}",
                tofile="/dev/null",
            )))
        elif op.kind == "rename":
            chunks.append(f"rename {op.path} -> {op.dst}\n")
        elif op.kind == "mkdir":
            chunks.append(f"mkdir {op.path}\n")
    return "\n".join(c for c in chunks if c)


@GLOBAL_REGISTRY.tool(
    description=(
        "Apply a coordinated multi-file edit atomically. Takes a JSON-encoded EditPlan "
        "(description, operations[], verification_commands[]). Snapshots every affected file "
        "before mutation; rolls back on any operation or verification failure."
    ),
    side_effect="destructive",
    timeout=300.0,
)
async def multi_edit(
    description: str,
    operations: list[dict[str, Any]],
    verification_commands: list[str] | None = None,
) -> dict[str, Any]:
    """Apply an EditPlan built from a list of operation dicts.

    Args:
        description: Short human-readable summary.
        operations: List of dicts with ``kind`` and operation-specific keys, or a
            JSON-encoded string of the same (auto-decoded when callers — e.g. an
            MCP client — pass the value as text).
        verification_commands: Optional list of shell commands run after all ops
            succeed. Also accepts a JSON-encoded string.

    Returns:
        A dict-ified :class:`EditResult` with status and per-op artifacts.
    """
    import json as _json

    if isinstance(operations, str):
        try:
            operations = _json.loads(operations)
        except _json.JSONDecodeError as exc:
            return {
                "edit_id": "",
                "status": "failed",
                "applied_ops": [],
                "verification_outputs": [],
                "error": f"operations was a string but not valid JSON: {exc}",
            }
    if isinstance(verification_commands, str):
        try:
            verification_commands = _json.loads(verification_commands)
        except _json.JSONDecodeError:
            verification_commands = [verification_commands]
    if not isinstance(operations, list):
        return {
            "edit_id": "",
            "status": "failed",
            "applied_ops": [],
            "verification_outputs": [],
            "error": f"operations must be a list, got {type(operations).__name__}",
        }
    ops = [EditOperation(**op) if isinstance(op, dict) else EditOperation(**_json.loads(op)) for op in operations]
    plan = EditPlan(
        description=description,
        operations=ops,
        verification_commands=verification_commands or [],
    )
    result = await apply_edit_plan(plan)
    return {
        "edit_id": result.edit_id,
        "status": result.status,
        "applied_ops": [op.model_dump() for op in result.applied_ops],
        "verification_outputs": result.verification_outputs,
        "error": result.error,
    }
