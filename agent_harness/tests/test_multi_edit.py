"""Tests for the multi-file edit engine."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.tools.multi_edit import (
    EditOperation,
    EditPlan,
    apply_edit_plan,
    render_plan_diff,
    rollback_edit_plan,
)


@pytest.mark.asyncio
async def test_apply_overwrite_and_create(workdir: Path) -> None:
    plan = EditPlan(
        description="overwrite + create",
        operations=[
            EditOperation(kind="overwrite", path=str(workdir / "a.txt"), content="A1\n"),
            EditOperation(kind="create", path=str(workdir / "b.txt"), content="B1\n"),
        ],
    )
    result = await apply_edit_plan(plan)
    assert result.status == "verified"
    assert (workdir / "a.txt").read_text() == "A1\n"
    assert (workdir / "b.txt").read_text() == "B1\n"


@pytest.mark.asyncio
async def test_rollback_on_failed_op(workdir: Path) -> None:
    target = workdir / "x.txt"
    target.write_text("before\n")
    plan = EditPlan(
        description="rollback test",
        operations=[
            EditOperation(kind="overwrite", path=str(target), content="after\n"),
            # 'create' must fail because the file already exists.
            EditOperation(kind="create", path=str(target), content="conflict\n"),
        ],
    )
    result = await apply_edit_plan(plan)
    assert result.status == "rolled_back"
    assert target.read_text() == "before\n"


@pytest.mark.asyncio
async def test_rollback_on_failed_verification(workdir: Path) -> None:
    target = workdir / "verify.txt"
    target.write_text("ok\n")
    plan = EditPlan(
        description="verify fail",
        operations=[EditOperation(kind="overwrite", path=str(target), content="changed\n")],
        verification_commands=["false"],
    )
    result = await apply_edit_plan(plan)
    assert result.status == "rolled_back"
    assert target.read_text() == "ok\n"


@pytest.mark.asyncio
async def test_delete_moves_to_trash(workdir: Path) -> None:
    target = workdir / "doomed.txt"
    target.write_text("bye\n")
    plan = EditPlan(
        description="delete",
        operations=[EditOperation(kind="delete", path=str(target))],
    )
    result = await apply_edit_plan(plan)
    assert result.status == "verified"
    assert not target.exists()
    trash_dirs = list((workdir / ".agent_trash").glob("*"))
    assert trash_dirs, "trash dir should be created"


@pytest.mark.asyncio
async def test_rename_and_mkdir(workdir: Path) -> None:
    src = workdir / "src.txt"
    src.write_text("hi\n")
    plan = EditPlan(
        description="mv + mkdir",
        operations=[
            EditOperation(kind="mkdir", path=str(workdir / "sub")),
            EditOperation(kind="rename", path=str(src), dst=str(workdir / "sub" / "dst.txt")),
        ],
    )
    result = await apply_edit_plan(plan)
    assert result.status == "verified"
    assert not src.exists()
    assert (workdir / "sub" / "dst.txt").read_text() == "hi\n"


def test_high_risk_detection() -> None:
    plan = EditPlan(
        description="ok",
        operations=[EditOperation(kind="overwrite", path="x", content="y")],
    )
    assert plan.is_high_risk() is False
    plan = EditPlan(
        description="risky",
        operations=[EditOperation(kind="delete", path="x")],
    )
    assert plan.is_high_risk() is True


def test_render_plan_diff_shows_changes(workdir: Path) -> None:
    target = workdir / "f.txt"
    target.write_text("a\nb\n")
    plan = EditPlan(
        description="diff",
        operations=[EditOperation(kind="overwrite", path=str(target), content="a\nB\n")],
    )
    diff = render_plan_diff(plan)
    assert "-b" in diff and "+B" in diff


def test_manual_rollback_restores_files(workdir: Path) -> None:
    target = workdir / "snap.txt"
    target.write_text("orig\n")
    plan = EditPlan(
        description="manual rollback",
        operations=[EditOperation(kind="overwrite", path=str(target), content="new\n")],
    )
    # populate snapshots manually then mutate then roll back
    from agent.tools.multi_edit import _take_snapshots
    _take_snapshots(plan)
    target.write_text("new\n")
    rollback_edit_plan(plan)
    assert target.read_text() == "orig\n"
