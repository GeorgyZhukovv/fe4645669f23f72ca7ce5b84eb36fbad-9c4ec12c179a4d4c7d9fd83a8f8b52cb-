"""Tests for swarm orchestration: roles, bus, conflict resolver, executor."""

from __future__ import annotations

import asyncio

import pytest

from agent.swarm import (
    AgentRole,
    EditConflictResolver,
    FileLockList,
    MessageBus,
    SwarmExecutor,
    SwarmMessage,
    assign_role_for_task,
)
from agent.types import Complexity, TaskNode


def test_assign_role_heuristics() -> None:
    coder = TaskNode(id="c", description="Add a function to compute X")
    tester = TaskNode(id="t", description="Write pytest tests for foo")
    documenter = TaskNode(id="d", description="Update README and docstring")
    reviewer = TaskNode(id="r", description="Review style and lint output")
    sec = TaskNode(id="s", description="Audit for OWASP vulnerabilities")
    arch = TaskNode(id="a", description="Plan the architecture")

    assert assign_role_for_task(coder) == AgentRole.CODER
    assert assign_role_for_task(tester) == AgentRole.TESTER
    assert assign_role_for_task(documenter) == AgentRole.DOCUMENTER
    assert assign_role_for_task(reviewer) == AgentRole.REVIEWER
    assert assign_role_for_task(sec) == AgentRole.SECURITY_AUDITOR
    assert assign_role_for_task(arch) == AgentRole.PLANNER


@pytest.mark.asyncio
async def test_message_bus_targeted_delivery() -> None:
    bus = MessageBus()
    a = await bus.subscribe("A")
    b = await bus.subscribe("B")
    await bus.publish(SwarmMessage(sender_role="B", target_role="A", message_type="STATUS"))
    msg = await asyncio.wait_for(a.get(), timeout=1)
    assert msg.sender_role == "B"
    assert b.empty()


@pytest.mark.asyncio
async def test_message_bus_broadcast() -> None:
    bus = MessageBus()
    a = await bus.subscribe("A")
    b = await bus.subscribe("B")
    await bus.publish(SwarmMessage(sender_role="X", target_role=None, message_type="FINDING"))
    assert (await asyncio.wait_for(a.get(), timeout=1)).message_type == "FINDING"
    assert (await asyncio.wait_for(b.get(), timeout=1)).message_type == "FINDING"


@pytest.mark.asyncio
async def test_file_lock_list_serialises_paths() -> None:
    locks = FileLockList()
    held = await locks.lock_paths(["a.py", "b.py"])
    with pytest.raises((asyncio.TimeoutError, TimeoutError)):
        await locks.lock_paths(["a.py"], timeout=0.05)
    locks.release(held)
    # Should now succeed
    held2 = await locks.lock_paths(["a.py"], timeout=1)
    locks.release(held2)


def test_edit_conflict_resolver_non_overlapping() -> None:
    base = "line1\nline2\nline3\n"
    a = "line1\nline2\nline3\nnew-a\n"
    b = "new-b\nline1\nline2\nline3\n"
    merged, ok = EditConflictResolver().merge(base, a, b)
    assert ok is True
    assert "new-a" in merged and "new-b" in merged


@pytest.mark.asyncio
async def test_swarm_executor_runs_workers_in_parallel() -> None:
    seen: list[str] = []

    async def worker(task, bus):
        seen.append(task.task.id)
        return {"task_id": task.task.id}

    executor = SwarmExecutor(worker=worker, max_parallel=2)
    tasks = [
        TaskNode(id="t1", description="Add code"),
        TaskNode(id="t2", description="Write tests"),
        TaskNode(id="t3", description="Update README"),
    ]
    plan = await executor.run(tasks)
    assert set(seen) == {"t1", "t2", "t3"}
    assert all(p.status == "complete" for p in plan)
