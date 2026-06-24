"""Swarm executor: parallel role-specialised sub-agents via ``asyncio.TaskGroup``."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable

from agent.swarm.bus import MessageBus, SwarmMessage
from agent.swarm.roles import AgentRole, assign_role_for_task
from agent.types import TaskNode


@dataclass
class SwarmTask:
    """One entry in the swarm work plan."""

    task: TaskNode
    role: AgentRole
    status: str = "pending"
    result: dict[str, Any] | None = None
    tokens: int = 0
    last_message: str = ""


WorkerFn = Callable[["SwarmTask", MessageBus], Any]


async def _maybe_await(value: Any) -> Any:
    """Await ``value`` if it is a coroutine, otherwise return it as-is."""
    if asyncio.iscoroutine(value):
        return await value
    return value


@dataclass
class SwarmExecutor:
    """Parallel orchestrator built on :class:`asyncio.TaskGroup` (Python 3.11+)."""

    worker: WorkerFn
    bus: MessageBus = field(default_factory=MessageBus)
    max_parallel: int = 4

    async def run(self, tasks: list[TaskNode]) -> list[SwarmTask]:
        """Assign roles and execute every task concurrently up to ``max_parallel``."""
        plan = [SwarmTask(task=t, role=assign_role_for_task(t)) for t in tasks]
        sem = asyncio.Semaphore(self.max_parallel)

        async def runner(swarm_task: SwarmTask) -> None:
            async with sem:
                swarm_task.status = "running"
                try:
                    await self.bus.publish(SwarmMessage(
                        sender_role=swarm_task.role.value,
                        target_role="PLANNER",
                        message_type="STATUS",
                        payload={"task_id": swarm_task.task.id, "phase": "start"},
                    ))
                    raw = self.worker(swarm_task, self.bus)
                    result = await _maybe_await(raw)
                    swarm_task.result = result if isinstance(result, dict) else {"value": result}
                    swarm_task.status = "complete"
                except Exception as exc:
                    swarm_task.status = "failed"
                    swarm_task.result = {"error": f"{type(exc).__name__}: {exc}"}
                    await self.bus.publish(SwarmMessage(
                        sender_role=swarm_task.role.value,
                        target_role="PLANNER",
                        message_type="BLOCKER",
                        payload={"task_id": swarm_task.task.id, "error": str(exc)},
                    ))

        async with asyncio.TaskGroup() as tg:
            for st in plan:
                tg.create_task(runner(st))
        return plan
