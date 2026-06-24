"""Multi-agent swarm orchestration."""

from agent.swarm.bus import MessageBus, SwarmMessage
from agent.swarm.executor import SwarmExecutor, SwarmTask
from agent.swarm.locks import EditConflictResolver, FileLockList
from agent.swarm.roles import AgentRole, assign_role_for_task

__all__ = [
    "AgentRole",
    "EditConflictResolver",
    "FileLockList",
    "MessageBus",
    "SwarmExecutor",
    "SwarmMessage",
    "SwarmTask",
    "assign_role_for_task",
]
