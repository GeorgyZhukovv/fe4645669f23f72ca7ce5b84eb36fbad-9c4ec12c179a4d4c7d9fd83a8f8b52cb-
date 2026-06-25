"""Agent role enum + assignment heuristics."""

from __future__ import annotations

from enum import Enum

from agent.types import TaskNode


class AgentRole(str, Enum):
    """Specialised agent role inside the swarm."""

    PLANNER = "PLANNER"
    CODER = "CODER"
    REVIEWER = "REVIEWER"
    TESTER = "TESTER"
    DOCUMENTER = "DOCUMENTER"
    SECURITY_AUDITOR = "SECURITY_AUDITOR"


def assign_role_for_task(task: TaskNode) -> AgentRole:
    """Pick a role for ``task`` using keyword heuristics over its description."""
    desc = task.description.lower()
    if any(k in desc for k in ("security", "vulnerab", "cve", "secret", "owasp")):
        return AgentRole.SECURITY_AUDITOR
    if any(k in desc for k in ("test", "pytest", "jest", "unit-test", "coverage")):
        return AgentRole.TESTER
    if any(k in desc for k in ("document", "readme", "docstring", "comment")):
        return AgentRole.DOCUMENTER
    if any(k in desc for k in ("review", "lint", "style")):
        return AgentRole.REVIEWER
    if any(k in desc for k in ("plan ", "design", "architecture")):
        return AgentRole.PLANNER
    return AgentRole.CODER
