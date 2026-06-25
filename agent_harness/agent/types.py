"""Shared structured types used by the orchestrator, loop, tools, and UI."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal


class TaskStatus(str, Enum):
    """Lifecycle state of a single task node."""

    PENDING = "pending"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    FAILED = "failed"
    SKIPPED = "skipped"


class Complexity(str, Enum):
    """Coarse complexity estimate produced by the Planner."""

    S = "S"
    M = "M"
    L = "L"
    XL = "XL"


_COMPLEXITY_COST = {Complexity.S: 1, Complexity.M: 3, Complexity.L: 8, Complexity.XL: 20}


@dataclass
class TaskNode:
    """A single node in the orchestrator task DAG."""

    id: str
    description: str
    complexity: Complexity = Complexity.M
    prerequisites: list[str] = field(default_factory=list)
    anticipated_tools: list[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    summary: str | None = None
    artifacts: list[str] = field(default_factory=list)
    error: str | None = None
    attempts: int = 0
    recoverability: float = 0.7

    def estimated_cost(self) -> int:
        """Return a unitless cost score based on complexity."""
        return _COMPLEXITY_COST[self.complexity]

    def priority(self) -> tuple[int, float]:
        """Priority key — lower is more urgent (cost asc, recoverability desc)."""
        return (self.estimated_cost(), -self.recoverability)


@dataclass
class ToolCall:
    """Represents a single tool invocation made by the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class ToolResult:
    """The structured outcome of a tool invocation."""

    call_id: str
    name: str
    ok: bool
    output: Any
    error: str | None = None
    latency_ms: float = 0.0
    finished_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_observation(self) -> str:
        """Render the result as a short LLM-friendly observation string."""
        if self.ok:
            return f"[{self.name}] OK ({self.latency_ms:.0f}ms): {self.output}"
        return f"[{self.name}] FAIL ({self.latency_ms:.0f}ms): {self.error}"


@dataclass
class LLMMessage:
    """A message exchanged with the LLM."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    tokens: int = 0


@dataclass
class LLMUsage:
    """Accumulated token usage for a single LLM response."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class LLMResponse:
    """Result of a non-streaming LLM call."""

    text: str
    tool_calls: list[ToolCall]
    usage: LLMUsage
    model: str
    stop_reason: str | None = None


@dataclass
class StreamChunk:
    """A single chunk yielded by an LLM streaming response."""

    kind: Literal["text", "tool_call", "tool_arg_delta", "stop", "thinking"]
    text: str = ""
    tool_call: ToolCall | None = None
    usage: LLMUsage | None = None


@dataclass
class Memory:
    """A semantic-memory record."""

    id: str
    kind: Literal["file", "decision", "error", "summary", "fact"]
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
