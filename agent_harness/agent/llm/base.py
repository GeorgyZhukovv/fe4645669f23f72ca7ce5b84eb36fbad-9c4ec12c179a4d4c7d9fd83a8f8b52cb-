"""Abstract LLM client base class with cost tracking and tool-use plumbing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from agent.types import LLMMessage, LLMResponse, LLMUsage, StreamChunk


PRICING_PER_M_TOKENS: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-opus-4-7": (15.0, 75.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "gpt-4o": (5.0, 15.0),
    "gpt-4o-mini": (0.15, 0.6),
    "gpt-4.1": (2.5, 10.0),
}


@dataclass
class CostEntry:
    """One LLM call's usage and computed cost."""

    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


@dataclass
class CostTracker:
    """Accumulates per-call usage and computes a running USD total."""

    entries: list[CostEntry] = field(default_factory=list)

    def record(self, model: str, usage: LLMUsage) -> CostEntry:
        """Add a cost entry from a single completion's usage."""
        in_price, out_price = PRICING_PER_M_TOKENS.get(model, (0.0, 0.0))
        cost = (usage.prompt_tokens * in_price + usage.completion_tokens * out_price) / 1_000_000
        entry = CostEntry(
            model=model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cost_usd=cost,
        )
        self.entries.append(entry)
        return entry

    @property
    def total_usd(self) -> float:
        return sum(e.cost_usd for e in self.entries)

    @property
    def total_prompt_tokens(self) -> int:
        return sum(e.prompt_tokens for e in self.entries)

    @property
    def total_completion_tokens(self) -> int:
        return sum(e.completion_tokens for e in self.entries)

    def summary(self) -> dict[str, Any]:
        """Return a serialisable summary of accumulated cost."""
        by_model: dict[str, dict[str, float]] = {}
        for e in self.entries:
            b = by_model.setdefault(
                e.model, {"prompt": 0, "completion": 0, "cost_usd": 0.0, "calls": 0}
            )
            b["prompt"] += e.prompt_tokens
            b["completion"] += e.completion_tokens
            b["cost_usd"] += e.cost_usd
            b["calls"] += 1
        return {"total_usd": self.total_usd, "by_model": by_model}


class LLMClient(ABC):
    """Common interface for chat completions with tool use."""

    def __init__(self, model: str, cost_tracker: CostTracker | None = None) -> None:
        """Bind ``model`` and an optional :class:`CostTracker`."""
        self.model = model
        self.cost_tracker = cost_tracker or CostTracker()

    @abstractmethod
    async def complete(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Non-streaming completion. Returns text + structured tool calls."""

    @abstractmethod
    def stream(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> AsyncIterator[StreamChunk]:
        """Streaming completion. Yields :class:`StreamChunk` events."""


class FallbackClient(LLMClient):
    """Wrap a primary client with a secondary used on failure or rate-limits."""

    def __init__(self, primary: LLMClient, secondary: LLMClient) -> None:
        super().__init__(model=primary.model, cost_tracker=primary.cost_tracker)
        self.primary = primary
        self.secondary = secondary

    def _is_terminal_auth_error(self, exc: Exception) -> bool:
        """Detect quota / auth errors that no fallback can recover from."""
        msg = str(exc).lower()
        name = type(exc).__name__.lower()
        if "insufficient_quota" in msg or "invalid_api_key" in msg or "incorrect api key" in msg:
            return True
        if "authenticationerror" in name and "anthropic" in name:
            return True
        if "authenticationerror" in name and "openai" in name:
            return True
        return False

    async def complete(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        try:
            return await self.primary.complete(messages, tools, system, temperature, max_tokens)
        except Exception as exc:
            if self._is_terminal_auth_error(exc):
                raise
            return await self.secondary.complete(messages, tools, system, temperature, max_tokens)

    async def stream(  # type: ignore[override]
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> AsyncIterator[StreamChunk]:
        try:
            async for chunk in self.primary.stream(messages, tools, system, temperature, max_tokens):
                yield chunk
        except Exception as exc:
            if self._is_terminal_auth_error(exc):
                raise
            async for chunk in self.secondary.stream(messages, tools, system, temperature, max_tokens):
                yield chunk
