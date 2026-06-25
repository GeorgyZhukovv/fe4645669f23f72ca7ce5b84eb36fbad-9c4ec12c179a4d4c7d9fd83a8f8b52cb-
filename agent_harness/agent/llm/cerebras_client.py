"""Cerebras client. OpenAI-compatible endpoint with throughput tracking."""

from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator
from typing import Any

from agent.llm.base import CostTracker
from agent.llm.openai_client import OpenAIClient
from agent.types import LLMMessage, StreamChunk


class CerebrasClient(OpenAIClient):
    """Thin OpenAI-compatible wrapper for ``https://api.cerebras.ai/v1``."""

    def __init__(
        self,
        model: str = "llama-3.3-70b",
        api_key: str | None = None,
        cost_tracker: CostTracker | None = None,
    ) -> None:
        """Construct the client.

        Args:
            model: Cerebras api_model_name.
            api_key: API key (otherwise reads ``CEREBRAS_API_KEY``).
            cost_tracker: Optional shared :class:`CostTracker`.
        """
        super().__init__(
            model=model,
            api_key=api_key or os.environ.get("CEREBRAS_API_KEY", ""),
            base_url="https://api.cerebras.ai/v1",
            cost_tracker=cost_tracker,
        )
        self._last_tps: float = 0.0
        self._tps_history: list[float] = []

    @property
    def last_tokens_per_second(self) -> float:
        """Throughput observed on the most recent streaming call."""
        return self._last_tps

    @property
    def tokens_per_second_history(self) -> list[float]:
        """Rolling history of per-call throughput."""
        return list(self._tps_history)

    async def stream(  # type: ignore[override]
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> AsyncIterator[StreamChunk]:
        start: float | None = None
        tokens_seen = 0
        async for chunk in super().stream(messages, tools, system, temperature, max_tokens):
            if start is None and chunk.kind in {"text", "thinking", "tool_arg_delta"}:
                start = time.perf_counter()
            if chunk.kind == "text" and chunk.text:
                tokens_seen += max(1, len(chunk.text) // 4)
            if chunk.kind == "stop" and chunk.usage is not None:
                tokens_seen = chunk.usage.completion_tokens or tokens_seen
            yield chunk
        if start is not None and tokens_seen > 0:
            elapsed = time.perf_counter() - start
            tps = tokens_seen / elapsed if elapsed > 0 else 0.0
            self._last_tps = tps
            self._tps_history.append(tps)
            if len(self._tps_history) > 50:
                self._tps_history.pop(0)
