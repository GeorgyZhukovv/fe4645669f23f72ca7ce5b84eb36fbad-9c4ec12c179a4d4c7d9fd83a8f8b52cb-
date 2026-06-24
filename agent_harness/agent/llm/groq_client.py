"""Groq client. OpenAI-compatible endpoint, with rate-limit and reasoning hooks."""

from __future__ import annotations

import os
import re
from collections.abc import AsyncIterator
from typing import Any

from agent.llm.base import CostTracker
from agent.llm.openai_client import OpenAIClient
from agent.types import LLMMessage, LLMResponse, StreamChunk


_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL)


class GroqClient(OpenAIClient):
    """Thin OpenAI-compatible wrapper for ``https://api.groq.com/openai/v1``."""

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        api_key: str | None = None,
        cost_tracker: CostTracker | None = None,
    ) -> None:
        """Construct the client.

        Args:
            model: Groq api_model_name (e.g. ``meta-llama/llama-4-scout-17b-16e-instruct``).
            api_key: API key (otherwise reads ``GROQ_API_KEY``).
            cost_tracker: Optional :class:`CostTracker`.
        """
        super().__init__(
            model=model,
            api_key=api_key or os.environ.get("GROQ_API_KEY", ""),
            base_url="https://api.groq.com/openai/v1",
            cost_tracker=cost_tracker,
        )
        self._last_reasoning: str = ""
        self._rate_limit_remaining: int | None = None
        self._rate_limit_reset: int | None = None

    @property
    def last_reasoning(self) -> str:
        """The most recent ``<think>...</think>`` body emitted by a reasoning model."""
        return self._last_reasoning

    def _extract_reasoning(self, text: str) -> tuple[str, str]:
        """Strip ``<think>...</think>`` blocks; return (cleaned_text, reasoning_concat)."""
        blocks = _THINK_BLOCK.findall(text)
        if not blocks:
            return text, ""
        cleaned = _THINK_BLOCK.sub("", text).strip()
        return cleaned, "\n".join(blocks).strip()

    async def complete(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        resp = await super().complete(messages, tools, system, temperature, max_tokens)
        cleaned, reasoning = self._extract_reasoning(resp.text)
        if reasoning:
            self._last_reasoning = reasoning
            resp.text = cleaned
        return resp

    async def stream(  # type: ignore[override]
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> AsyncIterator[StreamChunk]:
        in_think = False
        buf: list[str] = []
        reasoning_buf: list[str] = []
        async for chunk in super().stream(messages, tools, system, temperature, max_tokens):
            if chunk.kind == "text" and chunk.text:
                text = chunk.text
                if in_think:
                    end = text.find("</think>")
                    if end == -1:
                        reasoning_buf.append(text)
                        yield StreamChunk(kind="thinking", text=text)
                        continue
                    reasoning_buf.append(text[:end])
                    yield StreamChunk(kind="thinking", text=text[:end])
                    text = text[end + len("</think>"):]
                    in_think = False
                while True:
                    start = text.find("<think>")
                    if start == -1:
                        if text:
                            yield StreamChunk(kind="text", text=text)
                        break
                    pre = text[:start]
                    if pre:
                        yield StreamChunk(kind="text", text=pre)
                    rest = text[start + len("<think>"):]
                    end = rest.find("</think>")
                    if end == -1:
                        in_think = True
                        if rest:
                            reasoning_buf.append(rest)
                            yield StreamChunk(kind="thinking", text=rest)
                        text = ""
                        break
                    reasoning_buf.append(rest[:end])
                    yield StreamChunk(kind="thinking", text=rest[:end])
                    text = rest[end + len("</think>"):]
            else:
                yield chunk
        self._last_reasoning = "\n".join(reasoning_buf).strip()
