"""Reasoning-model adapter: handles ``<think>`` blocks + the no-system-role models."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from typing import Any

from agent.llm.base import LLMClient
from agent.types import LLMMessage, LLMResponse, StreamChunk


REASONING_MODEL_IDS = {
    "o3", "o4-mini",
    "deepseek-r1-distill", "qwen-qwq",
}


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def is_reasoning_model(model_id: str) -> bool:
    """Heuristic: is ``model_id`` a reasoning model that needs the adapter?"""
    if model_id in REASONING_MODEL_IDS:
        return True
    low = model_id.lower()
    return any(token in low for token in ("o3", "o4-mini", "reasoner", "qwq", "r1-distill"))


def _merge_system_into_first_user(messages: list[LLMMessage]) -> tuple[None, list[LLMMessage]]:
    """For models that reject a ``system`` role, merge system content into user[0]."""
    system_parts: list[str] = []
    rest: list[LLMMessage] = []
    for m in messages:
        if m.role == "system":
            system_parts.append(m.content)
        else:
            rest.append(m)
    if not system_parts:
        return None, list(rest)
    sys_blob = "\n\n".join(system_parts)
    if rest and rest[0].role == "user":
        rest[0] = LLMMessage(
            role="user",
            content=f"<system>\n{sys_blob}\n</system>\n\n{rest[0].content}",
            tool_calls=rest[0].tool_calls,
            tool_call_id=rest[0].tool_call_id,
            name=rest[0].name,
            tokens=rest[0].tokens,
        )
    else:
        rest.insert(0, LLMMessage(role="user", content=f"<system>\n{sys_blob}\n</system>"))
    return None, rest


def split_reasoning(text: str) -> tuple[str, str]:
    """Return ``(visible_text, concatenated_reasoning)`` by stripping ``<think>``."""
    blocks = _THINK_RE.findall(text or "")
    if not blocks:
        return text, ""
    cleaned = _THINK_RE.sub("", text).strip()
    return cleaned, "\n".join(b.strip() for b in blocks)


class ReasoningAdapter(LLMClient):
    """Wrap an :class:`LLMClient` so a reasoning model behaves like a regular one."""

    def __init__(self, inner: LLMClient, *, no_system_prompt: bool = False) -> None:
        """Wrap ``inner`` and configure adapter behaviour.

        Args:
            inner: The underlying client to adapt.
            no_system_prompt: When ``True`` (e.g. ``o3``/``o4-mini``), merges the
                system message into the first user message before forwarding.
        """
        super().__init__(model=inner.model, cost_tracker=inner.cost_tracker)
        self.inner = inner
        self.no_system_prompt = no_system_prompt
        self.last_reasoning: str = ""

    def _adapt(self, messages: list[LLMMessage], system: str | None) -> tuple[list[LLMMessage], str | None]:
        if not self.no_system_prompt:
            return messages, system
        all_messages = list(messages)
        if system:
            all_messages.insert(0, LLMMessage(role="system", content=system))
        _, merged = _merge_system_into_first_user(all_messages)
        return merged, None

    async def complete(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        adapted, sys_out = self._adapt(messages, system)
        resp = await self.inner.complete(adapted, tools, sys_out, temperature, max_tokens)
        visible, reasoning = split_reasoning(resp.text)
        if reasoning:
            self.last_reasoning = reasoning
            resp.text = visible
        return resp

    async def stream(  # type: ignore[override]
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> AsyncIterator[StreamChunk]:
        adapted, sys_out = self._adapt(messages, system)
        in_think = False
        reasoning_buf: list[str] = []
        async for chunk in self.inner.stream(adapted, tools, sys_out, temperature, max_tokens):
            if chunk.kind != "text" or not chunk.text:
                yield chunk
                continue
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
                if text[:start]:
                    yield StreamChunk(kind="text", text=text[:start])
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
        self.last_reasoning = "\n".join(reasoning_buf).strip()


def maybe_wrap_reasoning(client: LLMClient, model_id: str, no_system_prompt: bool = False) -> LLMClient:
    """Wrap ``client`` in :class:`ReasoningAdapter` if ``model_id`` is reasoning-class."""
    if is_reasoning_model(model_id) or no_system_prompt:
        return ReasoningAdapter(client, no_system_prompt=no_system_prompt)
    return client
