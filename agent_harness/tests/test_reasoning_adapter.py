"""Tests for the reasoning-model adapter."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from agent.llm.base import CostTracker, LLMClient
from agent.llm.reasoning import (
    ReasoningAdapter,
    is_reasoning_model,
    maybe_wrap_reasoning,
    split_reasoning,
)
from agent.types import LLMMessage, LLMResponse, LLMUsage, StreamChunk


class _Recording(LLMClient):
    def __init__(self, text: str = "", chunks: list[StreamChunk] | None = None) -> None:
        super().__init__(model="fake", cost_tracker=CostTracker())
        self.text = text
        self.chunks = chunks or []
        self.last_messages: list[LLMMessage] = []
        self.last_system: str | None = None

    async def complete(self, messages, tools=None, system=None, temperature=0.2, max_tokens=4096):
        self.last_messages = list(messages)
        self.last_system = system
        return LLMResponse(text=self.text, tool_calls=[], usage=LLMUsage(), model=self.model)

    async def stream(self, messages, tools=None, system=None, temperature=0.2, max_tokens=4096) -> AsyncIterator[StreamChunk]:
        self.last_messages = list(messages)
        self.last_system = system
        for c in self.chunks:
            yield c


def test_split_reasoning_extracts_blocks() -> None:
    visible, reasoning = split_reasoning("hello <think>step 1\nstep 2</think> world")
    assert visible == "hello  world"
    assert reasoning == "step 1\nstep 2"


def test_split_reasoning_passthrough_when_no_blocks() -> None:
    visible, reasoning = split_reasoning("just text")
    assert visible == "just text"
    assert reasoning == ""


def test_is_reasoning_model_detects_known_ids() -> None:
    assert is_reasoning_model("o3")
    assert is_reasoning_model("o4-mini")
    assert is_reasoning_model("deepseek-r1-distill-llama-70b")
    assert is_reasoning_model("qwen-qwq-32b")
    assert not is_reasoning_model("gpt-4o-mini")
    assert not is_reasoning_model("claude-sonnet-4-6")


@pytest.mark.asyncio
async def test_complete_strips_think_blocks_into_last_reasoning() -> None:
    inner = _Recording(text="<think>private</think>visible answer")
    adapter = ReasoningAdapter(inner)
    resp = await adapter.complete([LLMMessage(role="user", content="?")])
    assert resp.text == "visible answer"
    assert adapter.last_reasoning == "private"


@pytest.mark.asyncio
async def test_complete_merges_system_when_no_system_prompt() -> None:
    inner = _Recording(text="ok")
    adapter = ReasoningAdapter(inner, no_system_prompt=True)
    await adapter.complete(
        [LLMMessage(role="user", content="hello")],
        system="be terse",
    )
    # system should NOT be passed to inner; first user msg should contain it.
    assert inner.last_system is None
    assert "be terse" in inner.last_messages[0].content
    assert "hello" in inner.last_messages[0].content


@pytest.mark.asyncio
async def test_stream_separates_text_and_thinking() -> None:
    inner = _Recording(chunks=[
        StreamChunk(kind="text", text="pre "),
        StreamChunk(kind="text", text="<think>secret"),
        StreamChunk(kind="text", text=" thought</think>"),
        StreamChunk(kind="text", text=" after"),
        StreamChunk(kind="stop"),
    ])
    adapter = ReasoningAdapter(inner)
    out_kinds: list[tuple[str, str]] = []
    async for c in adapter.stream([LLMMessage(role="user", content="?")]):
        if c.text:
            out_kinds.append((c.kind, c.text))
    visible = "".join(t for k, t in out_kinds if k == "text")
    reasoning = "".join(t for k, t in out_kinds if k == "thinking")
    assert visible == "pre  after"
    assert "secret" in reasoning and "thought" in reasoning


def test_maybe_wrap_only_wraps_reasoning_models() -> None:
    inner = _Recording()
    assert maybe_wrap_reasoning(inner, "gpt-4o-mini") is inner
    wrapped = maybe_wrap_reasoning(inner, "o3")
    assert isinstance(wrapped, ReasoningAdapter)
    forced = maybe_wrap_reasoning(inner, "gpt-4o-mini", no_system_prompt=True)
    assert isinstance(forced, ReasoningAdapter)
