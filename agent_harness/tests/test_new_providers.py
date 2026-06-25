"""Tests for the new provider clients.

We test the *adapter logic* (reasoning extraction, throughput, routing config)
directly rather than running the OpenAI SDK end-to-end against a mocked
transport — the SDK captures its own ``httpx`` client at construction time
which makes a clean monkeypatch fragile.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from agent.llm.base import CostTracker
from agent.llm.cerebras_client import CerebrasClient
from agent.llm.groq_client import GroqClient
from agent.llm.together_client import TogetherClient
from agent.types import LLMMessage, LLMResponse, LLMUsage, StreamChunk


@pytest.mark.asyncio
async def test_groq_complete_strips_think_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    client = GroqClient(model="qwen-qwq-32b", api_key="t", cost_tracker=CostTracker())

    async def fake_super_complete(self, *a, **kw):
        return LLMResponse(
            text="<think>weighing options</think>final answer",
            tool_calls=[], usage=LLMUsage(), model=self.model,
        )

    from agent.llm.openai_client import OpenAIClient
    monkeypatch.setattr(OpenAIClient, "complete", fake_super_complete)
    resp = await client.complete([LLMMessage(role="user", content="?")])
    assert resp.text == "final answer"
    assert client.last_reasoning == "weighing options"


@pytest.mark.asyncio
async def test_groq_stream_routes_think_blocks_to_thinking_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    client = GroqClient(model="qwen-qwq-32b", api_key="t", cost_tracker=CostTracker())

    async def fake_super_stream(self, *a, **kw) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(kind="text", text="pre ")
        yield StreamChunk(kind="text", text="<think>secret")
        yield StreamChunk(kind="text", text=" stuff</think>")
        yield StreamChunk(kind="text", text=" after")
        yield StreamChunk(kind="stop", usage=LLMUsage())

    from agent.llm.openai_client import OpenAIClient
    monkeypatch.setattr(OpenAIClient, "stream", fake_super_stream)

    visible: list[str] = []
    thinking: list[str] = []
    async for c in client.stream([LLMMessage(role="user", content="?")]):
        if c.kind == "text" and c.text:
            visible.append(c.text)
        elif c.kind == "thinking" and c.text:
            thinking.append(c.text)
    assert "after" in "".join(visible)
    assert "secret" in "".join(thinking) and "stuff" in "".join(thinking)
    assert "secret" in client.last_reasoning


def test_groq_extract_reasoning_with_blocks() -> None:
    client = GroqClient(model="m", api_key="t")
    cleaned, reasoning = client._extract_reasoning("pre <think>r1</think> mid <think>r2</think> post")
    assert "pre" in cleaned and "post" in cleaned
    assert "<think>" not in cleaned
    assert "r1" in reasoning and "r2" in reasoning


def test_groq_extract_reasoning_no_blocks() -> None:
    client = GroqClient(model="m", api_key="t")
    cleaned, reasoning = client._extract_reasoning("hello")
    assert cleaned == "hello"
    assert reasoning == ""


@pytest.mark.asyncio
async def test_cerebras_records_tps_on_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    client = CerebrasClient(model="llama-3.3-70b", api_key="t", cost_tracker=CostTracker())

    async def fake_super_stream(self, *a, **kw) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(kind="text", text="hello world this is some text" * 4)
        yield StreamChunk(kind="stop", usage=LLMUsage(prompt_tokens=10, completion_tokens=50))

    from agent.llm.openai_client import OpenAIClient
    monkeypatch.setattr(OpenAIClient, "stream", fake_super_stream)

    async for _ in client.stream([LLMMessage(role="user", content="?")]):
        pass
    assert client.last_tokens_per_second > 0
    assert client.tokens_per_second_history[-1] == client.last_tokens_per_second


def test_cerebras_uses_cerebras_base_url() -> None:
    client = CerebrasClient(model="llama-3.3-70b", api_key="t")
    assert "api.cerebras.ai" in str(client._client.base_url)


def test_together_uses_together_base_url() -> None:
    client = TogetherClient(model="meta-llama/Llama-3.3-70B-Instruct-Turbo", api_key="t")
    assert "api.together.xyz" in str(client._client.base_url)


def test_groq_uses_groq_base_url() -> None:
    client = GroqClient(model="llama-3.3-70b-versatile", api_key="t")
    assert "api.groq.com" in str(client._client.base_url)
