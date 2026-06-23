"""OpenAI-compatible client (also targets local servers like ollama / lm-studio)."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from agent.llm.base import CostTracker, LLMClient
from agent.types import LLMMessage, LLMResponse, LLMUsage, StreamChunk, ToolCall


def _to_openai_messages(messages: list[LLMMessage], system: str | None) -> list[dict[str, Any]]:
    """Convert :class:`LLMMessage` list to OpenAI Chat Completions format."""
    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})
    for m in messages:
        if m.role == "system":
            out.append({"role": "system", "content": m.content})
        elif m.role == "user":
            out.append({"role": "user", "content": m.content})
        elif m.role == "assistant":
            msg: dict[str, Any] = {"role": "assistant", "content": m.content or None}
            if m.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in m.tool_calls
                ]
            out.append(msg)
        elif m.role == "tool":
            out.append(
                {"role": "tool", "tool_call_id": m.tool_call_id, "name": m.name, "content": m.content}
            )
    return out


def _tools_for_openai(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """Convert harness tool schemas to OpenAI ``function`` tool definitions."""
    if not tools:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        }
        for t in tools
    ]


class OpenAIClient(LLMClient):
    """LLM client that speaks the OpenAI Chat Completions protocol."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        cost_tracker: CostTracker | None = None,
    ) -> None:
        """Construct the client.

        Args:
            model: Model identifier passed to the API.
            api_key: API key (otherwise reads ``OPENAI_API_KEY``).
            base_url: Optional base URL for local servers (e.g. ``http://localhost:11434/v1``).
            cost_tracker: Optional :class:`CostTracker` instance.
        """
        super().__init__(model=model, cost_tracker=cost_tracker)
        from openai import AsyncOpenAI

        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncOpenAI(**kwargs)

    async def complete(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        oai_messages = _to_openai_messages(messages, system)
        oai_tools = _tools_for_openai(tools)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": oai_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if oai_tools:
            kwargs["tools"] = oai_tools
            kwargs["tool_choice"] = "auto"
        resp = await self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        msg = choice.message
        text = msg.content or ""
        tool_calls: list[ToolCall] = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
        usage = LLMUsage(
            prompt_tokens=getattr(resp.usage, "prompt_tokens", 0),
            completion_tokens=getattr(resp.usage, "completion_tokens", 0),
        )
        self.cost_tracker.record(self.model, usage)
        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            usage=usage,
            model=self.model,
            stop_reason=choice.finish_reason,
        )

    async def stream(  # type: ignore[override]
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> AsyncIterator[StreamChunk]:
        oai_messages = _to_openai_messages(messages, system)
        oai_tools = _tools_for_openai(tools)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": oai_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if oai_tools:
            kwargs["tools"] = oai_tools
            kwargs["tool_choice"] = "auto"
        stream = await self._client.chat.completions.create(**kwargs)
        pending_tools: dict[int, dict[str, Any]] = {}
        usage = LLMUsage()
        async for chunk in stream:
            if chunk.usage is not None:
                usage = LLMUsage(
                    prompt_tokens=getattr(chunk.usage, "prompt_tokens", 0),
                    completion_tokens=getattr(chunk.usage, "completion_tokens", 0),
                )
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                yield StreamChunk(kind="text", text=delta.content)
            for tc_delta in delta.tool_calls or []:
                idx = tc_delta.index
                buf = pending_tools.setdefault(idx, {"id": "", "name": "", "args": ""})
                if tc_delta.id:
                    buf["id"] = tc_delta.id
                if tc_delta.function:
                    if tc_delta.function.name:
                        buf["name"] = tc_delta.function.name
                    if tc_delta.function.arguments:
                        buf["args"] += tc_delta.function.arguments
                        yield StreamChunk(kind="tool_arg_delta", text=tc_delta.function.arguments)
        for buf in pending_tools.values():
            try:
                args = json.loads(buf["args"] or "{}")
            except json.JSONDecodeError:
                args = {}
            tc = ToolCall(id=buf["id"] or str(uuid.uuid4()), name=buf["name"], arguments=args)
            yield StreamChunk(kind="tool_call", tool_call=tc)
        self.cost_tracker.record(self.model, usage)
        yield StreamChunk(kind="stop", usage=usage)
