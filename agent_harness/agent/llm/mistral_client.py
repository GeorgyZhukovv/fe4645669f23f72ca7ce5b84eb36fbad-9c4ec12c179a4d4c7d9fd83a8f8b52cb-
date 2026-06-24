"""Mistral client with tool use, streaming, and Codestral fill-in-the-middle."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import AsyncIterator
from typing import Any

from agent.llm.base import CostTracker, LLMClient
from agent.types import LLMMessage, LLMResponse, LLMUsage, StreamChunk, ToolCall


def _to_mistral_messages(messages: list[LLMMessage], system: str | None) -> list[dict[str, Any]]:
    """Convert harness messages to Mistral's Chat API format."""
    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})
    for m in messages:
        if m.role == "system":
            out.append({"role": "system", "content": m.content})
        elif m.role == "user":
            out.append({"role": "user", "content": m.content})
        elif m.role == "assistant":
            payload: dict[str, Any] = {"role": "assistant", "content": m.content or ""}
            if m.tool_calls:
                payload["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in m.tool_calls
                ]
            out.append(payload)
        elif m.role == "tool":
            out.append({
                "role": "tool",
                "tool_call_id": m.tool_call_id,
                "name": m.name,
                "content": m.content,
            })
    return out


def _tools_for_mistral(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """Convert harness tool schemas to Mistral function-tool definitions."""
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


class MistralClient(LLMClient):
    """LLM client backed by the official ``mistralai`` SDK."""

    def __init__(
        self,
        model: str = "mistral-small-latest",
        api_key: str | None = None,
        cost_tracker: CostTracker | None = None,
    ) -> None:
        """Construct the client.

        Args:
            model: Mistral api_model_name (e.g. ``codestral-latest``).
            api_key: API key (otherwise reads ``MISTRAL_API_KEY``).
            cost_tracker: Optional shared :class:`CostTracker`.
        """
        super().__init__(model=model, cost_tracker=cost_tracker)
        self._api_key = api_key or os.environ.get("MISTRAL_API_KEY") or ""
        try:
            from mistralai import Mistral

            self._client = Mistral(api_key=self._api_key)
            self._sdk_available = True
        except ImportError:
            self._client = None
            self._sdk_available = False

    async def complete(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        if not self._sdk_available:
            raise RuntimeError("mistralai SDK not installed. pip install mistralai")
        mistral_messages = _to_mistral_messages(messages, system)
        mistral_tools = _tools_for_mistral(tools)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": mistral_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if mistral_tools:
            kwargs["tools"] = mistral_tools
            kwargs["tool_choice"] = "auto"
        response = await self._client.chat.complete_async(**kwargs)
        choice = response.choices[0]
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
            prompt_tokens=getattr(response.usage, "prompt_tokens", 0),
            completion_tokens=getattr(response.usage, "completion_tokens", 0),
        )
        self.cost_tracker.record(self.model, usage)
        return LLMResponse(
            text=text, tool_calls=tool_calls, usage=usage, model=self.model,
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
        if not self._sdk_available:
            raise RuntimeError("mistralai SDK not installed.")
        mistral_messages = _to_mistral_messages(messages, system)
        mistral_tools = _tools_for_mistral(tools)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": mistral_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if mistral_tools:
            kwargs["tools"] = mistral_tools
            kwargs["tool_choice"] = "auto"
        usage = LLMUsage()
        pending: dict[int, dict[str, Any]] = {}
        async for chunk in await self._client.chat.stream_async(**kwargs):
            data = getattr(chunk, "data", chunk)
            if not getattr(data, "choices", None):
                continue
            delta = data.choices[0].delta
            if delta.content:
                yield StreamChunk(kind="text", text=delta.content)
            for tc_delta in getattr(delta, "tool_calls", []) or []:
                idx = getattr(tc_delta, "index", 0)
                buf = pending.setdefault(idx, {"id": "", "name": "", "args": ""})
                if getattr(tc_delta, "id", None):
                    buf["id"] = tc_delta.id
                fn = getattr(tc_delta, "function", None)
                if fn is not None:
                    if getattr(fn, "name", None):
                        buf["name"] = fn.name
                    if getattr(fn, "arguments", None):
                        buf["args"] += fn.arguments
                        yield StreamChunk(kind="tool_arg_delta", text=fn.arguments)
            meta = getattr(data, "usage", None)
            if meta is not None:
                usage = LLMUsage(
                    prompt_tokens=getattr(meta, "prompt_tokens", 0),
                    completion_tokens=getattr(meta, "completion_tokens", 0),
                )
        for buf in pending.values():
            try:
                args = json.loads(buf["args"] or "{}")
            except json.JSONDecodeError:
                args = {}
            yield StreamChunk(
                kind="tool_call",
                tool_call=ToolCall(id=buf["id"] or str(uuid.uuid4()), name=buf["name"], arguments=args),
            )
        self.cost_tracker.record(self.model, usage)
        yield StreamChunk(kind="stop", usage=usage)

    async def fill_in_the_middle(self, prefix: str, suffix: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        """Codestral fill-in-the-middle completion.

        Args:
            prefix: Text before the gap.
            suffix: Text after the gap.
            max_tokens: Max tokens to generate for the middle.
            temperature: Sampling temperature.

        Returns:
            The text that fills the gap.
        """
        if not self._sdk_available:
            raise RuntimeError("mistralai SDK not installed.")
        response = await self._client.fim.complete_async(
            model=self.model,
            prompt=prefix,
            suffix=suffix,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""
