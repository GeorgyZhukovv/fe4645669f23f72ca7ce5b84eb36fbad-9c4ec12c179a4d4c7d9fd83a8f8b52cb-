"""Anthropic Claude client with streaming and tool use."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from agent.llm.base import CostTracker, LLMClient
from agent.types import LLMMessage, LLMResponse, LLMUsage, StreamChunk, ToolCall


def _to_anthropic_messages(messages: list[LLMMessage]) -> tuple[str | None, list[dict[str, Any]]]:
    """Convert the harness's :class:`LLMMessage` list to the Anthropic schema.

    Returns:
        A ``(system, messages)`` tuple where ``system`` is the concatenated
        system prompt and ``messages`` is the API-shaped message list.
    """
    system_parts: list[str] = []
    out: list[dict[str, Any]] = []
    pending_user: list[dict[str, Any]] = []
    pending_assistant: list[dict[str, Any]] = []

    def flush_user() -> None:
        if pending_user:
            out.append({"role": "user", "content": list(pending_user)})
            pending_user.clear()

    def flush_assistant() -> None:
        if pending_assistant:
            out.append({"role": "assistant", "content": list(pending_assistant)})
            pending_assistant.clear()

    for m in messages:
        if m.role == "system":
            system_parts.append(m.content)
        elif m.role == "user":
            flush_assistant()
            pending_user.append({"type": "text", "text": m.content})
        elif m.role == "assistant":
            flush_user()
            if m.content:
                pending_assistant.append({"type": "text", "text": m.content})
            for tc in m.tool_calls:
                pending_assistant.append(
                    {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
                )
        elif m.role == "tool":
            flush_assistant()
            pending_user.append(
                {"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.content}
            )
    flush_user()
    flush_assistant()
    system = "\n\n".join(system_parts) if system_parts else None
    return system, out


def _tools_for_anthropic(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """Pass-through the tool list (already in Anthropic ``input_schema`` shape)."""
    if not tools:
        return None
    return [
        {
            "name": t["name"],
            "description": t.get("description", ""),
            "input_schema": t.get("input_schema", {"type": "object", "properties": {}}),
        }
        for t in tools
    ]


class AnthropicClient(LLMClient):
    """LLM client backed by the official ``anthropic`` SDK."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
        cost_tracker: CostTracker | None = None,
    ) -> None:
        """Construct the client.

        Args:
            model: Anthropic model id.
            api_key: Optional API key (otherwise reads ``ANTHROPIC_API_KEY``).
            cost_tracker: Optional tracker to record per-call costs.
        """
        super().__init__(model=model, cost_tracker=cost_tracker)
        from anthropic import AsyncAnthropic

        self._client = AsyncAnthropic(api_key=api_key) if api_key else AsyncAnthropic()

    async def complete(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        sys_part, anth_messages = _to_anthropic_messages(messages)
        final_system = "\n\n".join(p for p in [system, sys_part] if p) or None
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": anth_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if final_system:
            kwargs["system"] = final_system
        if tools:
            kwargs["tools"] = _tools_for_anthropic(tools)
        response = await self._client.messages.create(**kwargs)
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
                )
        usage = LLMUsage(
            prompt_tokens=getattr(response.usage, "input_tokens", 0),
            completion_tokens=getattr(response.usage, "output_tokens", 0),
        )
        self.cost_tracker.record(self.model, usage)
        return LLMResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            usage=usage,
            model=self.model,
            stop_reason=response.stop_reason,
        )

    async def stream(  # type: ignore[override]
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> AsyncIterator[StreamChunk]:
        sys_part, anth_messages = _to_anthropic_messages(messages)
        final_system = "\n\n".join(p for p in [system, sys_part] if p) or None
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": anth_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if final_system:
            kwargs["system"] = final_system
        if tools:
            kwargs["tools"] = _tools_for_anthropic(tools)
        async with self._client.messages.stream(**kwargs) as stream:
            current_tool: dict[str, Any] | None = None
            tool_buf = ""
            async for event in stream:
                evt_type = getattr(event, "type", None)
                if evt_type == "content_block_start":
                    block = event.content_block
                    if block.type == "tool_use":
                        current_tool = {"id": block.id, "name": block.name, "args": ""}
                        tool_buf = ""
                elif evt_type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        yield StreamChunk(kind="text", text=delta.text)
                    elif delta.type == "input_json_delta" and current_tool:
                        tool_buf += delta.partial_json
                        yield StreamChunk(kind="tool_arg_delta", text=delta.partial_json)
                elif evt_type == "content_block_stop" and current_tool is not None:
                    try:
                        args = json.loads(tool_buf) if tool_buf else {}
                    except json.JSONDecodeError:
                        args = {}
                    tc = ToolCall(
                        id=current_tool["id"] or str(uuid.uuid4()),
                        name=current_tool["name"],
                        arguments=args,
                    )
                    yield StreamChunk(kind="tool_call", tool_call=tc)
                    current_tool = None
                    tool_buf = ""
            final = await stream.get_final_message()
            usage = LLMUsage(
                prompt_tokens=getattr(final.usage, "input_tokens", 0),
                completion_tokens=getattr(final.usage, "output_tokens", 0),
            )
            self.cost_tracker.record(self.model, usage)
            yield StreamChunk(kind="stop", usage=usage)
