"""Google Gemini client with tool use, vision, streaming, and thinking mode."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import AsyncIterator
from typing import Any

from agent.llm.base import CostTracker, LLMClient
from agent.types import LLMMessage, LLMResponse, LLMUsage, StreamChunk, ToolCall


_FINISH_MAP = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "OTHER": "other",
}


def _convert_messages_to_gemini(messages: list[LLMMessage]) -> tuple[str | None, list[dict[str, Any]]]:
    """Convert harness messages to Gemini ``contents`` format.

    Returns:
        ``(system_instruction, contents)`` — Gemini takes the system prompt out-of-band.
    """
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "system":
            system_parts.append(m.content)
            continue
        if m.role == "tool":
            contents.append({
                "role": "user",
                "parts": [{
                    "functionResponse": {
                        "name": m.name or "tool",
                        "response": {"content": m.content},
                    }
                }],
            })
            continue
        role = "user" if m.role == "user" else "model"
        parts: list[dict[str, Any]] = []
        if m.content:
            parts.append({"text": m.content})
        for tc in m.tool_calls:
            parts.append({
                "functionCall": {"name": tc.name, "args": tc.arguments},
            })
        contents.append({"role": role, "parts": parts or [{"text": ""}]})
    system = "\n\n".join(p for p in system_parts if p) or None
    return system, contents


def _convert_tools_to_gemini_format(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """Convert harness tool schemas to Gemini's ``FunctionDeclaration`` list."""
    if not tools:
        return None
    declarations = []
    for t in tools:
        declarations.append({
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
        })
    return [{"functionDeclarations": declarations}]


class GeminiClient(LLMClient):
    """LLM client backed by the ``google-genai`` SDK."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash-preview",
        api_key: str | None = None,
        cost_tracker: CostTracker | None = None,
        enable_thinking: bool = False,
        thinking_budget_tokens: int = 8192,
        enable_grounding: bool = False,
    ) -> None:
        """Construct the client.

        Args:
            model: Gemini api_model_name (e.g. ``gemini-2.5-pro-preview``).
            api_key: Override key (otherwise reads ``GEMINI_API_KEY``).
            cost_tracker: Optional shared :class:`CostTracker`.
            enable_thinking: Pass ``thinkingConfig`` to 2.5-pro/2.5-flash.
            thinking_budget_tokens: Max thinking budget.
            enable_grounding: Add ``google_search_retrieval`` as a tool so Gemini
                can search the web during reasoning.
        """
        super().__init__(model=model, cost_tracker=cost_tracker)
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY") or ""
        self._enable_thinking = enable_thinking
        self._thinking_budget = thinking_budget_tokens
        self._enable_grounding = enable_grounding
        try:
            from google import genai

            self._sdk_available = True
            self._client = genai.Client(api_key=self._api_key)
        except ImportError:
            self._sdk_available = False
            self._client = None

    def _config(self, tools: list[dict[str, Any]] | None, system: str | None, temperature: float, max_tokens: int) -> dict[str, Any]:
        cfg: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if system:
            cfg["system_instruction"] = system
        gemini_tools = _convert_tools_to_gemini_format(tools) or []
        if self._enable_grounding:
            gemini_tools = list(gemini_tools) + [{"google_search_retrieval": {}}]
        if gemini_tools:
            cfg["tools"] = gemini_tools
        if self._enable_thinking and self.model.startswith("gemini-2.5"):
            cfg["thinking_config"] = {"thinking_budget": self._thinking_budget}
        return cfg

    async def complete(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        if not self._sdk_available:
            raise RuntimeError("google-genai not installed. pip install google-genai")
        sys_part, contents = _convert_messages_to_gemini(messages)
        merged_system = "\n\n".join(p for p in [system, sys_part] if p) or None
        cfg = self._config(tools, merged_system, temperature, max_tokens)
        response = await self._client.aio.models.generate_content(
            model=self.model, contents=contents, config=cfg,
        )
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        thinking_parts: list[str] = []
        candidates = getattr(response, "candidates", []) or []
        for cand in candidates:
            for part in getattr(cand.content, "parts", []) or []:
                if getattr(part, "text", None):
                    if getattr(part, "thought", False):
                        thinking_parts.append(part.text)
                    else:
                        text_parts.append(part.text)
                fn = getattr(part, "function_call", None)
                if fn is not None:
                    args = dict(getattr(fn, "args", {}) or {})
                    tool_calls.append(ToolCall(id=str(uuid.uuid4()), name=fn.name, arguments=args))
        usage_meta = getattr(response, "usage_metadata", None)
        usage = LLMUsage(
            prompt_tokens=getattr(usage_meta, "prompt_token_count", 0) if usage_meta else 0,
            completion_tokens=getattr(usage_meta, "candidates_token_count", 0) if usage_meta else 0,
        )
        self.cost_tracker.record(self.model, usage)
        text = "".join(text_parts)
        if thinking_parts:
            text = "<thinking>\n" + "\n".join(thinking_parts) + "\n</thinking>\n" + text
        finish = "stop"
        if candidates:
            raw_finish = getattr(candidates[0], "finish_reason", "STOP")
            finish = _FINISH_MAP.get(str(raw_finish), str(raw_finish).lower())
        return LLMResponse(
            text=text, tool_calls=tool_calls, usage=usage, model=self.model, stop_reason=finish,
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
            raise RuntimeError("google-genai not installed.")
        sys_part, contents = _convert_messages_to_gemini(messages)
        merged_system = "\n\n".join(p for p in [system, sys_part] if p) or None
        cfg = self._config(tools, merged_system, temperature, max_tokens)
        usage = LLMUsage()
        async for chunk in self._client.aio.models.generate_content_stream(
            model=self.model, contents=contents, config=cfg,
        ):
            for cand in getattr(chunk, "candidates", []) or []:
                for part in getattr(cand.content, "parts", []) or []:
                    if getattr(part, "text", None):
                        kind = "thinking" if getattr(part, "thought", False) else "text"
                        yield StreamChunk(kind=kind, text=part.text)
                    fn = getattr(part, "function_call", None)
                    if fn is not None:
                        args = dict(getattr(fn, "args", {}) or {})
                        tc = ToolCall(id=str(uuid.uuid4()), name=fn.name, arguments=args)
                        yield StreamChunk(kind="tool_call", tool_call=tc)
            meta = getattr(chunk, "usage_metadata", None)
            if meta is not None:
                usage = LLMUsage(
                    prompt_tokens=getattr(meta, "prompt_token_count", 0),
                    completion_tokens=getattr(meta, "candidates_token_count", 0),
                )
        self.cost_tracker.record(self.model, usage)
        yield StreamChunk(kind="stop", usage=usage)
