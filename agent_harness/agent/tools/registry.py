"""Universal tool registry with JSON Schema generation and retry/timeout policy."""

from __future__ import annotations

import asyncio
import inspect
import time
import types
import typing as t
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal, get_args, get_origin

from agent.audit import AuditLog
from agent.types import ToolCall, ToolResult


SideEffect = Literal["read_only", "destructive", "network", "compute"]


@dataclass
class ToolSpec:
    """Static metadata about a registered tool."""

    name: str
    description: str
    side_effect: SideEffect
    schema: dict[str, Any]
    timeout: float
    func: Callable[..., Awaitable[Any]]


_PY_TO_JSON = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    dict: "object",
    list: "array",
    type(None): "null",
}


def _python_type_to_schema(annotation: Any) -> dict[str, Any]:
    """Best-effort conversion of a Python type annotation to a JSON Schema fragment."""
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {"type": "string"}
    origin = get_origin(annotation)
    if origin is None:
        if annotation in _PY_TO_JSON:
            return {"type": _PY_TO_JSON[annotation]}
        if isinstance(annotation, type):
            return {"type": "string"}
        return {"type": "string"}
    if origin is t.Literal:
        choices = list(get_args(annotation))
        return {"type": "string", "enum": [str(c) for c in choices]}
    if origin in (list, t.List):
        (inner,) = get_args(annotation) or (str,)
        return {"type": "array", "items": _python_type_to_schema(inner)}
    if origin in (dict, t.Dict):
        return {"type": "object"}
    if origin is t.Union or origin is getattr(types, "UnionType", ()):
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return _python_type_to_schema(args[0])
        return {"type": "string"}
    return {"type": "string"}


def _build_schema(func: Callable[..., Any], description: str) -> dict[str, Any]:
    """Inspect ``func`` and produce a JSON Schema compatible with LLM tool-use."""
    sig = inspect.signature(func)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, param in sig.parameters.items():
        if name in {"self", "ctx"}:
            continue
        schema = _python_type_to_schema(param.annotation)
        properties[name] = schema
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return {
        "name": func.__name__,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


class ToolRegistry:
    """Holds registered tools, runs them with retry/timeout, and audits results."""

    def __init__(self, audit: AuditLog | None = None) -> None:
        """Create a registry with optional audit log; supply one to record activity."""
        self._tools: dict[str, ToolSpec] = {}
        self._audit = audit

    @property
    def audit(self) -> AuditLog | None:
        return self._audit

    def set_audit(self, audit: AuditLog) -> None:
        """Attach a structured audit log (used by the orchestrator at startup)."""
        self._audit = audit

    def tool(
        self,
        *,
        description: str,
        side_effect: SideEffect = "read_only",
        timeout: float = 60.0,
        name: str | None = None,
    ) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
        """Decorator that registers an async function as a tool."""

        def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
            if not asyncio.iscoroutinefunction(func):
                raise TypeError(f"Tool {func.__name__} must be async")
            tool_name = name or func.__name__
            schema = _build_schema(func, description)
            schema["name"] = tool_name
            self._tools[tool_name] = ToolSpec(
                name=tool_name,
                description=description,
                side_effect=side_effect,
                schema=schema,
                timeout=timeout,
                func=func,
            )
            return func

        return decorator

    def register(self, spec: ToolSpec) -> None:
        """Register an already-built :class:`ToolSpec`."""
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        """Return the spec for ``name`` or raise ``KeyError``."""
        return self._tools[name]

    def names(self) -> list[str]:
        """All registered tool names."""
        return list(self._tools.keys())

    def schemas(self) -> list[dict[str, Any]]:
        """Return JSON Schemas suitable for passing to the LLM tool-use API."""
        return [spec.schema for spec in self._tools.values()]

    def specs(self) -> list[ToolSpec]:
        """All registered tool specs (used by CLI ``agent tools list``)."""
        return list(self._tools.values())

    async def invoke(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        retries: int = 2,
        backoff: float = 0.5,
    ) -> ToolResult:
        """Invoke a tool by name with retry/backoff and timeout enforcement."""
        spec = self.get(name)
        call = ToolCall(id=str(uuid.uuid4()), name=name, arguments=arguments or {})
        last_err: Exception | None = None
        started = time.perf_counter()
        for attempt in range(retries + 1):
            try:
                output = await asyncio.wait_for(
                    spec.func(**call.arguments), timeout=spec.timeout
                )
                latency = (time.perf_counter() - started) * 1000.0
                result = ToolResult(
                    call_id=call.id,
                    name=name,
                    ok=True,
                    output=output,
                    latency_ms=latency,
                )
                if self._audit:
                    self._audit.append(
                        "tool_call",
                        {
                            "name": name,
                            "args": call.arguments,
                            "ok": True,
                            "latency_ms": latency,
                            "attempt": attempt,
                            "side_effect": spec.side_effect,
                        },
                    )
                return result
            except TimeoutError as exc:
                last_err = exc
                break
            except Exception as exc:  # noqa: BLE001 - propagated below
                last_err = exc
                if attempt < retries:
                    await asyncio.sleep(backoff * (2**attempt))
                    continue
                break

        latency = (time.perf_counter() - started) * 1000.0
        result = ToolResult(
            call_id=call.id,
            name=name,
            ok=False,
            output=None,
            error=f"{type(last_err).__name__}: {last_err}",
            latency_ms=latency,
        )
        if self._audit:
            self._audit.append(
                "tool_call",
                {
                    "name": name,
                    "args": call.arguments,
                    "ok": False,
                    "error": result.error,
                    "latency_ms": latency,
                    "side_effect": spec.side_effect,
                },
            )
        return result


GLOBAL_REGISTRY = ToolRegistry()
"""Module-level registry shared by the default tool implementations."""
