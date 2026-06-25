"""The ReAct micro-loop: thought → action → observation, plus parallel tool exec."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from agent.context import ContextManager
from agent.llm.base import LLMClient
from agent.prompts.executor import build_executor_prompt
from agent.tools.registry import ToolRegistry
from agent.types import LLMMessage, TaskNode, ToolCall, ToolResult


TASK_COMPLETE_TOOL = {
    "name": "task_complete",
    "description": "Signal that the current sub-task is complete. Provide a short summary "
    "and a list of artifact file paths produced or modified.",
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "artifacts": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["summary"],
    },
}


@dataclass
class LoopResult:
    """Outcome of running a single task's ReAct loop."""

    completed: bool
    summary: str
    artifacts: list[str]
    steps: int
    tool_results: list[ToolResult] = field(default_factory=list)
    blocked: bool = False
    error: str | None = None


class AgentLoop:
    """Drives one task through the ReAct micro-loop until completion or step limit."""

    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        context: ContextManager,
        cwd: str,
        step_limit: int = 25,
        ui_hook: Any | None = None,
    ) -> None:
        """Bind dependencies for the loop.

        Args:
            llm: LLM client used for executor calls.
            tools: Tool registry used to dispatch actions.
            context: :class:`ContextManager` holding working/episodic/semantic memory.
            cwd: Working directory string passed to prompts.
            step_limit: Hard step cap before forced summarisation.
            ui_hook: Optional object with ``on_tool_start`` / ``on_tool_end`` /
                ``on_assistant_text`` callbacks for live UI updates.
        """
        self.llm = llm
        self.tools = tools
        self.context = context
        self.cwd = cwd
        self.step_limit = step_limit
        self.ui = ui_hook

    async def _execute_tool_calls(self, calls: list[ToolCall]) -> list[ToolResult]:
        """Run independent tool calls concurrently and return results in order."""
        async def run_one(call: ToolCall) -> ToolResult:
            if self.ui is not None and hasattr(self.ui, "on_tool_start"):
                self.ui.on_tool_start(call)
            res = await self.tools.invoke(call.name, call.arguments)
            res.call_id = call.id
            self.context.episodic.record(call.name, call.arguments, res.output, res.ok)
            if self.ui is not None and hasattr(self.ui, "on_tool_end"):
                self.ui.on_tool_end(res)
            return res

        return await asyncio.gather(*(run_one(c) for c in calls))

    async def run(self, task: TaskNode) -> LoopResult:
        """Run the ReAct loop for ``task`` until ``task_complete`` or step limit."""
        executor_system = build_executor_prompt(
            task_id=task.id,
            description=task.description,
            complexity=task.complexity.value,
            anticipated_tools=task.anticipated_tools,
            cwd=self.cwd,
            step_limit=self.step_limit,
        )
        self.context.system(executor_system)
        tool_schemas = self.tools.schemas() + [TASK_COMPLETE_TOOL]
        tool_results: list[ToolResult] = []
        steps = 0
        while steps < self.step_limit:
            steps += 1
            await self.context.working.compress_if_needed()
            response = await self.llm.complete(
                self.context.working.messages, tools=tool_schemas
            )
            if response.text and self.ui is not None and hasattr(self.ui, "on_assistant_text"):
                self.ui.on_assistant_text(response.text)
            assistant = LLMMessage(role="assistant", content=response.text, tool_calls=response.tool_calls)
            self.context.working.add(assistant)

            done_call: ToolCall | None = next(
                (c for c in response.tool_calls if c.name == "task_complete"), None
            )
            if done_call is not None:
                summary = str(done_call.arguments.get("summary", ""))
                artifacts = list(done_call.arguments.get("artifacts", []) or [])
                blocked = summary.upper().startswith("BLOCKED:")
                return LoopResult(
                    completed=not blocked,
                    summary=summary,
                    artifacts=artifacts,
                    steps=steps,
                    tool_results=tool_results,
                    blocked=blocked,
                )

            actionable = [c for c in response.tool_calls if c.name != "task_complete"]
            if not actionable:
                self.context.user(
                    "No tool was called. Either invoke a tool or call `task_complete` if done."
                )
                continue

            results = await self._execute_tool_calls(actionable)
            tool_results.extend(results)
            for r in results:
                payload = r.output if r.ok else {"error": r.error}
                content = json.dumps(payload, default=str)[:32_000]
                self.context.tool(r.name, r.call_id, content)

        forced_summary = await self._force_summary(task)
        return LoopResult(
            completed=False,
            summary=f"STEP_LIMIT_REACHED: {forced_summary}",
            artifacts=[],
            steps=steps,
            tool_results=tool_results,
            blocked=True,
        )

    async def _force_summary(self, task: TaskNode) -> str:
        """Ask the LLM for a brief progress summary when the step limit is hit."""
        msg = LLMMessage(
            role="user",
            content=(
                f"Step limit hit for task {task.id}. In ONE sentence, summarise what was "
                "accomplished and what remains. Do not call any tool."
            ),
        )
        self.context.working.add(msg)
        response = await self.llm.complete(self.context.working.messages, tools=None)
        return response.text.strip()
