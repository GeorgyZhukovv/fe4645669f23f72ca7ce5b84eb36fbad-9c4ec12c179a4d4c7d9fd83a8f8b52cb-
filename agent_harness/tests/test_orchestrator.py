"""Tests for the orchestrator: plan parsing, scheduling, persistence."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from agent.config import AgentConfig
from agent.llm.base import CostTracker, LLMClient
from agent.orchestrator import Orchestrator, Session, _strict_json_extract, assess_command
from agent.tools.registry import ToolRegistry
from agent.types import Complexity, LLMMessage, LLMResponse, LLMUsage, StreamChunk, TaskNode, TaskStatus, ToolCall


class ScriptedLLM(LLMClient):
    """LLM stub returning a queue of pre-baked responses."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        super().__init__(model="scripted", cost_tracker=CostTracker())
        self._queue = list(responses)
        self.calls: list[list[LLMMessage]] = []

    async def complete(
        self,
        messages,
        tools=None,
        system=None,
        temperature=0.2,
        max_tokens=4096,
    ) -> LLMResponse:
        self.calls.append(list(messages))
        if not self._queue:
            return LLMResponse(text="", tool_calls=[], usage=LLMUsage(), model=self.model)
        return self._queue.pop(0)

    async def stream(  # type: ignore[override]
        self, messages, tools=None, system=None, temperature=0.2, max_tokens=4096
    ) -> AsyncIterator[StreamChunk]:
        resp = await self.complete(messages, tools, system, temperature, max_tokens)
        yield StreamChunk(kind="text", text=resp.text)
        yield StreamChunk(kind="stop", usage=resp.usage)


def _config(tmp_path: Path) -> AgentConfig:
    cfg = AgentConfig()
    cfg.memory.working_dir = str(tmp_path / "state")
    cfg.safety.max_debug_iterations = 1
    cfg.safety.max_step_limit = 5
    return cfg


def test_strict_json_extract_parses_bare_object() -> None:
    assert _strict_json_extract('{"a": 1}') == {"a": 1}
    assert _strict_json_extract("```json\n{\"a\": 1}\n```") == {"a": 1}
    assert _strict_json_extract("prefix {\"a\": 1} suffix") == {"a": 1}
    assert _strict_json_extract("not json") == {}


def test_assess_command_flags_destructive() -> None:
    assert assess_command("rm -rf /tmp/x")["destructive"] is True
    assert assess_command("ls")["destructive"] is False


@pytest.mark.asyncio
async def test_plan_parses_structured_response(tmp_path: Path) -> None:
    plan = {
        "objective": "demo",
        "clarifications": [],
        "tasks": [
            {"id": "t1", "description": "first", "complexity": "S", "prerequisites": []},
            {"id": "t2", "description": "second", "complexity": "M", "prerequisites": ["t1"]},
        ],
    }
    llm = ScriptedLLM([LLMResponse(text=json.dumps(plan), tool_calls=[], usage=LLMUsage(), model="x")])
    cfg = _config(tmp_path)
    orch = Orchestrator(config=cfg, registry=ToolRegistry(), llm=llm)
    tasks, clarifications = await orch._plan("demo")
    assert [t.id for t in tasks] == ["t1", "t2"]
    assert tasks[0].complexity == Complexity.S
    assert tasks[1].prerequisites == ["t1"]
    assert clarifications == []


def test_pick_next_respects_dependencies(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    orch = Orchestrator(config=cfg, registry=ToolRegistry(), llm=ScriptedLLM([]))
    a = TaskNode(id="a", description="A", complexity=Complexity.S)
    b = TaskNode(id="b", description="B", complexity=Complexity.M, prerequisites=["a"])
    c = TaskNode(id="c", description="C", complexity=Complexity.L)
    orch.session = Session(
        id="x", objective="o", tasks={"a": a, "b": b, "c": c},
        order_hint=["a", "b", "c"], started_at=0,
    )
    nxt = orch._pick_next()
    assert nxt is not None and nxt.id == "a"
    a.status = TaskStatus.COMPLETE
    nxt = orch._pick_next()
    assert nxt is not None and nxt.id == "b"


def test_pick_next_uses_priority_for_ready_set(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    orch = Orchestrator(config=cfg, registry=ToolRegistry(), llm=ScriptedLLM([]))
    s = TaskNode(id="s", description="small", complexity=Complexity.S)
    xl = TaskNode(id="x", description="huge", complexity=Complexity.XL)
    orch.session = Session(
        id="x", objective="o", tasks={"s": s, "x": xl},
        order_hint=["x", "s"], started_at=0,
    )
    assert orch._pick_next().id == "s"


def test_apply_plan_changes_modifies_dag(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    orch = Orchestrator(config=cfg, registry=ToolRegistry(), llm=ScriptedLLM([]))
    a = TaskNode(id="a", description="A")
    b = TaskNode(id="b", description="B")
    orch.session = Session(
        id="x", objective="o", tasks={"a": a, "b": b}, order_hint=["a", "b"], started_at=0,
    )
    orch._apply_plan_changes([
        {"action": "remove", "task_id": "a"},
        {"action": "reprioritize", "task_id": "b"},
        {"action": "add", "task_id": "c", "rationale": "extra"},
    ])
    assert orch.session.tasks["a"].status == TaskStatus.SKIPPED
    assert orch.session.order_hint[0] == "b"
    assert "c" in orch.session.tasks


def test_save_and_load_session_round_trip(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    orch = Orchestrator(config=cfg, registry=ToolRegistry(), llm=ScriptedLLM([]))
    sess = Session(
        id="abcdef",
        objective="demo",
        tasks={"a": TaskNode(id="a", description="A")},
        order_hint=["a"],
        started_at=1.0,
    )
    orch.session = sess
    from agent.context import ContextManager
    orch.context = ContextManager.from_config(cfg, session_id="abcdef")
    orch.save_session()

    other = Orchestrator(config=cfg, registry=ToolRegistry(), llm=ScriptedLLM([]))
    loaded = other.load_session("abcdef")
    assert loaded.id == "abcdef"
    assert loaded.tasks["a"].description == "A"


def test_list_sessions_returns_serialized_view(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    orch = Orchestrator(config=cfg, registry=ToolRegistry(), llm=ScriptedLLM([]))
    sess = Session(
        id="zzz", objective="o", tasks={}, order_hint=[], started_at=0,
    )
    orch.session = sess
    from agent.context import ContextManager
    orch.context = ContextManager.from_config(cfg, session_id="zzz")
    orch.save_session()
    rows = orch.list_sessions()
    assert any(r["id"] == "zzz" for r in rows)


@pytest.mark.asyncio
async def test_loop_task_complete_short_circuits(tmp_path: Path) -> None:
    from agent.context import ContextManager
    from agent.loop import AgentLoop

    cfg = _config(tmp_path)
    completion_call = ToolCall(
        id="done-1", name="task_complete", arguments={"summary": "ok", "artifacts": []}
    )
    llm = ScriptedLLM([
        LLMResponse(text="thinking", tool_calls=[completion_call], usage=LLMUsage(), model="x"),
    ])
    ctx = ContextManager.from_config(cfg, session_id="sess1")
    loop = AgentLoop(llm=llm, tools=ToolRegistry(), context=ctx, cwd=str(tmp_path), step_limit=5)
    task = TaskNode(id="t", description="d", complexity=Complexity.S)
    res = await loop.run(task)
    assert res.completed
    assert res.summary == "ok"
    assert res.steps == 1


@pytest.mark.asyncio
async def test_loop_runs_tool_and_completes(tmp_path: Path) -> None:
    from agent.context import ContextManager
    from agent.loop import AgentLoop

    cfg = _config(tmp_path)
    reg = ToolRegistry()

    @reg.tool(description="add", side_effect="read_only", timeout=2.0)
    async def add(a: int, b: int) -> int:
        return a + b

    call = ToolCall(id="c1", name="add", arguments={"a": 1, "b": 2})
    done = ToolCall(id="d1", name="task_complete", arguments={"summary": "done", "artifacts": []})
    llm = ScriptedLLM([
        LLMResponse(text="will add", tool_calls=[call], usage=LLMUsage(), model="x"),
        LLMResponse(text="now done", tool_calls=[done], usage=LLMUsage(), model="x"),
    ])
    ctx = ContextManager.from_config(cfg, session_id="sess2")
    loop = AgentLoop(llm=llm, tools=reg, context=ctx, cwd=str(tmp_path), step_limit=5)
    task = TaskNode(id="t", description="d", complexity=Complexity.S, anticipated_tools=["add"])
    res = await loop.run(task)
    assert res.completed
    assert len(res.tool_results) == 1
    assert res.tool_results[0].output == 3


@pytest.mark.asyncio
async def test_loop_step_limit_forces_summary(tmp_path: Path) -> None:
    from agent.context import ContextManager
    from agent.loop import AgentLoop

    cfg = _config(tmp_path)
    reg = ToolRegistry()

    @reg.tool(description="noop", side_effect="read_only", timeout=2.0)
    async def noop() -> str:
        return "ok"

    call = ToolCall(id="c", name="noop", arguments={})
    responses = [LLMResponse(text=f"step {i}", tool_calls=[call], usage=LLMUsage(), model="x") for i in range(3)]
    responses.append(LLMResponse(text="forced summary", tool_calls=[], usage=LLMUsage(), model="x"))
    llm = ScriptedLLM(responses)
    ctx = ContextManager.from_config(cfg, session_id="sess3")
    loop = AgentLoop(llm=llm, tools=reg, context=ctx, cwd=str(tmp_path), step_limit=3)
    task = TaskNode(id="t", description="d", complexity=Complexity.S)
    res = await loop.run(task)
    assert res.blocked is True
    assert "STEP_LIMIT_REACHED" in res.summary
