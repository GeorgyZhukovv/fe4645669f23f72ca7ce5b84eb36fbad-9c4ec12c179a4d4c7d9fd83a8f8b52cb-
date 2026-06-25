"""Tests for the live ``/model`` switch path in the orchestrator + dispatcher."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.config import AgentConfig
from agent.llm.registry import ModelRegistry
from agent.orchestrator import Orchestrator
from agent.tools.registry import ToolRegistry
from agent.ui.interactive import CommandDispatcher, SteerState, parse_command


@pytest.fixture(autouse=True)
def _registry_reset():
    ModelRegistry.reset()
    yield
    ModelRegistry.reset()


def _config(tmp_path: Path) -> AgentConfig:
    cfg = AgentConfig()
    cfg.memory.working_dir = str(tmp_path / "state")
    cfg.memory.global_memory_dir = str(tmp_path / "global")
    return cfg


class _StubLLM:
    """Minimal LLM stub. switch_model() doesn't actually call it; we just need an object."""

    def __init__(self, model: str = "stub-original") -> None:
        self.model = model

    async def complete(self, *a, **kw): ...
    async def stream(self, *a, **kw): ...


def test_orchestrator_rejects_unknown_model(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    orch = Orchestrator(config=cfg, registry=ToolRegistry(), llm=_StubLLM())
    ok, msg = orch.switch_model("does-not-exist")
    assert ok is False
    assert "unknown" in msg


def test_orchestrator_rejects_unavailable_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = _config(tmp_path)
    orch = Orchestrator(config=cfg, registry=ToolRegistry(), llm=_StubLLM())
    ok, msg = orch.switch_model("gpt-4o-mini")
    assert ok is False
    assert "credentials" in msg.lower() or "provider" in msg.lower()


def test_orchestrator_switches_when_available(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    cfg = _config(tmp_path)
    orch = Orchestrator(config=cfg, registry=ToolRegistry(), llm=_StubLLM())
    # Don't actually instantiate the AnthropicClient (network/SDK irrelevant here) —
    # patch build_client to return a stub.
    monkeypatch.setattr(
        "agent.llm.registry.ModelRegistry.build_client",
        lambda self, mid, cost_tracker=None: _StubLLM(model=mid),
    )
    ok, msg = orch.switch_model("claude-sonnet-4")
    assert ok is True, msg
    assert orch.llm.model == "claude-sonnet-4"


def test_dispatcher_invokes_model_hook() -> None:
    state = SteerState()
    seen: list[str] = []

    def hook(args: list[str]) -> str:
        seen.append(",".join(args))
        return f"switched to {args[0]}"

    disp = CommandDispatcher(state, hooks={"model": hook})
    msg = disp.dispatch(parse_command("/model claude-sonnet-4"))
    assert "claude-sonnet-4" in msg
    assert seen == ["claude-sonnet-4"]


def test_dispatcher_model_without_args_returns_usage() -> None:
    disp = CommandDispatcher(SteerState())
    msg = disp.dispatch(parse_command("/model"))
    assert "usage" in msg.lower()
