"""Tests for the observability layer."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.observability.dashboard import aggregate, historical_total
from agent.observability.events import EventSink, SessionEvent
from agent.observability.replay import ReplayPlayer, render_session_report


def test_event_sink_round_trip(tmp_path: Path) -> None:
    sink = EventSink(tmp_path / "events.jsonl")
    sink.helper("session_started", objective="demo")
    sink.helper("task_started", task_id="t1", description="do thing")
    sink.helper("llm_call", model="m", prompt_tokens=10, completion_tokens=5, cost_usd=0.01)
    sink.helper("task_completed", task_id="t1", summary="done")
    sink.helper("session_ended", outcome="complete")
    events = sink.read_all()
    assert [e.type for e in events] == [
        "session_started", "task_started", "llm_call", "task_completed", "session_ended",
    ]


def test_event_listener_runs_synchronously(tmp_path: Path) -> None:
    sink = EventSink(tmp_path / "events.jsonl")
    seen: list[str] = []
    sink.add_listener(lambda evt: seen.append(evt.type))
    sink.helper("session_started", objective="x")
    sink.helper("session_ended", outcome="complete")
    assert seen == ["session_started", "session_ended"]


def test_aggregate_breakdown() -> None:
    events = [
        SessionEvent(type="task_started", payload={"task_id": "t1"}),
        SessionEvent(type="llm_call", payload={"model": "m", "prompt_tokens": 100, "completion_tokens": 50, "cost_usd": 0.5}),
        SessionEvent(type="tool_called", payload={"tool_name": "file_read"}),
        SessionEvent(type="tool_called", payload={"tool_name": "file_read"}),
    ]
    agg = aggregate(events)
    assert agg["total_cost"] == pytest.approx(0.5)
    assert agg["by_tool"]["file_read"] == 2
    assert agg["by_task"]["t1"] == 150


def test_render_session_report_summarizes(tmp_path: Path) -> None:
    events = [
        SessionEvent(type="session_started", payload={"objective": "demo"}),
        SessionEvent(type="task_completed", payload={"task_id": "t1", "summary": "ok"}),
        SessionEvent(type="edit_applied", payload={"edit_id": "abcd1234", "files_changed": 2, "lines_added": 5, "lines_removed": 1}),
        SessionEvent(type="llm_call", payload={"model": "m", "prompt_tokens": 10, "completion_tokens": 5, "cost_usd": 0.01}),
        SessionEvent(type="session_ended", payload={"outcome": "complete"}),
    ]
    md = render_session_report(events)
    assert "# Session report" in md
    assert "demo" in md
    assert "t1" in md
    assert "abcd1234" in md


def test_replay_player_iterates_in_order() -> None:
    events = [
        SessionEvent(type="session_started", payload={}),
        SessionEvent(type="session_ended", payload={}),
    ]
    player = ReplayPlayer(events, speed=0)
    assert player.step().type == "session_started"
    assert player.step().type == "session_ended"
    assert player.step() is None


def test_historical_total(tmp_path: Path) -> None:
    (tmp_path / "events" / "s1.jsonl").parent.mkdir(parents=True)
    sink = EventSink(tmp_path / "events" / "s1.jsonl")
    sink.helper("llm_call", model="m", prompt_tokens=10, completion_tokens=5, cost_usd=0.02)
    total = historical_total(tmp_path / "events")
    assert total["total_cost"] == pytest.approx(0.02)
