"""Tests for the interactive steering command dispatcher."""

from __future__ import annotations

import asyncio

import pytest

from agent.ui.interactive import (
    CommandDispatcher,
    PendingQuestion,
    SteerEvent,
    SteerQueue,
    SteerState,
    parse_command,
)


def test_parse_command_basic() -> None:
    assert parse_command("/pause") == SteerEvent(command="pause", args=[], raw="/pause")
    parsed = parse_command("/add refactor the auth module")
    assert parsed is not None
    assert parsed.command == "add"
    assert parsed.args == ["refactor", "the", "auth", "module"]


def test_parse_command_ignores_non_slash() -> None:
    assert parse_command("just text") is None
    assert parse_command("") is None
    assert parse_command("/") is None


def test_dispatch_mutates_state() -> None:
    state = SteerState()
    disp = CommandDispatcher(state)
    disp.dispatch(parse_command("/pause"))
    assert state.paused is True
    disp.dispatch(parse_command("/resume"))
    assert state.paused is False
    disp.dispatch(parse_command("/verbose"))
    assert state.verbose is True
    disp.dispatch(parse_command("/skip"))
    assert state.skip_current is True
    disp.dispatch(parse_command("/abort"))
    assert state.aborted is True


def test_dispatch_add_and_focus() -> None:
    state = SteerState()
    disp = CommandDispatcher(state)
    disp.dispatch(parse_command("/add write tests for auth"))
    assert state.new_tasks == ["write tests for auth"]
    disp.dispatch(parse_command("/focus t3"))
    assert state.focus_next == "t3"


def test_dispatch_model_and_budget() -> None:
    state = SteerState()
    disp = CommandDispatcher(state)
    disp.dispatch(parse_command("/model claude-sonnet-4-6"))
    assert state.model_override == "claude-sonnet-4-6"
    disp.dispatch(parse_command("/budget 50000"))
    assert state.budget_override == 50000


def test_dispatch_unknown_returns_message() -> None:
    state = SteerState()
    disp = CommandDispatcher(state)
    msg = disp.dispatch(parse_command("/wat"))
    assert "unknown" in msg.lower()


def test_dispatch_yolo_toggle() -> None:
    state = SteerState()
    disp = CommandDispatcher(state)
    disp.dispatch(parse_command("/yolo"))
    assert state.yolo is True
    disp.dispatch(parse_command("/yolo"))
    assert state.yolo is False


@pytest.mark.asyncio
async def test_steer_queue_push_pop() -> None:
    q = SteerQueue()
    await q.push(SteerEvent(command="pause"))
    evt = q.pop_nowait()
    assert evt is not None and evt.command == "pause"
    assert q.pop_nowait() is None


@pytest.mark.asyncio
async def test_pending_question_default_on_timeout() -> None:
    pq = PendingQuestion(question="x?", default="hello", timeout=0.05)
    answer = await pq.wait()
    assert answer == "hello"


@pytest.mark.asyncio
async def test_pending_question_answered() -> None:
    pq = PendingQuestion(question="y?", timeout=5)
    pq.answer("typed-answer")
    assert await pq.wait() == "typed-answer"
