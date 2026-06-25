"""Non-blocking conversational steering for the live agent loop."""

from __future__ import annotations

import asyncio
import shlex
from dataclasses import dataclass, field
from typing import Any, Callable


STEER_COMMANDS = {
    "status", "plan", "pause", "resume", "skip", "abort", "add", "focus",
    "retry", "explain", "undo", "diff", "memory", "cost", "model",
    "budget", "verbose", "yolo", "help",
}


@dataclass
class SteerEvent:
    """A parsed command typed by the user during the session."""

    command: str
    args: list[str] = field(default_factory=list)
    raw: str = ""


def parse_command(line: str) -> SteerEvent | None:
    """Parse a ``/`` command line into a :class:`SteerEvent`."""
    line = line.strip()
    if not line.startswith("/"):
        return None
    parts = shlex.split(line[1:])
    if not parts:
        return None
    cmd, *args = parts
    return SteerEvent(command=cmd, args=args, raw=line)


class SteerQueue:
    """Async-safe queue of :class:`SteerEvent` objects."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[SteerEvent] = asyncio.Queue()

    async def push(self, evt: SteerEvent) -> None:
        await self._queue.put(evt)

    def pop_nowait(self) -> SteerEvent | None:
        try:
            return self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def pop(self, timeout: float = 0.0) -> SteerEvent | None:
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=timeout)
        except (asyncio.TimeoutError, TimeoutError):
            return None


@dataclass
class SteerState:
    """Live flags driven by user commands."""

    paused: bool = False
    aborted: bool = False
    verbose: bool = False
    yolo: bool = False
    skip_current: bool = False
    retry_current: bool = False
    focus_next: str | None = None
    new_tasks: list[str] = field(default_factory=list)
    model_override: str | None = None
    budget_override: int | None = None


class CommandDispatcher:
    """Translates parsed commands into mutations of :class:`SteerState`."""

    def __init__(self, state: SteerState, hooks: dict[str, Callable[[list[str]], str]] | None = None) -> None:
        """Build a dispatcher; ``hooks`` lets the caller plug in side effects."""
        self.state = state
        self.hooks: dict[str, Callable[[list[str]], str]] = hooks or {}

    def dispatch(self, evt: SteerEvent) -> str:
        """Apply ``evt`` to the state and return a short human-readable response."""
        cmd = evt.command
        if cmd not in STEER_COMMANDS:
            return f"unknown command: /{cmd}"
        if cmd == "pause":
            self.state.paused = True
            return "paused after current task"
        if cmd == "resume":
            self.state.paused = False
            return "resumed"
        if cmd == "skip":
            self.state.skip_current = True
            return "skipping current task"
        if cmd == "abort":
            self.state.aborted = True
            return "abort requested"
        if cmd == "retry":
            self.state.retry_current = True
            return "retry requested"
        if cmd == "verbose":
            self.state.verbose = not self.state.verbose
            return f"verbose = {self.state.verbose}"
        if cmd == "yolo":
            self.state.yolo = not self.state.yolo
            return f"yolo = {self.state.yolo}"
        if cmd == "add":
            if not evt.args:
                return "usage: /add <task description>"
            desc = " ".join(evt.args)
            self.state.new_tasks.append(desc)
            return f"task queued: {desc[:60]}"
        if cmd == "focus":
            if not evt.args:
                return "usage: /focus <task_id>"
            self.state.focus_next = evt.args[0]
            return f"focus next on {evt.args[0]}"
        if cmd == "model":
            if not evt.args:
                return "usage: /model <name> (or /model list)"
            if evt.args[0] == "list":
                hook = self.hooks.get("model")
                return hook(evt.args) if hook else "(no orchestrator attached)"
            self.state.model_override = evt.args[0]
            hook = self.hooks.get("model")
            if hook is not None:
                return hook(evt.args)
            return f"model override → {evt.args[0]} (queued; will apply on next task)"
        if cmd == "budget":
            if not evt.args or not evt.args[0].isdigit():
                return "usage: /budget <int>"
            self.state.budget_override = int(evt.args[0])
            return f"budget → {self.state.budget_override}"
        hook = self.hooks.get(cmd)
        if hook is not None:
            return hook(evt.args)
        return f"/{cmd} acknowledged"


async def interactive_input_loop(queue: SteerQueue, prompt: str = "> ") -> None:
    """Read lines from stdin in the background and push them onto ``queue``.

    Uses ``aioconsole.ainput`` when available; otherwise schedules ``input()``
    on a worker thread so the event loop is never blocked.
    """
    try:
        from aioconsole import ainput  # type: ignore[import-not-found]

        async def get_line() -> str:
            return await ainput(prompt)
    except ImportError:
        async def get_line() -> str:
            return await asyncio.to_thread(input, prompt)

    while True:
        try:
            line = await get_line()
        except (EOFError, KeyboardInterrupt):
            return
        evt = parse_command(line)
        if evt is not None:
            await queue.push(evt)


ASK_USER_TOOL = {
    "name": "ask_user",
    "description": (
        "Ask the human a clarifying question mid-task. Pauses this task until "
        "the user answers via the interactive prompt. Returns the answer string."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "context": {"type": "string"},
            "default": {"type": "string"},
        },
        "required": ["question"],
    },
}


class PendingQuestion:
    """Tracks an outstanding ``ask_user`` request awaiting a typed answer."""

    def __init__(self, question: str, default: str | None = None, timeout: float = 300.0) -> None:
        self.question = question
        self.default = default
        self.timeout = timeout
        self._future: asyncio.Future[str] = asyncio.get_event_loop().create_future()

    def answer(self, value: str) -> None:
        if not self._future.done():
            self._future.set_result(value)

    async def wait(self) -> str:
        try:
            return await asyncio.wait_for(self._future, timeout=self.timeout)
        except (asyncio.TimeoutError, TimeoutError):
            if self.default is not None:
                return self.default
            raise
