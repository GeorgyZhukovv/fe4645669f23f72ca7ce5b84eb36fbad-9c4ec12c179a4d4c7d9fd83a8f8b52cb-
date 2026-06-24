"""Replay a session's event stream and produce a structured narrative report."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from agent.observability.events import EventSink, SessionEvent


class ReplayPlayer:
    """Plays back a session's events at configurable speed."""

    def __init__(self, events: list[SessionEvent], speed: float = 1.0, console: Console | None = None) -> None:
        """Build a player from a pre-loaded event list."""
        self.events = events
        self.speed = max(speed, 0.0)
        self.console = console or Console()
        self.paused = False
        self.cursor = 0

    @classmethod
    def from_file(cls, path: Path | str, speed: float = 1.0) -> "ReplayPlayer":
        """Build a player from a session JSONL file."""
        sink = EventSink(path)
        return cls(sink.read_all(), speed=speed)

    def step(self) -> SessionEvent | None:
        """Advance one event and return it, or ``None`` at end-of-stream."""
        if self.cursor >= len(self.events):
            return None
        evt = self.events[self.cursor]
        self.cursor += 1
        return evt

    def seek(self, index: int) -> None:
        """Jump cursor to ``index``."""
        self.cursor = max(0, min(index, len(self.events)))

    def iter_with_gaps(self) -> Iterator[tuple[SessionEvent, float]]:
        """Yield each event with the wall-clock gap to the previous event."""
        prev_ts: float | None = None
        for evt in self.events:
            gap = (evt.ts - prev_ts) if prev_ts is not None else 0.0
            prev_ts = evt.ts
            yield evt, gap

    async def play(self) -> None:
        """Render every event in order to the console, respecting ``speed``."""
        for evt, gap in self.iter_with_gaps():
            if self.speed > 0 and gap > 0:
                await asyncio.sleep(gap / self.speed)
            self.console.print(_format_event(evt))


def _format_event(evt: SessionEvent) -> str:
    """Render an event as a single colored console line."""
    color = {
        "task_started": "cyan",
        "task_completed": "green",
        "task_failed": "red",
        "tool_called": "yellow",
        "tool_result": "magenta",
        "llm_call": "blue",
        "edit_applied": "green",
        "edit_rolled_back": "red",
        "session_started": "bold cyan",
        "session_ended": "bold green",
    }.get(evt.type, "white")
    payload = str(evt.payload)[:140]
    return f"[{color}][{time.strftime('%H:%M:%S', time.localtime(evt.ts))}] {evt.type}[/{color}] {payload}"


def render_session_report(events: list[SessionEvent]) -> str:
    """Produce a markdown narrative summary of a session's events."""
    if not events:
        return "# Session report\n\n(no events)\n"
    objective = ""
    started_at = events[0].ts
    ended_at = events[-1].ts
    task_lines: list[str] = []
    edits: list[str] = []
    tools_count: dict[str, int] = {}
    total_cost = 0.0
    total_tokens = 0
    for evt in events:
        if evt.type == "session_started":
            objective = evt.payload.get("objective", "")
        elif evt.type == "task_completed":
            task_lines.append(f"- ✅ **{evt.payload.get('task_id', '?')}**: {evt.payload.get('summary', '')[:200]}")
        elif evt.type == "task_failed":
            task_lines.append(f"- ❌ **{evt.payload.get('task_id', '?')}**: {evt.payload.get('error', '')[:200]}")
        elif evt.type == "edit_applied":
            edits.append(
                f"- {evt.payload.get('edit_id', '?')[:8]}: "
                f"{evt.payload.get('files_changed', 0)} files, "
                f"+{evt.payload.get('lines_added', 0)}/-{evt.payload.get('lines_removed', 0)}"
            )
        elif evt.type == "tool_called":
            tools_count[evt.payload.get("tool_name", "?")] = tools_count.get(evt.payload.get("tool_name", "?"), 0) + 1
        elif evt.type == "llm_call":
            total_cost += float(evt.payload.get("cost_usd", 0))
            total_tokens += int(evt.payload.get("prompt_tokens", 0)) + int(evt.payload.get("completion_tokens", 0))

    duration = ended_at - started_at
    out: list[str] = []
    out.append(f"# Session report\n")
    out.append(f"**Objective:** {objective}\n")
    out.append(f"**Duration:** {duration:.1f}s\n")
    out.append(f"**Total tokens:** {total_tokens}\n")
    out.append(f"**Total cost:** ${total_cost:.4f}\n")
    out.append(f"\n## Tasks\n")
    out.extend(task_lines or ["(none)"])
    out.append(f"\n## Edits\n")
    out.extend(edits or ["(none)"])
    out.append(f"\n## Tool usage\n")
    for tool, count in sorted(tools_count.items(), key=lambda kv: kv[1], reverse=True):
        out.append(f"- {tool}: {count}")
    return "\n".join(out) + "\n"
