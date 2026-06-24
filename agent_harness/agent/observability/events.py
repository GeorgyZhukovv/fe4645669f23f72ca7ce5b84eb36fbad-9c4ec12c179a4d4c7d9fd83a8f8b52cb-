"""Structured session events + append-only JSONL sink."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


EventType = Literal[
    "task_started", "task_completed", "task_failed",
    "tool_called", "tool_result",
    "llm_call",
    "edit_applied", "edit_rolled_back",
    "memory_retrieved",
    "swarm_message",
    "user_command",
    "plan_revised",
    "session_started", "session_ended",
]


@dataclass
class SessionEvent:
    """Single structured event emitted during a session."""

    type: EventType
    ts: float = field(default_factory=time.time)
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"type": self.type, "ts": self.ts, "payload": self.payload}


class EventSink:
    """Thread-safe append-only JSONL event sink with in-memory mirror."""

    def __init__(self, path: Path | str) -> None:
        """Open (and lazily create) the JSONL file at ``path``."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._listeners: list = []

    def add_listener(self, callback) -> None:  # type: ignore[no-untyped-def]
        """Register a callable invoked synchronously for every event written."""
        self._listeners.append(callback)

    def emit(self, evt: SessionEvent) -> None:
        """Write one event to disk and notify listeners."""
        line = json.dumps(evt.to_dict(), default=str)
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        for cb in self._listeners:
            try:
                cb(evt)
            except Exception:
                continue

    def read_all(self) -> list[SessionEvent]:
        """Read every persisted event."""
        if not self.path.exists():
            return []
        out: list[SessionEvent] = []
        for raw in self.path.read_text(encoding="utf-8").splitlines():
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            out.append(SessionEvent(
                type=data["type"], ts=data["ts"], payload=data.get("payload", {}),
            ))
        return out

    def helper(self, type: EventType, **payload: Any) -> None:
        """Convenience emitter that builds the event for you."""
        self.emit(SessionEvent(type=type, payload=payload))
