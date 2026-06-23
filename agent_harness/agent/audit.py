"""Append-only structured audit trail (JSONL)."""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class AuditLog:
    """Thread-safe append-only audit log writer."""

    def __init__(self, path: Path | str) -> None:
        """Create a log writer pointing at ``path``; parents are created."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, event: str, payload: dict[str, Any]) -> None:
        """Write a single JSON event line."""
        record = {
            "ts": datetime.now(UTC).isoformat(),
            "event": event,
            "payload": payload,
        }
        line = json.dumps(record, default=str)
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def tail(self, n: int = 50) -> list[dict[str, Any]]:
        """Return the last ``n`` events as parsed dicts."""
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as fh:
            lines = fh.readlines()
        out: list[dict[str, Any]] = []
        for line in lines[-n:]:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out
