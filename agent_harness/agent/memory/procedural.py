"""Procedural memory: the agent's library of *how* to do recurring tasks."""

from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ProcedureStep:
    """One step inside a saved procedure."""

    tool: str
    args_summary: str = ""
    ok: bool = True


@dataclass
class Procedure:
    """A reusable sequence of tool calls that has succeeded before."""

    id: int
    fingerprint: str
    task_description: str
    steps: list[ProcedureStep]
    success_rate: float
    avg_duration_ms: float
    last_used: float
    times_used: int
    last_modified_by_session: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "fingerprint": self.fingerprint,
            "task_description": self.task_description,
            "steps": [s.__dict__ for s in self.steps],
            "success_rate": self.success_rate,
            "avg_duration_ms": self.avg_duration_ms,
            "last_used": self.last_used,
            "times_used": self.times_used,
            "last_modified_by_session": self.last_modified_by_session,
        }


_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "is",
    "by", "at", "this", "that", "from", "into", "as", "be", "it",
}


def fingerprint_task(description: str) -> str:
    """Reduce a task description to a stable token-set fingerprint for matching."""
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_]+", description.lower())
    keep = sorted({t for t in tokens if t not in _STOPWORDS and len(t) > 2})
    return " ".join(keep[:12])


class ProceduralMemory:
    """SQLite-backed procedure store with auto-extraction and retrieval."""

    def __init__(self, path: Path | str) -> None:
        """Open (and lazily create) the procedure database at ``path``."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._init_schema()

    def _init_schema(self) -> None:
        c = self.conn
        c.execute("""
            CREATE TABLE IF NOT EXISTS procedures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fingerprint TEXT NOT NULL UNIQUE,
                task_description TEXT NOT NULL,
                steps_json TEXT NOT NULL,
                success_rate REAL NOT NULL DEFAULT 1.0,
                avg_duration_ms REAL NOT NULL DEFAULT 0.0,
                last_used REAL NOT NULL,
                times_used INTEGER NOT NULL DEFAULT 1,
                last_modified_by_session TEXT NOT NULL DEFAULT ''
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_proc_fp ON procedures(fingerprint)")
        c.commit()

    def record_successful_procedure(
        self,
        task_description: str,
        steps: list[ProcedureStep],
        duration_ms: float,
        session_id: str = "",
    ) -> Procedure:
        """Insert / update a procedure derived from a successful task."""
        if not steps:
            steps = []
        fp = fingerprint_task(task_description)
        existing = self._get_by_fingerprint(fp)
        steps_json = json.dumps([s.__dict__ for s in steps])
        now = time.time()
        if existing is None:
            cur = self.conn.execute(
                "INSERT INTO procedures(fingerprint, task_description, steps_json, "
                "success_rate, avg_duration_ms, last_used, times_used, last_modified_by_session) "
                "VALUES (?, ?, ?, 1.0, ?, ?, 1, ?)",
                (fp, task_description, steps_json, duration_ms, now, session_id),
            )
            self.conn.commit()
            return Procedure(
                id=cur.lastrowid or 0, fingerprint=fp, task_description=task_description,
                steps=steps, success_rate=1.0, avg_duration_ms=duration_ms,
                last_used=now, times_used=1, last_modified_by_session=session_id,
            )
        new_times = existing.times_used + 1
        new_rate = (existing.success_rate * existing.times_used + 1.0) / new_times
        new_avg = (existing.avg_duration_ms * existing.times_used + duration_ms) / new_times
        self.conn.execute(
            "UPDATE procedures SET steps_json=?, success_rate=?, avg_duration_ms=?, "
            "last_used=?, times_used=?, last_modified_by_session=? WHERE id=?",
            (steps_json, new_rate, new_avg, now, new_times, session_id, existing.id),
        )
        self.conn.commit()
        existing.steps = steps
        existing.success_rate = new_rate
        existing.avg_duration_ms = new_avg
        existing.last_used = now
        existing.times_used = new_times
        existing.last_modified_by_session = session_id
        return existing

    def record_failure(self, task_description: str) -> None:
        """Decrement success rate for a previously-known procedure when it fails."""
        fp = fingerprint_task(task_description)
        existing = self._get_by_fingerprint(fp)
        if existing is None:
            return
        new_times = existing.times_used + 1
        new_rate = (existing.success_rate * existing.times_used) / new_times
        self.conn.execute(
            "UPDATE procedures SET success_rate=?, times_used=?, last_used=? WHERE id=?",
            (new_rate, new_times, time.time(), existing.id),
        )
        self.conn.commit()

    def retrieve_relevant_procedure(
        self, task_description: str, min_success_rate: float = 0.8
    ) -> Procedure | None:
        """Best match for ``task_description``; ``None`` if no procedure passes the bar."""
        fp = fingerprint_task(task_description)
        best: Procedure | None = None
        best_score = 0.0
        tokens_q = set(fp.split())
        for row in self.conn.execute(
            "SELECT id, fingerprint, task_description, steps_json, success_rate, "
            "avg_duration_ms, last_used, times_used, last_modified_by_session "
            "FROM procedures WHERE success_rate >= ?",
            (min_success_rate,),
        ):
            tokens_r = set(row[1].split())
            if not tokens_q or not tokens_r:
                continue
            overlap = len(tokens_q & tokens_r) / max(1, len(tokens_q | tokens_r))
            if overlap < 0.4:
                continue
            score = overlap * row[4]
            if score > best_score:
                best_score = score
                best = self._row_to_procedure(row)
        return best

    def all_procedures(self) -> list[Procedure]:
        """Return every recorded procedure (admin / debug helper)."""
        return [self._row_to_procedure(r) for r in self.conn.execute(
            "SELECT id, fingerprint, task_description, steps_json, success_rate, "
            "avg_duration_ms, last_used, times_used, last_modified_by_session FROM procedures "
            "ORDER BY last_used DESC"
        )]

    def _get_by_fingerprint(self, fp: str) -> Procedure | None:
        row = self.conn.execute(
            "SELECT id, fingerprint, task_description, steps_json, success_rate, "
            "avg_duration_ms, last_used, times_used, last_modified_by_session "
            "FROM procedures WHERE fingerprint=?",
            (fp,),
        ).fetchone()
        return self._row_to_procedure(row) if row else None

    def _row_to_procedure(self, row: tuple) -> Procedure:
        try:
            steps_raw = json.loads(row[3])
        except json.JSONDecodeError:
            steps_raw = []
        steps = [ProcedureStep(**s) for s in steps_raw if isinstance(s, dict)]
        return Procedure(
            id=row[0], fingerprint=row[1], task_description=row[2], steps=steps,
            success_rate=float(row[4]), avg_duration_ms=float(row[5]), last_used=float(row[6]),
            times_used=int(row[7]), last_modified_by_session=row[8],
        )
