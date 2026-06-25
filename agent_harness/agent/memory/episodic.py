"""Episodic memory: SQLite log of every tool call in the current session."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Episode:
    """One recorded episode (a tool call + result)."""

    id: int
    session_id: str
    ts: float
    tool: str
    args_json: str
    result_json: str
    ok: bool

    def args(self) -> dict[str, Any]:
        return json.loads(self.args_json)

    def result(self) -> Any:
        return json.loads(self.result_json)


class EpisodicMemory:
    """SQLite-backed log of tool calls, with FTS over args + result text."""

    def __init__(self, path: Path | str, session_id: str) -> None:
        """Open the episodic DB. FTS is enabled if the SQLite build supports it.

        Args:
            path: Path to the SQLite file.
            session_id: Logical session identifier used to scope queries.
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._init_schema()

    def _init_schema(self) -> None:
        c = self.conn
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                ts REAL NOT NULL,
                tool TEXT NOT NULL,
                args_json TEXT NOT NULL,
                result_json TEXT NOT NULL,
                ok INTEGER NOT NULL
            )
            """
        )
        c.execute("CREATE INDEX IF NOT EXISTS idx_session ON episodes(session_id)")
        try:
            c.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5("
                "  tool, args_json, result_json, content='episodes', content_rowid='id')"
            )
            c.execute(
                """
                CREATE TRIGGER IF NOT EXISTS episodes_ai AFTER INSERT ON episodes BEGIN
                    INSERT INTO episodes_fts(rowid, tool, args_json, result_json)
                    VALUES (new.id, new.tool, new.args_json, new.result_json);
                END
                """
            )
            self.fts_enabled = True
        except sqlite3.OperationalError:
            self.fts_enabled = False
        c.commit()

    def record(self, tool: str, args: dict[str, Any], result: Any, ok: bool) -> int:
        """Insert one episode and return its rowid."""
        cur = self.conn.execute(
            "INSERT INTO episodes(session_id, ts, tool, args_json, result_json, ok) VALUES (?,?,?,?,?,?)",
            (
                self.session_id,
                time.time(),
                tool,
                json.dumps(args, default=str),
                json.dumps(result, default=str),
                1 if ok else 0,
            ),
        )
        self.conn.commit()
        return cur.lastrowid or 0

    def search(self, query: str, limit: int = 20) -> list[Episode]:
        """Full-text or fallback substring search."""
        rows: list[tuple] = []
        if self.fts_enabled:
            try:
                rows = self.conn.execute(
                    "SELECT e.id, e.session_id, e.ts, e.tool, e.args_json, e.result_json, e.ok "
                    "FROM episodes e JOIN episodes_fts f ON e.id = f.rowid "
                    "WHERE episodes_fts MATCH ? LIMIT ?",
                    (query, limit),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []
        if not rows:
            like = f"%{query}%"
            rows = self.conn.execute(
                "SELECT id, session_id, ts, tool, args_json, result_json, ok "
                "FROM episodes WHERE tool LIKE ? OR args_json LIKE ? OR result_json LIKE ? "
                "ORDER BY ts DESC LIMIT ?",
                (like, like, like, limit),
            ).fetchall()
        return [
            Episode(
                id=r[0], session_id=r[1], ts=r[2], tool=r[3],
                args_json=r[4], result_json=r[5], ok=bool(r[6]),
            )
            for r in rows
        ]

    def recent(self, limit: int = 20) -> list[Episode]:
        """Return the most recent episodes for this session."""
        rows = self.conn.execute(
            "SELECT id, session_id, ts, tool, args_json, result_json, ok "
            "FROM episodes WHERE session_id=? ORDER BY ts DESC LIMIT ?",
            (self.session_id, limit),
        ).fetchall()
        return [
            Episode(
                id=r[0], session_id=r[1], ts=r[2], tool=r[3],
                args_json=r[4], result_json=r[5], ok=bool(r[6]),
            )
            for r in rows
        ]
