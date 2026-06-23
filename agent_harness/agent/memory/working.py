"""Working-memory tier: token-budgeted sliding context window + KV store."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.types import LLMMessage


def _count_tokens(text: str, model: str = "claude-sonnet-4-6") -> int:
    """Best-effort token count using ``tiktoken`` with a heuristic fallback."""
    try:
        import tiktoken

        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


@dataclass
class ScoredMessage:
    """A context message paired with its importance score."""

    message: LLMMessage
    importance: float
    age_index: int


class WorkingMemory:
    """Sliding-window context manager with token budget and importance scoring."""

    def __init__(self, budget_tokens: int = 180_000, model: str = "claude-sonnet-4-6") -> None:
        """Create a working memory with a max token budget.

        Args:
            budget_tokens: Total token budget (prompt side).
            model: Model name used for tiktoken encoding lookup.
        """
        self.budget_tokens = budget_tokens
        self.model = model
        self.messages: list[LLMMessage] = []
        self.current_task_id: str | None = None
        self._compression_callback = None

    def set_compression_callback(self, callback) -> None:
        """Register an async callable that summarises a list of messages → str."""
        self._compression_callback = callback

    def add(self, message: LLMMessage) -> None:
        """Append a message and recompute its token count."""
        message.tokens = _count_tokens(message.content, self.model)
        self.messages.append(message)

    def token_total(self) -> int:
        """Sum tokens across all retained messages."""
        return sum(m.tokens for m in self.messages)

    def score(self, idx: int, message: LLMMessage, current_task: str | None) -> float:
        """Score a message higher when it is recent, on-task, or contains tool output."""
        recency = (idx + 1) / len(self.messages) if self.messages else 0.0
        is_tool_result = 1.0 if message.role == "tool" else 0.0
        is_system = 5.0 if message.role == "system" else 0.0
        relevance = 0.0
        if current_task and current_task.lower() in message.content.lower():
            relevance = 1.0
        return recency * 1.0 + is_tool_result * 1.5 + is_system + relevance * 1.5

    async def compress_if_needed(self) -> int:
        """If over budget, summarise low-scoring messages and replace them.

        Returns:
            Number of messages that were compressed away.
        """
        total = self.token_total()
        if total <= self.budget_tokens:
            return 0
        scored = [
            ScoredMessage(m, self.score(i, m, self.current_task_id), i)
            for i, m in enumerate(self.messages)
        ]
        scored.sort(key=lambda s: s.importance)
        protected_ids: set[int] = set()
        if self.messages and self.messages[0].role == "system":
            protected_ids.add(id(self.messages[0]))
        to_compress: list[LLMMessage] = []
        target = total - int(self.budget_tokens * 0.7)
        freed = 0
        for s in scored:
            if id(s.message) in protected_ids:
                continue
            to_compress.append(s.message)
            freed += s.message.tokens
            if freed >= target:
                break
        if not to_compress:
            return 0
        summary_text = ""
        if self._compression_callback:
            summary_text = await self._compression_callback(to_compress)
        else:
            summary_text = "[compressed]\n" + "\n".join(
                f"{m.role}: {m.content[:200]}" for m in to_compress
            )
        summary = LLMMessage(role="system", content=f"<summary>\n{summary_text}\n</summary>")
        summary.tokens = _count_tokens(summary.content, self.model)
        removed = {id(m) for m in to_compress}
        self.messages = [m for m in self.messages if id(m) not in removed]
        self.messages.insert(1 if protected_ids else 0, summary)
        return len(to_compress)

    def to_serializable(self) -> list[dict[str, Any]]:
        """Return a JSON-serialisable view of the working set."""
        return [
            {
                "role": m.role,
                "content": m.content,
                "tool_call_id": m.tool_call_id,
                "name": m.name,
                "tokens": m.tokens,
            }
            for m in self.messages
        ]

    def restore(self, payload: list[dict[str, Any]]) -> None:
        """Replace state from a serialised list of messages."""
        self.messages = [
            LLMMessage(
                role=item["role"],
                content=item["content"],
                tool_call_id=item.get("tool_call_id"),
                name=item.get("name"),
                tokens=item.get("tokens", 0),
            )
            for item in payload
        ]


class KVStore:
    """Persistent key/value store with optional TTLs, backed by SQLite."""

    def __init__(self, path: Path | str) -> None:
        """Open (and initialise) the SQLite database at ``path``."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS kv ("
            "  k TEXT PRIMARY KEY,"
            "  v TEXT NOT NULL,"
            "  expires REAL"
            ")"
        )
        self.conn.commit()

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Store ``value`` (JSON-encoded) under ``key`` with optional TTL seconds.

        A ``ttl`` of ``None`` means never expire; a ``ttl`` of ``0`` expires immediately
        on the next read.
        """
        expires = time.time() + ttl if ttl is not None else None
        self.conn.execute(
            "INSERT OR REPLACE INTO kv(k, v, expires) VALUES (?,?,?)",
            (key, json.dumps(value), expires),
        )
        self.conn.commit()

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value for ``key`` or ``default`` if missing/expired."""
        row = self.conn.execute(
            "SELECT v, expires FROM kv WHERE k=?", (key,)
        ).fetchone()
        if row is None:
            return default
        v, expires = row
        if expires is not None and expires < time.time():
            self.conn.execute("DELETE FROM kv WHERE k=?", (key,))
            self.conn.commit()
            return default
        return json.loads(v)

    def delete(self, key: str) -> None:
        """Remove ``key`` if present."""
        self.conn.execute("DELETE FROM kv WHERE k=?", (key,))
        self.conn.commit()

    def keys(self) -> list[str]:
        """All non-expired keys."""
        now = time.time()
        rows = self.conn.execute(
            "SELECT k FROM kv WHERE expires IS NULL OR expires > ?", (now,)
        ).fetchall()
        return [r[0] for r in rows]
