"""Async message bus shared by all swarm members."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Literal


MessageType = Literal["FINDING", "QUESTION", "BLOCKER", "HANDOFF", "STATUS"]


@dataclass
class SwarmMessage:
    """A structured message posted on the bus."""

    sender_role: str
    target_role: str | None
    message_type: MessageType
    payload: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


class MessageBus:
    """Fan-out async pub/sub bus with per-role mailboxes + a broadcast log."""

    def __init__(self) -> None:
        self._mailboxes: dict[str, asyncio.Queue[SwarmMessage]] = {}
        self._broadcast_log: list[SwarmMessage] = []
        self._lock = asyncio.Lock()

    async def subscribe(self, role: str) -> asyncio.Queue[SwarmMessage]:
        """Register / fetch the queue for ``role``."""
        async with self._lock:
            q = self._mailboxes.get(role)
            if q is None:
                q = asyncio.Queue()
                self._mailboxes[role] = q
            return q

    async def publish(self, msg: SwarmMessage) -> None:
        """Route ``msg`` to its target queue (or every queue if target is ``None``)."""
        async with self._lock:
            self._broadcast_log.append(msg)
            targets = (
                [self._mailboxes[msg.target_role]]
                if msg.target_role and msg.target_role in self._mailboxes
                else list(self._mailboxes.values())
            )
        for q in targets:
            await q.put(msg)

    def log(self, limit: int = 50) -> list[SwarmMessage]:
        """Return the tail of the broadcast log."""
        return list(self._broadcast_log[-limit:])
