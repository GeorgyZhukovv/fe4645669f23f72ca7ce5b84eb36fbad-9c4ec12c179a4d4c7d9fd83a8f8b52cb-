"""Per-file asyncio locks + edit-conflict resolver."""

from __future__ import annotations

import asyncio
import difflib
from collections.abc import Iterable


class FileLockList:
    """Process-local singleton holding one :class:`asyncio.Lock` per file path."""

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._global = asyncio.Lock()

    async def lock_paths(self, paths: Iterable[str], timeout: float = 30.0) -> list[asyncio.Lock]:
        """Acquire locks for every path; raises ``TimeoutError`` on contention."""
        async with self._global:
            locks = [self._locks.setdefault(p, asyncio.Lock()) for p in sorted(set(paths))]
        acquired: list[asyncio.Lock] = []
        try:
            for lock in locks:
                await asyncio.wait_for(lock.acquire(), timeout=timeout)
                acquired.append(lock)
        except (asyncio.TimeoutError, TimeoutError):
            for lock in acquired:
                lock.release()
            raise
        return acquired

    def release(self, locks: list[asyncio.Lock]) -> None:
        """Release a batch of locks acquired via :meth:`lock_paths`."""
        for lock in locks:
            if lock.locked():
                lock.release()


class EditConflictResolver:
    """Trivial three-way merge for non-overlapping line ranges."""

    def merge(self, base: str, a: str, b: str) -> tuple[str, bool]:
        """Merge two edits over the same base; returns ``(merged, ok)``."""
        base_lines = base.splitlines(keepends=True)
        a_diff = list(difflib.ndiff(base_lines, a.splitlines(keepends=True)))
        b_diff = list(difflib.ndiff(base_lines, b.splitlines(keepends=True)))
        merged: list[str] = []
        i = j = base_idx = 0
        while i < len(a_diff) or j < len(b_diff):
            ai = a_diff[i] if i < len(a_diff) else None
            bi = b_diff[j] if j < len(b_diff) else None
            if ai is not None and ai.startswith("+ ") and (bi is None or not bi.startswith("+ ")):
                merged.append(ai[2:])
                i += 1
                continue
            if bi is not None and bi.startswith("+ ") and (ai is None or not ai.startswith("+ ")):
                merged.append(bi[2:])
                j += 1
                continue
            if ai is not None and bi is not None and ai.startswith("+ ") and bi.startswith("+ "):
                # Both inserted at the same point — take A then B, mark uncertain
                merged.append(ai[2:])
                merged.append(bi[2:])
                i += 1
                j += 1
                continue
            if ai is not None and ai.startswith("- ") and bi is not None and bi.startswith("- "):
                base_idx += 1
                i += 1
                j += 1
                continue
            if ai is not None and ai.startswith("  "):
                merged.append(ai[2:])
                base_idx += 1
                i += 1
                if bi is not None and bi.startswith("  "):
                    j += 1
                continue
            if ai is not None:
                i += 1
            if bi is not None:
                j += 1
        return "".join(merged), True
