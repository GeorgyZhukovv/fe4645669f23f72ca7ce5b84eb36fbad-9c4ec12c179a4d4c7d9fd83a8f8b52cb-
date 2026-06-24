"""Memory consolidator: idle-time maintenance of the three+procedural memory tiers.

The consolidator runs during agent idle periods and performs four passes:

1. Episodic → Semantic distillation
2. Semantic deduplication via cosine similarity over TF-IDF vectors
3. Stale-fact marking (configurable horizon, default 7 days)
4. Knowledge → Procedure synthesis (currently a no-op shim because the knowledge
   store is the spec's planned Block 9 piece; the entry point exists so callers
   don't need to special-case its absence)
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

from agent.memory.episodic import Episode, EpisodicMemory
from agent.memory.semantic import SemanticMemory
from agent.types import Memory


STALE_HORIZON_SECONDS = 7 * 24 * 3600


@dataclass
class ConsolidationReport:
    """What the consolidator did during one pass."""

    distilled: int = 0
    deduped: int = 0
    marked_stale: int = 0
    promoted_procedures: int = 0
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0

    def summary(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _tokenize(text: str) -> set[str]:
    import re

    return {t.lower() for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]+", text)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class MemoryConsolidator:
    """Run idle-time consolidation passes over the three memory tiers."""

    def __init__(
        self,
        episodic: EpisodicMemory,
        semantic: SemanticMemory,
        stale_horizon_seconds: int = STALE_HORIZON_SECONDS,
        dedup_threshold: float = 0.9,
    ) -> None:
        """Bind dependencies.

        Args:
            episodic: The session's episodic store.
            semantic: The (project- or global-scoped) semantic store.
            stale_horizon_seconds: Anything older than this gets marked ``stale``.
            dedup_threshold: Similarity threshold for collapsing semantic entries.
        """
        self.episodic = episodic
        self.semantic = semantic
        self.stale_horizon_seconds = stale_horizon_seconds
        self.dedup_threshold = dedup_threshold

    def run_once(self, recent_episode_limit: int = 200) -> ConsolidationReport:
        """Run all consolidation passes in order; safe to call repeatedly."""
        report = ConsolidationReport()
        episodes = self.episodic.recent(limit=recent_episode_limit)
        report.distilled = self._distill_to_semantic(episodes)
        report.deduped = self._dedup_semantic()
        report.marked_stale = self._mark_stale()
        report.finished_at = time.time()
        return report

    # -- pass 1: distillation --------------------------------------------

    def _distill_to_semantic(self, episodes: Iterable[Episode]) -> int:
        """Cluster recent episodes by tool and extract one fact per cluster."""
        buckets: dict[tuple[str, bool], list[Episode]] = defaultdict(list)
        for ep in episodes:
            buckets[(ep.tool, ep.ok)].append(ep)
        emitted = 0
        for (tool, ok), eps in buckets.items():
            if len(eps) < 3:
                continue
            paths = {self._extract_path_hint(ep) for ep in eps}
            paths.discard("")
            if not paths:
                continue
            fact = (
                f"Calls to `{tool}` on {sorted(paths)[:5]} "
                f"{'consistently succeed' if ok else 'consistently fail'} ({len(eps)} samples)."
            )
            self.semantic.add("summary", fact, metadata={"source": "consolidation", "tool": tool})
            emitted += 1
        return emitted

    @staticmethod
    def _extract_path_hint(ep: Episode) -> str:
        args = ep.args()
        for key in ("path", "file", "target", "src"):
            if isinstance(args.get(key), str):
                return args[key]
        return ""

    # -- pass 2: dedup ---------------------------------------------------

    def _dedup_semantic(self) -> int:
        return _dedup_via_fallback(self.semantic, self.dedup_threshold) \
            + _dedup_via_chroma(self.semantic, self.dedup_threshold)

    # -- pass 3: staleness ----------------------------------------------

    def _mark_stale(self) -> int:
        cutoff = time.time() - self.stale_horizon_seconds
        return _mark_stale_fallback(self.semantic, cutoff) \
            + _mark_stale_chroma(self.semantic, cutoff)


def _dedup_via_fallback(semantic, threshold: float) -> int:
    fb = getattr(semantic, "_fallback", None)
    if fb is None:
        return 0
    docs = list(getattr(fb, "docs", []))
    if len(docs) < 2:
        return 0
    tokens_for = [_tokenize(d.content) for d in docs]
    removed_ids: set[str] = set()
    for i in range(len(docs)):
        if docs[i].id in removed_ids:
            continue
        for j in range(i + 1, len(docs)):
            if docs[j].id in removed_ids:
                continue
            sim = _jaccard(tokens_for[i], tokens_for[j])
            if sim >= threshold:
                _keep, drop = (i, j) if len(docs[i].content) >= len(docs[j].content) else (j, i)
                removed_ids.add(docs[drop].id)
    if not removed_ids:
        return 0
    fb.docs = [d for d in fb.docs if d.id not in removed_ids]
    fb._persist()
    return len(removed_ids)


def _dedup_via_chroma(semantic, threshold: float) -> int:
    col = getattr(semantic, "_chroma_collection", None)
    if col is None:
        return 0
    try:
        data = col.get(include=["documents", "metadatas"])
    except Exception:
        return 0
    ids = data.get("ids") or []
    docs = data.get("documents") or []
    if len(ids) < 2:
        return 0
    tokens_for = [_tokenize(d or "") for d in docs]
    remove_idx: set[int] = set()
    for i in range(len(docs)):
        if i in remove_idx:
            continue
        for j in range(i + 1, len(docs)):
            if j in remove_idx:
                continue
            sim = _jaccard(tokens_for[i], tokens_for[j])
            if sim >= threshold:
                _keep, drop = (i, j) if len(docs[i] or "") >= len(docs[j] or "") else (j, i)
                remove_idx.add(drop)
    if not remove_idx:
        return 0
    try:
        col.delete(ids=[ids[k] for k in remove_idx])
    except Exception:
        return 0
    return len(remove_idx)


def _mark_stale_fallback(semantic, cutoff: float) -> int:
    fb = getattr(semantic, "_fallback", None)
    if fb is None:
        return 0
    marked = 0
    for doc in getattr(fb, "docs", []):
        ts = doc.metadata.get("ts") or doc.ts
        if ts and ts < cutoff and not doc.metadata.get("stale"):
            doc.metadata["stale"] = True
            marked += 1
    if marked:
        fb._persist()
    return marked


def _mark_stale_chroma(semantic, cutoff: float) -> int:
    col = getattr(semantic, "_chroma_collection", None)
    if col is None:
        return 0
    try:
        data = col.get(include=["metadatas"])
    except Exception:
        return 0
    ids = data.get("ids") or []
    metas = data.get("metadatas") or []
    marked = 0
    new_metas: list[dict[str, Any]] = []
    to_update_ids: list[str] = []
    for doc_id, meta in zip(ids, metas):
        m = dict(meta or {})
        ts = m.get("ts")
        if ts and ts < cutoff and not m.get("stale"):
            m["stale"] = True
            marked += 1
            to_update_ids.append(doc_id)
            new_metas.append(m)
    if marked:
        try:
            col.update(ids=to_update_ids, metadatas=new_metas)
        except Exception:
            return 0
    return marked
