"""Tests for the memory consolidator."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from agent.memory.consolidation import MemoryConsolidator
from agent.memory.episodic import EpisodicMemory
from agent.memory.semantic import SemanticMemory


def _seed_semantic_with_duplicates(sem: SemanticMemory) -> None:
    sem.add("fact", "Python uses indentation for blocks")
    sem.add("fact", "Python uses indentation for blocks indeed")  # near-duplicate
    sem.add("fact", "JavaScript uses curly braces for blocks")


def test_dedup_removes_near_duplicates(tmp_path: Path) -> None:
    sem = SemanticMemory(tmp_path / "sem")
    _seed_semantic_with_duplicates(sem)
    ep = EpisodicMemory(tmp_path / "ep.sqlite", session_id="s")
    cons = MemoryConsolidator(ep, sem, dedup_threshold=0.7)
    report = cons.run_once()
    assert report.deduped >= 1


def test_distillation_emits_summary(tmp_path: Path) -> None:
    sem = SemanticMemory(tmp_path / "sem2")
    ep = EpisodicMemory(tmp_path / "ep.sqlite", session_id="s")
    for _ in range(4):
        ep.record("file_read", {"path": "auth/service.py"}, {"size": 200}, ok=True)
    cons = MemoryConsolidator(ep, sem)
    report = cons.run_once()
    assert report.distilled >= 1
    found = sem.retrieve_relevant_memories("file_read", top_k=5)
    assert any("file_read" in m.content for m in found)


def test_mark_stale_when_old(tmp_path: Path) -> None:
    sem = SemanticMemory(tmp_path / "sem3")
    sem.add("fact", "ancient knowledge", metadata={"ts": time.time() - (8 * 24 * 3600)})
    sem.add("fact", "fresh knowledge", metadata={"ts": time.time()})
    ep = EpisodicMemory(tmp_path / "ep.sqlite", session_id="s")
    cons = MemoryConsolidator(ep, sem, stale_horizon_seconds=7 * 24 * 3600)
    report = cons.run_once()
    assert report.marked_stale >= 1
    if sem._chroma_collection is not None:
        data = sem._chroma_collection.get(include=["metadatas"])
        assert any((m or {}).get("stale") for m in (data.get("metadatas") or []))
    else:
        fb = sem._fallback
        assert fb is not None
        assert any(d.metadata.get("stale") for d in fb.docs)


def test_run_once_returns_report_even_when_empty(tmp_path: Path) -> None:
    sem = SemanticMemory(tmp_path / "sem4")
    ep = EpisodicMemory(tmp_path / "ep.sqlite", session_id="s")
    cons = MemoryConsolidator(ep, sem)
    report = cons.run_once()
    assert report.distilled == 0
    assert report.deduped == 0
    assert report.finished_at >= report.started_at
