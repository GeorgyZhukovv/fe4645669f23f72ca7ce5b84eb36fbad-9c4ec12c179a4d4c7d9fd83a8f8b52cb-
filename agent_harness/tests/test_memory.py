"""Tests for the three memory tiers."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from agent.memory.episodic import EpisodicMemory
from agent.memory.semantic import SemanticMemory
from agent.memory.working import KVStore, WorkingMemory, _count_tokens
from agent.types import LLMMessage


def test_token_count_falls_back_when_no_tiktoken_for_model() -> None:
    n = _count_tokens("hello world", model="unknown-model-xyz")
    assert n > 0


def test_working_memory_token_total_accumulates() -> None:
    wm = WorkingMemory(budget_tokens=200)
    wm.add(LLMMessage(role="system", content="be helpful"))
    wm.add(LLMMessage(role="user", content="hi"))
    total = wm.token_total()
    assert total >= 2


def test_working_memory_serialize_and_restore() -> None:
    wm = WorkingMemory(budget_tokens=2000)
    wm.add(LLMMessage(role="user", content="x"))
    payload = wm.to_serializable()
    wm2 = WorkingMemory(budget_tokens=2000)
    wm2.restore(payload)
    assert len(wm2.messages) == 1
    assert wm2.messages[0].role == "user"


@pytest.mark.asyncio
async def test_working_memory_compresses_when_over_budget() -> None:
    wm = WorkingMemory(budget_tokens=20)

    async def summarise(_msgs):
        return "summarised"

    wm.set_compression_callback(summarise)
    wm.add(LLMMessage(role="system", content="sys"))
    for i in range(40):
        wm.add(LLMMessage(role="user", content=f"long message number {i} " * 20))
    removed = await wm.compress_if_needed()
    assert removed > 0
    assert any("summary" in m.content for m in wm.messages)


def test_kvstore_set_get_ttl(tmp_path: Path) -> None:
    kv = KVStore(tmp_path / "kv.sqlite")
    kv.set("a", {"k": 1})
    assert kv.get("a") == {"k": 1}
    kv.set("b", 99, ttl=0)
    time.sleep(0.05)
    assert kv.get("b", default="missing") == "missing"
    assert "a" in kv.keys()


def test_episodic_memory_records_and_searches(tmp_path: Path) -> None:
    em = EpisodicMemory(tmp_path / "ep.sqlite", session_id="s1")
    rowid = em.record("file_read", {"path": "/a/b.py"}, {"size": 12}, ok=True)
    assert rowid > 0
    em.record("bash_exec", {"command": "ls"}, {"stdout": "a.py b.py"}, ok=True)
    found = em.search("file_read")
    assert any("file_read" in e.tool or "file_read" in e.args_json for e in found)
    recent = em.recent(limit=10)
    assert len(recent) == 2


def test_semantic_memory_round_trip(tmp_path: Path) -> None:
    sem = SemanticMemory(tmp_path / "sem", collection="test")
    sem.add("fact", "Python uses indentation for blocks", metadata={"lang": "python"})
    sem.add("fact", "JavaScript uses braces", metadata={"lang": "js"})
    sem.add("error", "TypeError: cannot read property of undefined", metadata={"lang": "js"})
    out = sem.retrieve_relevant_memories("python indentation", top_k=2)
    assert len(out) >= 1
    assert any("indent" in m.content.lower() for m in out)


def test_semantic_memory_persists_across_instances(tmp_path: Path) -> None:
    sem = SemanticMemory(tmp_path / "sem2", collection="t")
    sem.add("fact", "agent harness is awesome")
    sem2 = SemanticMemory(tmp_path / "sem2", collection="t")
    out = sem2.retrieve_relevant_memories("agent harness", top_k=3)
    assert any("agent harness" in m.content for m in out)
