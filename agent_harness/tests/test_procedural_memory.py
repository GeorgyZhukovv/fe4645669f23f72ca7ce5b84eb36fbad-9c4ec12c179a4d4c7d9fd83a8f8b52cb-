"""Tests for procedural memory."""

from __future__ import annotations

from pathlib import Path

from agent.memory.procedural import (
    ProceduralMemory,
    ProcedureStep,
    fingerprint_task,
)


def test_fingerprint_is_stable_across_phrasing() -> None:
    a = fingerprint_task("Add a new API endpoint for users")
    b = fingerprint_task("Add a new endpoint for users API")
    # word order is normalised, common stopwords stripped
    assert a == b


def test_record_and_retrieve_procedure(tmp_path: Path) -> None:
    pm = ProceduralMemory(tmp_path / "p.sqlite")
    steps = [
        ProcedureStep(tool="file_read", args_summary="path=auth.py", ok=True),
        ProcedureStep(tool="multi_edit", args_summary="2 ops", ok=True),
        ProcedureStep(tool="test_runner", args_summary="tests/", ok=True),
    ]
    proc = pm.record_successful_procedure("Add unit tests for the auth module", steps, duration_ms=1500.0)
    assert proc.id > 0
    assert proc.times_used == 1
    assert proc.success_rate == 1.0

    # retrieving with a similar phrasing should find the same procedure
    found = pm.retrieve_relevant_procedure("Write unit tests for auth module")
    assert found is not None
    assert found.id == proc.id


def test_record_again_increments_times_used(tmp_path: Path) -> None:
    pm = ProceduralMemory(tmp_path / "p.sqlite")
    pm.record_successful_procedure("Refactor the parser", [], duration_ms=100.0)
    again = pm.record_successful_procedure("Refactor the parser", [], duration_ms=200.0)
    assert again.times_used == 2
    assert again.avg_duration_ms == 150.0


def test_record_failure_lowers_success_rate(tmp_path: Path) -> None:
    pm = ProceduralMemory(tmp_path / "p.sqlite")
    pm.record_successful_procedure("Update CHANGES md", [], duration_ms=10.0)
    pm.record_failure("Update CHANGES md")
    proc = pm.retrieve_relevant_procedure("Update CHANGES md", min_success_rate=0.0)
    assert proc is not None
    assert proc.success_rate < 1.0


def test_retrieve_returns_none_below_threshold(tmp_path: Path) -> None:
    pm = ProceduralMemory(tmp_path / "p.sqlite")
    pm.record_successful_procedure("Do thing", [], duration_ms=10.0)
    pm.record_failure("Do thing")
    pm.record_failure("Do thing")
    # success rate is now 1/3 ≈ 0.33 — below 0.8 default threshold
    assert pm.retrieve_relevant_procedure("Do thing") is None


def test_all_procedures_sorted_by_recency(tmp_path: Path) -> None:
    pm = ProceduralMemory(tmp_path / "p.sqlite")
    pm.record_successful_procedure("First thing", [], duration_ms=1.0)
    pm.record_successful_procedure("Second thing", [], duration_ms=1.0)
    all_procs = pm.all_procedures()
    assert len(all_procs) == 2
    assert all_procs[0].task_description == "Second thing"
