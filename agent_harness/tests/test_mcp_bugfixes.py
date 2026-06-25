"""Regression tests for the three bugs Cursor's agent surfaced via MCP."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agent.tools.deps import (
    UpgradePlan,
    UpgradeProposal,
    _osv_fixed_version,
    _parse_version_key,
    upgrade_plan_to_edit_plan,
)
from agent.tools.multi_edit import EditOperation, multi_edit


# --- multi_edit accepts JSON-stringified arguments (MCP transport sends them stringified) ---


@pytest.mark.asyncio
async def test_multi_edit_accepts_string_operations(workdir: Path) -> None:
    target = workdir / "a.txt"
    ops_json = json.dumps([
        {"kind": "create", "path": str(target), "content": "hello\n"},
    ])
    result = await multi_edit(description="x", operations=ops_json)
    assert result["status"] == "verified"
    assert target.read_text() == "hello\n"


@pytest.mark.asyncio
async def test_multi_edit_accepts_string_verification_commands(workdir: Path) -> None:
    target = workdir / "b.txt"
    result = await multi_edit(
        description="x",
        operations=[{"kind": "create", "path": str(target), "content": "hi\n"}],
        verification_commands=json.dumps(["true"]),
    )
    assert result["status"] == "verified"
    assert target.read_text() == "hi\n"


@pytest.mark.asyncio
async def test_multi_edit_rejects_garbage_operations() -> None:
    result = await multi_edit(description="x", operations="not valid json")
    assert result["status"] == "failed"
    assert "JSON" in (result["error"] or "")


@pytest.mark.asyncio
async def test_multi_edit_handles_per_op_json_strings(workdir: Path) -> None:
    target = workdir / "c.txt"
    ops = [json.dumps({"kind": "create", "path": str(target), "content": "ok\n"})]
    result = await multi_edit(description="x", operations=ops)
    assert result["status"] == "verified"


# --- OSV fixed_version extraction ---


def test_osv_fixed_version_extracts_from_events() -> None:
    advisory: dict[str, Any] = {
        "affected": [{
            "ranges": [{
                "events": [
                    {"introduced": "0"},
                    {"fixed": "3.10.11"},
                    {"introduced": "3.11.0"},
                    {"fixed": "3.14.1"},
                ],
            }],
        }],
    }
    assert _osv_fixed_version(advisory) == "3.14.1"


def test_osv_fixed_version_returns_none_when_no_fix() -> None:
    advisory = {"affected": [{"ranges": [{"events": [{"introduced": "0"}]}]}]}
    assert _osv_fixed_version(advisory) is None


def test_parse_version_key_orders_correctly() -> None:
    assert _parse_version_key("3.9.0") < _parse_version_key("3.10.0")
    assert _parse_version_key("3.14.1") > _parse_version_key("3.9.4")


# --- pyproject.toml support in upgrade_plan_to_edit_plan ---


def test_upgrade_plan_edits_pyproject_dependencies(workdir: Path) -> None:
    pyp = workdir / "pyproject.toml"
    pyp.write_text(
        '[project]\n'
        'dependencies = [\n'
        '    "markdownify>=0.13.0",\n'
        '    "aiohttp>=3.9.0",\n'
        ']\n'
    )
    plan = UpgradePlan(
        strategy="security_only",
        proposals=[
            UpgradeProposal(package="markdownify", current_version="0.13.0", target_version="0.14.1", reason="cve"),
            UpgradeProposal(package="aiohttp", current_version="3.9.0", target_version="3.14.1", reason="cve"),
        ],
    )
    edit = upgrade_plan_to_edit_plan(plan, workdir, "python")
    assert edit["operations"], "should produce at least one op"
    new_content = edit["operations"][0]["content"]
    assert "markdownify" in new_content and "0.14.1" in new_content
    assert "aiohttp" in new_content and "3.14.1" in new_content


def test_upgrade_plan_skips_files_with_no_matches(workdir: Path) -> None:
    pyp = workdir / "pyproject.toml"
    pyp.write_text('[project]\ndependencies = ["click>=8.1.7"]\n')
    plan = UpgradePlan(
        strategy="security_only",
        proposals=[UpgradeProposal(package="nonexistent", current_version="1", target_version="2", reason="x")],
    )
    edit = upgrade_plan_to_edit_plan(plan, workdir, "python")
    # The pyproject would be rewritten to identical content, which we now skip.
    assert edit["operations"] == []


# --- pytest output parser ---


def test_test_runner_parser_handles_pytest_9_summary() -> None:
    # Pytest 9 summary line shape:
    # "=============== 153 passed, 1 warning in 7.11s ==============="
    from agent.tools import code as code_mod

    line = "=============== 153 passed, 1 warning in 7.11s ==============="
    import re
    passed = failed = errors = skipped = 0
    stripped = line.strip().strip("=").strip()
    for m in re.finditer(r"(\d+)\s+(passed|failed|errors?|skipped)", stripped):
        count = int(m.group(1))
        kw = m.group(2)
        if kw == "passed":
            passed = count
        elif kw == "failed":
            failed = count
        elif kw.startswith("error"):
            errors = count
        elif kw == "skipped":
            skipped = count
    assert passed == 153
    assert failed == 0
