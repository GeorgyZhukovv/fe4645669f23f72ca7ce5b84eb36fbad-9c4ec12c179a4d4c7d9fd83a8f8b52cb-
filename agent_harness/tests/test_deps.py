"""Tests for the dependency intelligence layer (against fixture project files)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.tools.deps import (
    _parse_cargo_lock,
    _parse_package_lock,
    _parse_pyproject,
    _parse_requirements,
    detect_ecosystem,
    plan_dependency_upgrades,
    scan_dependencies,
    upgrade_plan_to_edit_plan,
    UpgradePlan,
    UpgradeProposal,
    Vulnerability,
    VulnerabilityReport,
)


def test_detect_ecosystem(workdir: Path) -> None:
    eco, lock, locked = detect_ecosystem(workdir)
    assert eco == "unknown"
    (workdir / "requirements.txt").write_text("requests==2.28.1\n")
    eco, lock, locked = detect_ecosystem(workdir)
    assert eco == "python"
    (workdir / "package-lock.json").write_text("{}")
    eco, _, locked = detect_ecosystem(workdir)
    assert eco == "python"  # python wins because requirements.txt exists


def test_parse_requirements_strips_comments_and_specifiers() -> None:
    pkgs = _parse_requirements("# comment\nrequests==2.28.1\nclick>=8.0  # cli\n-e .\n\n")
    names = {p.name for p in pkgs}
    assert "requests" in names and "click" in names
    versions = {p.name: p.version for p in pkgs}
    assert versions["requests"] == "2.28.1"


def test_parse_pyproject_extracts_deps() -> None:
    toml = (
        '[project]\n'
        'name = "x"\n'
        'dependencies = ["requests>=2.28", "click==8.1.7"]\n'
        '\n'
        '[project.optional-dependencies]\n'
        'dev = ["pytest>=8"]\n'
    )
    pkgs = _parse_pyproject(toml)
    names = {p.name for p in pkgs}
    assert {"requests", "click", "pytest"}.issubset(names)


def test_parse_package_lock_v3() -> None:
    pl = {
        "name": "demo",
        "lockfileVersion": 3,
        "packages": {
            "": {"name": "demo", "version": "1.0.0"},
            "node_modules/express": {"version": "4.18.0"},
            "node_modules/lodash": {"version": "4.17.21", "dev": True},
        },
    }
    pkgs = _parse_package_lock(json.dumps(pl))
    names = {p.name for p in pkgs}
    assert "express" in names


def test_parse_cargo_lock_extracts_packages() -> None:
    toml = (
        '[[package]]\n'
        'name = "serde"\n'
        'version = "1.0.0"\n'
        '[[package]]\n'
        'name = "serde_json"\n'
        'version = "1.0.0"\n'
        'dependencies = ["serde 1.0.0"]\n'
    )
    pkgs = _parse_cargo_lock(toml)
    names = {p.name for p in pkgs}
    assert "serde" in names and "serde_json" in names


def test_scan_dependencies_python(workdir: Path) -> None:
    (workdir / "pyproject.toml").write_text(
        '[project]\nname="x"\ndependencies=["requests==2.28.1"]\n'
    )
    report = scan_dependencies(workdir)
    assert report.ecosystem == "python"
    assert any(p.name == "requests" for p in report.packages)


def test_plan_dependency_upgrades_security_only() -> None:
    vrep = VulnerabilityReport(
        ecosystem="python",
        total_packages_scanned=1,
        scan_time=0.0,
        vulnerabilities=[
            Vulnerability(
                package="requests",
                installed_version="2.28.1",
                fixed_version="2.31.0",
                severity="high",
                cve_ids=["CVE-2023-32681"],
                description="cookie leak",
            ),
            Vulnerability(
                package="x",
                installed_version="1.0",
                fixed_version=None,
                severity="low",
                cve_ids=[],
                description="",
            ),
        ],
    )
    plan = plan_dependency_upgrades(vrep, "security_only")
    pkgs = [p.package for p in plan.proposals]
    assert "requests" in pkgs and "x" not in pkgs


def test_upgrade_plan_to_edit_plan_python(workdir: Path) -> None:
    (workdir / "requirements.txt").write_text("requests==2.28.1\nclick>=8.0\n")
    plan = UpgradePlan(
        strategy="security_only",
        proposals=[UpgradeProposal(package="requests", current_version="2.28.1", target_version="2.31.0", reason="cve")],
    )
    edit = upgrade_plan_to_edit_plan(plan, workdir, "python")
    assert "requests==2.31.0" in edit["operations"][0]["content"]


def test_upgrade_plan_to_edit_plan_node(workdir: Path) -> None:
    pkg = {"name": "demo", "version": "1.0.0", "dependencies": {"express": "4.18.0"}}
    (workdir / "package.json").write_text(json.dumps(pkg))
    plan = UpgradePlan(
        strategy="security_only",
        proposals=[UpgradeProposal(package="express", current_version="4.18.0", target_version="4.19.2", reason="cve")],
    )
    edit = upgrade_plan_to_edit_plan(plan, workdir, "node")
    body = json.loads(edit["operations"][0]["content"])
    assert body["dependencies"]["express"] == "4.19.2"
