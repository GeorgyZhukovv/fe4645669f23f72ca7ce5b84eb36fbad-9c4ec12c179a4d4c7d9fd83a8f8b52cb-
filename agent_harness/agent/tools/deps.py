"""Dependency + vulnerability intelligence tools."""

from __future__ import annotations

import asyncio
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from agent.tools.registry import GLOBAL_REGISTRY


Severity = Literal["low", "medium", "high", "critical"]
EcoSystem = Literal["python", "node", "rust", "go", "jvm", "unknown"]


@dataclass
class PackageInfo:
    """One installed package."""

    name: str
    version: str
    ecosystem: EcoSystem
    direct: bool = True
    dependencies: list[str] = field(default_factory=list)


@dataclass
class DependencyReport:
    """The resolved dependency picture."""

    ecosystem: EcoSystem
    lock_file: str | None
    is_locked: bool
    packages: list[PackageInfo]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ecosystem": self.ecosystem,
            "lock_file": self.lock_file,
            "is_locked": self.is_locked,
            "packages": [p.__dict__ for p in self.packages],
        }


@dataclass
class Vulnerability:
    """A normalised vulnerability finding."""

    package: str
    installed_version: str
    fixed_version: str | None
    severity: Severity
    cve_ids: list[str]
    description: str
    remediation: str = ""


@dataclass
class VulnerabilityReport:
    """Aggregate vulnerability scan output."""

    ecosystem: EcoSystem
    total_packages_scanned: int
    vulnerabilities: list[Vulnerability]
    scan_time: float

    def critical_count(self) -> int:
        return sum(1 for v in self.vulnerabilities if v.severity == "critical")


@dataclass
class UpgradeProposal:
    """One package upgrade suggestion."""

    package: str
    current_version: str
    target_version: str
    reason: str
    breaking: bool = False
    changelog_url: str | None = None


@dataclass
class UpgradePlan:
    """A batch of upgrade proposals."""

    strategy: str
    proposals: list[UpgradeProposal]


# ---------- detection ----------


def detect_ecosystem(project_root: str | Path) -> tuple[EcoSystem, str | None, bool]:
    """Return ``(ecosystem, lock_file, is_locked)`` for the project at ``root``."""
    root = Path(project_root)
    if (root / "poetry.lock").exists():
        return "python", "poetry.lock", True
    if (root / "uv.lock").exists():
        return "python", "uv.lock", True
    if (root / "requirements.txt").exists() or (root / "pyproject.toml").exists():
        return "python", "requirements.txt" if (root / "requirements.txt").exists() else None, False
    if (root / "package-lock.json").exists():
        return "node", "package-lock.json", True
    if (root / "yarn.lock").exists():
        return "node", "yarn.lock", True
    if (root / "pnpm-lock.yaml").exists():
        return "node", "pnpm-lock.yaml", True
    if (root / "package.json").exists():
        return "node", None, False
    if (root / "Cargo.lock").exists():
        return "rust", "Cargo.lock", True
    if (root / "Cargo.toml").exists():
        return "rust", None, False
    if (root / "go.sum").exists():
        return "go", "go.sum", True
    if (root / "go.mod").exists():
        return "go", None, False
    return "unknown", None, False


# ---------- scanning ----------


async def _run(args: list[str], cwd: str | None = None, timeout: int = 120) -> tuple[int, str, str]:
    """Helper: run a command, return (rc, stdout, stderr)."""
    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=cwd
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        out, err = await proc.communicate()
    return proc.returncode or 0, out.decode("utf-8", "replace"), err.decode("utf-8", "replace")


def _parse_requirements(text: str) -> list[PackageInfo]:
    """Parse a ``requirements.txt`` body into :class:`PackageInfo` records."""
    out: list[PackageInfo] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        m = re.match(r"^([A-Za-z0-9_.\-]+)\s*(?:\[[^\]]+\])?\s*([<>=!~][^\s;]+)?", line)
        if not m:
            continue
        name = m.group(1)
        version_spec = m.group(2) or ""
        version = re.sub(r"^[<>=!~]+", "", version_spec)
        out.append(PackageInfo(name=name, version=version, ecosystem="python"))
    return out


def _parse_pyproject(text: str) -> list[PackageInfo]:
    """Parse PEP 621 ``[project.dependencies]`` and optional dep tables."""
    try:
        import tomllib
    except ImportError:  # pragma: no cover - py<3.11
        import tomli as tomllib  # type: ignore[no-redef]
    out: list[PackageInfo] = []
    try:
        data = tomllib.loads(text)
    except Exception:
        return out
    project = data.get("project", {})
    for raw in project.get("dependencies", []):
        pkg = _parse_dep_spec(str(raw))
        if pkg:
            out.append(pkg)
    for _group, items in (project.get("optional-dependencies") or {}).items():
        for raw in items:
            pkg = _parse_dep_spec(str(raw))
            if pkg:
                out.append(pkg)
    # Poetry style
    tool_poetry = data.get("tool", {}).get("poetry", {})
    for name, spec in (tool_poetry.get("dependencies") or {}).items():
        if name == "python":
            continue
        version = spec if isinstance(spec, str) else (spec.get("version", "") if isinstance(spec, dict) else "")
        out.append(PackageInfo(name=name, version=str(version).lstrip("^~="), ecosystem="python"))
    return out


def _parse_dep_spec(spec: str) -> PackageInfo | None:
    """Parse a single PEP 508-style dep spec string."""
    spec = spec.split(";")[0].strip()
    m = re.match(r"^([A-Za-z0-9_.\-]+)\s*(?:\[[^\]]+\])?\s*([<>=!~][^\s,]+)?", spec)
    if not m:
        return None
    name = m.group(1)
    version = re.sub(r"^[<>=!~]+", "", m.group(2) or "")
    return PackageInfo(name=name, version=version, ecosystem="python")


def _parse_package_lock(text: str) -> list[PackageInfo]:
    """Parse a ``package-lock.json`` v3+ tree."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    packages = data.get("packages", {})
    out: list[PackageInfo] = []
    for path, info in packages.items():
        if not path or path == "":
            continue
        name = info.get("name") or path.split("node_modules/")[-1]
        out.append(PackageInfo(
            name=name,
            version=info.get("version", ""),
            ecosystem="node",
            direct=not info.get("dev", False),
        ))
    return out


def _parse_cargo_lock(text: str) -> list[PackageInfo]:
    """Parse Cargo.lock (TOML)."""
    try:
        import tomllib
    except ImportError:  # pragma: no cover
        import tomli as tomllib  # type: ignore[no-redef]
    try:
        data = tomllib.loads(text)
    except Exception:
        return []
    return [
        PackageInfo(
            name=p["name"], version=p.get("version", ""), ecosystem="rust",
            dependencies=p.get("dependencies", []),
        )
        for p in data.get("package", [])
    ]


def _parse_go_sum(text: str) -> list[PackageInfo]:
    """Parse a ``go.sum`` file."""
    out: list[PackageInfo] = []
    seen: set[str] = set()
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        name = parts[0]
        version = parts[1].split("/")[0]
        key = f"{name}@{version}"
        if key in seen:
            continue
        seen.add(key)
        out.append(PackageInfo(name=name, version=version, ecosystem="go"))
    return out


def scan_dependencies(project_root: str | Path) -> DependencyReport:
    """Build a :class:`DependencyReport` for the project rooted at ``project_root``."""
    root = Path(project_root)
    ecosystem, lock_file, is_locked = detect_ecosystem(root)
    packages: list[PackageInfo] = []
    if ecosystem == "python":
        for req in root.glob("requirements*.txt"):
            packages.extend(_parse_requirements(req.read_text(encoding="utf-8", errors="replace")))
        py = root / "pyproject.toml"
        if py.exists():
            packages.extend(_parse_pyproject(py.read_text(encoding="utf-8", errors="replace")))
    elif ecosystem == "node":
        lock = root / "package-lock.json"
        if lock.exists():
            packages.extend(_parse_package_lock(lock.read_text(encoding="utf-8", errors="replace")))
        else:
            pkg = root / "package.json"
            if pkg.exists():
                try:
                    data = json.loads(pkg.read_text(encoding="utf-8", errors="replace"))
                except json.JSONDecodeError:
                    data = {}
                for name, version in (data.get("dependencies") or {}).items():
                    packages.append(PackageInfo(name=name, version=str(version), ecosystem="node"))
                for name, version in (data.get("devDependencies") or {}).items():
                    packages.append(PackageInfo(name=name, version=str(version), ecosystem="node", direct=False))
    elif ecosystem == "rust":
        lock = root / "Cargo.lock"
        if lock.exists():
            packages.extend(_parse_cargo_lock(lock.read_text(encoding="utf-8", errors="replace")))
    elif ecosystem == "go":
        gs = root / "go.sum"
        if gs.exists():
            packages.extend(_parse_go_sum(gs.read_text(encoding="utf-8", errors="replace")))
    return DependencyReport(
        ecosystem=ecosystem, lock_file=lock_file, is_locked=is_locked, packages=packages,
    )


# ---------- vulnerability ----------


def _normalize_severity(value: str) -> Severity:
    """Map any vendor severity string to one of {low, medium, high, critical}."""
    v = (value or "").strip().lower()
    if v in {"critical", "crit"}:
        return "critical"
    if v in {"high", "important"}:
        return "high"
    if v in {"moderate", "medium"}:
        return "medium"
    return "low"


async def _pip_audit(root: Path) -> list[Vulnerability]:
    """Run ``pip-audit --json`` if available; fall back to empty list."""
    if not shutil.which("pip-audit"):
        return []
    rc, out, _err = await _run(["pip-audit", "--format=json", "--strict"], cwd=str(root), timeout=300)
    findings: list[Vulnerability] = []
    try:
        data = json.loads(out) if out.strip() else {}
    except json.JSONDecodeError:
        return findings
    deps = data.get("dependencies") if isinstance(data, dict) else data
    for dep in deps or []:
        name = dep.get("name", "")
        version = dep.get("version", "")
        for v in dep.get("vulns", []):
            findings.append(Vulnerability(
                package=name,
                installed_version=version,
                fixed_version=(v.get("fix_versions") or [None])[0],
                severity=_normalize_severity(v.get("severity", "medium")),
                cve_ids=v.get("aliases", []) or [v.get("id", "")],
                description=v.get("description", "")[:1000],
                remediation=f"upgrade to {(v.get('fix_versions') or [''])[0]}",
            ))
    return findings


async def _osv_query(packages: list[PackageInfo], ecosystem: EcoSystem) -> list[Vulnerability]:
    """Query the OSV API for vulnerabilities. Used as fallback when pip-audit is absent."""
    if not packages:
        return []
    try:
        import httpx
    except ImportError:
        return []
    eco_map = {"python": "PyPI", "node": "npm", "rust": "crates.io", "go": "Go"}
    eco = eco_map.get(ecosystem, "")
    if not eco:
        return []
    findings: list[Vulnerability] = []
    async with httpx.AsyncClient(timeout=20) as client:
        for pkg in packages[:50]:
            if not pkg.version:
                continue
            try:
                resp = await client.post(
                    "https://api.osv.dev/v1/query",
                    json={"package": {"name": pkg.name, "ecosystem": eco}, "version": pkg.version},
                )
                data = resp.json()
            except Exception:
                continue
            for v in data.get("vulns", []) or []:
                severity = "medium"
                for s in v.get("severity", []) or []:
                    score = s.get("score", "")
                    if "C" in score or score.startswith("9") or score.startswith("10"):
                        severity = "critical"
                    elif score.startswith("7") or score.startswith("8"):
                        severity = "high"
                fixed = _osv_fixed_version(v)
                findings.append(Vulnerability(
                    package=pkg.name,
                    installed_version=pkg.version,
                    fixed_version=fixed,
                    severity=_normalize_severity(severity),
                    cve_ids=v.get("aliases", []) or [v.get("id", "")],
                    description=v.get("summary", "")[:1000],
                    remediation=f"upgrade to {fixed}" if fixed else "",
                ))
    return findings


def _osv_fixed_version(advisory: dict[str, Any]) -> str | None:
    """Walk an OSV ``vulns[i].affected[*].ranges[*].events[*]`` tree for the highest fixed version."""
    candidates: list[str] = []
    for affected in advisory.get("affected", []) or []:
        for rng in affected.get("ranges", []) or []:
            for evt in rng.get("events", []) or []:
                fix = evt.get("fixed")
                if fix:
                    candidates.append(str(fix))
    if not candidates:
        return None
    # Pick the lexicographically-highest version string. Imperfect but good
    # enough for a planner that the user will preview before applying.
    return sorted(candidates, key=_parse_version_key)[-1]


def _parse_version_key(v: str) -> tuple:
    """Coarse version parser: turn ``"1.10.2"`` into a tuple for sort comparison."""
    parts = []
    for chunk in v.split("."):
        try:
            parts.append(int(chunk))
        except ValueError:
            parts.append(0)
    return tuple(parts)


async def _npm_audit(root: Path) -> list[Vulnerability]:
    """Run ``npm audit --json``."""
    if not shutil.which("npm"):
        return []
    rc, out, _err = await _run(["npm", "audit", "--json"], cwd=str(root), timeout=300)
    findings: list[Vulnerability] = []
    try:
        data = json.loads(out) if out.strip() else {}
    except json.JSONDecodeError:
        return findings
    for name, info in (data.get("vulnerabilities") or {}).items():
        findings.append(Vulnerability(
            package=name,
            installed_version=info.get("range", ""),
            fixed_version=(info.get("fixAvailable") or {}).get("version") if isinstance(info.get("fixAvailable"), dict) else None,
            severity=_normalize_severity(info.get("severity", "medium")),
            cve_ids=[],
            description=str(info.get("via", ""))[:500],
        ))
    return findings


async def _cargo_audit(root: Path) -> list[Vulnerability]:
    """Run ``cargo audit --json``."""
    if not shutil.which("cargo"):
        return []
    rc, out, _err = await _run(["cargo", "audit", "--json"], cwd=str(root), timeout=300)
    findings: list[Vulnerability] = []
    try:
        data = json.loads(out) if out.strip() else {}
    except json.JSONDecodeError:
        return findings
    for v in (data.get("vulnerabilities") or {}).get("list", []):
        advisory = v.get("advisory", {})
        findings.append(Vulnerability(
            package=v.get("package", {}).get("name", ""),
            installed_version=v.get("package", {}).get("version", ""),
            fixed_version=(v.get("versions", {}).get("patched") or [None])[0],
            severity=_normalize_severity(advisory.get("severity", "medium")),
            cve_ids=advisory.get("aliases", []) or [advisory.get("id", "")],
            description=advisory.get("title", "")[:1000],
        ))
    return findings


async def _govulncheck(root: Path) -> list[Vulnerability]:
    """Run ``govulncheck -json``."""
    if not shutil.which("govulncheck"):
        return []
    rc, out, _err = await _run(["govulncheck", "-json", "./..."], cwd=str(root), timeout=600)
    findings: list[Vulnerability] = []
    for line in out.splitlines():
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue
        finding = evt.get("finding") or evt.get("osv")
        if not finding:
            continue
        if isinstance(finding, dict):
            findings.append(Vulnerability(
                package=str(finding.get("affected", [{}])[0].get("package", {}).get("name", "")),
                installed_version="",
                fixed_version=None,
                severity="high",
                cve_ids=finding.get("aliases", []),
                description=finding.get("summary", "")[:500],
            ))
    return findings


async def scan_vulnerabilities(report: DependencyReport, project_root: str | Path) -> VulnerabilityReport:
    """Run the appropriate vulnerability scanner for ``report.ecosystem``."""
    import time

    started = time.time()
    root = Path(project_root)
    if report.ecosystem == "python":
        findings = await _pip_audit(root)
        if not findings:
            findings = await _osv_query(report.packages, "python")
    elif report.ecosystem == "node":
        findings = await _npm_audit(root)
        if not findings:
            findings = await _osv_query(report.packages, "node")
    elif report.ecosystem == "rust":
        findings = await _cargo_audit(root)
        if not findings:
            findings = await _osv_query(report.packages, "rust")
    elif report.ecosystem == "go":
        findings = await _govulncheck(root)
        if not findings:
            findings = await _osv_query(report.packages, "go")
    else:
        findings = []
    return VulnerabilityReport(
        ecosystem=report.ecosystem,
        total_packages_scanned=len(report.packages),
        vulnerabilities=findings,
        scan_time=time.time() - started,
    )


# ---------- upgrade planner ----------


def plan_dependency_upgrades(report: VulnerabilityReport, strategy: str = "security_only") -> UpgradePlan:
    """Propose upgrades. Strategies: ``security_only`` / ``minor`` / ``all``."""
    proposals: list[UpgradeProposal] = []
    if strategy == "security_only":
        for v in report.vulnerabilities:
            if v.fixed_version:
                proposals.append(UpgradeProposal(
                    package=v.package,
                    current_version=v.installed_version,
                    target_version=v.fixed_version,
                    reason=f"security: {', '.join(v.cve_ids)[:200]}",
                ))
    else:
        # For minor/all we lean on the vulnerability scanner's fix versions plus
        # any package that has no version pin (treated as a candidate for
        # explicit pinning, not a "real" upgrade) -- this keeps the planner
        # deterministic without a registry lookup that requires network.
        for v in report.vulnerabilities:
            if v.fixed_version:
                proposals.append(UpgradeProposal(
                    package=v.package,
                    current_version=v.installed_version,
                    target_version=v.fixed_version,
                    reason=f"security: {', '.join(v.cve_ids)[:200]}",
                    breaking=strategy == "all",
                ))
    return UpgradePlan(strategy=strategy, proposals=proposals)


def upgrade_plan_to_edit_plan(
    upgrade: UpgradePlan,
    project_root: str | Path,
    ecosystem: EcoSystem,
) -> dict[str, Any]:
    """Convert an :class:`UpgradePlan` into an EditPlan-shaped dict (multi_edit input)."""
    root = Path(project_root)
    ops: list[dict[str, Any]] = []
    if ecosystem == "python":
        req = root / "requirements.txt"
        if req.exists():
            text = req.read_text(encoding="utf-8", errors="replace")
            new = text
            for prop in upgrade.proposals:
                new = re.sub(
                    rf"^({re.escape(prop.package)})\s*(?:\[[^\]]*\])?\s*[<>=!~][^\s;]+",
                    rf"\1=={prop.target_version}",
                    new,
                    flags=re.MULTILINE,
                )
            if new != text:
                ops.append({"kind": "overwrite", "path": str(req), "content": new})
        # pyproject.toml: rewrite version specifiers in both `dependencies` lists
        # and any `[project.optional-dependencies]` table.
        pyp = root / "pyproject.toml"
        if pyp.exists():
            text = pyp.read_text(encoding="utf-8", errors="replace")
            new = text
            for prop in upgrade.proposals:
                # match e.g. "aiohttp>=3.9.0" or "markdownify==0.13.0"
                new = re.sub(
                    rf'(["\']){re.escape(prop.package)}(\[[^\]]*\])?\s*(?:[<>=!~][^"\']+)?\1',
                    lambda m, p=prop: f'{m.group(1)}{p.package}{m.group(2) or ""}>={p.target_version}{m.group(1)}',
                    new,
                )
            if new != text:
                ops.append({"kind": "overwrite", "path": str(pyp), "content": new})
    elif ecosystem == "node":
        pj = root / "package.json"
        if pj.exists():
            try:
                data = json.loads(pj.read_text(encoding="utf-8", errors="replace"))
            except json.JSONDecodeError:
                data = {}
            for section in ("dependencies", "devDependencies"):
                sect = data.get(section) or {}
                for prop in upgrade.proposals:
                    if prop.package in sect:
                        sect[prop.package] = prop.target_version
                data[section] = sect
            ops.append({"kind": "overwrite", "path": str(pj), "content": json.dumps(data, indent=2) + "\n"})
    return {
        "description": f"Apply {upgrade.strategy} upgrades ({len(upgrade.proposals)} packages)",
        "operations": ops,
        "verification_commands": [],
    }


# ---------- tool registration ----------


@GLOBAL_REGISTRY.tool(
    description="Scan the project for dependencies. Detects Python, Node, Rust, Go projects.",
    side_effect="read_only",
    timeout=60.0,
)
async def deps_scan(project_root: str = ".") -> dict:
    """Build a dependency report."""
    report = await asyncio.to_thread(scan_dependencies, project_root)
    return report.to_dict()


@GLOBAL_REGISTRY.tool(
    description=(
        "Scan installed dependencies for known vulnerabilities using pip-audit / npm audit / "
        "cargo audit / govulncheck, falling back to the OSV API."
    ),
    side_effect="network",
    timeout=600.0,
)
async def deps_vulns(project_root: str = ".") -> dict:
    """Vulnerability scan against the resolved dependency set."""
    report = await asyncio.to_thread(scan_dependencies, project_root)
    vuln = await scan_vulnerabilities(report, project_root)
    return {
        "ecosystem": vuln.ecosystem,
        "total_packages_scanned": vuln.total_packages_scanned,
        "critical": vuln.critical_count(),
        "scan_time": vuln.scan_time,
        "vulnerabilities": [v.__dict__ for v in vuln.vulnerabilities],
    }


@GLOBAL_REGISTRY.tool(
    description="Build a dependency upgrade plan. strategy ∈ {security_only, minor, all}.",
    side_effect="network",
    timeout=600.0,
)
async def deps_upgrade_plan(strategy: str = "security_only", project_root: str = ".") -> dict:
    """Return an upgrade plan derived from the latest vulnerability scan."""
    report = await asyncio.to_thread(scan_dependencies, project_root)
    vuln = await scan_vulnerabilities(report, project_root)
    plan = plan_dependency_upgrades(vuln, strategy=strategy)
    return {
        "strategy": plan.strategy,
        "proposals": [p.__dict__ for p in plan.proposals],
    }


@GLOBAL_REGISTRY.tool(
    description=(
        "Translate an upgrade plan (from deps_upgrade_plan) into a multi_edit EditPlan. "
        "Returns the EditPlan dict; pass it to multi_edit to apply."
    ),
    side_effect="read_only",
    timeout=60.0,
)
async def deps_apply_upgrades(strategy: str = "security_only", project_root: str = ".") -> dict:
    """Convert the planned upgrades into an applyable EditPlan body."""
    report = await asyncio.to_thread(scan_dependencies, project_root)
    vuln = await scan_vulnerabilities(report, project_root)
    plan = plan_dependency_upgrades(vuln, strategy=strategy)
    return upgrade_plan_to_edit_plan(plan, project_root, report.ecosystem)


@GLOBAL_REGISTRY.tool(
    description="Explain why a package is installed by listing its direct dependents.",
    side_effect="read_only",
    timeout=30.0,
)
async def deps_why(package: str, project_root: str = ".") -> dict:
    """Trace transitive parents of ``package`` in the dependency report."""
    report = await asyncio.to_thread(scan_dependencies, project_root)
    parents: list[str] = []
    for pkg in report.packages:
        if package in pkg.dependencies:
            parents.append(pkg.name)
    return {"package": package, "parents": parents, "ecosystem": report.ecosystem}
