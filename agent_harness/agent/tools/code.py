"""Linting and test running tools."""

from __future__ import annotations

import asyncio
import json
import shutil
from dataclasses import dataclass

from agent.tools.registry import GLOBAL_REGISTRY


@dataclass
class LintReport:
    """Structured lint output."""

    path: str
    language: str
    tool: str
    diagnostics: list[dict]
    ok: bool


@dataclass
class TestReport:
    """Structured test runner output."""

    framework: str
    passed: int
    failed: int
    errors: int
    skipped: int
    duration_s: float
    failures: list[dict]


async def _run(cmd: list[str], cwd: str | None = None, timeout: int = 120) -> tuple[int, str, str]:
    """Run ``cmd`` and capture (exit_code, stdout, stderr)."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
    )
    try:
        out_b, err_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        out_b, err_b = await proc.communicate()
    return proc.returncode or 0, out_b.decode("utf-8", "replace"), err_b.decode("utf-8", "replace")


@GLOBAL_REGISTRY.tool(
    description=(
        "Lint a file with the appropriate tool: ruff for Python, eslint for JS/TS, "
        "clippy for Rust. Returns structured diagnostics."
    ),
    side_effect="read_only",
    timeout=120.0,
)
async def code_lint(path: str, language: str = "auto") -> dict:
    """Lint ``path``.

    Args:
        path: File path to lint.
        language: ``"python"``, ``"javascript"``, ``"typescript"``, ``"rust"``,
            or ``"auto"`` to infer from extension.
    """
    if language == "auto":
        if path.endswith(".py"):
            language = "python"
        elif path.endswith((".js", ".jsx", ".mjs", ".cjs")):
            language = "javascript"
        elif path.endswith((".ts", ".tsx")):
            language = "typescript"
        elif path.endswith(".rs"):
            language = "rust"
    diagnostics: list[dict] = []
    tool = "unknown"
    if language == "python" and shutil.which("ruff"):
        tool = "ruff"
        code, out, _err = await _run(["ruff", "check", "--output-format=json", path])
        try:
            parsed = json.loads(out) if out.strip() else []
            for item in parsed:
                diagnostics.append(
                    {
                        "file": item.get("filename", path),
                        "line": item.get("location", {}).get("row", 0),
                        "column": item.get("location", {}).get("column", 0),
                        "code": item.get("code"),
                        "message": item.get("message", ""),
                        "severity": "warning",
                    }
                )
        except json.JSONDecodeError:
            diagnostics.append({"file": path, "message": out.strip()[:1000], "severity": "error"})
    elif language in {"javascript", "typescript"} and shutil.which("eslint"):
        tool = "eslint"
        code, out, _err = await _run(["eslint", "--format=json", path])
        try:
            parsed = json.loads(out) if out.strip() else []
            for file_report in parsed:
                for msg in file_report.get("messages", []):
                    diagnostics.append(
                        {
                            "file": file_report.get("filePath", path),
                            "line": msg.get("line", 0),
                            "column": msg.get("column", 0),
                            "code": msg.get("ruleId"),
                            "message": msg.get("message", ""),
                            "severity": "error" if msg.get("severity") == 2 else "warning",
                        }
                    )
        except json.JSONDecodeError:
            diagnostics.append({"file": path, "message": out.strip()[:1000], "severity": "error"})
    elif language == "rust" and shutil.which("cargo"):
        tool = "clippy"
        code, out, err = await _run(["cargo", "clippy", "--message-format=json"], cwd=path if "/" not in path else None)
        for line in out.splitlines():
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("reason") == "compiler-message":
                m = msg.get("message", {})
                diagnostics.append(
                    {
                        "file": m.get("spans", [{}])[0].get("file_name", path),
                        "line": m.get("spans", [{}])[0].get("line_start", 0),
                        "code": m.get("code", {}).get("code") if m.get("code") else None,
                        "message": m.get("message", ""),
                        "severity": m.get("level", "warning"),
                    }
                )
    else:
        return {
            "path": path,
            "language": language,
            "tool": tool,
            "ok": True,
            "diagnostics": [],
            "note": f"No linter available for language {language!r}",
        }
    report = LintReport(
        path=path, language=language, tool=tool, diagnostics=diagnostics, ok=not diagnostics
    )
    return {
        "path": report.path,
        "language": report.language,
        "tool": report.tool,
        "ok": report.ok,
        "diagnostics": report.diagnostics,
    }


@GLOBAL_REGISTRY.tool(
    description=(
        "Run tests using a framework (pytest, jest, cargo, go). Optional filter applies "
        "the framework's pattern flag. Returns counts, duration, and failure tracebacks."
    ),
    side_effect="compute",
    timeout=600.0,
)
async def test_runner(path: str = ".", framework: str = "pytest", filter: str | None = None) -> dict:
    """Discover and run tests.

    Args:
        path: Path to run tests against.
        framework: ``"pytest"``, ``"jest"``, ``"cargo"``, or ``"go"``.
        filter: Optional substring/expression filter passed to the framework.
    """
    import time

    started = time.perf_counter()
    if framework == "pytest" and shutil.which("pytest"):
        cmd = ["pytest", "-q", path]
        if filter:
            cmd += ["-k", filter]
        code, out, err = await _run(cmd, timeout=600)
        passed = failed = errors = skipped = 0
        for line in out.splitlines():
            if "passed" in line or "failed" in line:
                for token in line.split():
                    if token.endswith("passed"):
                        passed = int(token[:-6]) if token[:-6].isdigit() else passed
                    elif token.endswith("failed"):
                        failed = int(token[:-6]) if token[:-6].isdigit() else failed
                    elif token.endswith("error") or token.endswith("errors"):
                        suffix = "errors" if token.endswith("errors") else "error"
                        errors = int(token[: -len(suffix)]) if token[: -len(suffix)].isdigit() else errors
                    elif token.endswith("skipped"):
                        skipped = int(token[:-7]) if token[:-7].isdigit() else skipped
        failures = _extract_pytest_failures(out)
        report = TestReport(
            framework=framework,
            passed=passed,
            failed=failed,
            errors=errors,
            skipped=skipped,
            duration_s=time.perf_counter() - started,
            failures=failures,
        )
    elif framework == "jest" and shutil.which("jest"):
        cmd = ["jest", "--json", path]
        if filter:
            cmd += ["-t", filter]
        code, out, err = await _run(cmd, timeout=600)
        try:
            data = json.loads(out)
        except json.JSONDecodeError:
            data = {}
        report = TestReport(
            framework=framework,
            passed=data.get("numPassedTests", 0),
            failed=data.get("numFailedTests", 0),
            errors=0,
            skipped=data.get("numPendingTests", 0),
            duration_s=time.perf_counter() - started,
            failures=[
                {"name": t.get("fullName", ""), "trace": "\n".join(t.get("failureMessages", []))}
                for r in data.get("testResults", [])
                for t in r.get("assertionResults", [])
                if t.get("status") == "failed"
            ],
        )
    elif framework == "cargo" and shutil.which("cargo"):
        cmd = ["cargo", "test", "--", "--nocapture"]
        if filter:
            cmd += [filter]
        code, out, err = await _run(cmd, cwd=path if path != "." else None, timeout=600)
        passed = out.count(" ok")
        failed = out.count("FAILED")
        report = TestReport(
            framework=framework,
            passed=passed,
            failed=failed,
            errors=0,
            skipped=0,
            duration_s=time.perf_counter() - started,
            failures=[],
        )
    elif framework == "go" and shutil.which("go"):
        cmd = ["go", "test", "./..."]
        if filter:
            cmd += ["-run", filter]
        code, out, err = await _run(cmd, cwd=path if path != "." else None, timeout=600)
        passed = out.count("--- PASS")
        failed = out.count("--- FAIL")
        report = TestReport(
            framework=framework,
            passed=passed,
            failed=failed,
            errors=0,
            skipped=0,
            duration_s=time.perf_counter() - started,
            failures=[],
        )
    else:
        return {
            "framework": framework,
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "skipped": 0,
            "duration_s": 0.0,
            "failures": [],
            "note": f"Test framework {framework!r} not available on PATH",
        }
    return {
        "framework": report.framework,
        "passed": report.passed,
        "failed": report.failed,
        "errors": report.errors,
        "skipped": report.skipped,
        "duration_s": report.duration_s,
        "failures": report.failures,
    }


def _extract_pytest_failures(output: str) -> list[dict]:
    """Extract failure tracebacks from pytest output."""
    failures: list[dict] = []
    in_failures = False
    current: dict | None = None
    buf: list[str] = []
    for line in output.splitlines():
        if line.startswith("=") and "FAILURES" in line:
            in_failures = True
            continue
        if not in_failures:
            continue
        if line.startswith("=") and ("short test summary" in line or "passed" in line or "failed" in line):
            if current is not None:
                current["trace"] = "\n".join(buf)
                failures.append(current)
            break
        if line.startswith("_") and "_" in line.strip("_"):
            if current is not None:
                current["trace"] = "\n".join(buf)
                failures.append(current)
            current = {"name": line.strip("_ "), "trace": ""}
            buf = []
        else:
            buf.append(line)
    return failures
