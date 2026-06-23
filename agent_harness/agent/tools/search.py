"""File search using ripgrep with a pure-Python fallback."""

from __future__ import annotations

import asyncio
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from agent.tools.registry import GLOBAL_REGISTRY


@dataclass
class SearchMatch:
    """A single search hit."""

    path: str
    line: int
    text: str
    context_before: list[str]
    context_after: list[str]


async def _run_ripgrep(
    pattern: str, path: str, case_sensitive: bool, max_results: int
) -> list[SearchMatch]:
    """Invoke ``rg`` and parse its ``--json`` output."""
    rg = shutil.which("rg")
    if rg is None:
        return []
    args = [rg, "--json", "--context", "2", "--max-count", str(max_results), pattern, path]
    if not case_sensitive:
        args.append("-i")
    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, _ = await proc.communicate()
    import json

    matches: list[SearchMatch] = []
    context_buf: list[tuple[int, str]] = []
    pending: SearchMatch | None = None
    for raw_line in stdout.splitlines():
        try:
            evt = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if evt.get("type") == "match":
            data = evt["data"]
            path_str = data["path"].get("text", "")
            line_no = data.get("line_number", 0)
            text = data["lines"].get("text", "").rstrip("\n")
            before = [t for ln, t in context_buf if ln < line_no][-2:]
            if pending is not None:
                matches.append(pending)
            pending = SearchMatch(
                path=path_str,
                line=line_no,
                text=text,
                context_before=before,
                context_after=[],
            )
            context_buf.clear()
        elif evt.get("type") == "context":
            data = evt["data"]
            line_no = data.get("line_number", 0)
            text = data["lines"].get("text", "").rstrip("\n")
            if pending is not None and len(pending.context_after) < 2:
                pending.context_after.append(text)
            else:
                context_buf.append((line_no, text))
        if len(matches) >= max_results:
            break
    if pending is not None:
        matches.append(pending)
    return matches[:max_results]


def _python_search(
    pattern: str, path: str, case_sensitive: bool, max_results: int
) -> list[SearchMatch]:
    """Recursive pure-Python search used when ripgrep is unavailable."""
    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        regex = re.compile(pattern, flags)
    except re.error as exc:
        raise ValueError(f"Invalid regex: {exc}") from exc
    root = Path(path)
    matches: list[SearchMatch] = []
    iterator = [root] if root.is_file() else root.rglob("*")
    for file in iterator:
        if not file.is_file():
            continue
        try:
            with file.open("r", encoding="utf-8", errors="replace") as fh:
                lines = fh.readlines()
        except OSError:
            continue
        for idx, line in enumerate(lines):
            if regex.search(line):
                before = [ln.rstrip("\n") for ln in lines[max(0, idx - 2) : idx]]
                after = [ln.rstrip("\n") for ln in lines[idx + 1 : idx + 3]]
                matches.append(
                    SearchMatch(
                        path=str(file),
                        line=idx + 1,
                        text=line.rstrip("\n"),
                        context_before=before,
                        context_after=after,
                    )
                )
                if len(matches) >= max_results:
                    return matches
    return matches


@GLOBAL_REGISTRY.tool(
    description=(
        "Search files for a regex pattern. Uses ripgrep when available, with a "
        "pure-Python fallback. Returns matches with file path, line number, and context."
    ),
    side_effect="read_only",
    timeout=60.0,
)
async def file_search(
    pattern: str,
    path: str = ".",
    case_sensitive: bool = False,
    max_results: int = 50,
) -> dict:
    """Search for ``pattern`` under ``path``.

    Args:
        pattern: A regular expression.
        path: Root directory (or file) to search.
        case_sensitive: Case sensitivity flag.
        max_results: Cap on number of hits.
    """
    results = await _run_ripgrep(pattern, path, case_sensitive, max_results)
    if not results and shutil.which("rg") is None:
        results = await asyncio.to_thread(
            _python_search, pattern, path, case_sensitive, max_results
        )
    return {
        "pattern": pattern,
        "path": path,
        "case_sensitive": case_sensitive,
        "count": len(results),
        "matches": [
            {
                "path": m.path,
                "line": m.line,
                "text": m.text,
                "context_before": m.context_before,
                "context_after": m.context_after,
            }
            for m in results
        ],
    }
