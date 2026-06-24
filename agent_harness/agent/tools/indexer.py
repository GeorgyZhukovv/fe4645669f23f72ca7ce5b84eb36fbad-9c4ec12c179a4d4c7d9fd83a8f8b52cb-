"""Tool wrappers around the codebase indexer."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

from agent.indexer.builder import (
    build_index,
    index_dependency_chain,
    index_find_callers,
    index_find_usages,
    index_search_symbol,
    update_index,
)
from agent.indexer.model import CodebaseIndex
from agent.indexer.persist import load_index, save_index
from agent.tools.registry import GLOBAL_REGISTRY


_LOCK = threading.Lock()
_CACHED_INDEX: dict[str, CodebaseIndex] = {}


def get_or_build_index(root: str | Path = ".") -> CodebaseIndex:
    """Return a cached or freshly built+persisted index for ``root``."""
    root_str = str(Path(root).resolve())
    with _LOCK:
        cached = _CACHED_INDEX.get(root_str)
        if cached is not None:
            return cached
        loaded = load_index(root_str)
        if loaded is None:
            loaded = build_index(root_str)
            save_index(loaded, root_str)
        else:
            loaded = update_index(loaded, root_str)
            save_index(loaded, root_str)
        _CACHED_INDEX[root_str] = loaded
        return loaded


def invalidate_cached_index(root: str | Path = ".") -> None:
    """Drop the cached index so the next call rebuilds it from disk."""
    root_str = str(Path(root).resolve())
    with _LOCK:
        _CACHED_INDEX.pop(root_str, None)


@GLOBAL_REGISTRY.tool(
    description=(
        "Search the persistent codebase index for symbols matching a substring. "
        "Optionally filter by kind (class, function, method, variable, type_alias, "
        "constant, interface, enum) or language. Far faster than file_search for "
        "structured lookups."
    ),
    side_effect="read_only",
    timeout=15.0,
)
async def index_search(name: str, kind: str | None = None, language: str | None = None, root: str = ".") -> dict:
    """Symbol search across the indexed project."""
    idx = await asyncio.to_thread(get_or_build_index, root)
    matches = index_search_symbol(idx, name, kind=kind, language=language)
    return {
        "query": name,
        "count": len(matches),
        "matches": [
            {
                "name": m.name, "kind": m.kind, "file": m.file, "line": m.line,
                "signature": m.signature, "language": m.language,
                "docstring": m.docstring[:300],
            }
            for m in matches[:200]
        ],
    }


@GLOBAL_REGISTRY.tool(
    description="Find symbols whose bodies call the given symbol (reverse call-graph lookup).",
    side_effect="read_only",
    timeout=15.0,
)
async def index_callers(symbol: str, root: str = ".") -> dict:
    """Reverse call-graph lookup."""
    idx = await asyncio.to_thread(get_or_build_index, root)
    callers = index_find_callers(idx, symbol)
    return {
        "symbol": symbol,
        "count": len(callers),
        "callers": [
            {"name": c.name, "file": c.file, "line": c.line, "kind": c.kind}
            for c in callers[:200]
        ],
    }


@GLOBAL_REGISTRY.tool(
    description="Find every textual usage of a symbol across indexed files.",
    side_effect="read_only",
    timeout=30.0,
)
async def index_usages(symbol: str, root: str = ".") -> dict:
    """Whole-project textual usage scan, scoped by the index file set."""
    idx = await asyncio.to_thread(get_or_build_index, root)
    hits = await asyncio.to_thread(index_find_usages, idx, root, symbol)
    return {"symbol": symbol, "count": len(hits), "usages": hits[:500]}


@GLOBAL_REGISTRY.tool(
    description="Return a structured summary of a file: symbol table + imports + line count.",
    side_effect="read_only",
    timeout=10.0,
)
async def index_summarize_file(path: str, root: str = ".") -> dict:
    """Return the symbol/import view of ``path`` without reading the whole file."""
    idx = await asyncio.to_thread(get_or_build_index, root)
    rel = str(Path(path)).replace("\\", "/")
    summary = idx.file_summary(rel)
    if summary is None:
        return {"path": path, "found": False}
    return {
        "path": path,
        "found": True,
        "language": summary.language,
        "line_count": summary.line_count,
        "symbols": [
            {"name": s.name, "kind": s.kind, "line": s.line, "signature": s.signature, "docstring": s.docstring[:200]}
            for s in summary.symbols
        ],
        "imports": [{"module": i.module, "alias": i.alias, "line": i.line} for i in summary.imports],
    }


@GLOBAL_REGISTRY.tool(
    description="Return the dependency tree (imports of imports) rooted at a file, up to depth.",
    side_effect="read_only",
    timeout=10.0,
)
async def index_dependencies(file: str, depth: int = 2, root: str = ".") -> dict:
    """Walk the dependency graph from ``file``."""
    idx = await asyncio.to_thread(get_or_build_index, root)
    tree = index_dependency_chain(idx, file, depth=depth)
    return {"root": file, "depth": depth, "tree": tree}


@GLOBAL_REGISTRY.tool(
    description="Return summary statistics about the codebase index.",
    side_effect="read_only",
    timeout=10.0,
)
async def index_stats(root: str = ".") -> dict:
    """Return basic statistics: file count, symbol count, languages, freshness."""
    idx = await asyncio.to_thread(get_or_build_index, root)
    stats = idx.stats()
    return {
        "total_files": stats.total_files,
        "total_symbols": stats.total_symbols,
        "by_language": stats.by_language,
        "last_updated": stats.last_updated,
    }


@GLOBAL_REGISTRY.tool(
    description="Rebuild the codebase index from scratch. Useful after large refactors.",
    side_effect="compute",
    timeout=300.0,
)
async def index_rebuild(root: str = ".") -> dict:
    """Discard the cached index and rebuild it."""
    invalidate_cached_index(root)

    def _do() -> CodebaseIndex:
        idx = build_index(root)
        save_index(idx, root)
        return idx

    idx = await asyncio.to_thread(_do)
    with _LOCK:
        _CACHED_INDEX[str(Path(root).resolve())] = idx
    stats = idx.stats()
    return {"rebuilt": True, "total_files": stats.total_files, "total_symbols": stats.total_symbols}


def resolve_mentions(objective: str, root: str | Path = ".") -> list[dict]:
    """Parse ``@symbol`` / ``@path/to/file`` references in ``objective``.

    Args:
        objective: The user-provided objective string.
        root: Project root used to build / load the index.

    Returns:
        A list of resolution dicts ready to be injected into the planner context.
    """
    import re

    mentions = re.findall(r"@([A-Za-z_][\w./-]*)", objective)
    if not mentions:
        return []
    idx = get_or_build_index(root)
    root_p = Path(root).resolve()
    resolved: list[dict] = []
    for token in mentions:
        candidate = root_p / token
        if candidate.exists() and candidate.is_file():
            rel = str(candidate.relative_to(root_p)).replace("\\", "/")
            summary = idx.file_summary(rel)
            if summary is not None:
                resolved.append({
                    "mention": token,
                    "type": "file",
                    "path": rel,
                    "language": summary.language,
                    "symbols": [
                        {"name": s.name, "kind": s.kind, "line": s.line, "signature": s.signature}
                        for s in summary.symbols
                    ],
                })
                continue
        matches = index_search_symbol(idx, token)
        if matches:
            resolved.append({
                "mention": token,
                "type": "symbol",
                "matches": [
                    {"name": m.name, "kind": m.kind, "file": m.file, "line": m.line, "signature": m.signature}
                    for m in matches[:5]
                ],
            })
    return resolved
