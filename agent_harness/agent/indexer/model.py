"""Data model for the codebase index."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SymbolKind = str  # class | function | method | variable | type_alias | constant | interface | enum


@dataclass
class Location:
    """A file + line + column triple."""

    file: str
    line: int
    column: int = 0


@dataclass
class SymbolEntry:
    """One indexed top-level or nested symbol."""

    name: str
    kind: SymbolKind
    file: str
    line: int
    signature: str = ""
    docstring: str = ""
    language: str = ""

    def qualified(self) -> str:
        """Return a ``file:symbol`` qualified id."""
        return f"{self.file}:{self.name}"


@dataclass
class ImportEntry:
    """One import statement in a source file."""

    module: str
    alias: str = ""
    line: int = 0


@dataclass
class FileMeta:
    """Per-file metadata used to detect changes."""

    language: str
    size_bytes: int
    line_count: int
    last_modified: float
    content_hash: str


@dataclass
class FileSummary:
    """Complete symbol view of a single file."""

    path: str
    language: str
    symbols: list[SymbolEntry]
    imports: list[ImportEntry]
    line_count: int


@dataclass
class DependencyTree:
    """Recursive import dependency tree rooted at one file."""

    file: str
    depth: int
    children: list["DependencyTree"] = field(default_factory=list)


@dataclass
class IndexStats:
    """Snapshot statistics about the index."""

    total_files: int
    total_symbols: int
    by_language: dict[str, int]
    last_updated: float


@dataclass
class CodebaseIndex:
    """In-memory index of the whole project."""

    symbols: dict[str, list[SymbolEntry]] = field(default_factory=dict)
    imports: dict[str, list[ImportEntry]] = field(default_factory=dict)
    call_graph: dict[str, set[str]] = field(default_factory=dict)
    file_metadata: dict[str, FileMeta] = field(default_factory=dict)
    dependency_graph: dict[str, set[str]] = field(default_factory=dict)
    last_updated: float = 0.0

    def stats(self) -> IndexStats:
        """Return :class:`IndexStats`."""
        by_lang: dict[str, int] = {}
        for meta in self.file_metadata.values():
            by_lang[meta.language] = by_lang.get(meta.language, 0) + 1
        total_symbols = sum(len(v) for v in self.symbols.values())
        return IndexStats(
            total_files=len(self.file_metadata),
            total_symbols=total_symbols,
            by_language=by_lang,
            last_updated=self.last_updated,
        )

    def file_summary(self, path: str) -> FileSummary | None:
        """Build a :class:`FileSummary` for one file."""
        meta = self.file_metadata.get(path)
        if meta is None:
            return None
        symbols = [s for lst in self.symbols.values() for s in lst if s.file == path]
        return FileSummary(
            path=path,
            language=meta.language,
            symbols=sorted(symbols, key=lambda s: s.line),
            imports=list(self.imports.get(path, [])),
            line_count=meta.line_count,
        )

    def to_serializable(self) -> dict[str, Any]:
        """Return a plain-dict view (used by the persistence layer)."""
        return {
            "symbols": {
                k: [s.__dict__ for s in v] for k, v in self.symbols.items()
            },
            "imports": {
                k: [i.__dict__ for i in v] for k, v in self.imports.items()
            },
            "call_graph": {k: sorted(v) for k, v in self.call_graph.items()},
            "file_metadata": {k: v.__dict__ for k, v in self.file_metadata.items()},
            "dependency_graph": {k: sorted(v) for k, v in self.dependency_graph.items()},
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_serializable(cls, payload: dict[str, Any]) -> "CodebaseIndex":
        """Inverse of :meth:`to_serializable`."""
        return cls(
            symbols={
                k: [SymbolEntry(**s) for s in v] for k, v in payload.get("symbols", {}).items()
            },
            imports={
                k: [ImportEntry(**i) for i in v] for k, v in payload.get("imports", {}).items()
            },
            call_graph={k: set(v) for k, v in payload.get("call_graph", {}).items()},
            file_metadata={
                k: FileMeta(**v) for k, v in payload.get("file_metadata", {}).items()
            },
            dependency_graph={
                k: set(v) for k, v in payload.get("dependency_graph", {}).items()
            },
            last_updated=payload.get("last_updated", 0.0),
        )
