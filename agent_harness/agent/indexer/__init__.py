"""Persistent codebase indexer (symbol table, imports, call graph)."""

from agent.indexer.builder import IndexBuilder, build_index, update_index
from agent.indexer.model import (
    CodebaseIndex,
    DependencyTree,
    FileMeta,
    FileSummary,
    ImportEntry,
    IndexStats,
    Location,
    SymbolEntry,
)
from agent.indexer.persist import load_index, save_index

__all__ = [
    "CodebaseIndex",
    "DependencyTree",
    "FileMeta",
    "FileSummary",
    "ImportEntry",
    "IndexBuilder",
    "IndexStats",
    "Location",
    "SymbolEntry",
    "build_index",
    "load_index",
    "save_index",
    "update_index",
]
