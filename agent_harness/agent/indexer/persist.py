"""Pickle+gzip persistence for the codebase index."""

from __future__ import annotations

import gzip
import pickle
from pathlib import Path

from agent.indexer.model import CodebaseIndex


def index_path(project_root: Path | str) -> Path:
    """Return the canonical on-disk index location for ``project_root``."""
    return Path(project_root) / ".agent_index" / "index.pkl.gz"


def save_index(index: CodebaseIndex, project_root: Path | str) -> Path:
    """Serialise ``index`` to ``<project_root>/.agent_index/index.pkl.gz``."""
    path = index_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as fh:
        pickle.dump(index.to_serializable(), fh)
    return path


def load_index(project_root: Path | str) -> CodebaseIndex | None:
    """Load a previously persisted index, or ``None`` if missing/corrupt."""
    path = index_path(project_root)
    if not path.exists():
        return None
    try:
        with gzip.open(path, "rb") as fh:
            payload = pickle.load(fh)
    except (OSError, pickle.PickleError, EOFError):
        return None
    return CodebaseIndex.from_serializable(payload)
