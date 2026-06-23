"""Shared fixtures."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture()
def tmp_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated `.agent_state` directory routed via env override."""
    state = tmp_path / "state"
    state.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("AGENT_MEMORY__WORKING_DIR", str(state))
    yield state
    if state.exists():
        shutil.rmtree(state, ignore_errors=True)


@pytest.fixture()
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Empty CWD for tools that operate on the working tree."""
    monkeypatch.chdir(tmp_path)
    return tmp_path
