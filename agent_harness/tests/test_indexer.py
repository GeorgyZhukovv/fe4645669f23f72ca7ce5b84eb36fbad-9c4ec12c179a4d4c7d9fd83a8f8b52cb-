"""Tests for the codebase indexer."""

from __future__ import annotations

from pathlib import Path

from agent.indexer import build_index, load_index, save_index, update_index
from agent.indexer.builder import (
    index_dependency_chain,
    index_find_callers,
    index_search_symbol,
)
from agent.tools.indexer import invalidate_cached_index, resolve_mentions


def _make_project(root: Path) -> None:
    (root / "pkg").mkdir()
    (root / "pkg" / "__init__.py").write_text("")
    (root / "pkg" / "auth.py").write_text(
        '''"""auth module."""
import os

CONSTANT = 1

def login(user: str, password: str) -> bool:
    """Authenticate."""
    return validate(user, password)


def validate(user: str, password: str) -> bool:
    return bool(user) and bool(password)


class AuthService:
    """High-level auth API."""

    def authenticate(self, user: str) -> bool:
        return login(user, "")
''')
    (root / "pkg" / "ui.ts").write_text(
        '''import { foo } from "./foo";

export function renderHeader(title: string) {
    return `# ${title}`;
}

export const greet = (name: string) => `hi ${name}`;

export interface User { id: number }
export type Id = number;
''')


def test_build_index_finds_python_and_ts(workdir: Path) -> None:
    _make_project(workdir)
    idx = build_index(workdir)
    stats = idx.stats()
    assert stats.total_files >= 3
    assert stats.by_language["python"] >= 2
    assert stats.by_language["typescript"] == 1
    login = index_search_symbol(idx, "login")
    assert any(s.name == "login" for s in login)
    auth_svc = index_search_symbol(idx, "AuthService", kind="class")
    assert auth_svc and auth_svc[0].kind == "class"


def test_callers_and_dependency_chain(workdir: Path) -> None:
    _make_project(workdir)
    idx = build_index(workdir)
    # AuthService.authenticate calls login
    callers = index_find_callers(idx, "login")
    names = [c.name for c in callers]
    assert "AuthService.authenticate" in names
    tree = index_dependency_chain(idx, "pkg/auth.py", depth=2)
    assert tree["file"] == "pkg/auth.py"


def test_incremental_update_after_edit(workdir: Path) -> None:
    _make_project(workdir)
    idx = build_index(workdir)
    auth = workdir / "pkg" / "auth.py"
    auth.write_text(auth.read_text() + "\n\ndef new_helper() -> int:\n    return 42\n")
    updated = update_index(idx, workdir)
    assert any(s.name == "new_helper" for s in updated.symbols.get("new_helper", []))


def test_persistence_round_trip(workdir: Path) -> None:
    _make_project(workdir)
    idx = build_index(workdir)
    save_index(idx, workdir)
    loaded = load_index(workdir)
    assert loaded is not None
    assert loaded.stats().total_files == idx.stats().total_files


def test_resolve_mentions_for_file_and_symbol(workdir: Path, monkeypatch) -> None:
    _make_project(workdir)
    monkeypatch.chdir(workdir)
    invalidate_cached_index(".")
    out = resolve_mentions("Refactor @AuthService and @pkg/auth.py", root=".")
    assert any(r["mention"] == "AuthService" and r["type"] == "symbol" for r in out)
    assert any(r["mention"] == "pkg/auth.py" and r["type"] == "file" for r in out)
