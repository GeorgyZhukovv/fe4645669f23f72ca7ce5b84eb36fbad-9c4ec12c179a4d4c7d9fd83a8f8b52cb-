"""File read/write tools with encoding detection, chunking, and atomic writes."""

from __future__ import annotations

import asyncio
import difflib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from agent.tools.registry import GLOBAL_REGISTRY


@dataclass
class FileContent:
    """Structured representation of a file's contents."""

    path: str
    encoding: str
    is_binary: bool
    size: int
    content: str
    truncated: bool = False


@dataclass
class WriteResult:
    """Result of a file write operation."""

    path: str
    bytes_written: int
    mode: str
    applied: bool


MAX_READ_BYTES = 1_000_000


def _detect_encoding(data: bytes) -> tuple[str, bool]:
    """Detect encoding via ``chardet``; return (encoding, is_binary)."""
    try:
        import chardet
    except ImportError:  # pragma: no cover
        chardet = None  # type: ignore[assignment]
    if not data:
        return "utf-8", False
    sample = data[:8192]
    if b"\x00" in sample:
        return "binary", True
    if chardet is None:
        return "utf-8", False
    guess = chardet.detect(sample)
    enc = guess.get("encoding") or "utf-8"
    confidence = guess.get("confidence") or 0.0
    return enc, confidence < 0.5


@GLOBAL_REGISTRY.tool(
    description="Read a file from disk with automatic encoding detection and chunking for large files.",
    side_effect="read_only",
    timeout=30.0,
)
async def file_read(path: str, encoding: str = "auto") -> dict:
    """Read ``path`` and return a :class:`FileContent`-shaped dict.

    Args:
        path: Filesystem path to read.
        encoding: ``"auto"`` to detect, or a codec name to force.
    """
    p = Path(path)

    def _read() -> FileContent:
        size = p.stat().st_size
        with p.open("rb") as fh:
            raw = fh.read(MAX_READ_BYTES + 1)
        truncated = len(raw) > MAX_READ_BYTES
        raw = raw[:MAX_READ_BYTES]
        if encoding == "auto":
            enc, is_bin = _detect_encoding(raw)
        else:
            enc, is_bin = encoding, False
        if is_bin:
            return FileContent(
                path=str(p),
                encoding=enc,
                is_binary=True,
                size=size,
                content=f"<binary file, {size} bytes>",
                truncated=truncated,
            )
        text = raw.decode(enc, errors="replace")
        return FileContent(
            path=str(p),
            encoding=enc,
            is_binary=False,
            size=size,
            content=text,
            truncated=truncated,
        )

    fc = await asyncio.to_thread(_read)
    return {
        "path": fc.path,
        "encoding": fc.encoding,
        "is_binary": fc.is_binary,
        "size": fc.size,
        "content": fc.content,
        "truncated": fc.truncated,
    }


def _apply_unified_diff(original: str, diff: str) -> str:
    """Apply a unified diff to ``original`` and return the patched text.

    Raises:
        ValueError: If the patch cannot be cleanly applied.
    """
    original_lines = original.splitlines(keepends=True)
    patched: list[str] = []
    src_idx = 0
    lines = diff.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("---") or line.startswith("+++"):
            i += 1
            continue
        if line.startswith("@@"):
            parts = line.split(" ")
            try:
                src_meta = parts[1]
                start = int(src_meta.split(",")[0].lstrip("-")) - 1
            except (IndexError, ValueError) as exc:
                raise ValueError(f"Malformed hunk header: {line}") from exc
            patched.extend(original_lines[src_idx:start])
            src_idx = start
            i += 1
            while i < len(lines) and not lines[i].startswith("@@"):
                hunk = lines[i]
                if hunk.startswith(" "):
                    if src_idx >= len(original_lines):
                        raise ValueError("Hunk context past end of file")
                    patched.append(original_lines[src_idx])
                    src_idx += 1
                elif hunk.startswith("-"):
                    src_idx += 1
                elif hunk.startswith("+"):
                    body = hunk[1:]
                    if not body.endswith("\n"):
                        body += "\n"
                    patched.append(body)
                elif hunk == "":
                    pass
                else:
                    raise ValueError(f"Unexpected diff line: {hunk!r}")
                i += 1
        else:
            i += 1
    patched.extend(original_lines[src_idx:])
    return "".join(patched)


@GLOBAL_REGISTRY.tool(
    description=(
        "Write a file to disk. Mode 'overwrite' uses an atomic temp+rename. "
        "Mode 'patch' applies a unified diff. Mode 'append' appends to the end."
    ),
    side_effect="destructive",
    timeout=30.0,
)
async def file_write(path: str, content: str, mode: Literal["overwrite", "patch", "append"] = "overwrite") -> dict:
    """Write to ``path``; see ``mode`` for semantics.

    Args:
        path: Target file path.
        content: New content, or unified diff text for ``mode='patch'``.
        mode: Write mode.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    def _do() -> WriteResult:
        if mode == "overwrite":
            fd, tmp = tempfile.mkstemp(
                prefix=p.name + ".", suffix=".tmp", dir=str(p.parent)
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(content)
                os.replace(tmp, p)
            except Exception:
                if os.path.exists(tmp):
                    os.unlink(tmp)
                raise
            return WriteResult(path=str(p), bytes_written=len(content.encode()), mode=mode, applied=True)
        if mode == "append":
            with p.open("a", encoding="utf-8") as fh:
                fh.write(content)
            return WriteResult(path=str(p), bytes_written=len(content.encode()), mode=mode, applied=True)
        if mode == "patch":
            original = p.read_text(encoding="utf-8") if p.exists() else ""
            patched = _apply_unified_diff(original, content)
            tmp_fd, tmp = tempfile.mkstemp(prefix=p.name + ".", suffix=".tmp", dir=str(p.parent))
            try:
                with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                    fh.write(patched)
                os.replace(tmp, p)
            except Exception:
                if os.path.exists(tmp):
                    os.unlink(tmp)
                raise
            return WriteResult(path=str(p), bytes_written=len(patched.encode()), mode=mode, applied=True)
        raise ValueError(f"Unknown mode {mode!r}")

    res = await asyncio.to_thread(_do)
    return {
        "path": res.path,
        "bytes_written": res.bytes_written,
        "mode": res.mode,
        "applied": res.applied,
    }


def diff_strings(a: str, b: str, a_label: str = "a", b_label: str = "b") -> str:
    """Return a unified diff between ``a`` and ``b`` for display."""
    return "".join(
        difflib.unified_diff(
            a.splitlines(keepends=True),
            b.splitlines(keepends=True),
            fromfile=a_label,
            tofile=b_label,
            n=3,
        )
    )
