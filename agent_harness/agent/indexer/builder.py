"""Build / update the codebase index. AST-based for Python, regex for others."""

from __future__ import annotations

import ast
import fnmatch
import hashlib
import re
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from agent.indexer.model import (
    CodebaseIndex,
    FileMeta,
    ImportEntry,
    SymbolEntry,
)


DEFAULT_INCLUDES = ["**/*.py", "**/*.js", "**/*.jsx", "**/*.ts", "**/*.tsx", "**/*.rs", "**/*.go"]
DEFAULT_EXCLUDES = [
    "**/__pycache__/**", "**/.git/**", "**/node_modules/**", "**/target/**",
    "**/.venv/**", "**/venv/**", "**/dist/**", "**/build/**", "**/.agent_index/**",
    "**/.agent_state/**", "**/.agent_trash/**", "**/site-packages/**",
]


def _language_for(path: Path) -> str:
    """Map a file extension to a coarse language tag."""
    ext = path.suffix.lower()
    return {
        ".py": "python",
        ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
        ".ts": "typescript", ".tsx": "typescript",
        ".rs": "rust",
        ".go": "go",
    }.get(ext, "unknown")


def _matches_any(path: str, patterns: Iterable[str]) -> bool:
    """Glob-style match against any pattern."""
    return any(fnmatch.fnmatch(path, pat) for pat in patterns)


def _file_hash(path: Path) -> str:
    """SHA-1 of file bytes, used to detect changes between runs."""
    h = hashlib.sha1()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class IndexBuilder:
    """Build a :class:`CodebaseIndex` from a project root."""

    def __init__(
        self,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
    ) -> None:
        """Configure the include/exclude glob lists."""
        self.include_patterns = include_patterns or DEFAULT_INCLUDES
        self.exclude_patterns = exclude_patterns or DEFAULT_EXCLUDES

    def discover(self, root: Path) -> list[Path]:
        """Walk ``root`` honouring the include/exclude lists."""
        out: list[Path] = []
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            rel = str(path.relative_to(root)).replace("\\", "/")
            if _matches_any(rel, self.exclude_patterns):
                continue
            if not _matches_any(rel, self.include_patterns):
                continue
            out.append(path)
        return out

    def parse_file(self, root: Path, path: Path) -> tuple[list[SymbolEntry], list[ImportEntry], FileMeta, set[str], set[str]]:
        """Parse ``path`` and return (symbols, imports, meta, calls, dep_files)."""
        rel = str(path.relative_to(root)).replace("\\", "/")
        language = _language_for(path)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        meta = FileMeta(
            language=language,
            size_bytes=len(text.encode("utf-8")),
            line_count=text.count("\n") + (1 if text and not text.endswith("\n") else 0),
            last_modified=path.stat().st_mtime if path.exists() else 0.0,
            content_hash=_file_hash(path) if path.exists() else "",
        )
        if language == "python":
            syms, imps, calls, deps = _parse_python(text, rel)
        elif language in {"javascript", "typescript"}:
            syms, imps, calls, deps = _parse_js_ts(text, rel, language)
        elif language == "rust":
            syms, imps, calls, deps = _parse_rust(text, rel)
        elif language == "go":
            syms, imps, calls, deps = _parse_go(text, rel)
        else:
            syms, imps, calls, deps = [], [], set(), set()
        return syms, imps, meta, calls, deps


def _docstring_of(node: ast.AST) -> str:
    """Extract the first-statement string literal as docstring text."""
    if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef, ast.ClassDef, ast.Module)):
        return ast.get_docstring(node) or ""
    return ""


def _python_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Render a compact, source-faithful signature for a Python def."""
    args = node.args
    parts: list[str] = []
    for a in args.posonlyargs:
        parts.append(_arg_str(a))
    if args.posonlyargs:
        parts.append("/")
    for a in args.args:
        parts.append(_arg_str(a))
    if args.vararg:
        parts.append("*" + _arg_str(args.vararg))
    elif args.kwonlyargs:
        parts.append("*")
    for a in args.kwonlyargs:
        parts.append(_arg_str(a))
    if args.kwarg:
        parts.append("**" + _arg_str(args.kwarg))
    ret = ""
    if node.returns is not None:
        ret = " -> " + ast.unparse(node.returns)
    prefix = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "
    return f"{prefix}{node.name}({', '.join(parts)}){ret}"


def _arg_str(arg: ast.arg) -> str:
    """Render a single argument as ``name: annotation``."""
    if arg.annotation is not None:
        return f"{arg.arg}: {ast.unparse(arg.annotation)}"
    return arg.arg


class _CallVisitor(ast.NodeVisitor):
    """Collect callee names referenced inside a function body."""

    def __init__(self) -> None:
        self.calls: set[str] = set()

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        func = node.func
        if isinstance(func, ast.Name):
            self.calls.add(func.id)
        elif isinstance(func, ast.Attribute):
            self.calls.add(func.attr)
        self.generic_visit(node)


def _parse_python(text: str, rel: str) -> tuple[list[SymbolEntry], list[ImportEntry], set[str], set[str]]:
    """Parse a Python file via ``ast``."""
    try:
        tree = ast.parse(text, filename=rel)
    except SyntaxError:
        return [], [], set(), set()
    symbols: list[SymbolEntry] = []
    imports: list[ImportEntry] = []
    calls: set[str] = set()
    deps: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(ImportEntry(module=alias.name, alias=alias.asname or "", line=node.lineno))
                deps.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for alias in node.names:
                imports.append(ImportEntry(module=f"{mod}.{alias.name}" if mod else alias.name, alias=alias.asname or "", line=node.lineno))
            if mod:
                deps.add(mod.split(".")[0])

    def walk_body(body: list[ast.stmt], cls_name: str | None = None) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                kind = "method" if cls_name else "function"
                name = f"{cls_name}.{node.name}" if cls_name else node.name
                sig = _python_signature(node)
                doc = _docstring_of(node)
                symbols.append(SymbolEntry(
                    name=name, kind=kind, file=rel, line=node.lineno,
                    signature=sig, docstring=doc, language="python",
                ))
                cv = _CallVisitor()
                cv.visit(node)
                for c in cv.calls:
                    calls.add(c)
            elif isinstance(node, ast.ClassDef):
                bases = [ast.unparse(b) for b in node.bases]
                sig = f"class {node.name}" + (f"({', '.join(bases)})" if bases else "")
                symbols.append(SymbolEntry(
                    name=node.name, kind="class", file=rel, line=node.lineno,
                    signature=sig, docstring=_docstring_of(node), language="python",
                ))
                walk_body(node.body, cls_name=node.name)
            elif isinstance(node, ast.Assign) and cls_name is None:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        symbols.append(SymbolEntry(
                            name=target.id, kind="constant" if target.id.isupper() else "variable",
                            file=rel, line=node.lineno,
                            signature=ast.unparse(node)[:200], language="python",
                        ))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and cls_name is None:
                symbols.append(SymbolEntry(
                    name=node.target.id, kind="type_alias" if isinstance(node.annotation, ast.Name) and node.annotation.id == "TypeAlias" else "variable",
                    file=rel, line=node.lineno,
                    signature=ast.unparse(node)[:200], language="python",
                ))

    walk_body(tree.body)
    return symbols, imports, calls, deps


_JS_FUNC_RE = re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)", re.MULTILINE)
_JS_ARROW_RE = re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\(([^)]*)\)\s*=>", re.MULTILINE)
_JS_CLASS_RE = re.compile(r"^\s*(?:export\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?", re.MULTILINE)
_TS_INTERFACE_RE = re.compile(r"^\s*(?:export\s+)?interface\s+(\w+)", re.MULTILINE)
_TS_TYPE_RE = re.compile(r"^\s*(?:export\s+)?type\s+(\w+)\s*=", re.MULTILINE)
_TS_ENUM_RE = re.compile(r"^\s*(?:export\s+)?enum\s+(\w+)", re.MULTILINE)
_JS_IMPORT_RE = re.compile(r"^\s*import\s+(?:[^'\"\n]+\s+from\s+)?['\"]([^'\"]+)['\"]", re.MULTILINE)


def _line_of(text: str, offset: int) -> int:
    """Map a string offset to a 1-based line number."""
    return text.count("\n", 0, offset) + 1


def _parse_js_ts(text: str, rel: str, language: str) -> tuple[list[SymbolEntry], list[ImportEntry], set[str], set[str]]:
    """Regex-based JS / TS symbol extraction."""
    symbols: list[SymbolEntry] = []
    imports: list[ImportEntry] = []
    deps: set[str] = set()

    for m in _JS_FUNC_RE.finditer(text):
        symbols.append(SymbolEntry(
            name=m.group(1), kind="function", file=rel, line=_line_of(text, m.start()),
            signature=f"function {m.group(1)}({m.group(2).strip()})", language=language,
        ))
    for m in _JS_ARROW_RE.finditer(text):
        symbols.append(SymbolEntry(
            name=m.group(1), kind="function", file=rel, line=_line_of(text, m.start()),
            signature=f"const {m.group(1)} = ({m.group(2).strip()}) => ...", language=language,
        ))
    for m in _JS_CLASS_RE.finditer(text):
        extends = f" extends {m.group(2)}" if m.group(2) else ""
        symbols.append(SymbolEntry(
            name=m.group(1), kind="class", file=rel, line=_line_of(text, m.start()),
            signature=f"class {m.group(1)}{extends}", language=language,
        ))
    if language == "typescript":
        for m in _TS_INTERFACE_RE.finditer(text):
            symbols.append(SymbolEntry(
                name=m.group(1), kind="interface", file=rel, line=_line_of(text, m.start()),
                signature=f"interface {m.group(1)}", language=language,
            ))
        for m in _TS_TYPE_RE.finditer(text):
            symbols.append(SymbolEntry(
                name=m.group(1), kind="type_alias", file=rel, line=_line_of(text, m.start()),
                signature=f"type {m.group(1)}", language=language,
            ))
        for m in _TS_ENUM_RE.finditer(text):
            symbols.append(SymbolEntry(
                name=m.group(1), kind="enum", file=rel, line=_line_of(text, m.start()),
                signature=f"enum {m.group(1)}", language=language,
            ))
    for m in _JS_IMPORT_RE.finditer(text):
        mod = m.group(1)
        imports.append(ImportEntry(module=mod, line=_line_of(text, m.start())))
        if not mod.startswith("."):
            deps.add(mod.split("/")[0])
    return symbols, imports, set(), deps


_RUST_FN_RE = re.compile(r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+(\w+)\s*(<[^>]*>)?\s*\(([^)]*)\)", re.MULTILINE)
_RUST_STRUCT_RE = re.compile(r"^\s*(?:pub\s+)?struct\s+(\w+)", re.MULTILINE)
_RUST_ENUM_RE = re.compile(r"^\s*(?:pub\s+)?enum\s+(\w+)", re.MULTILINE)
_RUST_TRAIT_RE = re.compile(r"^\s*(?:pub\s+)?trait\s+(\w+)", re.MULTILINE)
_RUST_TYPE_RE = re.compile(r"^\s*(?:pub\s+)?type\s+(\w+)\s*=", re.MULTILINE)
_RUST_USE_RE = re.compile(r"^\s*use\s+([\w:]+)", re.MULTILINE)


def _parse_rust(text: str, rel: str) -> tuple[list[SymbolEntry], list[ImportEntry], set[str], set[str]]:
    """Regex-based Rust symbol extraction."""
    symbols: list[SymbolEntry] = []
    imports: list[ImportEntry] = []
    deps: set[str] = set()
    for m in _RUST_FN_RE.finditer(text):
        symbols.append(SymbolEntry(
            name=m.group(1), kind="function", file=rel, line=_line_of(text, m.start()),
            signature=f"fn {m.group(1)}({m.group(3).strip()})", language="rust",
        ))
    for m in _RUST_STRUCT_RE.finditer(text):
        symbols.append(SymbolEntry(
            name=m.group(1), kind="class", file=rel, line=_line_of(text, m.start()),
            signature=f"struct {m.group(1)}", language="rust",
        ))
    for m in _RUST_ENUM_RE.finditer(text):
        symbols.append(SymbolEntry(
            name=m.group(1), kind="enum", file=rel, line=_line_of(text, m.start()),
            signature=f"enum {m.group(1)}", language="rust",
        ))
    for m in _RUST_TRAIT_RE.finditer(text):
        symbols.append(SymbolEntry(
            name=m.group(1), kind="interface", file=rel, line=_line_of(text, m.start()),
            signature=f"trait {m.group(1)}", language="rust",
        ))
    for m in _RUST_TYPE_RE.finditer(text):
        symbols.append(SymbolEntry(
            name=m.group(1), kind="type_alias", file=rel, line=_line_of(text, m.start()),
            signature=f"type {m.group(1)}", language="rust",
        ))
    for m in _RUST_USE_RE.finditer(text):
        path = m.group(1)
        imports.append(ImportEntry(module=path, line=_line_of(text, m.start())))
        deps.add(path.split("::")[0])
    return symbols, imports, set(), deps


_GO_FN_RE = re.compile(r"^\s*func\s+(?:\([^)]*\)\s+)?(\w+)\s*\(([^)]*)\)", re.MULTILINE)
_GO_TYPE_RE = re.compile(r"^\s*type\s+(\w+)\s+(struct|interface|\w+)", re.MULTILINE)
_GO_VAR_RE = re.compile(r"^\s*(?:var|const)\s+(\w+)", re.MULTILINE)
_GO_IMPORT_RE = re.compile(r'^\s*"([^"]+)"\s*$', re.MULTILINE)


def _parse_go(text: str, rel: str) -> tuple[list[SymbolEntry], list[ImportEntry], set[str], set[str]]:
    """Regex-based Go symbol extraction."""
    symbols: list[SymbolEntry] = []
    imports: list[ImportEntry] = []
    deps: set[str] = set()
    for m in _GO_FN_RE.finditer(text):
        symbols.append(SymbolEntry(
            name=m.group(1), kind="function", file=rel, line=_line_of(text, m.start()),
            signature=f"func {m.group(1)}({m.group(2).strip()})", language="go",
        ))
    for m in _GO_TYPE_RE.finditer(text):
        name, what = m.group(1), m.group(2)
        kind = {"struct": "class", "interface": "interface"}.get(what, "type_alias")
        symbols.append(SymbolEntry(
            name=name, kind=kind, file=rel, line=_line_of(text, m.start()),
            signature=f"type {name} {what}", language="go",
        ))
    for m in _GO_VAR_RE.finditer(text):
        symbols.append(SymbolEntry(
            name=m.group(1), kind="variable", file=rel, line=_line_of(text, m.start()),
            language="go",
        ))
    # Imports — best effort, capture quoted strings within `import (...)` blocks.
    if "import" in text:
        for m in _GO_IMPORT_RE.finditer(text):
            mod = m.group(1)
            imports.append(ImportEntry(module=mod, line=_line_of(text, m.start())))
            deps.add(mod.split("/")[0])
    return symbols, imports, set(), deps


def _merge_file_into_index(index: CodebaseIndex, rel: str, symbols: list[SymbolEntry], imports: list[ImportEntry], meta: FileMeta, calls: set[str], deps: set[str]) -> None:
    """Insert / replace a file's contribution within the index."""
    # Strip previous entries for this file
    for name in list(index.symbols.keys()):
        index.symbols[name] = [s for s in index.symbols[name] if s.file != rel]
        if not index.symbols[name]:
            del index.symbols[name]
    for sym in symbols:
        index.symbols.setdefault(sym.name, []).append(sym)
        # Reverse call-graph: this file:sym -> calls
        index.call_graph[f"{rel}:{sym.name}"] = set(calls)
    index.imports[rel] = imports
    index.file_metadata[rel] = meta
    index.dependency_graph[rel] = deps


def build_index(
    root: str | Path,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
) -> CodebaseIndex:
    """Walk ``root`` and produce a fresh index."""
    root_p = Path(root).resolve()
    builder = IndexBuilder(include_patterns, exclude_patterns)
    index = CodebaseIndex()
    for path in builder.discover(root_p):
        symbols, imports, meta, calls, deps = builder.parse_file(root_p, path)
        rel = str(path.relative_to(root_p)).replace("\\", "/")
        _merge_file_into_index(index, rel, symbols, imports, meta, calls, deps)
    index.last_updated = time.time()
    return index


def update_index(
    index: CodebaseIndex,
    root: str | Path,
    changed_files: list[str] | None = None,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
) -> CodebaseIndex:
    """Incrementally re-parse changed files, falling back to hash detection."""
    root_p = Path(root).resolve()
    builder = IndexBuilder(include_patterns, exclude_patterns)
    if changed_files is None:
        # Detect via hash comparison against the existing metadata.
        candidates = builder.discover(root_p)
        changed_files = []
        seen: set[str] = set()
        for path in candidates:
            rel = str(path.relative_to(root_p)).replace("\\", "/")
            seen.add(rel)
            prev = index.file_metadata.get(rel)
            new_hash = _file_hash(path)
            if prev is None or prev.content_hash != new_hash:
                changed_files.append(rel)
        # Files that vanished from disk should be evicted from the index.
        for rel in list(index.file_metadata.keys()):
            if rel not in seen:
                changed_files.append(rel)

    for rel in changed_files:
        path = root_p / rel
        if not path.exists():
            for name in list(index.symbols.keys()):
                index.symbols[name] = [s for s in index.symbols[name] if s.file != rel]
                if not index.symbols[name]:
                    del index.symbols[name]
            index.imports.pop(rel, None)
            index.file_metadata.pop(rel, None)
            index.dependency_graph.pop(rel, None)
            continue
        symbols, imports, meta, calls, deps = builder.parse_file(root_p, path)
        _merge_file_into_index(index, rel, symbols, imports, meta, calls, deps)
    index.last_updated = time.time()
    return index


def index_search_symbol(
    index: CodebaseIndex,
    name: str,
    kind: str | None = None,
    language: str | None = None,
) -> list[SymbolEntry]:
    """Substring lookup against the indexed symbol table."""
    out: list[SymbolEntry] = []
    needle = name.lower()
    for sym_name, entries in index.symbols.items():
        if needle not in sym_name.lower():
            continue
        for s in entries:
            if kind and s.kind != kind:
                continue
            if language and s.language != language:
                continue
            out.append(s)
    return out


def index_find_callers(index: CodebaseIndex, symbol: str) -> list[SymbolEntry]:
    """Return any indexed symbols whose call set references ``symbol``."""
    out: list[SymbolEntry] = []
    for caller_key, callees in index.call_graph.items():
        if symbol not in callees:
            continue
        file_part, _, sym_name = caller_key.partition(":")
        for s in index.symbols.get(sym_name, []):
            if s.file == file_part:
                out.append(s)
    return out


def index_find_usages(index: CodebaseIndex, root: str | Path, symbol: str) -> list[dict[str, Any]]:
    """Plain text-search for ``symbol`` across indexed files (best effort)."""
    root_p = Path(root)
    out: list[dict[str, Any]] = []
    pattern = re.compile(r"\b" + re.escape(symbol) + r"\b")
    for rel in index.file_metadata.keys():
        path = root_p / rel
        if not path.exists():
            continue
        try:
            for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if pattern.search(line):
                    out.append({"file": rel, "line": i, "text": line.strip()[:200]})
        except OSError:
            continue
    return out


def index_dependency_chain(index: CodebaseIndex, file: str, depth: int = 2) -> dict[str, Any]:
    """Walk the dependency graph from ``file`` up to ``depth`` levels."""
    visited: set[str] = set()

    def walk(node: str, remaining: int) -> dict[str, Any]:
        if remaining <= 0 or node in visited:
            return {"file": node, "children": []}
        visited.add(node)
        children = [walk(c, remaining - 1) for c in sorted(index.dependency_graph.get(node, set()))]
        return {"file": node, "children": children}

    return walk(file, depth)
