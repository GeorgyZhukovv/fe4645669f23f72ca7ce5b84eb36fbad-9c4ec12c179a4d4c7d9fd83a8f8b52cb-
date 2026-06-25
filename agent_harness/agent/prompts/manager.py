"""Prompt template registry, mini-template renderer, persona registry."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


try:
    import tomllib  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


AgentRole = Literal["PLANNER", "CODER", "REVIEWER", "TESTER", "DOCUMENTER", "SECURITY_AUDITOR"]


class PromptExample(BaseModel):
    """One example input/output pair for a prompt."""

    input: dict[str, Any] = Field(default_factory=dict)
    expected_output: str = ""


class PromptTemplate(BaseModel):
    """Versioned named prompt with declared variables."""

    id: str
    version: str = "1"
    role: str = "PLANNER"
    template_str: str
    variables: list[str] = Field(default_factory=list)
    description: str = ""
    examples: list[PromptExample] = Field(default_factory=list)


class Persona(BaseModel):
    """A reusable behaviour preset (system-prompt overrides + tool whitelist + sampling)."""

    name: str
    base_role: str = "PLANNER"
    system_prompt_overrides: dict[str, str] = Field(default_factory=dict)
    tool_whitelist: list[str] | None = None
    tool_blacklist: list[str] | None = None
    temperature: float = 0.2
    max_tokens: int = 4096


BUILT_IN_PERSONAS: dict[str, Persona] = {
    "default": Persona(name="default", base_role="PLANNER"),
    "strict": Persona(
        name="strict",
        base_role="PLANNER",
        system_prompt_overrides={
            "executor": "You are strict: zero tolerance for speculative code. Every "
                       "function added must have a corresponding test. Explain every "
                       "decision in plain English before acting."
        },
        temperature=0.0,
    ),
    "fast": Persona(
        name="fast",
        base_role="PLANNER",
        system_prompt_overrides={
            "executor": "Be terse. One-line thoughts. Prefer single-step solutions over "
                       "elaborate plans. Skip reflections."
        },
        temperature=0.0,
        max_tokens=1024,
    ),
    "security": Persona(
        name="security",
        base_role="SECURITY_AUDITOR",
        system_prompt_overrides={
            "executor": "You are a security-first agent. Every file read triggers a scan "
                       "for hardcoded secrets and injection risks. Use trufflehog / "
                       "detect-secrets via bash_exec when present."
        },
        tool_whitelist=None,
        temperature=0.0,
    ),
    "minimal": Persona(
        name="minimal",
        base_role="PLANNER",
        system_prompt_overrides={
            "executor": "Optimise for cost. Use the smallest viable LLM. Keep context "
                       "minimal. Avoid re-reads."
        },
        temperature=0.0,
        max_tokens=512,
    ),
    "architect": Persona(
        name="architect",
        base_role="PLANNER",
        system_prompt_overrides={
            "executor": "You are an architect. Bias toward refactor and abstraction over "
                       "direct implementation. Surface design tradeoffs explicitly before "
                       "writing code."
        },
        temperature=0.4,
    ),
}


_VAR_RE = re.compile(r"\{\{\s*([\w.]+)\s*\}\}")
_INCLUDE_RE = re.compile(r"\{%\s*include\s+\"([^\"]+)\"\s*%\}")
_IF_RE = re.compile(r"\{%\s*if\s+([\w.]+)\s*%\}(.*?)\{%\s*endif\s*%\}", re.DOTALL)
_FOR_RE = re.compile(r"\{%\s*for\s+(\w+)\s+in\s+([\w.]+)\s*%\}(.*?)\{%\s*endfor\s*%\}", re.DOTALL)


def _lookup(context: dict[str, Any], dotted: str) -> Any:
    """Walk a dotted path through ``context`` and return the value (or ``None``)."""
    cur: Any = context
    for part in dotted.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            cur = getattr(cur, part, None)
        if cur is None:
            return None
    return cur


def render_template(
    template_str: str,
    context: dict[str, Any],
    library: "PromptManager | None" = None,
    _visited: set[str] | None = None,
) -> str:
    """Render a template string with ``{{ var }}`` / ``include`` / ``if`` / ``for`` support."""
    visited = _visited or set()

    def include_replacer(match: re.Match[str]) -> str:
        prompt_id = match.group(1)
        if library is None or prompt_id in visited:
            return ""
        included = library.get(prompt_id)
        if included is None:
            return ""
        return render_template(
            included.template_str, context, library, _visited=visited | {prompt_id}
        )

    text = _INCLUDE_RE.sub(include_replacer, template_str)

    def for_replacer(match: re.Match[str]) -> str:
        item_var, iter_var, body = match.group(1), match.group(2), match.group(3)
        items = _lookup(context, iter_var) or []
        out: list[str] = []
        for item in items:
            sub_ctx = dict(context)
            sub_ctx[item_var] = item
            out.append(render_template(body, sub_ctx, library, _visited=visited))
        return "".join(out)

    text = _FOR_RE.sub(for_replacer, text)

    def if_replacer(match: re.Match[str]) -> str:
        cond, body = match.group(1), match.group(2)
        return body if _lookup(context, cond) else ""

    text = _IF_RE.sub(if_replacer, text)

    def var_replacer(match: re.Match[str]) -> str:
        value = _lookup(context, match.group(1))
        return "" if value is None else str(value)

    return _VAR_RE.sub(var_replacer, text)


class PromptManager:
    """Loads / renders prompt templates from a TOML library directory."""

    def __init__(self, library_dir: Path | str | None = None) -> None:
        """Load every ``*.toml`` template under ``library_dir``."""
        self.templates: dict[str, PromptTemplate] = {}
        self.personas: dict[str, Persona] = dict(BUILT_IN_PERSONAS)
        if library_dir is None:
            library_dir = Path(__file__).parent / "library"
        self.library_dir = Path(library_dir)
        if self.library_dir.exists():
            self._load_library()

    def _load_library(self) -> None:
        for path in self.library_dir.rglob("*.toml"):
            try:
                data = tomllib.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            for raw in data.get("prompt", []) or []:
                try:
                    tpl = PromptTemplate(**raw)
                except Exception:
                    continue
                self.templates[tpl.id] = tpl

    def register(self, template: PromptTemplate) -> None:
        """Add or replace a template by id."""
        self.templates[template.id] = template

    def register_persona(self, persona: Persona) -> None:
        """Add or replace a persona by name."""
        self.personas[persona.name] = persona

    def get(self, id: str) -> PromptTemplate | None:
        """Return the template with this id (or ``None``)."""
        return self.templates.get(id)

    def render(self, id: str, **context: Any) -> str:
        """Render template ``id`` with ``context``."""
        tpl = self.get(id)
        if tpl is None:
            raise KeyError(f"unknown prompt id: {id}")
        return render_template(tpl.template_str, context, library=self)

    def persona(self, name: str) -> Persona:
        """Look up a persona; falls back to ``default``."""
        return self.personas.get(name) or self.personas["default"]

    def load_personas_from_config(self, raw: dict[str, dict[str, Any]]) -> None:
        """Inject user-defined personas from an ``.agent.toml`` ``[personas.*]`` mapping."""
        for name, body in raw.items():
            try:
                p = Persona(name=name, **body)
            except Exception:
                continue
            self.personas[name] = p
