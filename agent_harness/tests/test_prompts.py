"""Tests for the prompt manager + persona system."""

from __future__ import annotations

from agent.prompts.manager import (
    BUILT_IN_PERSONAS,
    Persona,
    PromptManager,
    PromptTemplate,
    render_template,
)
from agent.prompts.roles import ROLE_PROMPTS


def test_render_variable_substitution() -> None:
    out = render_template("Hello, {{ name }}.", {"name": "world"})
    assert out == "Hello, world."


def test_render_if_block() -> None:
    out = render_template(
        "{% if active %}yes{% endif %}{% if missing %}no{% endif %}",
        {"active": True, "missing": False},
    )
    assert out == "yes"


def test_render_for_block() -> None:
    out = render_template(
        "{% for x in items %}[{{ x }}]{% endfor %}",
        {"items": ["a", "b", "c"]},
    )
    assert out == "[a][b][c]"


def test_render_include_with_cycle_protection() -> None:
    pm = PromptManager()
    pm.register(PromptTemplate(id="inner", template_str="ok"))
    pm.register(PromptTemplate(id="outer", template_str='X{% include "inner" %}'))
    pm.register(PromptTemplate(id="loop", template_str='{% include "loop" %}'))
    assert pm.render("outer") == "Xok"
    assert pm.render("loop") == ""


def test_load_builtin_library() -> None:
    pm = PromptManager()
    assert pm.get("executor.default") is not None
    rendered = pm.render(
        "executor.default",
        task_id="t1", description="do thing", complexity="M",
        anticipated_tools="file_read", cwd=".", step_limit=10,
    )
    assert "t1" in rendered and "M" in rendered


def test_built_in_personas() -> None:
    pm = PromptManager()
    for name in ("default", "strict", "fast", "security", "minimal", "architect"):
        p = pm.persona(name)
        assert isinstance(p, Persona)
        assert p.name == name


def test_user_personas_loaded_from_config() -> None:
    pm = PromptManager()
    pm.load_personas_from_config({"team": {"base_role": "CODER", "temperature": 0.1}})
    assert pm.persona("team").base_role == "CODER"


def test_role_prompts_present() -> None:
    for role in ("CODER", "REVIEWER", "TESTER", "DOCUMENTER", "SECURITY_AUDITOR", "PLANNER"):
        assert role in ROLE_PROMPTS
        assert ROLE_PROMPTS[role]
