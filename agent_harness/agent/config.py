"""Layered configuration for the agent harness.

The resolution order is: hard-coded defaults → ``~/.config/agent/config.toml`` →
project-local ``.agent.toml`` → environment variables (``AGENT_*``) → CLI flags.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import tomllib  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - Python <3.11 fallback
    import tomli as tomllib  # type: ignore[no-redef]

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseModel):
    """LLM provider and routing configuration."""

    provider: str = "anthropic"
    model: str = "claude-sonnet-4-6"
    fallback_provider: str | None = "openai"
    fallback_model: str | None = "gpt-4o-mini"
    temperature: float = 0.2
    max_output_tokens: int = 4096
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    openai_base_url: str | None = None


class BudgetConfig(BaseModel):
    """Token and cost budget limits."""

    session_token_limit: int = 1_000_000
    context_token_limit: int = 180_000
    cost_warn_usd: float = 5.0
    cost_abort_usd: float = 50.0


class ToolsConfig(BaseModel):
    """Per-tool execution policy."""

    default_timeout: int = 60
    bash_timeout: int = 120
    allow_destructive_commands: bool = False
    ripgrep_binary: str = "rg"


class MemoryConfig(BaseModel):
    """Persistence locations for the memory tiers."""

    working_dir: str = ".agent_state"
    semantic_collection: str = "agent_default"
    episodic_db: str = "episodic.sqlite"
    working_ttl_seconds: int = 0


class UIConfig(BaseModel):
    """Terminal UI preferences."""

    refresh_per_second: int = 10
    show_memory_panel: bool = True
    show_dag_panel: bool = True
    enabled: bool = True


class SafetyConfig(BaseModel):
    """Safety / human-in-the-loop policy."""

    require_confirm_for_high_risk: bool = True
    hypothetical_sandbox: bool = True
    max_debug_iterations: int = 5
    max_step_limit: int = 40


class AgentConfig(BaseSettings):
    """Top-level agent configuration aggregating every subsection."""

    model_config = SettingsConfigDict(
        env_prefix="AGENT_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    llm: LLMConfig = Field(default_factory=LLMConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)


def _load_toml(path: Path) -> dict[str, Any]:
    """Read a TOML file and return its parsed mapping, or {} if absent."""
    if not path.exists():
        return {}
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` into ``base`` returning a new dict."""
    out = dict(base)
    for key, value in overlay.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(
    project_dir: Path | str | None = None,
    overrides: dict[str, Any] | None = None,
) -> AgentConfig:
    """Build :class:`AgentConfig` by layering config sources.

    Args:
        project_dir: Optional project root used to locate ``.agent.toml``.
        overrides: Optional override dict (typically from CLI flags).

    Returns:
        The fully resolved configuration object.
    """
    project_dir = Path(project_dir or os.getcwd())
    user_config = Path.home() / ".config" / "agent" / "config.toml"
    project_config = project_dir / ".agent.toml"

    merged: dict[str, Any] = {}
    merged = _deep_merge(merged, _load_toml(user_config))
    merged = _deep_merge(merged, _load_toml(project_config))
    if overrides:
        merged = _deep_merge(merged, overrides)

    # AgentConfig BaseSettings reads env vars automatically on instantiation.
    return AgentConfig(**merged)
