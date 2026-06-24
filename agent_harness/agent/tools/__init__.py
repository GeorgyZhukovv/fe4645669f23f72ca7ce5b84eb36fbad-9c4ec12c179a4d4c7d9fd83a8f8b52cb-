"""Built-in tools and registry for the agent harness."""

from agent.tools.registry import GLOBAL_REGISTRY, ToolRegistry, ToolSpec
from agent.tools import bash, code, deps, files, git, indexer, lsp, multi_edit, search, vcs_hosting, web  # noqa: F401 - side-effect: registration


def default_registry() -> ToolRegistry:
    """Return the populated default registry (built-in tools registered)."""
    return GLOBAL_REGISTRY


__all__ = ["GLOBAL_REGISTRY", "ToolRegistry", "ToolSpec", "default_registry"]
