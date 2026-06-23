"""Click-based CLI: run, resume, history, tools, memory, cost."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from agent.config import AgentConfig, load_config
from agent.memory.semantic import SemanticMemory
from agent.orchestrator import Orchestrator
from agent.tools import default_registry


console = Console()


def _build_orchestrator(config: AgentConfig, with_ui: bool) -> tuple[Orchestrator, Any]:
    """Build an orchestrator (+ optional UI) using shared dependencies."""
    registry = default_registry()
    ui = None
    if with_ui and config.ui.enabled:
        from agent.ui.terminal import TerminalUI

        ui = TerminalUI(
            cost_tracker=None,  # set after orchestrator creation
            token_budget=config.budget.session_token_limit,
            refresh_per_second=config.ui.refresh_per_second,
            console=console,
        )
    orch = Orchestrator(
        config=config,
        registry=registry,
        clarification_handler=_clarify_interactively,
        ui_hook=ui,
    )
    if ui is not None:
        ui.cost_tracker = orch.cost
    return orch, ui


def _clarify_interactively(questions: list[dict[str, str]]) -> list[dict[str, str]]:
    """Prompt the user on the terminal for planner-surfaced ambiguities."""
    answers: list[dict[str, str]] = []
    for q in questions:
        ans = click.prompt(q.get("question", ""), default="", show_default=False)
        answers.append({"id": q.get("id", ""), "question": q.get("question", ""), "answer": ans})
    return answers


@click.group(help="Agent Harness CLI.")
@click.option("--config", "config_file", type=click.Path(path_type=Path), default=None)
@click.pass_context
def main(ctx: click.Context, config_file: Path | None) -> None:
    """Top-level CLI group."""
    overrides: dict[str, Any] = {}
    if config_file:
        import tomllib

        overrides = tomllib.loads(config_file.read_text())
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(overrides=overrides)


@main.command(help='Run the agent on a new objective.')
@click.argument("objective", nargs=-1, required=True)
@click.option("--no-ui", is_flag=True, help="Disable the rich terminal UI.")
@click.pass_context
def run(ctx: click.Context, objective: tuple[str, ...], no_ui: bool) -> None:
    config: AgentConfig = ctx.obj["config"]
    obj_str = " ".join(objective)
    orch, ui = _build_orchestrator(config, with_ui=not no_ui)

    async def go() -> None:
        if ui is not None:
            with ui:
                session = await orch.run(obj_str)
        else:
            session = await orch.run(obj_str)
        console.print(f"[bold green]Session {session.id} → {session.outcome}[/bold green]")
        console.print(json.dumps(orch.cost.summary(), indent=2))

    asyncio.run(go())


@main.command(help="Resume a previous session by id.")
@click.argument("session_id")
@click.option("--no-ui", is_flag=True)
@click.pass_context
def resume(ctx: click.Context, session_id: str, no_ui: bool) -> None:
    config: AgentConfig = ctx.obj["config"]
    orch, ui = _build_orchestrator(config, with_ui=not no_ui)

    async def go() -> None:
        if ui is not None:
            with ui:
                session = await orch.resume(session_id)
        else:
            session = await orch.resume(session_id)
        console.print(f"[bold green]Session {session.id} → {session.outcome}[/bold green]")

    asyncio.run(go())


@main.command(help="List previously saved sessions.")
@click.pass_context
def history(ctx: click.Context) -> None:
    config: AgentConfig = ctx.obj["config"]
    orch = Orchestrator(config=config)
    rows = orch.list_sessions()
    if not rows:
        console.print("[grey50](no sessions found)[/grey50]")
        return
    tbl = Table(title="sessions")
    tbl.add_column("id")
    tbl.add_column("objective", overflow="fold")
    tbl.add_column("outcome")
    tbl.add_column("tasks")
    for r in rows:
        tbl.add_row(
            r.get("id", ""),
            r.get("objective", "")[:80],
            r.get("outcome", ""),
            str(len(r.get("tasks", {}))),
        )
    console.print(tbl)


@main.group(help="Tool management.")
def tools() -> None:
    """Tool subcommands."""


@tools.command("list", help="List registered tools and their schemas.")
def tools_list() -> None:
    reg = default_registry()
    tbl = Table(title="tools")
    tbl.add_column("name")
    tbl.add_column("side-effect")
    tbl.add_column("timeout")
    tbl.add_column("description", overflow="fold")
    for spec in reg.specs():
        tbl.add_row(spec.name, spec.side_effect, f"{spec.timeout:.0f}s", spec.description)
    console.print(tbl)


@main.group(help="Semantic memory subcommands.")
def memory() -> None:
    """Memory subcommands."""


@memory.command("search", help="Search the semantic memory store.")
@click.argument("query")
@click.option("--top", default=5)
@click.pass_context
def memory_search(ctx: click.Context, query: str, top: int) -> None:
    config: AgentConfig = ctx.obj["config"]
    state_dir = Path(config.memory.working_dir).resolve() / "semantic"
    sem = SemanticMemory(state_dir, collection=config.memory.semantic_collection)
    results = sem.retrieve_relevant_memories(query, top_k=top)
    if not results:
        console.print("[grey50](no results)[/grey50]")
        return
    tbl = Table(title=f"semantic search: {query!r}")
    tbl.add_column("score", justify="right")
    tbl.add_column("kind")
    tbl.add_column("content", overflow="fold")
    for m in results:
        tbl.add_row(f"{m.score:.3f}", m.kind, m.content[:200])
    console.print(tbl)


@main.command(help="Show cost and token breakdown for the most recent session.")
@click.pass_context
def cost(ctx: click.Context) -> None:
    config: AgentConfig = ctx.obj["config"]
    state_dir = Path(config.memory.working_dir).resolve() / "sessions"
    if not state_dir.exists():
        console.print("[grey50](no sessions yet)[/grey50]")
        return
    files = sorted(state_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        console.print("[grey50](no sessions yet)[/grey50]")
        return
    data = json.loads(files[0].read_text())
    summary = data.get("cost_summary") or {}
    console.print_json(json.dumps(summary, indent=2))


def cli_entry() -> None:
    """Console-script entry point."""
    main(obj={})


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(obj={}) or 0)
