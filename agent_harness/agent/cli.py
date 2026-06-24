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


def _friendly_llm_error(exc: BaseException) -> str | None:
    """Translate well-known LLM SDK errors into a one-line user-facing message."""
    name = type(exc).__name__
    msg = str(exc)
    if "insufficient_quota" in msg or "exceeded your current quota" in msg.lower():
        return (
            "OpenAI returned 'insufficient_quota'. Your account has no available credit. "
            "Add billing at https://platform.openai.com/account/billing — the API key is fine, "
            "the wallet is empty."
        )
    if name == "RateLimitError" or "rate limit" in msg.lower():
        return "Rate-limited by the provider. Wait a moment and retry, or pick a smaller model."
    if "model_not_found" in msg or "does not exist" in msg.lower() or "no such model" in msg.lower():
        return (
            f"Model id rejected by the provider: {msg.splitlines()[-1][:200]}. "
            "Set AGENT_LLM__MODEL to a real id (e.g. gpt-4o-mini, gpt-4o, gpt-4.1)."
        )
    if name in {"AuthenticationError", "PermissionDeniedError"} or "invalid_api_key" in msg.lower():
        return (
            "The API key was rejected. Re-check OPENAI_API_KEY / ANTHROPIC_API_KEY for typos "
            "and that the key has not been revoked."
        )
    return None


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


@main.command(help='Run the agent on a new objective. Supports @symbol and @path/to/file mentions.')
@click.argument("objective", nargs=-1, required=True)
@click.option("--no-ui", is_flag=True, help="Disable the rich terminal UI.")
@click.option("--persona", default=None, help="Persona name (default | strict | fast | security | minimal | architect).")
@click.pass_context
def run(ctx: click.Context, objective: tuple[str, ...], no_ui: bool, persona: str | None) -> None:
    config: AgentConfig = ctx.obj["config"]
    obj_str = " ".join(objective)

    # Resolve any @mentions in the objective via the indexer.
    try:
        from agent.tools.indexer import resolve_mentions

        resolved = resolve_mentions(obj_str)
    except Exception:
        resolved = []
    if resolved:
        ctx_block = "\n".join(
            f"- @{r['mention']}: {json.dumps({k: v for k, v in r.items() if k != 'mention'})[:400]}"
            for r in resolved
        )
        obj_str = f"{obj_str}\n\nMention context:\n{ctx_block}"

    if persona:
        from agent.prompts.manager import PromptManager

        p = PromptManager().persona(persona)
        config.llm.temperature = p.temperature
        config.llm.max_output_tokens = p.max_tokens
        config.persona.active = persona

    orch, ui = _build_orchestrator(config, with_ui=not no_ui)

    async def go() -> None:
        try:
            if ui is not None:
                with ui:
                    session = await orch.run(obj_str)
            else:
                session = await orch.run(obj_str)
        except Exception as exc:
            friendly = _friendly_llm_error(exc)
            if friendly:
                console.print(f"\n[bold red]✗ {friendly}[/bold red]")
                sys.exit(2)
            raise
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


@main.group(help="UI utilities.")
def ui() -> None:
    """UI subcommands."""


@ui.command("demo", help="Preview the tantalus UI with canned fake state (no API calls).")
@click.option("--seconds", default=20, type=int, help="How long to render the demo for.")
def ui_demo(seconds: int) -> None:
    """Render the live UI with synthetic data for ``seconds`` then exit."""
    import asyncio
    import random
    import time
    import uuid

    from agent.audit import AuditLog
    from agent.llm.base import CostTracker
    from agent.tools.multi_edit import EditOperation, EditPlan, render_plan_diff
    from agent.types import Complexity, Memory, TaskNode, TaskStatus, ToolCall, ToolResult
    from agent.ui.terminal import TerminalUI

    async def go() -> None:
        cost = CostTracker()
        cost._active_model = "gpt-4o-mini"
        audit_dir = Path(".agent_state") / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        audit = AuditLog(audit_dir / "demo.jsonl")
        ui_view = TerminalUI(
            cost_tracker=cost,
            audit=audit,
            token_budget=1_000_000,
            refresh_per_second=10,
            console=console,
        )
        ui_view.set_backend("docker")

        # Build a fake session that looks like a real plan in progress.
        class _Session:
            id = "demo123"
            objective = "Refactor @AuthService to use dependency injection and add JWT tests"
            tasks: dict[str, TaskNode] = {}
            order_hint: list[str] = []

        sess = _Session()
        tasks = [
            ("t1", "Read AuthService and map its dependencies", Complexity.S, [], TaskStatus.COMPLETE, "located 3 callers in api/, ui/, batch/"),
            ("t2", "Extract AuthService interface", Complexity.M, ["t1"], TaskStatus.COMPLETE, "created auth/interface.py"),
            ("t3", "Wire DI container in main.py", Complexity.M, ["t2"], TaskStatus.IN_PROGRESS, ""),
            ("t4", "Migrate JWT signing to RS256", Complexity.L, ["t2"], TaskStatus.PENDING, ""),
            ("t5", "Add pytest suite for auth.token", Complexity.M, ["t4"], TaskStatus.PENDING, ""),
            ("t6", "Update README + CHANGES.md", Complexity.S, ["t3", "t4", "t5"], TaskStatus.PENDING, ""),
            ("t7", "Run security audit", Complexity.S, ["t4"], TaskStatus.FAILED, "BLOCKED: missing trufflehog binary"),
        ]
        for tid, desc, comp, prereqs, status, summary in tasks:
            node = TaskNode(id=tid, description=desc, complexity=comp, prerequisites=list(prereqs))
            node.status = status
            node.summary = summary
            sess.tasks[tid] = node
            sess.order_hint.append(tid)
        ui_view.set_session(sess)

        # Seed the stream / memory / diff panels with believable content.
        for line in [
            "Thought: I'll start by inspecting the AuthService entry points.",
            "Action: file_search(pattern='class AuthService')",
            "Observation: 1 hit at auth/service.py:14",
            "Thought: Mapping callers via the codebase index.",
            "Action: index_callers(symbol='AuthService')",
            "Observation: 3 callers across api/, ui/, batch/",
        ]:
            ui_view.on_assistant_text(line)

        ui_view.on_memories([
            Memory(id="m1", kind="decision", content="Switched from PyJWT to authlib for RS256 support last week", metadata={}, score=0.91),
            Memory(id="m2", kind="error", content="Earlier session: trufflehog binary missing on macOS — install via brew", metadata={}, score=0.78),
            Memory(id="m3", kind="file", content="auth/service.py defines a 200-line god class — candidate for split", metadata={}, score=0.71),
        ])

        ui_view.on_swarm_update([
            {"role": "CODER", "status": "running", "task": "extracting AuthService interface", "tokens": 1820},
            {"role": "REVIEWER", "status": "waiting", "task": "queued after CODER", "tokens": 0},
            {"role": "TESTER", "status": "waiting", "task": "queued after CODER", "tokens": 0},
            {"role": "SECURITY_AUDITOR", "status": "failed", "task": "trufflehog missing", "tokens": 412},
        ])

        plan = EditPlan(
            description="DI container wiring",
            operations=[
                EditOperation(kind="overwrite", path="main.py", content="from auth.container import build_container\n\napp = build_container()\n"),
                EditOperation(kind="create", path="auth/container.py", content="def build_container():\n    return AuthService(JwtSigner())\n"),
            ],
        )
        ui_view.on_diff_preview(render_plan_diff(plan))

        for evt, payload in [
            ("session_start", {"objective": sess.objective}),
            ("plan", {"tasks": [t.id for t in sess.tasks.values()]}),
            ("task_start", {"id": "t1"}),
            ("tool_call", {"name": "file_search", "ok": True, "latency_ms": 42}),
            ("tool_call", {"name": "index_callers", "ok": True, "latency_ms": 18}),
            ("task_end", {"id": "t1", "status": "complete"}),
            ("review", {"task": "t2", "approve": True}),
        ]:
            audit.append(evt, payload)

        call = ToolCall(id=uuid.uuid4().hex, name="multi_edit", arguments={"description": "wire DI container", "ops": 2})
        ui_view.on_tool_start(call)

        ui_view.on_status("running tantalus demo · no API calls being made")

        with ui_view:
            start = time.time()
            i = 0
            from agent.llm.base import CostEntry

            while time.time() - start < seconds:
                # Mutate state a little each tick so the UI looks alive.
                i += 1
                cost.entries.append(CostEntry(
                    model="gpt-4o-mini",
                    prompt_tokens=200 + random.randint(0, 400),
                    completion_tokens=80 + random.randint(0, 120),
                    cost_usd=0.0008 + random.random() * 0.0015,
                ))

                if i == 5:
                    ui_view.on_tool_end(ToolResult(call_id=call.id, name=call.name, ok=True, output={"applied": True}, latency_ms=124.0))
                if i == 10:
                    sess.tasks["t3"].status = TaskStatus.COMPLETE
                    sess.tasks["t3"].summary = "DI container live"
                    sess.tasks["t4"].status = TaskStatus.IN_PROGRESS
                if i == 18:
                    sess.tasks["t4"].status = TaskStatus.COMPLETE
                    sess.tasks["t4"].summary = "RS256 signing wired"
                    sess.tasks["t5"].status = TaskStatus.IN_PROGRESS

                ui_view.on_assistant_text(
                    random.choice([
                        "Thought: planning next step",
                        "Action: code_lint(path='auth/service.py')",
                        "Observation: 0 issues",
                        "Thought: writing the test scaffold",
                        "Action: file_write(path='tests/test_auth.py')",
                        "Observation: 47 bytes written",
                    ])
                )
                audit.append("tool_call", {"name": "code_lint", "ok": True, "latency_ms": random.randint(10, 90)})
                ui_view.refresh_dag()
                await asyncio.sleep(0.5)

        console.print(f"[bold green]demo finished after {seconds}s — no API calls were made.[/bold green]")

    asyncio.run(go())


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


@memory.command("search", help="Search the memory tiers (semantic + episodic + procedural).")
@click.argument("query")
@click.option("--top", default=5)
@click.option("--tier", default="all", type=click.Choice(["all", "semantic", "episodic", "procedural"]))
@click.pass_context
def memory_search(ctx: click.Context, query: str, top: int, tier: str) -> None:
    config: AgentConfig = ctx.obj["config"]
    state_dir = Path(config.memory.working_dir).resolve()
    if config.memory.semantic_memory_scope == "global":
        sem_root = Path(config.memory.global_memory_dir).expanduser().resolve() / "semantic"
    else:
        sem_root = state_dir / "semantic"
    if tier in {"all", "semantic"}:
        sem = SemanticMemory(sem_root, collection=config.memory.semantic_collection)
        results = sem.retrieve_relevant_memories(query, top_k=top)
        if results:
            tbl = Table(title=f"semantic — {query!r}")
            tbl.add_column("score", justify="right")
            tbl.add_column("kind")
            tbl.add_column("content", overflow="fold")
            for m in results:
                tag = m.kind + (" ⚠stale" if m.metadata.get("stale") else "")
                tbl.add_row(f"{m.score:.3f}", tag, m.content[:200])
            console.print(tbl)
    if tier in {"all", "procedural"}:
        from agent.memory.procedural import ProceduralMemory

        global_root = Path(config.memory.global_memory_dir).expanduser().resolve()
        pm = ProceduralMemory(global_root / config.memory.procedural_db)
        proc = pm.retrieve_relevant_procedure(query, min_success_rate=0.0)
        if proc is not None:
            tbl = Table(title=f"procedural — {query!r}")
            tbl.add_column("rate", justify="right")
            tbl.add_column("uses", justify="right")
            tbl.add_column("description", overflow="fold")
            tbl.add_row(f"{proc.success_rate:.2f}", str(proc.times_used), proc.task_description[:160])
            console.print(tbl)
    if tier in {"all", "episodic"}:
        from agent.memory.episodic import EpisodicMemory

        ep = EpisodicMemory(state_dir / config.memory.episodic_db, session_id="search")
        hits = ep.search(query, limit=top)
        if hits:
            tbl = Table(title=f"episodic — {query!r}")
            tbl.add_column("ok")
            tbl.add_column("tool")
            tbl.add_column("args", overflow="fold")
            for h in hits:
                tbl.add_row("✓" if h.ok else "✗", h.tool, h.args_json[:160])
            console.print(tbl)
    # Backwards-compat: also print the raw semantic results in the legacy format
    sem = SemanticMemory(sem_root, collection=config.memory.semantic_collection)
    results = sem.retrieve_relevant_memories(query, top_k=top)
    if not results and tier == "all":
        console.print("[grey50](no semantic results)[/grey50]")
        return
    tbl = Table(title=f"semantic search: {query!r}")
    tbl.add_column("score", justify="right")
    tbl.add_column("kind")
    tbl.add_column("content", overflow="fold")
    for m in results:
        tbl.add_row(f"{m.score:.3f}", m.kind, m.content[:200])
    console.print(tbl)


@memory.command("export", help="Export memory tiers to a tar.gz archive.")
@click.argument("archive", type=click.Path())
@click.option("--scope", default="project", type=click.Choice(["project", "global"]))
@click.pass_context
def memory_export(ctx: click.Context, archive: str, scope: str) -> None:
    import tarfile

    config: AgentConfig = ctx.obj["config"]
    if scope == "global":
        root = Path(config.memory.global_memory_dir).expanduser().resolve()
    else:
        root = Path(config.memory.working_dir).resolve()
    if not root.exists():
        console.print(f"[red]nothing to export from {root}[/red]")
        return
    out = Path(archive)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as tar:
        tar.add(root, arcname=root.name)
    console.print(f"[green]wrote {out} ({out.stat().st_size:,} bytes)[/green]")


@memory.command("import", help="Import memory from a tar.gz archive into the local store.")
@click.argument("archive", type=click.Path(exists=True))
@click.option("--scope", default="project", type=click.Choice(["project", "global"]))
@click.pass_context
def memory_import(ctx: click.Context, archive: str, scope: str) -> None:
    import tarfile

    config: AgentConfig = ctx.obj["config"]
    if scope == "global":
        target = Path(config.memory.global_memory_dir).expanduser().resolve()
    else:
        target = Path(config.memory.working_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        # Safe-extract: refuse any entry whose path escapes the target.
        members = []
        for m in tar.getmembers():
            full = (target / m.name).resolve()
            try:
                full.relative_to(target)
            except ValueError:
                console.print(f"[red]skipping unsafe entry: {m.name}[/red]")
                continue
            members.append(m)
        tar.extractall(target, members=members)
    console.print(f"[green]imported into {target}[/green]")


@main.command(help="Show cost and token breakdown for the most recent session.")
@click.option("--dashboard", is_flag=True, help="Render the live cost dashboard.")
@click.pass_context
def cost(ctx: click.Context, dashboard: bool) -> None:
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
    if dashboard:
        from agent.observability.dashboard import render_cost_dashboard, historical_total
        from agent.observability.events import EventSink

        events_path = state_dir.parent / "events" / f"{data['id']}.jsonl"
        events = EventSink(events_path).read_all() if events_path.exists() else []
        render_cost_dashboard(events, console=console)
        hist = historical_total(state_dir.parent / "events")
        console.rule("[bold cyan]all-time")
        console.print(f"Total cost across sessions: [bold green]${hist['total_cost']:.4f}[/bold green]")
        console.print(f"Total tokens across sessions: [bold]{int(hist['total_tokens'])}[/bold]")
        return
    summary = data.get("cost_summary") or {}
    console.print_json(json.dumps(summary, indent=2))


@main.command(help="Replay a session by id from its event stream.")
@click.argument("session_id")
@click.option("--speed", default="1", help="Playback speed: 1, 2, 5, or 'instant'.")
@click.option("--report", is_flag=True, help="Write session_report.md and exit.")
@click.pass_context
def replay(ctx: click.Context, session_id: str, speed: str, report: bool) -> None:
    """Replay a previously recorded session."""
    from agent.observability.events import EventSink
    from agent.observability.replay import ReplayPlayer, render_session_report

    config: AgentConfig = ctx.obj["config"]
    events_path = Path(config.memory.working_dir).resolve() / "events" / f"{session_id}.jsonl"
    if not events_path.exists():
        console.print(f"[red]No event stream for session {session_id}[/red]")
        return
    events = EventSink(events_path).read_all()
    if report:
        md = render_session_report(events)
        out = Path("session_report.md")
        out.write_text(md)
        console.print(f"[green]wrote {out}[/green]")
        return
    sp = 0.0 if speed == "instant" else float(speed)
    player = ReplayPlayer(events, speed=sp if sp > 0 else 1.0, console=console)
    asyncio.run(player.play())


@main.group(help="GitHub / GitLab PR operations.")
def pr() -> None:
    """PR subcommands."""


@pr.command("review", help="Review a PR: fetch diff and ask the agent to surface findings.")
@click.argument("pr_number", type=int)
@click.option("--repo", default="", help="owner/repo (default $GITHUB_REPOSITORY).")
@click.pass_context
def pr_review(ctx: click.Context, pr_number: int, repo: str) -> None:
    config: AgentConfig = ctx.obj["config"]

    async def go() -> None:
        from agent.tools.vcs_hosting import _build_client, _default_repo

        client = _build_client()
        r = repo or _default_repo()
        diff = await client.get_pr_diff(r, pr_number)
        orch, _ui = _build_orchestrator(config, with_ui=False)
        session = await orch.run(
            f"Review PR #{pr_number} in {r}. Diff follows.\n\n{diff[:60_000]}"
        )
        console.print(f"[bold green]Review session {session.id} → {session.outcome}[/bold green]")

    asyncio.run(go())


@pr.command("fix", help="Fix unresolved review comments on a PR autonomously.")
@click.argument("pr_number", type=int)
@click.option("--repo", default="")
@click.pass_context
def pr_fix(ctx: click.Context, pr_number: int, repo: str) -> None:
    config: AgentConfig = ctx.obj["config"]

    async def go() -> None:
        from agent.tools.vcs_hosting import _build_client, _default_repo

        client = _build_client()
        r = repo or _default_repo()
        pr_obj = await client.get_pull_request(r, pr_number)
        diff = await client.get_pr_diff(r, pr_number)
        orch, _ui = _build_orchestrator(config, with_ui=False)
        session = await orch.run(
            f"Fix all review comments on PR #{pr_number} ({pr_obj.title}). "
            f"Branch: {pr_obj.head_ref}. Diff:\n\n{diff[:30_000]}"
        )
        console.print(f"[bold green]Fix session {session.id} → {session.outcome}[/bold green]")

    asyncio.run(go())


@main.command("issue", help="Fetch an issue and implement it end-to-end.")
@click.argument("issue_number", type=int)
@click.option("--repo", default="")
@click.pass_context
def issue_impl(ctx: click.Context, issue_number: int, repo: str) -> None:
    config: AgentConfig = ctx.obj["config"]

    async def go() -> None:
        from agent.tools.vcs_hosting import _build_client, _default_repo

        client = _build_client()
        r = repo or _default_repo()
        issue = await client.get_issue(r, issue_number)
        objective = (
            f"Implement issue #{issue.number}: {issue.title}\n\n{issue.body[:6000]}"
        )
        orch, _ui = _build_orchestrator(config, with_ui=False)
        session = await orch.run(objective)
        console.print(f"[bold green]Issue session {session.id} → {session.outcome}[/bold green]")

    asyncio.run(go())


@main.group(help="CI helpers.")
def ci() -> None:
    """CI subcommands."""


@ci.command("fix", help="Diagnose and autonomously fix failing CI checks on current branch.")
@click.option("--commit", default="HEAD", help="Commit sha to inspect.")
@click.option("--repo", default="")
@click.pass_context
def ci_fix(ctx: click.Context, commit: str, repo: str) -> None:
    config: AgentConfig = ctx.obj["config"]

    async def go() -> None:
        import subprocess

        from agent.tools.vcs_hosting import _build_client, _default_repo

        client = _build_client()
        r = repo or _default_repo()
        sha = commit
        if sha == "HEAD":
            try:
                sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            except Exception:
                pass
        checks = await client.get_ci_status(r, sha)
        failing = [c for c in checks if (c.conclusion or "").lower() in {"failure", "cancelled", "timed_out"}]
        if not failing:
            console.print("[green]No failing checks.[/green]")
            return
        details = "\n".join(f"- {c.name}: {c.details[:400]}" for c in failing)
        orch, _ui = _build_orchestrator(config, with_ui=False)
        await orch.run(f"Fix failing CI checks on commit {sha}:\n{details}")

    asyncio.run(go())


@main.group(help="Prompt management.")
def prompt_cmd() -> None:
    """Prompt subcommands."""


@prompt_cmd.command("eval", help="A/B-compare prompt variants on the same objective.")
@click.argument("objective", nargs=-1, required=True)
@click.option("--variants", default="default,strict", help="Comma-separated persona names.")
@click.pass_context
def prompt_eval(ctx: click.Context, objective: tuple[str, ...], variants: str) -> None:
    config: AgentConfig = ctx.obj["config"]
    obj_str = " ".join(objective)

    async def go() -> None:
        from agent.prompts.manager import PromptManager

        pm = PromptManager()
        names = [v.strip() for v in variants.split(",") if v.strip()]
        rows: list[dict[str, Any]] = []
        for name in names:
            persona = pm.persona(name)
            orch, _ui = _build_orchestrator(config, with_ui=False)
            orch.llm.cost_tracker = orch.cost
            orch.config.llm.temperature = persona.temperature
            orch.config.llm.max_output_tokens = persona.max_tokens
            session = await orch.run(obj_str)
            rows.append({
                "persona": name,
                "outcome": session.outcome,
                "cost_usd": orch.cost.total_usd,
                "prompt_tokens": orch.cost.total_prompt_tokens,
                "completion_tokens": orch.cost.total_completion_tokens,
                "tasks_complete": sum(1 for t in session.tasks.values() if t.status.value == "complete"),
            })

        tbl = Table(title="prompt eval results")
        for col in ("persona", "outcome", "cost_usd", "prompt_tokens", "completion_tokens", "tasks_complete"):
            tbl.add_column(col)
        for r in rows:
            tbl.add_row(
                r["persona"], r["outcome"], f"${r['cost_usd']:.4f}",
                str(r["prompt_tokens"]), str(r["completion_tokens"]), str(r["tasks_complete"]),
            )
        console.print(tbl)

    asyncio.run(go())


def cli_entry() -> None:
    """Console-script entry point."""
    main(obj={})


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(obj={}) or 0)
