"""Cost dashboard renderer."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.progress_bar import ProgressBar

from agent.observability.events import EventSink, SessionEvent


def aggregate(events: list[SessionEvent]) -> dict[str, Any]:
    """Aggregate raw events into per-model / per-task / per-tool buckets."""
    by_model: dict[str, dict[str, float]] = defaultdict(lambda: {"prompt": 0, "completion": 0, "cost_usd": 0.0, "calls": 0})
    by_task: dict[str, float] = defaultdict(float)
    by_tool: dict[str, int] = defaultdict(int)
    timeline: list[tuple[str, float]] = []
    current_task = "?"
    for evt in events:
        if evt.type == "task_started":
            current_task = str(evt.payload.get("task_id", "?"))
        elif evt.type == "llm_call":
            model = str(evt.payload.get("model", "?"))
            tokens = int(evt.payload.get("prompt_tokens", 0)) + int(evt.payload.get("completion_tokens", 0))
            cost = float(evt.payload.get("cost_usd", 0))
            by_model[model]["prompt"] += int(evt.payload.get("prompt_tokens", 0))
            by_model[model]["completion"] += int(evt.payload.get("completion_tokens", 0))
            by_model[model]["cost_usd"] += cost
            by_model[model]["calls"] += 1
            by_task[current_task] += tokens
            timeline.append((current_task, float(tokens)))
        elif evt.type == "tool_called":
            by_tool[str(evt.payload.get("tool_name", "?"))] += 1
    total_cost = sum(b["cost_usd"] for b in by_model.values())
    total_tokens = sum(b["prompt"] + b["completion"] for b in by_model.values())
    return {
        "total_cost": total_cost,
        "total_tokens": total_tokens,
        "by_model": dict(by_model),
        "by_task": dict(by_task),
        "by_tool": dict(by_tool),
        "timeline": timeline,
    }


def render_cost_dashboard(events: list[SessionEvent], console: Console | None = None) -> None:
    """Render an aggregate breakdown to the console."""
    console = console or Console()
    agg = aggregate(events)
    console.rule("[bold cyan]cost dashboard")
    console.print(f"Total tokens: [bold]{agg['total_tokens']}[/bold]   Total cost: [bold green]${agg['total_cost']:.4f}[/bold green]")

    tbl_model = Table(title="by model")
    tbl_model.add_column("model")
    tbl_model.add_column("calls", justify="right")
    tbl_model.add_column("prompt", justify="right")
    tbl_model.add_column("completion", justify="right")
    tbl_model.add_column("cost (USD)", justify="right")
    for model, info in agg["by_model"].items():
        tbl_model.add_row(
            model,
            str(int(info["calls"])),
            str(int(info["prompt"])),
            str(int(info["completion"])),
            f"${info['cost_usd']:.4f}",
        )
    console.print(tbl_model)

    tbl_task = Table(title="by task (tokens)")
    tbl_task.add_column("task")
    tbl_task.add_column("tokens", justify="right")
    tbl_task.add_column("share")
    total_task_tokens = sum(agg["by_task"].values()) or 1
    for task, tokens in sorted(agg["by_task"].items(), key=lambda kv: kv[1], reverse=True):
        share_pct = 100 * tokens / total_task_tokens
        tbl_task.add_row(task, f"{int(tokens)}", f"{share_pct:.1f}%")
    console.print(tbl_task)

    tbl_tool = Table(title="by tool (calls)")
    tbl_tool.add_column("tool")
    tbl_tool.add_column("count", justify="right")
    for tool, count in sorted(agg["by_tool"].items(), key=lambda kv: kv[1], reverse=True):
        tbl_tool.add_row(tool, str(count))
    console.print(tbl_tool)

    if agg["timeline"]:
        console.rule("[bold cyan]timeline (tokens per llm call)")
        max_tokens = max(t for _, t in agg["timeline"]) or 1
        for i, (task, tokens) in enumerate(agg["timeline"][-30:], start=1):
            bar = ProgressBar(total=max_tokens, completed=tokens, width=40)
            console.print(f"{i:>3} {task[:16]:<16}", bar, f" {int(tokens)}")


def list_session_files(state_dir: Path | str) -> list[Path]:
    """Return every ``*.jsonl`` event file under the given state directory."""
    base = Path(state_dir)
    if not base.exists():
        return []
    return sorted(base.rglob("*.jsonl"))


def historical_total(state_dir: Path | str) -> dict[str, float]:
    """Sum cost + tokens across every historical session events file."""
    total_cost = 0.0
    total_tokens = 0
    for path in list_session_files(state_dir):
        events = EventSink(path).read_all()
        agg = aggregate(events)
        total_cost += agg["total_cost"]
        total_tokens += agg["total_tokens"]
    return {"total_cost": total_cost, "total_tokens": float(total_tokens)}
