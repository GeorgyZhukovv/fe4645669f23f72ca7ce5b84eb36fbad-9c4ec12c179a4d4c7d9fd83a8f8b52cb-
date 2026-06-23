"""Rich `Live` terminal UI with header, DAG, agent stream, tool, memory, log panels."""

from __future__ import annotations

import time
from collections import deque
from typing import TYPE_CHECKING, Any

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from agent.audit import AuditLog
from agent.llm.base import CostTracker
from agent.types import Memory, TaskStatus, ToolCall, ToolResult

if TYPE_CHECKING:
    from agent.orchestrator import Session


STATUS_COLOR = {
    TaskStatus.PENDING: ("○", "grey50"),
    TaskStatus.READY: ("◐", "yellow"),
    TaskStatus.IN_PROGRESS: ("◑", "cyan"),
    TaskStatus.COMPLETE: ("●", "green"),
    TaskStatus.FAILED: ("✗", "red"),
    TaskStatus.SKIPPED: ("⊘", "grey37"),
}


class TerminalUI:
    """Live multi-panel terminal renderer for the running agent."""

    def __init__(
        self,
        cost_tracker: CostTracker,
        audit: AuditLog | None = None,
        token_budget: int = 1_000_000,
        refresh_per_second: int = 10,
        console: Console | None = None,
    ) -> None:
        """Construct the UI.

        Args:
            cost_tracker: Shared :class:`CostTracker` used to render the cost bar.
            audit: Optional audit log to tail in the log panel.
            token_budget: Session token budget for the header progress bar.
            refresh_per_second: Live refresh rate.
            console: Optional :class:`rich.console.Console` (one is created otherwise).
        """
        self.console = console or Console()
        self.cost_tracker = cost_tracker
        self.audit = audit
        self.token_budget = token_budget
        self.refresh_per_second = refresh_per_second
        self.session: Session | None = None
        self.started_at = time.time()
        self.agent_stream: deque[str] = deque(maxlen=12)
        self.recent_tool: tuple[ToolCall, ToolResult | None] | None = None
        self.recent_memories: list[Memory] = []
        self.live: Live | None = None

    # -- hooks -------------------------------------------------------------

    def set_session(self, session: "Session") -> None:
        self.session = session

    def on_tool_start(self, call: ToolCall) -> None:
        self.recent_tool = (call, None)

    def on_tool_end(self, result: ToolResult) -> None:
        if self.recent_tool is not None:
            self.recent_tool = (self.recent_tool[0], result)

    def on_assistant_text(self, text: str) -> None:
        for line in text.splitlines():
            self.agent_stream.append(line)

    def on_memories(self, memories: list[Memory]) -> None:
        self.recent_memories = memories[:3]

    def refresh_dag(self) -> None:
        if self.live is not None:
            self.live.update(self._layout(), refresh=True)

    # -- layout ------------------------------------------------------------

    def _layout(self) -> Layout:
        layout = Layout(name="root")
        layout.split_column(
            Layout(self._render_header(), name="header", size=4),
            Layout(name="body"),
            Layout(self._render_log(), name="log", size=10),
        )
        body = layout["body"]
        body.split_row(
            Layout(self._render_dag(), name="dag", ratio=1),
            Layout(name="right", ratio=2),
        )
        right = body["right"]
        right.split_column(
            Layout(self._render_agent_stream(), name="stream", ratio=2),
            Layout(self._render_tool_panel(), name="tool", size=8),
            Layout(self._render_memory_panel(), name="memory", size=8),
        )
        return layout

    def _render_header(self) -> Panel:
        obj = self.session.objective if self.session else "(no objective)"
        elapsed = int(time.time() - self.started_at)
        cost = self.cost_tracker.total_usd
        used = self.cost_tracker.total_prompt_tokens + self.cost_tracker.total_completion_tokens
        bar = ProgressBar(total=max(1, self.token_budget), completed=min(used, self.token_budget), width=30)
        line1 = Text(f"Objective: {obj}", style="bold cyan")
        line2 = Text(
            f"Elapsed: {elapsed}s   Tokens: {used}/{self.token_budget}   Cost: ${cost:.4f}",
            style="white",
        )
        return Panel(Group(line1, line2, bar), border_style="cyan", title="agent harness")

    def _render_dag(self) -> Panel:
        if self.session is None or not self.session.tasks:
            return Panel(Text("(no plan yet)", style="grey50"), title="task DAG", border_style="magenta")
        tree = Tree("[bold]Plan[/bold]", guide_style="grey37")
        children: dict[str, list[str]] = {tid: [] for tid in self.session.tasks}
        roots: list[str] = []
        for tid in self.session.order_hint:
            t = self.session.tasks.get(tid)
            if t is None:
                continue
            if not t.prerequisites:
                roots.append(tid)
            else:
                for p in t.prerequisites:
                    if p in children:
                        children[p].append(tid)

        def render(tid: str, node: Tree) -> None:
            t = self.session.tasks[tid]
            glyph, color = STATUS_COLOR[t.status]
            label = Text(f"{glyph} {tid}: {t.description[:60]}", style=color)
            child = node.add(label)
            for c in children.get(tid, []):
                render(c, child)

        for r in roots:
            render(r, tree)
        return Panel(tree, title="task DAG", border_style="magenta")

    def _render_agent_stream(self) -> Panel:
        body = Text()
        for line in self.agent_stream:
            style = "white"
            lower = line.lstrip().lower()
            if lower.startswith("thought") or lower.startswith("plan"):
                style = "cyan"
            elif lower.startswith("action") or lower.startswith("tool"):
                style = "yellow"
            elif lower.startswith("observation"):
                style = "green"
            body.append(line + "\n", style=style)
        return Panel(body, title="agent stream", border_style="cyan")

    def _render_tool_panel(self) -> Panel:
        if self.recent_tool is None:
            body = Text("(no tool calls yet)", style="grey50")
        else:
            call, result = self.recent_tool
            tbl = Table.grid(padding=(0, 1))
            tbl.add_row(Text("tool", style="bold"), Text(call.name, style="yellow"))
            args_preview = str(call.arguments)
            if len(args_preview) > 120:
                args_preview = args_preview[:117] + "..."
            tbl.add_row(Text("args", style="bold"), Text(args_preview, style="white"))
            if result is None:
                tbl.add_row(Text("status", style="bold"), Text("running…", style="cyan"))
            else:
                status_style = "green" if result.ok else "red"
                tbl.add_row(Text("status", style="bold"), Text("OK" if result.ok else "FAIL", style=status_style))
                tbl.add_row(Text("latency", style="bold"), Text(f"{result.latency_ms:.0f}ms"))
            body = tbl
        return Panel(body, title="tool execution", border_style="yellow")

    def _render_memory_panel(self) -> Panel:
        if not self.recent_memories:
            body: Any = Text("(no memories retrieved)", style="grey50")
        else:
            tbl = Table.grid(padding=(0, 1))
            for m in self.recent_memories:
                preview = m.content[:120].replace("\n", " ")
                tbl.add_row(Text(f"[{m.kind}]", style="magenta"), Text(preview))
            body = tbl
        return Panel(body, title="semantic memory", border_style="magenta")

    def _render_log(self) -> Panel:
        events = self.audit.tail(8) if self.audit else []
        if not events:
            body: Any = Text("(audit log empty)", style="grey50")
        else:
            tbl = Table.grid(padding=(0, 1))
            for evt in events:
                ts = evt.get("ts", "")[11:19]
                name = evt.get("event", "")
                payload = str(evt.get("payload", ""))[:120]
                tbl.add_row(Text(ts, style="grey50"), Text(name, style="bold cyan"), Text(payload))
            body = tbl
        return Panel(body, title="audit log", border_style="grey50")

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> "TerminalUI":
        self.live = Live(
            self._layout(),
            console=self.console,
            refresh_per_second=self.refresh_per_second,
            screen=False,
        )
        self.live.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.live is not None:
            self.live.__exit__(exc_type, exc, tb)
            self.live = None
