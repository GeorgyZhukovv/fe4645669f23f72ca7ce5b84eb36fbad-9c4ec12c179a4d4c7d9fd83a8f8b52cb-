"""Tantalus — the rich live terminal UI for the agent harness.

Layout:

    ┌──────── tantalus banner + session header (objective, elapsed, tokens, cost) ────────┐
    │                                                                                     │
    │  ┌── task DAG ────┐ ┌── agent stream (Thought / Action / Observation) ──────────┐   │
    │  │                │ │                                                            │   │
    │  ├── swarm ───────┤ ├── tool execution ──────────────────────────────────────────┤   │
    │  │                │ │                                                            │   │
    │  ├── memory ──────┤ ├── diff preview ────────────────────────────────────────────┤   │
    │  │                │ │                                                            │   │
    │  └────────────────┘ └────────────────────────────────────────────────────────────┘   │
    │  ┌── audit log ─────────────────────────────────────────────────────────────────┐   │
    │  └──────────────────────────────────────────────────────────────────────────────┘   │
    │  > steering input                                                                   │
    └─────────────────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import time
from collections import deque
from typing import TYPE_CHECKING, Any

from rich.align import Align
from rich.box import HEAVY, ROUNDED
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from agent.audit import AuditLog
from agent.llm.base import CostTracker
from agent.types import Memory, TaskStatus, ToolCall, ToolResult

if TYPE_CHECKING:
    from agent.orchestrator import Session


STATUS_GLYPH = {
    TaskStatus.PENDING: ("○", "grey50"),
    TaskStatus.READY: ("◐", "yellow"),
    TaskStatus.IN_PROGRESS: ("◑", "bright_cyan"),
    TaskStatus.COMPLETE: ("●", "bright_green"),
    TaskStatus.FAILED: ("✗", "bright_red"),
    TaskStatus.SKIPPED: ("⊘", "grey37"),
}


ROLE_COLOR = {
    "PLANNER": "bright_cyan",
    "CODER": "bright_blue",
    "REVIEWER": "bright_magenta",
    "TESTER": "bright_green",
    "DOCUMENTER": "yellow",
    "SECURITY_AUDITOR": "bright_red",
    "EXECUTOR": "white",
}


# ASCII banner — one row per character, joined per column for a compact glyph row.
_BANNER_GLYPHS = {
    "T": ["████████", "   ██   ", "   ██   ", "   ██   ", "   ██   "],
    "A": ["  ████  ", " ██  ██ ", "████████", "██    ██", "██    ██"],
    "N": ["██    ██", "███   ██", "██ ██ ██", "██   ███", "██    ██"],
    "L": ["██      ", "██      ", "██      ", "██      ", "████████"],
    "U": ["██    ██", "██    ██", "██    ██", "██    ██", " ██████ "],
    "S": [" ██████ ", "██      ", " ██████ ", "      ██", "███████ "],
    " ": ["   ", "   ", "   ", "   ", "   "],
}


def _render_banner(text: str = "TANTALUS") -> Text:
    """Render the TANTALUS wordmark in a 5-row block ASCII font with a gradient."""
    rows = ["", "", "", "", ""]
    for ch in text:
        glyph = _BANNER_GLYPHS.get(ch.upper(), _BANNER_GLYPHS[" "])
        for i in range(5):
            rows[i] += glyph[i] + " "
    palette = ["#5fafff", "#5fd7ff", "#87ffff", "#afffff", "#d7ffff"]
    out = Text(no_wrap=True, overflow="crop")
    for i, row in enumerate(rows):
        out.append(row + "\n", style=f"bold {palette[i % len(palette)]}")
    return out


class TerminalUI:
    """Tantalus: multi-panel terminal renderer for the running agent."""

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
        self.agent_stream: deque[str] = deque(maxlen=16)
        self.recent_tool: tuple[ToolCall, ToolResult | None] | None = None
        self.recent_memories: list[Memory] = []
        self.token_history: deque[int] = deque(maxlen=60)
        self.swarm_rows: list[dict[str, Any]] = []
        self.diff_preview: str = ""
        self.input_buffer: str = ""
        self.status_line: str = ""
        self.live: Live | None = None
        self.backend_label: str = "subprocess"

    # -- hooks (called by AgentLoop / Orchestrator) ----------------------

    def set_session(self, session: "Session") -> None:
        self.session = session

    def set_backend(self, name: str) -> None:
        self.backend_label = name

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

    def on_swarm_update(self, rows: list[dict[str, Any]]) -> None:
        self.swarm_rows = rows

    def on_diff_preview(self, diff: str) -> None:
        self.diff_preview = diff[:6000]

    def on_status(self, msg: str) -> None:
        self.status_line = msg

    def on_input(self, buffer: str) -> None:
        self.input_buffer = buffer

    def refresh_dag(self) -> None:
        used = self.cost_tracker.total_prompt_tokens + self.cost_tracker.total_completion_tokens
        self.token_history.append(used)
        if self.live is not None:
            self.live.update(self._layout(), refresh=True)

    # -- layout ----------------------------------------------------------

    def _layout(self) -> Layout:
        layout = Layout(name="root")
        layout.split_column(
            Layout(self._render_header(), name="header", size=10),
            Layout(name="body"),
            Layout(self._render_log(), name="log", size=8),
            Layout(self._render_input(), name="input", size=3),
        )
        body = layout["body"]
        body.split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=2),
        )
        left = body["left"]
        left.split_column(
            Layout(self._render_dag(), name="dag", ratio=2),
            Layout(self._render_swarm(), name="swarm", ratio=1),
            Layout(self._render_memory(), name="memory", ratio=1),
        )
        right = body["right"]
        right.split_column(
            Layout(self._render_stream(), name="stream", ratio=2),
            Layout(self._render_tool(), name="tool", size=8),
            Layout(self._render_diff(), name="diff", ratio=1),
        )
        return layout

    def _render_header(self) -> Panel:
        obj = self.session.objective[:140] if self.session else "(no objective)"
        elapsed = int(time.time() - self.started_at)
        cost = self.cost_tracker.total_usd
        used = self.cost_tracker.total_prompt_tokens + self.cost_tracker.total_completion_tokens

        banner = _render_banner()
        tagline = Text(
            "autonomous coding harness  ·  sandbox=", style="grey50"
        )
        tagline.append(self.backend_label, style="bright_yellow")
        tagline.append("  ·  model=", style="grey50")
        if self.session:
            tagline.append(getattr(self.cost_tracker, "_active_model", "?") or "?", style="bright_magenta")

        obj_line = Text("OBJECTIVE  ", style="bold grey50")
        obj_line.append(obj, style="bold white")

        meta = Text("ELAPSED ", style="grey50")
        meta.append(f"{elapsed:>5}s   ", style="bold")
        meta.append("TOKENS ", style="grey50")
        meta.append(f"{used:>7}/{self.token_budget}   ", style="bold")
        meta.append("COST ", style="grey50")
        meta.append(f"${cost:.4f}", style="bold green")
        if self.status_line:
            meta.append("   · " + self.status_line, style="yellow")

        bar = ProgressBar(total=max(1, self.token_budget), completed=min(used, self.token_budget), width=60, complete_style="bright_cyan")
        spark = _sparkline(list(self.token_history))
        spark_line = Text("Δ tokens  ", style="grey50")
        spark_line.append(spark, style="bright_cyan")

        return Panel(
            Group(
                Align.center(banner),
                Align.center(tagline),
                Rule(style="grey37"),
                obj_line,
                meta,
                bar,
                spark_line,
            ),
            border_style="bright_cyan",
            box=HEAVY,
            title="[bold bright_cyan] tantalus [/bold bright_cyan]",
        )

    def _render_dag(self) -> Panel:
        if self.session is None or not self.session.tasks:
            return Panel(Text("(no plan yet)", style="grey50"), title="task DAG", border_style="magenta", box=ROUNDED)
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
            glyph, color = STATUS_GLYPH[t.status]
            label = Text(f"{glyph} ", style=color)
            label.append(f"{tid}", style="bold")
            label.append(f"  {t.description[:60]}", style="white")
            label.append(f"  [{t.complexity.value}]", style="grey50")
            child = node.add(label)
            for c in children.get(tid, []):
                render(c, child)

        for r in roots:
            render(r, tree)
        return Panel(tree, title="task DAG", border_style="magenta", box=ROUNDED)

    def _render_swarm(self) -> Panel:
        if not self.swarm_rows:
            body: Any = Text("(no sub-agents)", style="grey50")
        else:
            tbl = Table.grid(padding=(0, 1))
            tbl.add_column()
            tbl.add_column()
            tbl.add_column(overflow="fold")
            for row in self.swarm_rows:
                role = str(row.get("role", "?"))
                color = ROLE_COLOR.get(role, "white")
                tbl.add_row(
                    Text(f"[{role}]", style=f"bold {color}"),
                    Text(str(row.get("status", "?")), style="grey50"),
                    Text(str(row.get("task", row.get("last_message", "")))[:80]),
                )
            body = tbl
        return Panel(body, title="swarm", border_style="bright_blue", box=ROUNDED)

    def _render_stream(self) -> Panel:
        body = Text()
        for line in self.agent_stream:
            style = "white"
            lower = line.lstrip().lower()
            if lower.startswith(("thought", "plan")):
                style = "bright_cyan"
                body.append("▶ ", style="cyan")
            elif lower.startswith(("action", "tool")):
                style = "yellow"
                body.append("⚙ ", style="yellow")
            elif lower.startswith("observation"):
                style = "green"
                body.append("◇ ", style="green")
            else:
                body.append("  ")
            body.append(line + "\n", style=style)
        return Panel(body, title="agent stream", border_style="bright_cyan", box=ROUNDED)

    def _render_tool(self) -> Panel:
        if self.recent_tool is None:
            body: Any = Text("(no tool calls yet)", style="grey50")
        else:
            call, result = self.recent_tool
            tbl = Table.grid(padding=(0, 1))
            tbl.add_column(width=10)
            tbl.add_column(overflow="fold")
            tbl.add_row(Text("tool", style="bold grey50"), Text(call.name, style="bold yellow"))
            args = str(call.arguments)
            if len(args) > 200:
                args = args[:197] + "..."
            tbl.add_row(Text("args", style="bold grey50"), Text(args))
            if result is None:
                tbl.add_row(Text("status", style="bold grey50"), Text("◌ running…", style="bright_cyan"))
            else:
                ok_style = "bright_green" if result.ok else "bright_red"
                glyph = "✓" if result.ok else "✗"
                tbl.add_row(Text("status", style="bold grey50"), Text(f"{glyph} {'OK' if result.ok else 'FAIL'}", style=ok_style))
                tbl.add_row(Text("latency", style="bold grey50"), Text(f"{result.latency_ms:.0f} ms"))
            body = tbl
        return Panel(body, title="tool execution", border_style="yellow", box=ROUNDED)

    def _render_memory(self) -> Panel:
        if not self.recent_memories:
            body: Any = Text("(no memories retrieved)", style="grey50")
        else:
            tbl = Table.grid(padding=(0, 1))
            tbl.add_column(width=12)
            tbl.add_column(overflow="fold")
            for m in self.recent_memories:
                preview = m.content[:140].replace("\n", " ")
                tbl.add_row(Text(f"[{m.kind}]", style="bright_magenta"), Text(preview))
            body = tbl
        return Panel(body, title="semantic memory", border_style="bright_magenta", box=ROUNDED)

    def _render_diff(self) -> Panel:
        if not self.diff_preview:
            body: Any = Text("(no pending edits)", style="grey50")
        else:
            colored = Text()
            for line in self.diff_preview.splitlines():
                if line.startswith("+++") or line.startswith("---"):
                    colored.append(line + "\n", style="bold")
                elif line.startswith("+"):
                    colored.append(line + "\n", style="green")
                elif line.startswith("-"):
                    colored.append(line + "\n", style="red")
                elif line.startswith("@@"):
                    colored.append(line + "\n", style="cyan")
                else:
                    colored.append(line + "\n", style="grey70")
            body = colored
        return Panel(body, title="diff preview", border_style="green", box=ROUNDED)

    def _render_log(self) -> Panel:
        events = self.audit.tail(8) if self.audit else []
        if not events:
            body: Any = Text("(audit log empty)", style="grey50")
        else:
            tbl = Table.grid(padding=(0, 1))
            tbl.add_column(width=8)
            tbl.add_column(width=18)
            tbl.add_column(overflow="fold")
            for evt in events:
                ts = (evt.get("ts") or "")[11:19]
                name = evt.get("event", "")
                payload = str(evt.get("payload", ""))[:150]
                tbl.add_row(
                    Text(ts, style="grey50"),
                    Text(name, style="bold bright_cyan"),
                    Text(payload, style="grey70"),
                )
            body = tbl
        return Panel(body, title="audit log", border_style="grey50", box=ROUNDED)

    def _render_input(self) -> Panel:
        prompt = Text("› ", style="bold bright_cyan")
        prompt.append(self.input_buffer or "type /help for steering commands", style="grey50" if not self.input_buffer else "white")
        return Panel(prompt, border_style="bright_cyan", box=ROUNDED, height=3)

    # -- lifecycle -------------------------------------------------------

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


_SPARK_CHARS = "▁▂▃▄▅▆▇█"


def _sparkline(values: list[int]) -> str:
    """Return a unicode sparkline summarising a numeric series."""
    if not values:
        return ""
    lo, hi = min(values), max(values)
    if hi == lo:
        return _SPARK_CHARS[0] * len(values)
    step = (hi - lo) / (len(_SPARK_CHARS) - 1)
    return "".join(_SPARK_CHARS[int((v - lo) / step)] for v in values)
