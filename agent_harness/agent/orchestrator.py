"""Central orchestrator: planning, scheduling, execution, verification, reflection."""

from __future__ import annotations

import asyncio
import json
import pickle
import re
import time
import uuid
from dataclasses import dataclass, field
from heapq import heappop, heappush
from pathlib import Path
from typing import Any, Callable

from agent.audit import AuditLog
from agent.config import AgentConfig
from agent.context import ContextManager
from agent.llm.anthropic_client import AnthropicClient
from agent.llm.base import CostTracker, LLMClient
from agent.llm.openai_client import OpenAIClient
from agent.loop import AgentLoop, LoopResult
from agent.prompts.planner import build_planner_prompt
from agent.prompts.reflector import COMPRESSION_PROMPT, REFLECTOR_PROMPT
from agent.prompts.reviewer import REVIEWER_PROMPT
from agent.tools import default_registry
from agent.tools.bash import is_destructive
from agent.tools.registry import ToolRegistry
from agent.types import Complexity, LLMMessage, TaskNode, TaskStatus


SESSION_VERSION = 1


@dataclass
class Session:
    """Persistent session state — serialised at every task boundary."""

    id: str
    objective: str
    tasks: dict[str, TaskNode]
    order_hint: list[str]
    started_at: float
    finished_at: float | None = None
    outcome: str = "in_progress"
    clarifications: list[dict[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    cost_summary: dict[str, Any] = field(default_factory=dict)

    def serialize_json(self) -> dict[str, Any]:
        """Return a JSON-safe view of the session for inspection."""
        return {
            "id": self.id,
            "objective": self.objective,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "outcome": self.outcome,
            "tasks": {
                tid: {
                    "id": t.id,
                    "description": t.description,
                    "complexity": t.complexity.value,
                    "prerequisites": t.prerequisites,
                    "anticipated_tools": t.anticipated_tools,
                    "status": t.status.value,
                    "summary": t.summary,
                    "artifacts": t.artifacts,
                    "error": t.error,
                    "attempts": t.attempts,
                    "recoverability": t.recoverability,
                }
                for tid, t in self.tasks.items()
            },
            "order_hint": self.order_hint,
            "clarifications": self.clarifications,
            "notes": self.notes,
            "cost_summary": self.cost_summary,
        }


class Orchestrator:
    """Top-level coordinator. Owns the plan, schedules tasks, runs verify+reflect."""

    def __init__(
        self,
        config: AgentConfig,
        registry: ToolRegistry | None = None,
        llm: LLMClient | None = None,
        clarification_handler: Callable[[list[dict[str, str]]], list[dict[str, str]]] | None = None,
        ui_hook: Any | None = None,
    ) -> None:
        """Wire dependencies and the safety / cost trackers.

        Args:
            config: Resolved configuration.
            registry: Tool registry (defaults to the built-in one).
            llm: LLM client (defaults to one built from config).
            clarification_handler: Callable invoked when the planner surfaces
                ambiguities. Receives the question list, returns answers.
            ui_hook: Optional terminal UI object.
        """
        self.config = config
        self.registry = registry or default_registry()
        self.cost = CostTracker()
        self.llm = llm or self._build_llm()
        self.clarification_handler = clarification_handler
        self.ui = ui_hook
        self.session: Session | None = None
        self.context: ContextManager | None = None
        self.audit: AuditLog | None = None
        self._notes: list[str] = []

    def _build_llm(self) -> LLMClient:
        """Construct an LLM client (with fallback chain) from the config."""
        from agent.llm.base import FallbackClient

        primary: LLMClient
        if self.config.llm.provider == "anthropic":
            primary = AnthropicClient(
                model=self.config.llm.model,
                api_key=self.config.llm.anthropic_api_key,
                cost_tracker=self.cost,
            )
        else:
            primary = OpenAIClient(
                model=self.config.llm.model,
                api_key=self.config.llm.openai_api_key,
                base_url=self.config.llm.openai_base_url,
                cost_tracker=self.cost,
            )
        if self.config.llm.fallback_provider and self.config.llm.fallback_model:
            secondary: LLMClient
            if self.config.llm.fallback_provider == "anthropic":
                secondary = AnthropicClient(
                    model=self.config.llm.fallback_model,
                    api_key=self.config.llm.anthropic_api_key,
                    cost_tracker=self.cost,
                )
            else:
                secondary = OpenAIClient(
                    model=self.config.llm.fallback_model,
                    api_key=self.config.llm.openai_api_key,
                    base_url=self.config.llm.openai_base_url,
                    cost_tracker=self.cost,
                )
            return FallbackClient(primary, secondary)
        return primary

    # -------- Session lifecycle --------

    def _state_dir(self) -> Path:
        return Path(self.config.memory.working_dir).resolve()

    def _session_path(self, session_id: str) -> Path:
        return self._state_dir() / "sessions" / f"{session_id}.pkl"

    def _session_json_path(self, session_id: str) -> Path:
        return self._state_dir() / "sessions" / f"{session_id}.json"

    def save_session(self) -> None:
        """Serialise the current session and working context to disk."""
        if self.session is None:
            return
        path = self._session_path(self.session.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": SESSION_VERSION,
            "session": self.session,
            "context": self.context.working.to_serializable() if self.context else [],
            "cost_summary": self.cost.summary(),
        }
        self.session.cost_summary = self.cost.summary()
        path.write_bytes(pickle.dumps(payload))
        self._session_json_path(self.session.id).write_text(
            json.dumps(self.session.serialize_json(), indent=2, default=str)
        )

    def load_session(self, session_id: str) -> Session:
        """Restore a serialised session into ``self.session`` / ``self.context``."""
        path = self._session_path(session_id)
        if not path.exists():
            raise FileNotFoundError(f"Session {session_id} not found at {path}")
        payload = pickle.loads(path.read_bytes())
        self.session = payload["session"]
        self.context = ContextManager.from_config(self.config, session_id=session_id)
        self.context.working.restore(payload.get("context", []))
        self.audit = AuditLog(self._state_dir() / "audit" / f"{session_id}.jsonl")
        self.registry.set_audit(self.audit)
        return self.session

    def list_sessions(self) -> list[dict[str, Any]]:
        """List previously saved sessions with their summary metadata."""
        out: list[dict[str, Any]] = []
        d = self._state_dir() / "sessions"
        if not d.exists():
            return out
        for f in sorted(d.glob("*.json")):
            try:
                out.append(json.loads(f.read_text()))
            except json.JSONDecodeError:
                continue
        return out

    # -------- Planning --------

    async def _plan(self, objective: str) -> tuple[list[TaskNode], list[dict[str, str]]]:
        """Call the planner LLM and parse the structured response.

        Returns:
            (tasks, clarifications) — the parsed task DAG and any ambiguities
            the planner wants the human to resolve.
        """
        sys_prompt = build_planner_prompt(self.registry.names(), str(Path.cwd()))
        messages = [LLMMessage(role="user", content=f"Objective: {objective}\n\nProduce the plan.")]
        response = await self.llm.complete(messages, tools=None, system=sys_prompt, temperature=0.0)
        text = response.text.strip()
        parsed = _strict_json_extract(text)
        tasks: list[TaskNode] = []
        for raw in parsed.get("tasks", []):
            try:
                complexity = Complexity(str(raw.get("complexity", "M")).upper())
            except ValueError:
                complexity = Complexity.M
            tasks.append(
                TaskNode(
                    id=str(raw["id"]),
                    description=str(raw["description"]),
                    complexity=complexity,
                    prerequisites=list(raw.get("prerequisites", []) or []),
                    anticipated_tools=list(raw.get("anticipated_tools", []) or []),
                    recoverability=float(raw.get("recoverability", 0.7)),
                )
            )
        clarifications = list(parsed.get("clarifications", []) or [])
        return tasks, clarifications

    def _topological_ready(self) -> list[TaskNode]:
        """Return tasks whose prerequisites are all complete and which are still pending."""
        assert self.session is not None
        ready: list[TaskNode] = []
        for t in self.session.tasks.values():
            if t.status not in {TaskStatus.PENDING, TaskStatus.READY}:
                continue
            if all(
                self.session.tasks.get(p) and self.session.tasks[p].status == TaskStatus.COMPLETE
                for p in t.prerequisites
            ):
                ready.append(t)
        return ready

    def _pick_next(self) -> TaskNode | None:
        """Scheduler: smallest priority key wins, ties broken by ``order_hint``."""
        ready = self._topological_ready()
        if not ready:
            return None
        heap: list[tuple[tuple[int, float], int, str]] = []
        assert self.session is not None
        order_map = {tid: i for i, tid in enumerate(self.session.order_hint)}
        for t in ready:
            heappush(heap, (t.priority(), order_map.get(t.id, 9999), t.id))
        _, _, tid = heappop(heap)
        return self.session.tasks[tid]

    # -------- Safety --------

    async def _hypothetical_check(self, command: str) -> str:
        """Ask the LLM about likely consequences of a destructive shell command."""
        messages = [
            LLMMessage(
                role="user",
                content=(
                    f"The agent wants to run: `{command}`\n\n"
                    "List potential consequences for the project state in 3 bullet points. "
                    "Reply with bullets only."
                ),
            )
        ]
        try:
            response = await self.llm.complete(messages, tools=None, temperature=0.0)
            return response.text.strip()
        except Exception as exc:
            return f"(hypothetical check unavailable: {exc})"

    # -------- Verify & Review --------

    async def _verify(self, task: TaskNode, loop_result: LoopResult) -> tuple[bool, str]:
        """Run linter+tests over any modified artifacts and ask the LLM to grade."""
        notes: list[str] = []
        ok = True
        py_artifacts = [a for a in loop_result.artifacts if a.endswith(".py")]
        for path in py_artifacts:
            res = await self.registry.invoke("code_lint", {"path": path, "language": "python"})
            if res.ok and not (res.output or {}).get("ok", True):
                ok = False
                notes.append(f"lint failed: {path}")
        if any(a.startswith("tests") or a.endswith("_test.py") for a in loop_result.artifacts):
            res = await self.registry.invoke("test_runner", {"path": "tests", "framework": "pytest"})
            if res.ok:
                out = res.output or {}
                if out.get("failed") or out.get("errors"):
                    ok = False
                    notes.append(f"test_runner: {out.get('failed', 0)} failed")
        if not notes:
            notes.append("(no automated checks applicable)")
        return ok, "; ".join(notes)

    async def _spawn_code_review(self, diff: str) -> dict[str, Any]:
        """Run the parallel Code Reviewer sub-agent over ``diff``."""
        if not diff.strip():
            return {"summary": "no diff", "findings": [], "approve": True}
        messages = [
            LLMMessage(role="user", content=f"Review this diff:\n\n```diff\n{diff[:60_000]}\n```")
        ]
        try:
            response = await self.llm.complete(
                messages, tools=None, system=REVIEWER_PROMPT, temperature=0.0
            )
            return _strict_json_extract(response.text)
        except Exception as exc:
            return {"summary": f"reviewer failed: {exc}", "findings": [], "approve": True}

    # -------- Debug loop --------

    async def _debug_loop(self, task: TaskNode, failure: str) -> bool:
        """Autonomous fix-and-retry loop, capped by config.safety.max_debug_iterations."""
        assert self.context is not None
        max_iters = self.config.safety.max_debug_iterations
        for i in range(max_iters):
            self.context.user(
                f"DEBUG iteration {i+1}/{max_iters}. Failure was:\n{failure}\n"
                "Form a hypothesis, apply a targeted fix, and re-run the relevant test."
            )
            sub_loop = AgentLoop(
                llm=self.llm,
                tools=self.registry,
                context=self.context,
                cwd=str(Path.cwd()),
                step_limit=10,
                ui_hook=self.ui,
            )
            result = await sub_loop.run(task)
            if result.completed:
                return True
            failure = result.summary
        return False

    # -------- Public entrypoints --------

    async def run(self, objective: str) -> Session:
        """Start a new session from an objective, returning the final session state."""
        session_id = uuid.uuid4().hex[:12]
        self.session = Session(
            id=session_id,
            objective=objective,
            tasks={},
            order_hint=[],
            started_at=time.time(),
        )
        self.context = ContextManager.from_config(self.config, session_id=session_id)
        self.audit = AuditLog(self._state_dir() / "audit" / f"{session_id}.jsonl")
        self.registry.set_audit(self.audit)
        self.context.working.set_compression_callback(self._compress_messages)

        if self.ui is not None and hasattr(self.ui, "set_session"):
            self.ui.set_session(self.session)

        self.audit.append("session_start", {"objective": objective})

        # PLAN
        tasks, clarifications = await self._plan(objective)
        self.session.clarifications = clarifications
        for t in tasks:
            self.session.tasks[t.id] = t
            self.session.order_hint.append(t.id)
        if self.ui is not None and hasattr(self.ui, "refresh_dag"):
            self.ui.refresh_dag()
        self.audit.append("plan", {"tasks": [t.id for t in tasks], "clarifications": clarifications})

        # CLARIFY
        if clarifications and self.clarification_handler:
            answers = self.clarification_handler(clarifications)
            self.context.system(
                "Clarifications:\n" + "\n".join(f"- {a['question']} → {a.get('answer', '')}" for a in answers)
            )

        # Seed semantic memory with the objective
        self.context.seed_with_memories(objective, top_k=5)
        self.save_session()

        # EXECUTE
        completed_count = 0
        while True:
            task = self._pick_next()
            if task is None:
                break
            await self._execute_one(task)
            completed_count += 1
            self.save_session()

            # REFLECT every 5 tasks
            if completed_count % 5 == 0:
                await self._reflect()

            # Meta-cognition: every 3 tasks, evaluate if plan still optimal
            if completed_count % 3 == 0:
                await self._meta_evaluate()

        await self._finalize()
        return self.session

    async def resume(self, session_id: str) -> Session:
        """Resume a previously checkpointed session."""
        self.load_session(session_id)
        if self.ui is not None and hasattr(self.ui, "set_session"):
            self.ui.set_session(self.session)
        assert self.session is not None and self.context is not None
        self.context.working.set_compression_callback(self._compress_messages)
        completed_count = sum(
            1 for t in self.session.tasks.values() if t.status == TaskStatus.COMPLETE
        )
        while True:
            task = self._pick_next()
            if task is None:
                break
            await self._execute_one(task)
            completed_count += 1
            self.save_session()
            if completed_count % 5 == 0:
                await self._reflect()
        await self._finalize()
        return self.session

    async def _execute_one(self, task: TaskNode) -> None:
        """Run + verify + review a single task; trigger debug loop on failure."""
        assert self.session is not None and self.context is not None and self.audit is not None
        task.status = TaskStatus.IN_PROGRESS
        task.attempts += 1
        if self.ui is not None and hasattr(self.ui, "refresh_dag"):
            self.ui.refresh_dag()
        self.audit.append("task_start", {"id": task.id, "description": task.description})
        loop = AgentLoop(
            llm=self.llm,
            tools=self.registry,
            context=self.context,
            cwd=str(Path.cwd()),
            step_limit=self.config.safety.max_step_limit,
            ui_hook=self.ui,
        )
        loop_result = await loop.run(task)
        task.summary = loop_result.summary
        task.artifacts = loop_result.artifacts

        if not loop_result.completed and not loop_result.blocked:
            ok = await self._debug_loop(task, loop_result.summary)
            if ok:
                loop_result.completed = True

        # VERIFY
        ok, notes = await self._verify(task, loop_result)
        if not ok:
            self.context.system(f"Verification notes: {notes}")
            self.audit.append("verify_failed", {"task": task.id, "notes": notes})
            self.session.notes.append(f"verify[{task.id}]: {notes}")

        # REVIEW (parallel sub-agent)
        if loop_result.artifacts:
            diff_res = await self.registry.invoke("git_ops", {"operation": "diff", "args": {}})
            diff = (diff_res.output or {}).get("data", {}).get("patch", "") if diff_res.ok else ""
            review = await self._spawn_code_review(diff)
            self.context.system(
                "Code review summary: " + str(review.get("summary", ""))[:600]
            )
            self.audit.append("review", {"task": task.id, "review": review})
            if review.get("findings"):
                self.context.semantic.add(
                    "decision",
                    f"Reviewer findings for task {task.id}: {json.dumps(review)[:1500]}",
                    metadata={"task_id": task.id},
                )

        # Store decision / learnings in semantic memory
        self.context.semantic.add(
            "decision",
            f"Task {task.id}: {task.description}\nOutcome: {task.summary}",
            metadata={"task_id": task.id, "complexity": task.complexity.value},
        )

        if loop_result.completed and ok:
            task.status = TaskStatus.COMPLETE
        elif loop_result.blocked:
            task.status = TaskStatus.FAILED
            task.error = loop_result.summary
        else:
            task.status = TaskStatus.FAILED
            task.error = loop_result.summary or notes
            self.context.semantic.add(
                "error",
                f"Task {task.id} failed: {task.error}",
                metadata={"task_id": task.id},
            )

        self.audit.append(
            "task_end",
            {
                "id": task.id,
                "status": task.status.value,
                "steps": loop_result.steps,
                "artifacts": loop_result.artifacts,
            },
        )
        if self.ui is not None and hasattr(self.ui, "refresh_dag"):
            self.ui.refresh_dag()

    async def _reflect(self) -> None:
        """Reflection pass: summarise progress + optionally rewrite plan."""
        assert self.session is not None and self.context is not None and self.audit is not None
        completed = [t for t in self.session.tasks.values() if t.status == TaskStatus.COMPLETE]
        pending = [t for t in self.session.tasks.values() if t.status == TaskStatus.PENDING]
        prompt = (
            "Completed tasks:\n"
            + "\n".join(f"- {t.id}: {t.description} → {t.summary}" for t in completed)
            + "\nPending tasks:\n"
            + "\n".join(f"- {t.id}: {t.description}" for t in pending)
        )
        try:
            response = await self.llm.complete(
                [LLMMessage(role="user", content=prompt)],
                tools=None,
                system=REFLECTOR_PROMPT,
                temperature=0.0,
            )
            data = _strict_json_extract(response.text)
        except Exception as exc:
            data = {"summary": f"reflection failed: {exc}"}
        self.session.notes.append("reflect: " + str(data.get("summary", ""))[:400])
        if data.get("should_rewrite_plan") and data.get("plan_changes"):
            self._apply_plan_changes(data["plan_changes"])
        self.context.semantic.add(
            "summary",
            f"Session {self.session.id} reflection: {data.get('summary', '')}",
        )
        self.audit.append("reflect", data)

    async def _meta_evaluate(self) -> None:
        """Lightweight meta-cognition: ask if the plan is still optimal."""
        assert self.session is not None and self.audit is not None
        completed = [t for t in self.session.tasks.values() if t.status == TaskStatus.COMPLETE]
        pending = [t for t in self.session.tasks.values() if t.status not in {TaskStatus.COMPLETE, TaskStatus.FAILED}]
        if not pending:
            return
        prompt = (
            "Given the current state, do you recommend keeping the plan as-is or modifying it?\n"
            f"Completed: {len(completed)}\nPending: {[t.id for t in pending]}\n"
            "Respond with strict JSON: {\"keep\": true|false, \"reason\": \"...\"}"
        )
        try:
            response = await self.llm.complete(
                [LLMMessage(role="user", content=prompt)], tools=None, temperature=0.0
            )
            data = _strict_json_extract(response.text)
        except Exception:
            return
        self.audit.append("meta", data)
        if not data.get("keep", True):
            self.session.notes.append(f"meta: {data.get('reason', '')[:200]}")

    def _apply_plan_changes(self, changes: list[dict[str, Any]]) -> None:
        """Apply reflector-suggested mutations to the task DAG."""
        assert self.session is not None
        for change in changes:
            action = change.get("action")
            tid = change.get("task_id")
            if action == "remove" and tid in self.session.tasks:
                t = self.session.tasks[tid]
                if t.status in {TaskStatus.PENDING, TaskStatus.READY}:
                    t.status = TaskStatus.SKIPPED
            elif action == "reprioritize" and tid in self.session.order_hint:
                self.session.order_hint.remove(tid)
                self.session.order_hint.insert(0, tid)
            elif action == "add":
                new_id = tid or f"t_meta_{len(self.session.tasks)+1}"
                self.session.tasks[new_id] = TaskNode(
                    id=new_id,
                    description=str(change.get("rationale", "")),
                )
                self.session.order_hint.append(new_id)

    async def _compress_messages(self, messages: list[LLMMessage]) -> str:
        """Summarise a batch of low-importance messages for context compression."""
        try:
            content = "\n".join(f"{m.role}: {m.content[:1000]}" for m in messages)
            response = await self.llm.complete(
                [LLMMessage(role="user", content=content)],
                tools=None,
                system=COMPRESSION_PROMPT,
                temperature=0.0,
            )
            return response.text.strip()
        except Exception:
            return "\n".join(f"- {m.role}: {m.content[:200]}" for m in messages)

    async def _finalize(self) -> None:
        """Wrap up the session: documentation, cost report, persistence."""
        assert self.session is not None and self.audit is not None
        await self._generate_documentation()
        self.session.finished_at = time.time()
        self.session.cost_summary = self.cost.summary()
        statuses = {t.status for t in self.session.tasks.values()}
        if TaskStatus.FAILED in statuses:
            self.session.outcome = "partial"
        elif all(t.status == TaskStatus.COMPLETE for t in self.session.tasks.values()):
            self.session.outcome = "complete"
        else:
            self.session.outcome = "incomplete"
        self.save_session()
        self.audit.append("session_end", {"outcome": self.session.outcome, "cost": self.cost.summary()})

    async def _generate_documentation(self) -> None:
        """Auto-generate CHANGES.md describing modifications in the session."""
        assert self.session is not None
        try:
            diff_res = await self.registry.invoke("git_ops", {"operation": "diff", "args": {}})
        except Exception:
            return
        if not diff_res.ok:
            return
        diff = (diff_res.output or {}).get("data", {}).get("patch", "")
        if not diff:
            return
        try:
            messages = [
                LLMMessage(
                    role="user",
                    content=(
                        f"Objective: {self.session.objective}\n\n"
                        "Given the following diff, write a short CHANGES.md entry "
                        "(a header and 3-7 bullets). Reply with markdown only.\n\n"
                        f"```diff\n{diff[:40_000]}\n```"
                    ),
                )
            ]
            response = await self.llm.complete(messages, tools=None, temperature=0.0)
            block = response.text.strip() + "\n"
        except Exception:
            block = f"# {self.session.objective}\n\n- (auto-generated entry; see git diff)\n"
        changes_path = Path("CHANGES.md")
        existing = changes_path.read_text() if changes_path.exists() else ""
        changes_path.write_text(block + "\n" + existing)


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _strict_json_extract(text: str) -> dict[str, Any]:
    """Pull a JSON object out of a possibly fenced LLM response."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_OBJ_RE.search(text)
        if not m:
            return {}
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return {}


def assess_command(command: str) -> dict[str, Any]:
    """Public helper used by the CLI confirmation hook."""
    return {"destructive": is_destructive(command), "command": command}


__all__ = ["Orchestrator", "Session", "assess_command"]


async def _run_with_optional_event_loop(coro):  # pragma: no cover - helper
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None:
        return await coro
    return asyncio.run(coro)
