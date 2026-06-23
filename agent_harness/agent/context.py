"""ContextManager bundling working memory, episodic memory, and semantic recall."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agent.config import AgentConfig
from agent.memory.episodic import EpisodicMemory
from agent.memory.semantic import SemanticMemory
from agent.memory.working import KVStore, WorkingMemory
from agent.types import LLMMessage, Memory


@dataclass
class ContextManager:
    """Glue class wiring the three memory tiers and exposing them to the loop."""

    working: WorkingMemory
    episodic: EpisodicMemory
    semantic: SemanticMemory
    kv: KVStore
    state_dir: Path

    @classmethod
    def from_config(cls, config: AgentConfig, session_id: str) -> "ContextManager":
        """Build a :class:`ContextManager` from the loaded :class:`AgentConfig`."""
        state_dir = Path(config.memory.working_dir).resolve()
        state_dir.mkdir(parents=True, exist_ok=True)
        episodic = EpisodicMemory(state_dir / config.memory.episodic_db, session_id=session_id)
        semantic = SemanticMemory(state_dir / "semantic", collection=config.memory.semantic_collection)
        working = WorkingMemory(budget_tokens=config.budget.context_token_limit, model=config.llm.model)
        kv = KVStore(state_dir / "kv.sqlite")
        return cls(working=working, episodic=episodic, semantic=semantic, kv=kv, state_dir=state_dir)

    def system(self, content: str) -> None:
        """Push a system message onto working memory."""
        self.working.add(LLMMessage(role="system", content=content))

    def user(self, content: str) -> None:
        """Push a user message."""
        self.working.add(LLMMessage(role="user", content=content))

    def assistant(self, content: str) -> None:
        """Push an assistant message."""
        self.working.add(LLMMessage(role="assistant", content=content))

    def tool(self, name: str, call_id: str, content: str) -> None:
        """Push a tool-result message."""
        msg = LLMMessage(role="tool", content=content, tool_call_id=call_id, name=name)
        self.working.add(msg)

    def seed_with_memories(self, query: str, top_k: int = 5) -> list[Memory]:
        """Retrieve relevant semantic memories and inject them as a system note."""
        mems = self.semantic.retrieve_relevant_memories(query, top_k=top_k)
        if not mems:
            return []
        lines = [f"- ({m.kind}) {m.content[:300]}" for m in mems]
        self.system("Relevant past memories:\n" + "\n".join(lines))
        return mems
