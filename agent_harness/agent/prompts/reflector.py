"""Reflector and compression prompts."""

from __future__ import annotations

REFLECTOR_PROMPT = """You are the REFLECTOR. The agent has just completed several sub-tasks.
Review the recent activity and produce a STRICT JSON object — no prose:

{
  "summary": "1-3 sentences of progress so far",
  "key_learnings": ["..."],
  "plan_changes": [
    {"action": "add|remove|reprioritize", "task_id": "tN?", "rationale": "..."}
  ],
  "should_rewrite_plan": true|false
}
"""


COMPRESSION_PROMPT = """Summarize the following conversation fragment into a SHORT
single paragraph that preserves: the goal in progress, decisions made, files
touched, and any unresolved errors. Do not invent details.
"""
