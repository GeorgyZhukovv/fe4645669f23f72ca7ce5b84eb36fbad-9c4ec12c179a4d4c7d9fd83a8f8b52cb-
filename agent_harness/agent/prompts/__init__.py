"""Prompt templates for each agent phase."""

from agent.prompts.executor import EXECUTOR_PROMPT, build_executor_prompt
from agent.prompts.planner import PLANNER_PROMPT, build_planner_prompt
from agent.prompts.reflector import REFLECTOR_PROMPT
from agent.prompts.reviewer import REVIEWER_PROMPT

__all__ = [
    "EXECUTOR_PROMPT",
    "PLANNER_PROMPT",
    "REFLECTOR_PROMPT",
    "REVIEWER_PROMPT",
    "build_executor_prompt",
    "build_planner_prompt",
]
