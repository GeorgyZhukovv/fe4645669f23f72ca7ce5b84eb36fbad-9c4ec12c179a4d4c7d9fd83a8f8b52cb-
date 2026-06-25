"""Together AI client. Thin OpenAI-compatible wrapper."""

from __future__ import annotations

import os

from agent.llm.base import CostTracker
from agent.llm.openai_client import OpenAIClient


class TogetherClient(OpenAIClient):
    """Thin OpenAI-compatible wrapper for ``https://api.together.xyz/v1``."""

    def __init__(
        self,
        model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        api_key: str | None = None,
        cost_tracker: CostTracker | None = None,
    ) -> None:
        """Construct the client.

        Args:
            model: Together api_model_name.
            api_key: API key (otherwise reads ``TOGETHER_API_KEY``).
            cost_tracker: Optional shared :class:`CostTracker`.
        """
        super().__init__(
            model=model,
            api_key=api_key or os.environ.get("TOGETHER_API_KEY", ""),
            base_url="https://api.together.xyz/v1",
            cost_tracker=cost_tracker,
        )
