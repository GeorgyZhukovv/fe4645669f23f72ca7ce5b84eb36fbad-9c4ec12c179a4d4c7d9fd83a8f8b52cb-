"""Model registry: loads ``models.toml``, autodetects local backends, builds clients."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

try:
    import tomllib  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

from agent.llm.base import CostTracker, LLMClient


@dataclass
class ModelSpec:
    """One model definition resolved from ``models.toml``."""

    id: str
    provider: str
    api_model_name: str
    context_window_tokens: int
    input_cost_per_million: float
    output_cost_per_million: float
    supports_tool_use: bool
    supports_vision: bool
    supports_streaming: bool
    tier: str
    notes: str = ""
    reasoning: bool = False
    no_system_prompt: bool = False
    fim: bool = False
    base_url: str | None = None
    api_key_env: str | None = None
    available: bool = True

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class CustomEndpoint:
    """User-defined OpenAI-compatible endpoint loaded from ``.agent.toml``."""

    name: str
    base_url: str
    api_key_env: str
    model_id: str
    context_window: int = 32000


@dataclass
class LocalAutodetect:
    """Spec for a local backend that exposes a ``/models``-style listing endpoint."""

    id: str
    provider: str
    base_url: str
    list_endpoint: str
    list_key: str
    list_name_key: str
    prefix: str
    tier: str = "local"


def _models_toml_path() -> Path:
    return Path(__file__).parent / "models.toml"


def _truthy_env(name: str) -> bool:
    """Return True if the env var ``name`` is set and non-empty."""
    val = os.environ.get(name, "")
    return bool(val.strip())


def _provider_available(provider: str) -> bool:
    """Heuristic: whether the env var for ``provider`` is set."""
    env_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "groq": "GROQ_API_KEY",
        "cerebras": "CEREBRAS_API_KEY",
        "together": "TOGETHER_API_KEY",
    }
    env = env_map.get(provider)
    if env is None:
        return True  # local backends; their availability is checked separately
    return _truthy_env(env)


class ModelRegistry:
    """Singleton-style registry of every model the harness can talk to."""

    _instance: "ModelRegistry | None" = None

    def __init__(self) -> None:
        """Build the registry by loading ``models.toml`` and probing local backends."""
        self._models: dict[str, ModelSpec] = {}
        self._autodetect: list[LocalAutodetect] = []
        self._custom_endpoints: list[CustomEndpoint] = []
        self._load_toml(_models_toml_path())

    @classmethod
    def instance(cls) -> "ModelRegistry":
        """Lazy singleton accessor."""
        if cls._instance is None:
            cls._instance = cls()
            cls._instance.autodetect_local()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Tear down the singleton (for tests)."""
        cls._instance = None

    def _load_toml(self, path: Path) -> None:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        for section, items in data.items():
            if section == "local_autodetect":
                for raw in items:
                    self._autodetect.append(LocalAutodetect(**raw))
                continue
            if not isinstance(items, list):
                continue
            for raw in items:
                spec = ModelSpec(**raw)
                spec.available = _provider_available(spec.provider)
                self._models[spec.id] = spec

    # -- custom endpoints (from user config) --------------------------------

    def register_custom_endpoint(self, ep: CustomEndpoint) -> None:
        """Add an OpenAI-compatible endpoint at runtime."""
        self._custom_endpoints.append(ep)
        spec = ModelSpec(
            id=ep.name,
            provider="custom",
            api_model_name=ep.model_id,
            context_window_tokens=ep.context_window,
            input_cost_per_million=0.0,
            output_cost_per_million=0.0,
            supports_tool_use=True,
            supports_vision=False,
            supports_streaming=True,
            tier="local",
            notes=f"Custom endpoint @ {ep.base_url}",
            base_url=ep.base_url,
            api_key_env=ep.api_key_env,
            available=_truthy_env(ep.api_key_env),
        )
        self._models[ep.name] = spec

    def load_custom_endpoints_from_config(self, raw: list[dict[str, Any]] | None) -> None:
        """Load custom endpoints from a ``[[llm.custom_endpoints]]`` list."""
        for entry in raw or []:
            try:
                ep = CustomEndpoint(
                    name=entry["name"],
                    base_url=entry["base_url"],
                    api_key_env=entry.get("api_key_env", "AGENT_CUSTOM_API_KEY"),
                    model_id=entry.get("model_id", entry["name"]),
                    context_window=int(entry.get("context_window", 32000)),
                )
            except KeyError:
                continue
            self.register_custom_endpoint(ep)

    # -- local autodetect ---------------------------------------------------

    def autodetect_local(self) -> None:
        """Probe local backends (Ollama, LM Studio) and register their loaded models."""
        for spec in self._autodetect:
            self._probe_one_local(spec)

    def _probe_one_local(self, spec: LocalAutodetect) -> None:
        try:
            import httpx
        except ImportError:
            return
        try:
            with httpx.Client(timeout=2) as client:
                resp = client.get(spec.list_endpoint)
                if resp.status_code != 200:
                    return
                data = resp.json()
        except Exception:
            return
        items = data.get(spec.list_key, []) if isinstance(data, dict) else []
        for item in items:
            if isinstance(item, dict):
                name = item.get(spec.list_name_key)
            else:
                name = str(item)
            if not name:
                continue
            model_id = f"{spec.prefix}{name}"
            self._models[model_id] = ModelSpec(
                id=model_id,
                provider=spec.provider,
                api_model_name=name,
                context_window_tokens=32000,
                input_cost_per_million=0.0,
                output_cost_per_million=0.0,
                supports_tool_use=True,
                supports_vision=False,
                supports_streaming=True,
                tier=spec.tier,
                notes=f"Auto-detected on {spec.base_url}",
                base_url=spec.base_url,
                available=True,
            )

    # -- accessors ----------------------------------------------------------

    def all(self) -> list[ModelSpec]:
        """All registered models."""
        return list(self._models.values())

    def get(self, model_id: str) -> ModelSpec | None:
        """Look up a model by id."""
        return self._models.get(model_id)

    def find_by_api_name(self, api_name: str) -> ModelSpec | None:
        """Reverse-lookup a model by its provider-side api_model_name."""
        for spec in self._models.values():
            if spec.api_model_name == api_name:
                return spec
        return None

    def by_tier(self, tier: str) -> list[ModelSpec]:
        """Filter by tier (frontier / strong / fast / local)."""
        return [m for m in self._models.values() if m.tier == tier]

    def by_provider(self, provider: str) -> list[ModelSpec]:
        """Filter by provider."""
        return [m for m in self._models.values() if m.provider == provider]

    def available_models(self) -> list[ModelSpec]:
        """Only the models whose provider credentials appear to be configured."""
        return [m for m in self._models.values() if m.available]

    # -- client construction ------------------------------------------------

    def build_client(
        self,
        model_id: str,
        cost_tracker: CostTracker | None = None,
    ) -> LLMClient:
        """Return a configured :class:`LLMClient` for ``model_id``.

        Args:
            model_id: A registered model id (the key from ``models.toml``).
            cost_tracker: Optional :class:`CostTracker` to attach.

        Raises:
            KeyError: If ``model_id`` is not registered.
            RuntimeError: If the required SDK isn't installed or env var is missing.
        """
        spec = self.get(model_id)
        if spec is None:
            raise KeyError(f"unknown model id: {model_id}")
        client = _build_client_for_spec(spec, cost_tracker)
        if hasattr(client, "_active_model"):
            client._active_model = spec.api_model_name  # type: ignore[attr-defined]
        else:
            client.cost_tracker._active_model = spec.api_model_name  # type: ignore[attr-defined]
        return client

    def print_availability_summary(self) -> None:  # pragma: no cover - bootstrap convenience
        """Bootstrap helper: print per-provider availability to stdout."""
        by_provider: dict[str, list[ModelSpec]] = {}
        for m in self._models.values():
            by_provider.setdefault(m.provider, []).append(m)
        for provider, models in sorted(by_provider.items()):
            avail = sum(1 for m in models if m.available)
            print(f"  {provider:<10}  {avail:>2}/{len(models)} configured")


def _build_client_for_spec(spec: ModelSpec, cost_tracker: CostTracker | None) -> LLMClient:
    """Dispatch to the provider-appropriate client constructor."""
    if spec.provider == "anthropic":
        from agent.llm.anthropic_client import AnthropicClient

        return AnthropicClient(model=spec.api_model_name, cost_tracker=cost_tracker)
    if spec.provider == "openai":
        from agent.llm.openai_client import OpenAIClient

        return OpenAIClient(model=spec.api_model_name, cost_tracker=cost_tracker)
    if spec.provider == "gemini":
        from agent.llm.gemini_client import GeminiClient

        return GeminiClient(model=spec.api_model_name, cost_tracker=cost_tracker)
    if spec.provider == "mistral":
        from agent.llm.mistral_client import MistralClient

        return MistralClient(model=spec.api_model_name, cost_tracker=cost_tracker)
    if spec.provider == "groq":
        from agent.llm.groq_client import GroqClient

        return GroqClient(model=spec.api_model_name, cost_tracker=cost_tracker)
    if spec.provider == "cerebras":
        from agent.llm.cerebras_client import CerebrasClient

        return CerebrasClient(model=spec.api_model_name, cost_tracker=cost_tracker)
    if spec.provider == "together":
        from agent.llm.together_client import TogetherClient

        return TogetherClient(model=spec.api_model_name, cost_tracker=cost_tracker)
    if spec.provider == "ollama":
        from agent.llm.openai_client import OpenAIClient

        return OpenAIClient(
            model=spec.api_model_name,
            api_key="ollama",
            base_url=spec.base_url or "http://localhost:11434/v1",
            cost_tracker=cost_tracker,
        )
    if spec.provider == "lmstudio":
        from agent.llm.openai_client import OpenAIClient

        return OpenAIClient(
            model=spec.api_model_name,
            api_key="lmstudio",
            base_url=spec.base_url or "http://localhost:1234/v1",
            cost_tracker=cost_tracker,
        )
    if spec.provider == "custom":
        from agent.llm.openai_client import OpenAIClient

        api_key = os.environ.get(spec.api_key_env or "", "")
        return OpenAIClient(
            model=spec.api_model_name,
            api_key=api_key or "custom",
            base_url=spec.base_url,
            cost_tracker=cost_tracker,
        )
    raise RuntimeError(f"unsupported provider: {spec.provider}")
