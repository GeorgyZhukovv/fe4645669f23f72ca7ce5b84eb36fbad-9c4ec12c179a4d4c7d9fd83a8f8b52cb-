"""Tests for the model registry and TOML loader."""

from __future__ import annotations

import pytest

from agent.llm.base import CostTracker
from agent.llm.registry import CustomEndpoint, ModelRegistry, ModelSpec


@pytest.fixture(autouse=True)
def _reset_registry():
    ModelRegistry.reset()
    yield
    ModelRegistry.reset()


def test_registry_loads_toml_with_known_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    r = ModelRegistry.instance()
    ids = {m.id for m in r.all()}
    for required in [
        "claude-opus-4", "claude-sonnet-4", "claude-haiku-4",
        "gpt-4o", "gpt-4o-mini", "o3", "o4-mini", "codex-mini",
        "gemini-2.5-pro", "gemini-2.5-flash",
        "mistral-large", "codestral",
        "llama-3.3-70b", "deepseek-r1-distill", "qwen-qwq",
        "llama-3.3-70b-cerebras",
        "deepseek-v3", "qwen-2.5-coder-32b",
    ]:
        assert required in ids, f"missing: {required}"


def test_registry_marks_provider_availability(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    r = ModelRegistry.instance()
    anthropic_avail = [m for m in r.by_provider("anthropic") if m.available]
    openai_avail = [m for m in r.by_provider("openai") if m.available]
    assert anthropic_avail
    assert not openai_avail


def test_registry_filters_by_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    r = ModelRegistry.instance()
    assert any(m.id == "claude-opus-4" for m in r.by_tier("frontier"))
    assert any(m.id == "gpt-4o-mini" for m in r.by_tier("fast"))


def test_registry_custom_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_KEY", "abc")
    r = ModelRegistry.instance()
    r.register_custom_endpoint(CustomEndpoint(
        name="my-llm",
        base_url="http://localhost:8000/v1",
        api_key_env="MY_KEY",
        model_id="my-model",
        context_window=64000,
    ))
    spec = r.get("my-llm")
    assert spec is not None
    assert spec.available is True
    assert spec.context_window_tokens == 64000


def test_build_client_routes_to_correct_class(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setenv("TOGETHER_API_KEY", "k")
    r = ModelRegistry.instance()
    from agent.llm.anthropic_client import AnthropicClient
    from agent.llm.groq_client import GroqClient
    from agent.llm.openai_client import OpenAIClient
    from agent.llm.together_client import TogetherClient
    assert isinstance(r.build_client("claude-sonnet-4", CostTracker()), AnthropicClient)
    assert isinstance(r.build_client("gpt-4o-mini", CostTracker()), OpenAIClient)
    assert isinstance(r.build_client("llama-3.3-70b", CostTracker()), GroqClient)
    assert isinstance(r.build_client("deepseek-v3", CostTracker()), TogetherClient)


def test_build_client_unknown_id_raises() -> None:
    r = ModelRegistry.instance()
    with pytest.raises(KeyError):
        r.build_client("nonexistent-model")


def test_local_autodetect_handles_missing_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    # ensure registry build doesn't crash even when both local probes fail
    import httpx

    real_client = httpx.Client

    def _fail_client(*a, **kw):
        raise ConnectionError("nope")

    monkeypatch.setattr(httpx, "Client", _fail_client)
    r = ModelRegistry.instance()
    r.autodetect_local()
    # No exception; registry still has the static entries.
    assert r.get("gpt-4o") is not None
