# Agent Harness

A production-grade agentic coding harness for fully autonomous software engineering.

## Features

- Plan-and-Execute orchestrator with a persistent task DAG and topological scheduling.
- ReAct micro-loop with thought / action / observation streaming.
- Three-tier memory: working (token-budgeted context), episodic (SQLite), semantic (chromadb).
- Tool registry with auto-generated JSON Schema, retry/backoff, timeout enforcement, and audit logging.
- Pluggable LLM backends (Anthropic, OpenAI-compatible) with streaming, parallel tool calls, fallback, and cost tracking.
- Rich live terminal UI with task DAG, agent stream, tool panel, and memory panel.
- Self-improvement: autonomous debug loop, parallel code reviewer, sandboxed destructive-command preview.

## Installation

```bash
./bootstrap.sh
```

Or manually:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

## Quick start

```bash
export ANTHROPIC_API_KEY=...
agent run "Add a TODO endpoint and tests to the api package"
```

## CLI

| Command | Description |
| --- | --- |
| `agent run "<objective>"` | Start a new agent session. |
| `agent resume <session-id>` | Resume a checkpointed session. |
| `agent history` | List prior sessions. |
| `agent tools list` | Show registered tools and JSON Schemas. |
| `agent memory search "<query>"` | Search the semantic memory store. |
| `agent cost` | Show the cost / token breakdown for the last session. |

## Configuration

Layered: hard-coded defaults → `~/.config/agent/config.toml` → `./.agent.toml` → environment variables (`AGENT_*`) → CLI flags.
See `.agent.toml.example` for the full surface.

## Project layout

See `agent/` for runtime modules and `tests/` for the test suite. Every module exposes a typed public interface and is independently importable.
