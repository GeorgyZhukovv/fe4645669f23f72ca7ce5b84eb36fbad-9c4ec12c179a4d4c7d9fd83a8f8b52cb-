# Agent Harness

A production-grade agentic coding harness for fully autonomous software engineering.

## Features

### Foundation
- Plan-and-Execute orchestrator with persistent task DAG, topological scheduler, meta-cognition.
- ReAct micro-loop with parallel tool execution and streaming.
- Three-tier memory: working (tiktoken-budgeted), episodic (SQLite FTS5), semantic (chromadb / TF-IDF fallback).
- Tool registry with auto-generated JSON Schema, retry/backoff, timeout, side-effect classification, JSONL audit.
- Pluggable LLM clients (Anthropic + OpenAI / ollama / lm-studio) with streaming, parallel tool calls, fallback, cost tracking.
- Rich live terminal UI with task DAG, agent stream, tool, memory, audit panels.

### Multi-file edit engine (`agent/tools/multi_edit.py`)
- `EditPlan` + `EditOperation` (overwrite / patch / create / delete / rename / mkdir).
- Snapshot-based atomic rollback on operation or verification failure.
- Soft-delete to `.agent_trash/<timestamp>/`.
- Diff preview renderer.

### Codebase indexer (`agent/indexer/`)
- AST-based Python parsing + regex extraction for JS/TS/Rust/Go.
- Persistent gzipped pickle index at `.agent_index/index.pkl.gz`.
- Incremental updates via content hashing.
- Tools: `index_search`, `index_callers`, `index_usages`, `index_summarize_file`, `index_dependencies`, `index_stats`, `index_rebuild`.
- `@symbol` and `@path/to/file` mention resolution in the run objective.

### Sandboxed execution (`agent/sandbox/`)
- Abstract `SandboxBackend` with three implementations:
  - `subprocess` (default) — ulimit / RLIMIT_AS + RLIMIT_CPU + RLIMIT_NPROC, blocked-env masking, process-group SIGKILL on timeout.
  - `docker` — long-lived container reused per session, project image auto-detection, cpu / memory / pids / network limits, writable overlay sync on teardown.
  - `firejail` — Linux-only, `--noprofile --noroot --private-tmp --net=none`.
- Auto-select with graceful fallback; `bash_exec` transparently routes through the active backend.

### Dependency intelligence (`agent/tools/deps.py`)
- Auto-detects Python / Node / Rust / Go from marker files and lockfiles.
- Vulnerability scan via `pip-audit` / `npm audit` / `cargo audit` / `govulncheck`, with OSV fallback.
- Upgrade planner with `security_only` / `minor` / `all` strategies.
- `deps_apply_upgrades` produces an `EditPlan` ready for `multi_edit`.

### Observability (`agent/observability/`)
- Structured `SessionEvent` stream → append-only JSONL.
- `agent replay <session-id>` plays back at `--speed 1|2|5|instant`, with `--report` writing `session_report.md`.
- Optional OTLP exporter (`OTelExporter`) for Grafana / Tempo.
- `agent cost --dashboard` with per-model / per-task / per-tool tables and a session token timeline.

### Multi-agent swarm (`agent/swarm/`)
- `AgentRole` (PLANNER / CODER / REVIEWER / TESTER / DOCUMENTER / SECURITY_AUDITOR) with keyword-driven assignment.
- `SwarmExecutor` using `asyncio.TaskGroup` and a `Semaphore` for bounded parallelism.
- `MessageBus` with per-role mailboxes and a broadcast log.
- `FileLockList` for per-path serialisation and `EditConflictResolver` for non-overlapping three-way merges.

### Interactive steering (`agent/ui/interactive.py`)
- Non-blocking input loop (aioconsole when available, threaded `input()` fallback).
- Commands: `/status`, `/plan`, `/pause`, `/resume`, `/skip`, `/abort`, `/add <desc>`, `/focus <id>`, `/retry`, `/explain`, `/undo`, `/diff`, `/memory <q>`, `/cost`, `/model <name>`, `/budget <n>`, `/verbose`, `/yolo`.
- `PendingQuestion` supports inline `ask_user(question, context, default)` tool calls with timeout-default fallback.

### LSP integration (`agent/tools/lsp.py`)
- JSON-RPC 2.0 over stdio.
- Auto-detect pyright / pylsp / typescript-language-server / rust-analyzer / gopls / clangd.
- `LSPServerPool` keeps long-lived clients per language and reinitialises on crash.
- Tools: `lsp_diagnostics`, `lsp_hover`, `lsp_definition`, `lsp_references`, `lsp_code_action`.

### VCS hosting (`agent/tools/vcs_hosting.py`)
- Abstract `VCSHostClient` with `GitHubClient` and `GitLabClient` (REST over httpx — no PyGithub / python-gitlab dependency).
- Tools: `gh_get_pr`, `gh_create_pr`, `gh_pr_diff`, `gh_get_issue`, `gh_ci_status`, `gh_comment`.
- CLI: `agent pr review`, `agent pr fix`, `agent issue <n>`, `agent ci fix`.

### Prompts + personas (`agent/prompts/`)
- TOML prompt library with versioned templates.
- Template language: `{{ var }}`, `{% include "id" %}` (cycle-detected), `{% if cond %}...{% endif %}`, `{% for x in list %}...{% endfor %}`.
- Built-in personas: `default`, `strict`, `fast`, `security`, `minimal`, `architect`.
- User-defined personas under `[personas.<name>]` in `.agent.toml`.
- `agent prompt eval "<objective>" --variants a,b` runs A/B comparisons.

## Installation

```bash
./bootstrap.sh                          # venv + install + tests + (optional) demo
PYTHON=python3.12 ./bootstrap.sh        # use a specific Python interpreter
```

Manual:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

## Quick start

```bash
export ANTHROPIC_API_KEY=...
agent run "Refactor @AuthService to use dependency injection"
agent run "Add tests for @src/utils/parser.py" --persona strict
agent pr review 42
agent issue 17
agent ci fix
agent replay <session-id> --report
agent cost --dashboard
```

## CLI

| Command | Description |
| --- | --- |
| `agent run "<objective>"` | Start a new session (supports `@mentions` and `--persona`). |
| `agent resume <session-id>` | Resume a checkpointed session. |
| `agent history` | List prior sessions. |
| `agent tools list` | Show registered tools and schemas. |
| `agent memory search "<q>"` | Search the semantic memory store. |
| `agent cost [--dashboard]` | Cost breakdown (rich dashboard with `--dashboard`). |
| `agent replay <id> [--speed S] [--report]` | Replay or summarise a session. |
| `agent pr review <n>` | Review a PR. |
| `agent pr fix <n>` | Autonomously address PR review comments. |
| `agent issue <n>` | Fetch and implement an issue. |
| `agent ci fix` | Detect failing CI checks on the current commit and fix them. |
| `agent prompt-cmd eval "<objective>" --variants a,b` | A/B-compare prompt variants. |

## @ mention syntax

`@SymbolName` resolves through the codebase index to the symbol's location + signature.
`@path/to/file.py` injects the file's symbol summary into planner context.
Both are auto-detected in the objective string.

## Configuration

Layered: defaults → `~/.config/agent/config.toml` → `./.agent.toml` → `AGENT_*` env vars → CLI flags. Key sections:

```toml
[sandbox]
backend = "docker"      # auto | subprocess | docker | firejail
memory_limit = "2g"
network_mode = "none"

[swarm]
enabled = false
max_parallel = 4

[observability]
otlp_endpoint = "http://localhost:4318/v1/traces"

[persona]
active = "strict"

[personas.team]
base_role = "CODER"
temperature = 0.1
```

## Sandbox setup

Docker backend requires `docker` on PATH. Pull pre-warm the image:

```bash
docker pull python:3.12-slim
AGENT_SANDBOX__BACKEND=docker agent run "..."
```

firejail backend requires `firejail` (Debian/Ubuntu: `sudo apt install firejail`).

## Swarm mode

```toml
[swarm]
enabled = true
max_parallel = 6
```

The orchestrator assigns roles per task. Sub-agents share the episodic + semantic memory; their working memories stay isolated. `FileLockList` prevents two coders from editing the same file at once.

## VCS integration

Set `GITHUB_TOKEN` (or `GITLAB_TOKEN`) and either `GITHUB_REPOSITORY=owner/repo` or pass `--repo owner/repo`. GitHub Enterprise / self-hosted GitLab: override the API base on the client constructor.

## Tests

```bash
pytest -q                    # all 100+ tests
pytest tests/test_indexer.py
pytest tests/test_multi_edit.py
```
