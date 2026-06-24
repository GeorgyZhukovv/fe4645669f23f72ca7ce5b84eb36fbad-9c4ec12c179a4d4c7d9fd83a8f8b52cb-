# Agent Harness (Tantalus)

A production-grade agentic coding harness for fully autonomous software engineering.

## Supported Models

Switch with `/model <id>` while a session is running, or set `AGENT_LLM__MODEL=<id>` before launch. The full set is defined in `agent/llm/models.toml`.

| id | provider | tier | context | tool use | vision | notes |
| --- | --- | --- | ---: | :---: | :---: | --- |
| claude-opus-4 | anthropic | frontier | 200k | ✓ | ✓ | highest capability |
| claude-sonnet-4 | anthropic | strong | 200k | ✓ | ✓ | balanced default |
| claude-haiku-4 | anthropic | fast | 200k | ✓ | ✓ | cheapest Anthropic |
| gpt-4o | openai | frontier | 128k | ✓ | ✓ | multimodal frontier |
| gpt-4o-mini | openai | fast | 128k | ✓ | ✓ | cheap default |
| o3 | openai | frontier | 200k | ✓ | ✗ | reasoning, no system role |
| o4-mini | openai | strong | 200k | ✓ | ✗ | smaller reasoning |
| codex-mini | openai | fast | 128k | ✓ | ✗ | code-optimised |
| gemini-2.5-pro | gemini | frontier | 1M | ✓ | ✓ | thinking mode |
| gemini-2.5-flash | gemini | fast | 1M | ✓ | ✓ | cheap huge ctx |
| gemini-2.0-flash | gemini | fast | 1M | ✓ | ✓ | stable Flash |
| mistral-large | mistral | strong | 128k | ✓ | ✗ | EU frontier |
| mistral-small | mistral | fast | 32k | ✓ | ✗ | fast + cheap |
| codestral | mistral | fast | 32k | ✓ | ✗ | code, FIM endpoint |
| llama-4-scout | groq | fast | 131k | ✓ | ✗ | sub-second TTFT |
| llama-4-maverick | groq | strong | 131k | ✓ | ✗ | frontier-class Llama 4 |
| llama-3.3-70b | groq | strong | 128k | ✓ | ✗ | reliable workhorse |
| deepseek-r1-distill | groq | strong | 128k | ✗ | ✗ | reasoning |
| qwen-qwq | groq | strong | 128k | ✗ | ✗ | open reasoning |
| llama-4-scout-cerebras | cerebras | fast | 131k | ✓ | ✗ | ~2000 tok/s |
| llama-3.3-70b-cerebras | cerebras | fast | 128k | ✓ | ✗ | extreme throughput |
| llama-3.3-70b-together | together | strong | 128k | ✓ | ✗ | Turbo build |
| deepseek-v3 | together | frontier | 128k | ✓ | ✗ | DeepSeek V3 MoE |
| qwen-2.5-coder-32b | together | strong | 32k | ✓ | ✗ | code-specialised |
| ollama-* | ollama | local | varies | varies | ✗ | autodetected from `http://localhost:11434` |
| lmstudio-* | lmstudio | local | varies | varies | ✗ | autodetected from `http://localhost:1234` |
| custom endpoints | custom | local | configured | ✓ | ✗ | `[[llm.custom_endpoints]]` in `.agent.toml` |

Models whose required env var (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `MISTRAL_API_KEY`, `GROQ_API_KEY`, `CEREBRAS_API_KEY`, `TOGETHER_API_KEY`) is missing are shown but marked unavailable; `agent tools list` and `/model list` will tell you which are configured right now.

## Switching Models

```bash
agent run "..." --persona strict
# inside the session:
/model claude-sonnet-4          # switch to a different model right now
/model list                     # show all configured models
/model gemini-2.5-pro           # try a 1M-context model on this task
```

The orchestrator hot-swaps the active client; the conversation context, task DAG, and audit log all stay intact. Reasoning models (`o3`, `o4-mini`, `deepseek-r1-distill`, `qwen-qwq`) are auto-wrapped by `ReasoningAdapter`, which:

- Merges the system prompt into the first user message for OpenAI's `o*` models (which reject the `system` role).
- Routes `<think>...</think>` blocks to a separate stream so they show up in the reasoning sub-panel rather than the visible answer.

## Memory Architecture

```
                ┌─────────────────────────────────────┐
                │       Working memory (RAM)          │   token-budgeted; compresses
                │       — current LLM context          │   low-priority msgs to a summary
                └────────────────┬────────────────────┘
                                 │   every tool call
                                 ▼
                ┌─────────────────────────────────────┐
                │       Episodic memory (SQLite)      │   FTS5 over tool calls;
                │       — per-session, queryable       │   scoped to the session id
                └────────────────┬────────────────────┘
                                 │   consolidation pass
                                 ▼
                ┌─────────────────────────────────────┐
                │   Semantic memory (chromadb/json)   │   facts + decisions + errors
                │   — project- or global-scoped        │   retrieved by similarity
                └────────────────┬────────────────────┘
                                 │   knowledge → procedure
                                 ▼
                ┌─────────────────────────────────────┐
                │  Procedural memory (SQLite, global) │   "how to" sequences extracted
                │  — cross-project, success-weighted   │   from successful tasks
                └─────────────────────────────────────┘
```

The consolidator (`agent/memory/consolidation.py`) runs idle-time passes:

- **Distillation** — clusters recent episodic entries (same tool + outcome) into one semantic fact.
- **Deduplication** — removes semantic entries with token-set similarity above `dedup_threshold` (default 0.9).
- **Staleness** — entries older than `stale_horizon_seconds` (default 7 days) get a `stale=True` flag and a `⚠ stale` badge in the UI; they're still retrieved but the planner knows to verify.

### Cross-project memory

```toml
[memory]
semantic_memory_scope = "global"   # share semantic facts across all projects
knowledge_store_scope = "global"
# procedural memory is always global by design.
```

```bash
agent memory search "auth refactor" --tier all      # search all four tiers
agent memory search "..." --tier procedural         # only procedures
agent memory export memory.tgz --scope global       # portable archive
agent memory import memory.tgz --scope global       # restore on a new machine
```

Imports refuse any archive entry that resolves outside the target directory.

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
