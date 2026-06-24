#!/usr/bin/env bash
# One-command bootstrap: venv + install + smoke checks + test suite + optional demo.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
    echo "[bootstrap] python3 not found on PATH" >&2
    exit 1
fi

VENV="$ROOT_DIR/.venv"
if [ ! -d "$VENV" ]; then
    echo "[bootstrap] creating venv at $VENV"
    "$PY" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "[bootstrap] upgrading pip"
pip install --upgrade pip >/dev/null

echo "[bootstrap] installing agent_harness (editable, dev extras)"
pip install -e ".[dev]"

echo "[bootstrap] sanity: model provider availability"
python -c "
from agent.llm.registry import ModelRegistry
r = ModelRegistry.instance()
r.print_availability_summary()
"

echo "[bootstrap] sanity: ollama / lm studio detection"
if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "  ✓ ollama running"
else
    echo "  ✗ ollama not detected (start with: ollama serve)"
fi
if curl -sf http://localhost:1234/v1/models >/dev/null 2>&1; then
    echo "  ✓ lm studio running"
else
    echo "  ✗ lm studio not detected"
fi

echo "[bootstrap] sanity: build the codebase index against this project"
python -c "
from agent.indexer import build_index, save_index
idx = build_index('.')
save_index(idx, '.')
stats = idx.stats()
print(f'  files indexed: {stats.total_files}, symbols: {stats.total_symbols}, languages: {sorted(stats.by_language.keys())}')
"

echo "[bootstrap] sanity: dependency scan against this project"
python -c "
import asyncio
from agent.tools.deps import deps_vulns
out = asyncio.run(deps_vulns('.'))
print(f'  ecosystem={out[\"ecosystem\"]} scanned={out[\"total_packages_scanned\"]} critical={out[\"critical\"]}')
"

if command -v docker >/dev/null 2>&1; then
    echo "[bootstrap] sanity: docker sandbox smoke test"
    python -c "
import asyncio
from agent.sandbox import select_sandbox_backend, SandboxConfig
async def go():
    sb = select_sandbox_backend(SandboxConfig(backend='docker', image='alpine:3', network_mode='none'))
    if sb.name != 'docker':
        print('  docker backend not selected (skipped)')
        return
    try:
        await sb.setup('.', SandboxConfig(backend='docker', image='alpine:3', network_mode='none'))
        r = await sb.exec('echo sandbox-ok', timeout=10)
        print(f'  docker exec → exit={r.exit_code} stdout={r.stdout.strip()!r}')
    finally:
        await sb.teardown()
asyncio.run(go())
" || echo "[bootstrap] docker smoke test failed (non-fatal)"
else
    echo "[bootstrap] docker not on PATH — skipping docker smoke test"
fi

echo "[bootstrap] running test suite"
pytest -q

DEMO_OBJECTIVE="${DEMO_OBJECTIVE:-Inspect the repository, summarise its layout, and write a SUMMARY.md.}"

if [ -n "${ANTHROPIC_API_KEY:-}" ] || [ -n "${OPENAI_API_KEY:-}" ]; then
    echo "[bootstrap] launching demo session"
    agent run "$DEMO_OBJECTIVE" --no-ui
else
    echo "[bootstrap] no LLM credentials in environment; skipping demo."
    echo "[bootstrap] export ANTHROPIC_API_KEY or OPENAI_API_KEY to enable 'agent run'."
fi

echo "[bootstrap] done."
