#!/usr/bin/env bash
# One-command bootstrap: venv + install + test + demo.
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
