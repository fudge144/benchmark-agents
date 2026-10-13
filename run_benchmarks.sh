#!/bin/bash
# Benchmark cron wrapper — runs benchmark_runner.py with the right API
# API key is loaded from /opt/data/.openrouter_key (OpenRouter),
# /opt/data/.upstage_key (direct Upstage API for Solar Pro 4), or
# /opt/data/.gemini_key (Google AI Studio for Gemini)

set -e

AGENTS_VENV="/opt/data/agents-venv"
RUNNER="/opt/data/agents/benchmark_runner.py"

# Determine which API to use
# Default: OpenRouter. Set BENCHMARK_API=upstage for Upstage, or =gemini for Gemini.
API="${BENCHMARK_API:-openrouter}"

# Load API key
if [ "$API" = "upstage" ]; then
    KEY_FILE="/opt/data/.upstage_key"
    if [ -f "$KEY_FILE" ]; then
        export UPSTAGE_API_KEY="$(cat "$KEY_FILE" | tr -d '[:space:]')"
    fi
    echo "Using Upstage API key" >&2
elif [ "$API" = "gemini" ]; then
    KEY_FILE="/opt/data/.gemini_key"
    if [ -f "$KEY_FILE" ]; then
        export GEMINI_API_KEY="$(cat "$KEY_FILE" | tr -d '[:space:]')"
    fi
    echo "Using Gemini API key" >&2
else
    KEY_FILE="/opt/data/.openrouter_key"
    if [ -f "$KEY_FILE" ]; then
        export OPENROUTER_API_KEY="$(cat "$KEY_FILE" | tr -d '[:space:]')"
    fi
    echo "Using OpenRouter API key" >&2
fi

# Activate venv and run
source "$AGENTS_VENV/bin/activate"
exec python3 "$RUNNER" --api "$API" --models solar-pro4 --tasks mmlu,gsm8k,hellaswag,arc_easy
