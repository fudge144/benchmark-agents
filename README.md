# Benchmark Agents

Automatically benchmarks AI models on standard tasks and tracks their performance over time.

## Models Supported

| Model | API | Short Name |
|-------|-----|------------|
| Solar Pro 4 | OpenRouter | `solar-pro4` |
| Solar Pro 4 | Upstage direct | `solar-pro4` (with `--api upstage`) |
| Kimi | OpenRouter | `kimi` |
| Claude Fable | OpenRouter | `fable` |
| GPT-4o | OpenRouter | `gpt4o` |
| Claude Sonnet | OpenRouter | `claude` |
| Gemini 2.5 Pro | Google AI Studio | `gemini` (with `--api gemini`) |
| DeepSeek V3/R1 | OpenRouter | `deepseek` |
| Llama 3.2 1B | OpenRouter | `llama-1b` |
| Llama 3.2 3B | OpenRouter | `llama-3b` |
| Llama 3.1 8B | OpenRouter | `llama-8b` |

## Benchmark Tasks

| Task | Description |
|------|-------------|
| `mmlu` | MMLU-style multiple choice |
| `gsm8k` | GSM8K-style math word problems |
| `hellaswag` | Sentence completion |
| `arc_easy` | ARC Easy science questions |
| `arc_challenge` | ARC Challenge science questions |
| `coding` | HumanEval-style coding |
| `coding2` | Code completion |
| `reasoning` | Multi-step logic |
| `reasoning2` | GPQA-style science |
| `logic` | Deduction puzzles |
| `math` | Math word problems |

## Usage

### Local
```bash
# Solar Pro 4 via OpenRouter
export OPENROUTER_API_KEY="sk-or-..."
python benchmark_runner.py --models solar-pro4 --tasks mmlu,coding,reasoning

# Gemini via Google AI Studio
export GEMINI_API_KEY="AIza..."
python benchmark_runner.py --api gemini --models gemini --tasks mmlu,coding

# Solar Pro 4 direct via Upstage
export UPSTAGE_API_KEY="up_..."
python benchmark_runner.py --api upstage --models solar-pro4 --tasks mmlu,coding
```

### Mobile Trigger API
```bash
# Start the API server
python mobile_trigger_api.py
# Then from your phone:
curl -X POST http://<your-server>:8766/trigger \
  -H "Content-Type: application/json" \
  -d '{"api": "openrouter", "models": ["solar-pro4", "llama-3b"], "tasks": ["mmlu", "coding"]}'
```

### GitHub Actions
The workflow runs automatically daily at 03:00 UTC. To run manually:
1. Go to Actions → Benchmark Agents → Run workflow
2. Select models, tasks, and API backend
3. Click "Run workflow"

## Setup

### GitHub Secrets
Add these in your repo settings → Secrets → Actions:

- `OPENROUTER_API_KEY` — your OpenRouter key (for Solar Pro 4, Kimi, etc.)
- `UPSTAGE_API_KEY` — your Upstage key (for direct Solar Pro 4 access)
- `GEMINI_API_KEY` — your Google AI Studio key (for Gemini)

### Local Dashboard
```bash
streamlit run benchmark_dashboard.py --server.port 8765 --server.address 0.0.0.0
```

## License

MIT
