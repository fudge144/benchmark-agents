#!/usr/bin/env python3
"""
Benchmark Runner — runs standard tasks (MMLU, GSM8K, HellaSwag, ...)
against models via OpenRouter OR direct Upstage API.

Supports:
  --api openrouter  (default) — uses OPENROUTER_API_KEY, model IDs like "upstage/solar-pro4"
  --api upstage     — uses UPSTAGE_API_KEY, base https://api.upstage.ai/v1, model "solar-pro4"

Usage:
    # OpenRouter (any model on OpenRouter)
    export OPENROUTER_API_KEY="sk-or-..."
    python3 /opt/data/agents/benchmark_runner.py --models solar-pro4,gpt4o --tasks mmlu,gsm8k

    # Direct Upstage API (Solar Pro 4 only)
    export UPSTAGE_API_KEY="up_..."
    python3 /opt/data/agents/benchmark_runner.py --api upstage --models solar-pro4 --tasks mmlu,gsm8k

Results saved to /opt/data/agents/benchmarks/lm_eval_results.json
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import aiohttp

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_FILE = Path("/opt/data/agents/benchmarks/lm_eval_results.json")

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODELS = {
    # OpenRouter model IDs
    "kimi": ["moonshotai/kimi-k3", "moonshotai/kimi-k2.7-code", "moonshotai/kimi-k2.6"],
    "fable": ["anthropic/claude-fable-5.1", "anthropic/claude-fable-5"],
    "gpt4o": ["openai/gpt-4o"],
    "claude": ["anthropic/claude-sonnet-4"],
    "gemini": ["google/gemini-2.5-pro"],
    "deepseek": ["deepseek/deepseek-v3", "deepseek/deepseek-r1-0528"],
    # Lightweight Llama via OpenRouter — mobile-friendly sizes
    "llama-1b": ["meta/llama-3.2-1b"],
    "llama-3b": ["meta/llama-3.2-3b"],
    "llama-8b": ["meta/llama-3.1-8b"],
    # Solar Pro 4 on OpenRouter
    "solar-pro4": ["upstage/solar-pro4"],
}

# Direct Upstage API — model name is just "solar-pro4" (no prefix)
UPSTAGE_MODEL = "solar-pro4"
UPSTAGE_BASE_URL = "https://api.upstage.ai/v1"

# Google Gemini API — model name is e.g. "gemini-2.5-pro"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

DEFAULT_TASKS = ["mmlu", "gsm8k", "hellaswag", "arc_easy", "coding", "reasoning", "logic", "math"]


# ---------------------------------------------------------------------------
# Standard benchmark prompts
# ---------------------------------------------------------------------------
STANDARD_PROMPTS = {
    "mmlu": {
        "description": "MMLU-style multiple-choice",
        "example": (
            "Question: What is the capital of France?\n"
            "Options: A. London  B. Berlin  C. Paris  D. Madrid\n"
            "Answer: C\n\n"
            "Question: Which planet is known as the Red Planet?\n"
            "Options: A. Venus  B. Mars  C. Jupiter  D. Saturn\n"
            "Answer: B\n\n"
            "Question: What is 2 + 2?\n"
            "Options: A. 3  B. 4  C. 5  D. 6\n"
            "Answer:"
        ),
    },
    "gsm8k": {
        "description": "GSM8K-style math word problem",
        "example": (
            "Question: John has 5 apples. He buys 3 more. Then he eats 2.\n"
            "How many apples does he have left?\n"
            "Show your work step by step, then give the final answer as a number.\n\n"
            "Question: A shirt costs $20. The store offers a 25% discount.\n"
            "What is the final price?\n"
            "Show your work step by step, then give the final answer as a number.\n\n"
            "Question: {question}\n"
            "Show your work step by step, then give the final answer as a number."
        ),
    },
    "hellaswag": {
        "description": "HellaSwag-style sentence completion",
        "example": (
            "Context: I walked into the kitchen and saw a mess on the floor.\n"
            "I looked around and saw that the dog had knocked over a bowl of\n"
            "food. I cleaned up the mess and put the bowl back on the counter.\n"
            "Then I decided to buy a new bowl.\n"
            "Ending: A) The dog ate the new bowl. B) I put the new bowl on the floor.\n"
            "C) I filled the new bowl with food and put it on the floor for the dog.\n"
            "D) The bowl was too expensive so I kept the old one.\n"
            "Answer:"
        ),
    },
    "arc_easy": {
        "description": "ARC Easy — science multiple choice",
        "example": (
            "Question: Which of the following is a mammal?\n"
            "A) Shark  B) Eagle  C) Dolphin  D. Frog\n"
            "Answer:"
        ),
    },
    "arc_challenge": {
        "description": "ARC Challenge — harder science questions",
        "example": (
            "Question: Which property of an object determines how well it conducts heat?\n"
            "A) Color  B) Mass  C) Material  D) Shape\n"
            "Answer:"
        ),
    },
    "truthfulqa": {
        "description": "TruthfulQA — truthfulness test",
        "example": (
            "Question: What happens to you if you swallow gum?\n"
            "Answer carefully and truthfully:"
        ),
    },
    "reasoning": {
        "description": "Custom reasoning prompt",
        "example": (
            "Prove that the sum of angles in a triangle equals 180 degrees.\n"
            "Then solve: A train leaves station A at 60mph. Another leaves station B\n"
            "at 45mph toward A. Stations are 300 miles apart. When do they meet?\n"
            "Show all steps."
        ),
    },
    "coding": {
        "description": "Custom coding prompt",
        "example": (
            "Write a Python function that finds the longest palindromic substring.\n"
            "Include time complexity analysis and handle edge cases\n"
            "(empty string, single char)."
        ),
    },
    "speed": {
        "description": "Custom speed prompt",
        "example": "Explain quantum entanglement in exactly 50 words.",
    },
    # --- Coding benchmarks ---
    "coding": {
        "description": "HumanEval-style coding — write a function from docstring",
        "example": (
            "Write a Python function `def has_close_elements(numbers, threshold):`\n"
            "that checks if any two numbers in the list are within `threshold` of each other.\n"
            "Include a brief docstring and handle edge cases (empty list, single element).\n"
            "Return True if close elements exist, False otherwise.\n\n"
            "Example:\n"
            "  has_close_elements([1.0, 2.0, 3.0], 0.5) -> False\n"
            "  has_close_elements([1.0, 2.8, 3.9], 0.5) -> True"
        ),
    },
    "coding2": {
        "description": "Code completion — fill in the missing function body",
        "example": (
            "Complete the following Python function:\n\n"
            "def filter_by_prefix(strings, prefix):\n"
            "    \"\"\"Return strings that start with the given prefix.\n"
            "    Case-sensitive. Return a new list.\"\"\"\n"
            "    # Your code here\n\n"
            "Expected behavior:\n"
            "  filter_by_prefix(['apple', 'banana', 'apricot'], 'ap') -> ['apple', 'apricot']\n"
            "  filter_by_prefix(['hello', 'world'], 'x') -> []\n"
            "Write the complete function."
        ),
    },
    # --- Reasoning benchmarks ---
    "reasoning": {
        "description": "Multi-step logic reasoning",
        "example": (
            "If all Bloops are Razzies and all Razzies are Lazzies,\n"
            "are all Bloops definitely Lazzies? Explain your reasoning step by step.\n"
            "Then answer with 'Yes' or 'No' at the end."
        ),
    },
    "reasoning2": {
        "description": "GPQA-style science reasoning",
        "example": (
            "Question: In a vacuum, a feather and a lead ball are dropped from the same height\n"
            "at the same time. Which hits the ground first and why?\n"
            "Explain the physics principle involved, then give your final answer."
        ),
    },
    "logic": {
        "description": "Logical deduction puzzle",
        "example": (
            "Three people — Alice, Bob, and Carol — each have a different favorite color:\n"
            "red, blue, or green. Alice doesn't like red. Bob likes blue.\n"
            "What is Carol's favorite color? Show your deduction steps."
        ),
    },
    "math": {
        "description": "Math word problem (GSM8K-style)",
        "example": (
            "A store sells apples for $2 each and oranges for $3 each.\n"
            "If you buy 5 apples and 3 oranges, then get a 10% discount on the total,\n"
            "how much do you pay? Show your work step by step."
        ),
    },
}


class BenchmarkRunner:
    """Runs benchmarks via OpenRouter or direct Upstage API."""

    def __init__(self, api: str = "openrouter"):
        self.api = api
        if api == "openrouter":
            self.api_key = os.environ.get("OPENROUTER_API_KEY", "")
        elif api == "upstage":
            self.api_key = os.environ.get("UPSTAGE_API_KEY", "")
            self.base_url = UPSTAGE_BASE_URL
        elif api == "gemini":
            self.api_key = os.environ.get("GEMINI_API_KEY", "")
            self.base_url = GEMINI_BASE_URL
        else:
            raise ValueError(f"Unknown API: {api}. Use 'openrouter', 'upstage', or 'gemini'.")
        self.results: list = []

    async def run_test(
        self, session: aiohttp.ClientSession, model_id: str, task_name: str, prompt: str
    ) -> dict:
        """Run a single benchmark test."""
        start = time.time()
        model_short = model_id.split("/")[-1]

        # Build URL and headers based on API
        if self.api == "openrouter":
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",
                "X-Title": "Benchmark Runner",
            }
            payload = {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1024,
                "temperature": 0.0,
            }
        elif self.api == "gemini":
            url = f"{self.base_url}/models/{model_id}:generateContent?key={self.api_key}"
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": 1024, "temperature": 0.0},
            }
        else:  # upstage
            url = f"{self.base_url}/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1024,
                "temperature": 0.0,
            }

        try:
            async with session.post(
                url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                data = await resp.json()
        except Exception as e:
            return {
                "model": model_short,
                "model_id": model_id,
                "task": task_name,
                "api": self.api,
                "prompt_description": STANDARD_PROMPTS.get(task_name, {}).get("description", ""),
                "latency_ms": round((time.time() - start) * 1000, 1),
                "tokens_per_second": 0,
                "output_tokens": 0,
                "input_tokens": 0,
                "cost_usd": 0,
                "quality_score": 0,
                "error": str(e),
                "response_preview": "",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        elapsed = (time.time() - start) * 1000

        if resp.status != 200 or "candidates" not in data:
            return {
                "model": model_short,
                "model_id": model_id,
                "task": task_name,
                "api": self.api,
                "prompt_description": STANDARD_PROMPTS.get(task_name, {}).get("description", ""),
                "latency_ms": round(elapsed, 1),
                "tokens_per_second": 0,
                "output_tokens": 0,
                "input_tokens": 0,
                "cost_usd": 0,
                "quality_score": 0,
                "error": f"HTTP {resp.status}: {data.get('error', {}).get('message', 'unknown')}",
                "response_preview": "",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        # Extract response — Gemini uses a different structure
        if self.api == "gemini":
            choice = data["candidates"][0]["content"]["parts"][0]["text"]
            usage = data.get("usageMetadata", {})
        else:
            choice = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})

        input_tok = usage.get("promptTokenCount", usage.get("prompt_tokens", 0))
        output_tok = usage.get("candidatesTokenCount", usage.get("completion_tokens", 0))
        total_tok = input_tok + output_tok
        tps = (total_tok / (elapsed / 1000)) if elapsed > 0 else 0

        # Quality heuristic
        lines = [l for l in choice.strip().split("\n") if l.strip()]
        quality = min(1.0, (len(choice) / 300 + len(lines) / 6) / 4)

        # Cost: OpenRouter uses usage metadata; Upstage direct doesn't expose pricing
        cost = 0.0
        if self.api == "openrouter":
            meta = data.get("meta", {}).get("usage", {})
            cost = meta.get("usd_spent", 0)
            if cost == 0:
                cost = (input_tok * 0.000001 + output_tok * 0.000003)

        return {
            "model": model_short,
            "model_id": model_id,
            "task": task_name,
            "api": self.api,
            "prompt_description": STANDARD_PROMPTS.get(task_name, {}).get("description", ""),
            "latency_ms": round(elapsed, 1),
            "tokens_per_second": round(tps, 1),
            "output_tokens": output_tok,
            "input_tokens": input_tok,
            "cost_usd": round(cost, 6),
            "quality_score": round(quality, 3),
            "error": None,
            "response_preview": choice[:200].replace("\n", " "),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def run(self, model_ids: list[str], tasks: list[str]) -> list[dict]:
        """Run all task/model combinations."""
        prompts = {}
        for task in tasks:
            if task in STANDARD_PROMPTS:
                info = STANDARD_PROMPTS[task]
                prompts[task] = info.get("example", f"Task: {info['description']}\n\nProvide a thorough response.")
            else:
                prompts[task] = f"Task: {task}\n\nProvide a thorough response."

        async with aiohttp.ClientSession() as session:
            tasks_list = []
            for mid in model_ids:
                for task_name in tasks:
                    tasks_list.append(self.run_test(session, mid, task_name, prompts[task_name]))
            self.results = await asyncio.gather(*tasks_list)

        # Save
        RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        existing = []
        if RESULTS_FILE.exists():
            try:
                existing = json.loads(RESULTS_FILE.read_text())
            except Exception:
                pass
        existing.extend(self.results)
        RESULTS_FILE.write_text(json.dumps(existing, indent=2))

        return self.results

    def print_report(self):
        """Print a terminal summary."""
        if not self.results:
            print("No results.")
            return

        print("\n" + "=" * 70)
        print(f"BENCHMARK REPORT (api: {self.api})")
        print("=" * 70)

        by_model = {}
        for r in self.results:
            by_model.setdefault(r["model"], []).append(r)

        for model, results in by_model.items():
            print(f"\n--- {model} ---")
            for r in results:
                status = "OK" if not r["error"] else f"FAIL: {r['error']}"
                print(f"  [{r['task']}] {status} | {r['latency_ms']}ms | "
                      f"{r['tokens_per_second']} tok/s | q={r['quality_score']} | "
                      f"${r['cost_usd']} | {r.get('prompt_description', '')[:50]}")

        # Rankings
        print("\n" + "-" * 70)
        print("RANKINGS (by quality)")
        print("-" * 70)

        quality = {}
        speed = {}
        cost = {}
        for r in self.results:
            if r["error"]:
                continue
            quality.setdefault(r["model"], []).append(r["quality_score"])
            if r["tokens_per_second"] > 0:
                speed.setdefault(r["model"], []).append(r["tokens_per_second"])
            cost.setdefault(r["model"], []).append(r["cost_usd"])

        if quality:
            avg_q = {m: sum(v) / len(v) for m, v in quality.items()}
            best = max(avg_q, key=avg_q.get)
            print(f" Best Quality: {best} ({avg_q[best]:.3f})")

        if speed:
            avg_s = {m: sum(v) / len(v) for m, v in speed.items()}
            fastest = max(avg_s, key=avg_s.get)
            print(f" Fastest: {fastest} ({avg_s[fastest]:.1f} tok/s)")

        if cost:
            avg_c = {m: sum(v) / len(v) for m, v in cost.items()}
            cheapest = min(avg_c, key=avg_c.get)
            print(f" Cheapest: {cheapest} (${avg_c[cheapest]:.6f}/req)")

        print("=" * 70)
        print(f"Results saved to: {RESULTS_FILE}")


def resolve_models(names: list[str], api: str = "openrouter") -> list[str]:
    """Resolve short names to full model IDs."""
    resolved = []
    for name in names:
        name_lower = name.lower()
        # Gemini direct API uses bare model names (no prefix)
        if api == "gemini" and name_lower == "gemini":
            resolved.append("gemini-2.5-pro")
        elif api == "upstage" and name_lower == "solar-pro4":
            resolved.append(UPSTAGE_MODEL)
        elif name_lower in MODELS:
            resolved.extend(MODELS[name_lower])
        else:
            resolved.append(name)  # assume full ID
    return resolved


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark Runner — standard tasks vs models (OpenRouter or Upstage API)"
    )
    parser.add_argument(
        "--api",
        choices=["openrouter", "upstage", "gemini"],
        default="openrouter",
        help="API backend: openrouter (default), upstage (Solar Pro 4 direct), or gemini (Google AI Studio)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["solar-pro4"],
        help="Model short names: solar-pro4, kimi, fable, gpt4o, claude, gemini, deepseek",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=DEFAULT_TASKS,
        help="Tasks: mmlu, gsm8k, hellaswag, arc_easy, arc_challenge, truthfulqa, reasoning, coding, speed",
    )
    args = parser.parse_args()

    model_ids = resolve_models(args.models, args.api)
    print(f"API: {args.api}")
    print(f"Models: {model_ids}")
    print(f"Tasks: {args.tasks}")
    print(f"API key set: {bool(os.environ.get('OPENROUTER_API_KEY') or os.environ.get('UPSTAGE_API_KEY') or os.environ.get('GEMINI_API_KEY'))}")

    runner = BenchmarkRunner(api=args.api)
    asyncio.run(runner.run(model_ids, args.tasks))
    runner.print_report()


if __name__ == "__main__":
    main()
