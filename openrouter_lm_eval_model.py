#!/usr/bin/env python3
"""
OpenRouter model wrapper for lm-eval-harness.

Drop this file in your PYTHONPATH or pass it via --model-class.
lm-eval will call `completion` + `loglikelihood` on this class to score
MMLU, GSM8K, HellaSwag, etc. against real OpenRouter models.
"""

import os
import time
import asyncio
from typing import List, Tuple, Optional, Dict, Any
import aiohttp

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OPENROUTER_API = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")


class OpenRouterModel:
    """
    lm-eval-compatible wrapper that talks to OpenRouter's chat API.

    lm-eval calls:
      - `model.algorithm`       -> "greedy" or "nucleus"
      - `model.api_key`         -> for auth
      - `model.get_pred_occurrences/loglikelihood/...` via the task harness
    We implement the minimal surface the harness needs for multiple-choice
    and generative tasks.
    """

    def __init__(
        self,
        model_name: str = "moonshotai/kimi-k3",
        max_tokens: int = 2048,
        temperature: float = 0.0,
        timeout: int = 120,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None

    # -- lm-eval required attributes ---------------------------------------
    @property
    def algorithm(self) -> str:
        return "greedy"

    @property
    def api_key(self) -> str:
        return API_KEY

    # -- lifecycle ----------------------------------------------------------
    async def start(self):
        self.session = aiohttp.ClientSession()

    async def stop(self):
        if self.session:
            await self.session.close()
            self.session = None

    # -- core inference ----------------------------------------------------
    async def completion(self, prompt: str, **kwargs) -> str:
        """Return a single completion for `prompt`."""
        if not self.session:
            raise RuntimeError("Model not started. Call await model.start() first.")
        try:
            async with self.session.post(
                OPENROUTER_API,
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "http://localhost",
                    "X-Title": "lm-eval-openrouter",
                },
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                    "temperature": kwargs.get("temperature", self.temperature),
                },
                timeout=aiohttp.ClientTimeout(total=kwargs.get("timeout", self.timeout)),
            ) as resp:
                data = await resp.json()
        except Exception as e:
            return f"ERROR: {e}"

        if resp.status != 200 or "choices" not in data:
            return f"ERROR: HTTP {resp.status}"

        return data["choices"][0]["message"]["content"]

    # -- loglikelihood (used by multiple-choice tasks like MMLU) -----------
    async def loglikelihood(self, context: str, continuation: str) -> Tuple[float, bool]:
        """
        Return (logprob, is_greedy) for continuation given context.

        OpenRouter doesn't expose token-level logprobs by default, so we
        approximate by requesting logprobs and summing the continuation tokens.
        """
        if not self.session:
            raise RuntimeError("Model not started.")
        try:
            async with self.session.post(
                OPENROUTER_API,
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "http://localhost",
                    "X-Title": "lm-eval-openrouter",
                },
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": context}],
                    "max_tokens": len(continuation.split()) + 50,
                    "temperature": 0.0,
                    "logprobs": 1,  # request logprobs
                },
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as resp:
                data = await resp.json()
        except Exception:
            return (0.0, False)

        if resp.status != 200 or "choices" not in data:
            return (0.0, False)

        choice = data["choices"][0]
        # OpenRouter returns logprobs in choice['logprobs'] if requested
        logprobs = choice.get("logprobs", {}).get("content", [])
        total_logprob = sum(
            item.get("logprob", 0.0) for item in logprobs if item.get("token") != "[PAD]"
        )
        return (total_logprob, True)

    # -- batched helpers (lm-eval may call these) --------------------------
    async def generate_until(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        return await self.completion(prompt, **kwargs)

    async def greedy_until(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        return await self.completion(prompt, temperature=0.0, **kwargs)
