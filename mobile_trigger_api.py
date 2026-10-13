#!/usr/bin/env python3
"""
Mobile Benchmark Trigger API — lightweight FastAPI server.

Allows your Android phone to trigger benchmark runs via HTTP.

Usage:
    # Start the server (alongside the dashboard)
    source /opt/data/agents-venv/bin/activate
    uvicorn mobile_trigger_api:app --host 0.0.0.0 --port 8766

    # Trigger a benchmark from your phone/browser:
    curl -X POST http://localhost:8766/trigger \
      -H "Content-Type: application/json" \
      -d '{"api": "openrouter", "models": ["solar-pro4", "llama-3b"], "tasks": ["mmlu", "coding", "reasoning"]}'

    # Check status:
    curl http://localhost:8766/status

    # Get last results:
    curl http://localhost:8766/results
"""

import asyncio
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_FILE = Path("/opt/data/agents/benchmarks/lm_eval_results.json")
RUNNER_SCRIPT = "/opt/data/agents/benchmark_runner.py"
VENV_PYTHON = "/opt/data/agents-venv/bin/python3"

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
last_job: Optional[dict] = None
job_history: list = []


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class TriggerRequest(BaseModel):
    api: str = "openrouter"
    models: list[str] = Field(default=["solar-pro4"])
    tasks: list[str] = Field(default=["mmlu", "coding", "reasoning"])


class TriggerResponse(BaseModel):
    job_id: str
    status: str
    message: str
    estimated_duration_seconds: int = 60


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Mobile Benchmark Trigger",
    description="Trigger AI model benchmarks from your phone",
    version="1.0.0",
)


def run_benchmark_sync(api: str, models: list[str], tasks: list[str]) -> dict:
    """Run benchmark in a subprocess and return results summary."""
    env = os.environ.copy()
    if api == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY", "")
        if key:
            env["OPENROUTER_API_KEY"] = key
    elif api == "upstage":
        key = os.environ.get("UPSTAGE_API_KEY", "")
        if key:
            env["UPSTAGE_API_KEY"] = key
    elif api == "gemini":
        key = os.environ.get("GEMINI_API_KEY", "")
        if key:
            env["GEMINI_API_KEY"] = key

    models_str = " ".join(models)
    tasks_str = " ".join(tasks)

    cmd = [
        VENV_PYTHON,
        RUNNER_SCRIPT,
        "--api", api,
        "--models", models_str,
        "--tasks", tasks_str,
    ]

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )
        elapsed = time.time() - start
        success = result.returncode == 0
        output = result.stdout[-2000:] if result.stdout else ""
        error = result.stderr[-500:] if result.stderr and not success else ""
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        success = False
        output = ""
        error = "Benchmark timed out after 300s"

    # Load results from file
    results_summary = []
    if RESULTS_FILE.exists():
        try:
            all_results = json.loads(RESULTS_FILE.read_text())
            # Get the most recent batch (last N entries matching this run's models)
            recent = [r for r in all_results if r.get("model") in models]
            results_summary = recent[-20:] if recent else []
        except Exception:
            pass

    return {
        "success": success,
        "elapsed_seconds": round(elapsed, 1),
        "output_preview": output,
        "error": error,
        "results_count": len(results_summary),
        "results_summary": results_summary,
    }


@app.post("/trigger", response_model=TriggerResponse)
async def trigger_benchmark(req: TriggerRequest):
    """Trigger a benchmark run from your mobile device."""
    global last_job

    job_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")[:-3]
    estimated = len(req.models) * len(req.tasks) * 30  # rough estimate

    # Run in background thread to not block the HTTP response
    def _run():
        global last_job
        result = run_benchmark_sync(req.api, req.models, req.tasks)
        last_job = {
            "job_id": job_id,
            "api": req.api,
            "models": req.models,
            "tasks": req.tasks,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            **result,
        }
        job_history.append(last_job)
        # Keep only last 50 jobs
        if len(job_history) > 50:
            job_history.pop(0)

    asyncio.get_event_loop().run_in_executor(None, _run)

    return TriggerResponse(
        job_id=job_id,
        status="started",
        message=f"Benchmark running: {req.api} / {', '.join(req.models)} / {', '.join(req.tasks)}",
        estimated_duration_seconds=estimated,
    )


@app.get("/status")
async def get_status():
    """Check the status of the last benchmark job."""
    if last_job is None:
        return {
            "status": "idle",
            "message": "No benchmark has been run yet. POST to /trigger to start one.",
        }
    return {
        "status": "completed" if last_job.get("success") else "failed",
        "job_id": last_job.get("job_id"),
        "api": last_job.get("api"),
        "models": last_job.get("models"),
        "tasks": last_job.get("tasks"),
        "elapsed_seconds": last_job.get("elapsed_seconds"),
        "results_count": last_job.get("results_count"),
        "finished_at": last_job.get("finished_at"),
        "error": last_job.get("error"),
    }


@app.get("/results")
async def get_results():
    """Get the most recent benchmark results."""
    if not RESULTS_FILE.exists():
        raise HTTPException(status_code=404, detail="No results file found")

    try:
        all_results = json.loads(RESULTS_FILE.read_text())
        # Return last 50
        return {
            "total_results": len(all_results),
            "recent": all_results[-50:],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8766, log_level="info")
