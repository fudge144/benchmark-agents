#!/usr/bin/env python3
"""
Streamlit Dashboard — benchmark results for your agents.

Access on your Android phone via:
    https://<your-hermes-dashboard-url>/app/benchmarks

Or locally: streamlit run /opt/data/agents/benchmark_dashboard.py
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st

RESULTS_FILE = Path(os.environ.get("BENCHMARK_RESULTS_FILE", "lm_eval_results.json"))
RESULTS_FILE = RESULTS_FILE.resolve() if RESULTS_FILE.is_absolute() else Path.cwd() / RESULTS_FILE
legacy_results_file = Path(os.environ.get("BENCHMARK_LEGACY_FILE", "/opt/data/agents/benchmarks/results.json"))


def load_results() -> list:
    """Load all benchmark results (new lm-eval format + legacy format)."""
    all_results = []

    # New format
    if RESULTS_FILE.exists():
        try:
            all_results.extend(json.loads(RESULTS_FILE.read_text()))
        except Exception:
            pass

    # Legacy format (from old benchmark_engine.py)
    if legacy_results_file.exists():
        try:
            legacy = json.loads(legacy_results_file.read_text())
            for r in legacy:
                # Normalize legacy fields
                normalized = {
                    "model": r.get("model", "unknown"),
                    "model_id": r.get("model_id", ""),
                    "task": r.get("test", "unknown"),
                    "latency_ms": r.get("latency_ms", 0),
                    "tokens_per_second": r.get("tokens_per_second", 0),
                    "output_tokens": r.get("output_tokens", 0),
                    "input_tokens": r.get("input_tokens", 0),
                    "cost_usd": r.get("cost_usd", 0),
                    "quality_score": r.get("quality_score", 0),
                    "error": r.get("error"),
                    "response_preview": "",
                    "prompt_description": "",
                    "timestamp": datetime.fromtimestamp(r.get("timestamp", 0)).isoformat()
                    if r.get("timestamp")
                    else datetime.utcnow().isoformat(),
                }
                all_results.append(normalized)
        except Exception:
            pass

    return all_results


def format_timestamp(ts_str: str) -> str:
    """Format ISO timestamp to readable."""
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ts_str


def main():
    st.set_page_config(
        page_title="Agent Benchmarks",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("📊 Agent Benchmark Dashboard")
    st.caption("Benchmark scores for your AI agents — view on mobile or desktop")

    # ── Sidebar ──────────────────────────────────────────────────────────
    st.sidebar.header("Controls")

    all_results = load_results()

    if not all_results:
        st.sidebar.warning("No benchmark results yet. Run a benchmark first.")
        st.info("""
        ### No data yet
        Run the benchmark runner to populate results:
        ```bash
        export OPENROUTER_API_KEY="your-key"
        python3 /opt/data/agents/benchmark_runner.py --models kimi,gpt4o --tasks mmlu,gsm8k
        ```
        Then refresh this page.
        """)
        return

    # Filters
    models = sorted(set(r["model"] for r in all_results))
    tasks = sorted(set(r["task"] for r in all_results))

    selected_models = st.sidebar.multiselect("Models", models, default=models)
    selected_tasks = st.sidebar.multiselect("Tasks", tasks, default=tasks)

    # Date filter
    now = datetime.utcnow()
    days = st.sidebar.slider("History (days)", 1, 30, 7)
    cutoff = now - timedelta(days=days)

    # Apply filters
    filtered = [
        r for r in all_results
        if r["model"] in selected_models
        and r["task"] in selected_tasks
        and datetime.fromisoformat(r["timestamp"]) >= cutoff
    ]

    if not filtered:
        st.warning("No results match your filters.")
        return

    # ── KPI row ──────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    successful = [r for r in filtered if not r.get("error")]
    error_count = len(filtered) - len(successful)

    avg_quality = (
        sum(r["quality_score"] for r in successful) / len(successful)
        if successful
        else 0
    )
    avg_latency = (
        sum(r["latency_ms"] for r in successful) / len(successful)
        if successful
        else 0
    )
    avg_tps = (
        sum(r["tokens_per_second"] for r in successful if r["tokens_per_second"] > 0)
        / len([r for r in successful if r["tokens_per_second"] > 0])
        if any(r["tokens_per_second"] > 0 for r in successful)
        else 0
    )
    total_cost = sum(r["cost_usd"] for r in filtered)

    col1.metric("Models Tested", len(set(r["model"] for r in filtered)))
    col2.metric("Avg Quality", f"{avg_quality:.3f}")
    col3.metric("Avg Latency", f"{avg_latency:.0f} ms")
    col4.metric("Total Cost", f"${total_cost:.4f}")

    if error_count > 0:
        st.sidebar.error(f"{error_count} failed runs (HTTP errors)")

    # ── Rankings ─────────────────────────────────────────────────────────
    st.subheader("🏆 Rankings")

    rank_col1, rank_col2, rank_col3 = st.columns(3)

    # Quality ranking
    quality_by_model = {}
    for r in successful:
        quality_by_model.setdefault(r["model"], []).append(r["quality_score"])
    avg_quality_by_model = {
        m: sum(v) / len(v) for m, v in quality_by_model.items()
    }
    best_quality = max(avg_quality_by_model, key=avg_quality_by_model.get) if avg_quality_by_model else "—"

    # Speed ranking (only models with tps > 0)
    speed_by_model = {}
    for r in successful:
        if r["tokens_per_second"] > 0:
            speed_by_model.setdefault(r["model"], []).append(r["tokens_per_second"])
    avg_speed_by_model = {
        m: sum(v) / len(v) for m, v in speed_by_model.items()
    }
    fastest = max(avg_speed_by_model, key=avg_speed_by_model.get) if avg_speed_by_model else "—"

    # Cost ranking
    cost_by_model = {}
    for r in filtered:
        cost_by_model.setdefault(r["model"], []).append(r["cost_usd"])
    avg_cost_by_model = {m: sum(v) / len(v) for m, v in cost_by_model.items()}
    cheapest = min(avg_cost_by_model, key=avg_cost_by_model.get) if avg_cost_by_model else "—"

    with rank_col1:
        st.markdown(f"**Best Quality**")
        st.markdown(f"# {best_quality}")
        st.caption(f"Score: {avg_quality_by_model.get(best_quality, 0):.3f}")

    with rank_col2:
        st.markdown(f"**Fastest**")
        st.markdown(f"# {fastest}")
        st.caption(
            f"{avg_speed_by_model.get(fastest, 0):.1f} tok/s"
            if fastest != "—"
            else "No speed data"
        )

    with rank_col3:
        st.markdown(f"**Cheapest**")
        st.markdown(f"# {cheapest}")
        st.caption(
            f"${avg_cost_by_model.get(cheapest, 0):.6f}/req"
            if cheapest != "—"
            else "No cost data"
        )

    # ── Per-model detail ─────────────────────────────────────────────────
    st.subheader("📋 Per-Model Scores")

    for model in sorted(set(r["model"] for r in filtered)):
        model_results = [r for r in filtered if r["model"] == model]
        with st.expander(f"**{model}** — {len(model_results)} runs"):
            # Summary table
            summary_data = []
            for r in model_results:
                summary_data.append(
                    {
                        "Task": r["task"],
                        "Status": "✅" if not r.get("error") else "❌",
                        "Latency (ms)": f"{r['latency_ms']:.0f}",
                        "Tok/s": f"{r['tokens_per_second']:.1f}",
                        "Quality": f"{r['quality_score']:.3f}",
                        "Cost ($)": f"{r['cost_usd']:.6f}",
                        "When": format_timestamp(r["timestamp"]),
                    }
                )

            st.dataframe(summary_data, use_container_width=True, hide_index=True)

            # Errors
            errors = [r for r in model_results if r.get("error")]
            if errors:
                st.markdown("**Errors:**")
                for e in errors[:5]:
                    st.code(f"{e['task']}: {e['error']}", language="text")

    # ── Raw data export ──────────────────────────────────────────────────
    st.subheader("📥 Export")
    st.download_button(
        label="Download JSON",
        data=json.dumps(filtered, indent=2),
        file_name=f"benchmarks_{now.strftime('%Y%m%d')}.json",
        mime="application/json",
    )

    st.caption(f"Showing {len(filtered)} results | Data: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
