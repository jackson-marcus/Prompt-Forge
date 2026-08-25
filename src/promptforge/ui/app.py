"""Streamlit demo: variant leaderboards + an interactive A/B prompt bench."""

from __future__ import annotations

import os

import httpx
import pandas as pd
import streamlit as st

API_URL = os.environ.get("PROMPTFORGE_API_URL", "http://localhost:8460")

st.set_page_config(page_title="promptforge", page_icon="🛠️", layout="wide")
st.title("🛠️ promptforge")
st.caption(
    "Prompt workbench: versioned registry, A/B evals with bootstrap CIs, regression gates, cost"
)


def _ok() -> bool:
    try:
        return httpx.get(f"{API_URL}/health", timeout=3).status_code == 200
    except httpx.HTTPError:
        return False


if not _ok():
    st.error(f"API not reachable at {API_URL}. Start it with `make api`.")
    st.stop()

tab_board, tab_ab, tab_history = st.tabs(["🏆 Leaderboards", "⚖️ A/B bench", "🕓 History"])

with tab_board:
    tasks = httpx.get(f"{API_URL}/tasks", timeout=30).json()["tasks"]
    task = st.selectbox("Task", list(tasks))
    board = httpx.get(f"{API_URL}/leaderboard/{task}", timeout=30).json()
    st.caption(f"Baseline variant: **{board['baseline']}**")
    df = pd.DataFrame(board["leaderboard"])
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.caption(
        "delta_vs_baseline is the paired-bootstrap pass-rate difference; beats_baseline "
        "means its 95% CI clears zero. Engineered prompts (format + few-shot + constraint) "
        "win — the harness proves it, it isn't asserted."
    )

with tab_ab:
    tasks = httpx.get(f"{API_URL}/tasks", timeout=30).json()["tasks"]
    task = st.selectbox("Task", list(tasks), key="ab-task")
    variants = httpx.get(f"{API_URL}/variants/{task}", timeout=30).json()["variants"]
    names = list(variants)
    c1, c2 = st.columns(2)
    a = c1.text_area("Prompt A", variants[names[0]], height=160)
    b = c2.text_area("Prompt B", variants[names[-1]], height=160)
    if st.button("Run A/B", type="primary"):
        body = httpx.post(
            f"{API_URL}/ab", json={"task": task, "template_a": a, "template_b": b}, timeout=120
        ).json()
        c1, c2, c3 = st.columns(3)
        c1.metric("A pass rate", f"{body['a']['pass_rate']:.0%}", f"${body['a']['cost_usd']:.4f}")
        c2.metric("B pass rate", f"{body['b']['pass_rate']:.0%}", f"${body['b']['cost_usd']:.4f}")
        ab = body["ab"]
        verdict = (
            "B wins"
            if ab["b_wins_significant"]
            else "A wins"
            if ab["a_wins_significant"]
            else "no significant difference"
        )
        c3.metric("Delta (B - A)", f"{ab['delta']:+.0%}", verdict)
        st.caption(f"95% CI on the difference: [{ab['ci_low']:+.0%}, {ab['ci_high']:+.0%}]")
        reg = body["regression"]
        if reg["passes_gate"]:
            st.success(
                f"Regression gate PASSED (drop {reg['regression']:.0%} ≤ {reg['tolerance']:.0%})"
            )
        else:
            st.error(
                f"Regression gate FAILED (drop {reg['regression']:.0%} > {reg['tolerance']:.0%})"
            )

with tab_history:
    tasks = httpx.get(f"{API_URL}/tasks", timeout=30).json()["tasks"]
    task = st.selectbox("Task", list(tasks), key="hist-task")
    history = httpx.get(f"{API_URL}/variants/{task}/history", timeout=30).json()["history"]
    name = st.selectbox("Variant", list(history), key="hist-variant")
    lineage = history[name]
    st.dataframe(
        pd.DataFrame(lineage)[["version", "parent_version", "created_by", "content_hash"]],
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "History is append-only: restoring an old prompt appends a new version whose "
        "parent is the restored one, so a rollback never erases what caused it."
    )
    versions = [s["version"] for s in lineage]
    if len(versions) > 1:
        c1, c2 = st.columns(2)
        a = c1.selectbox("Base version", versions, index=0, key="hist-a")
        b = c2.selectbox("Target version", versions, index=len(versions) - 1, key="hist-b")
        diff = httpx.get(
            f"{API_URL}/variants/{task}/diff",
            params={"name": name, "a": a, "b": b},
            timeout=30,
        ).json()
        if diff["identical"]:
            st.info("Identical content — same hash, nothing to diff.")
        else:
            st.code(diff["diff"], language="diff")
    else:
        st.info("Only one version so far. Save an edited template to build a lineage.")
