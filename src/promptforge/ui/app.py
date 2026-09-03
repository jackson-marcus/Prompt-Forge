"""Streamlit bench: leaderboards, an A/B prompt bench, a gated prompt optimiser, history."""

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

tab_board, tab_ab, tab_opt, tab_history = st.tabs(
    ["🏆 Leaderboards", "⚖️ A/B bench", "🧗 Optimize", "🕓 History"]
)

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

with tab_opt:
    tasks = httpx.get(f"{API_URL}/tasks", timeout=30).json()["tasks"]
    task = st.selectbox("Task", list(tasks), key="opt-task")
    variants = httpx.get(f"{API_URL}/variants/{task}", timeout=30).json()["variants"]
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    name = c1.selectbox("Variant to climb from", list(variants), key="opt-variant")
    policy = c2.selectbox("Acceptance rule", ["gated", "safe", "greedy"], key="opt-policy")
    seed = c3.number_input("Split seed", min_value=0, value=0, step=1, key="opt-seed")
    commit = c4.checkbox("Version each step", value=True, key="opt-commit")
    st.code(variants[name], language="text")
    st.caption(
        "Edits are chosen on half of the task's cases and judged on the other half. "
        "`gated` needs the bootstrap CI to clear zero (the leaderboard's bar); `greedy` "
        "takes any positive dev delta - including noise."
    )
    if st.button("Climb", type="primary"):
        body = httpx.post(
            f"{API_URL}/optimize",
            json={"task": task, "name": name, "policy": policy, "seed": int(seed), "commit": commit},
            timeout=300,
        ).json()
        hold = body["holdout"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Start (held-out)", f"{hold['start']['test_pass_rate']:.0%}")
        c2.metric(
            "Final (held-out)",
            f"{hold['final']['test_pass_rate']:.0%}",
            f"{hold['ab']['delta']:+.0%} vs start",
        )
        c3.metric(
            "Dev-set optimism",
            f"{hold['optimism']:+.0%}",
            "dev minus held-out on the final prompt",
            delta_color="off",
        )
        steps = " -> ".join(s["edit"] for s in body["steps"]) or "(none)"
        st.write(f"**Accepted:** {steps}  \n**Stopped because:** {body['stop_reason']}")
        if body["committed_versions"]:
            st.success(
                f"Appended versions {body['committed_versions']} to {task}/{name} "
                f"(from v{body['from_version']}). See the History tab to diff or roll back."
            )
        ab = hold["ab"]
        st.caption(
            f"Held-out CI on the gain: [{ab['ci_low']:+.0%}, {ab['ci_high']:+.0%}] over "
            f"{ab['n_groups']} distinct inputs ({hold['n_test']} cases); "
            f"{'significant' if ab['b_wins_significant'] else 'not significant'}. "
            f"Search spent {body['ledger']['model_calls']} model calls, "
            f"${body['ledger']['cost_usd']:.4f}."
        )
        st.dataframe(pd.DataFrame(body["ledger"]["events"]), hide_index=True, use_container_width=True)
        st.code(body["final_template"], language="text")
