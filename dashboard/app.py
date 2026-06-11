"""
WellCo Grader Dashboard

Sections:
  1. Leaderboard — sortable by precision@N (N from slider)
  2. Precision@N Chart — one line per candidate over all N
  3. Code Review Viewer — per-question scores and justifications
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# Allow running from repo root: streamlit run dashboard/app.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import Settings
from grader.scoring.metrics import random_baseline_precision
from grader.scoring.scorer import Scorer
from grader.storage.cache import ResultCache
from grader.storage.models import CandidateResult, PredictionStatus

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="WellCo Grader",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Auto-refresh
# ---------------------------------------------------------------------------
settings = Settings()
st_autorefresh(interval=settings.refresh_interval_seconds * 1000, key="autorefresh")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
@st.cache_resource
def get_cache() -> ResultCache:
    return ResultCache(settings.cache_db_path)


@st.cache_resource
def get_scorer() -> Scorer:
    return Scorer(settings.true_labels_path)


def load_results() -> list[CandidateResult]:
    cache = get_cache()
    return cache.get_all_latest()


results = load_results()
scorer = get_scorer()
baseline = scorer.baseline_precision

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("WellCo Grader")
    st.caption("Auto-refreshes every 60 seconds")
    st.divider()

    n_slider = st.slider(
        "Outreach N",
        min_value=1,
        max_value=10_000,
        value=1_000,
        step=50,
        help="Adjust N to see how leaderboard rankings change",
    )

    show_baseline = st.checkbox("Show random baseline", value=True)
    show_only_ok = st.checkbox("Show only valid submissions", value=False)

    st.divider()
    st.metric("Total candidates", len(results))
    ok_count = sum(1 for r in results if r.status == PredictionStatus.OK)
    st.metric("Valid submissions", ok_count)
    st.metric("Churn rate (baseline)", f"{baseline:.1%}")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_STATUS_COLOR = {
    PredictionStatus.OK: "🟢",
    PredictionStatus.DEGENERATE_PREDICTIONS: "🟡",
    PredictionStatus.MISSING_PREDICTIONS: "🔴",
    PredictionStatus.SCHEMA_ERROR: "🔴",
    PredictionStatus.INVALID_PREDICTIONS: "🔴",
    PredictionStatus.REPO_UNAVAILABLE: "⚫",
    PredictionStatus.GITHUB_ERROR: "⚫",
}

_N_SOURCE_LABELS = {
    "csv_row_count": "CSV rows",
    "csv_explicit_column": "CSV column",
    "readme": "README",
    "code": "Code",
    "pdf": "PDF",
    "inferred": "Inferred ⚠️",
}


def _build_leaderboard(results: list[CandidateResult], n: int) -> pd.DataFrame:
    rows = []
    for r in results:
        p_at_n = r.precision_at_n(n)
        rows.append(
            {
                "": _STATUS_COLOR.get(r.status, "❓"),
                "Candidate": r.candidate_name,
                "Precision@N": f"{p_at_n:.3f}" if p_at_n is not None else "—",
                "Precision@N_raw": p_at_n if p_at_n is not None else -1,
                "Review Score": f"{r.review_result.weighted_score:.2f}" if r.review_result else "—",
                "Rec. N": str(r.recommended_n) if r.recommended_n else "—",
                "N Source": _N_SOURCE_LABELS.get(r.n_extraction.source.value, "—") if r.n_extraction else "—",
                "N ⚠️": "⚠️" if r.n_extraction and r.n_extraction.n_warning else "",
                "Status": r.status.value,
            }
        )

    df = pd.DataFrame(rows)
    if show_only_ok:
        df = df[df["Status"] == PredictionStatus.OK.value]
    df = df.sort_values("Precision@N_raw", ascending=False).drop(columns=["Precision@N_raw"])
    df = df.reset_index(drop=True)
    df.index += 1
    return df


# ---------------------------------------------------------------------------
# Section 1: Leaderboard
# ---------------------------------------------------------------------------
st.header(f"Leaderboard — Precision @ N={n_slider:,}")

if not results:
    st.info("No results yet. Run `python -m grader` to process candidates.")
else:
    leaderboard_df = _build_leaderboard(results, n_slider)
    st.dataframe(
        leaderboard_df,
        use_container_width=True,
        hide_index=False,
        column_config={
            "": st.column_config.TextColumn("", width="small"),
            "Candidate": st.column_config.TextColumn("Candidate", width="medium"),
            "Precision@N": st.column_config.TextColumn(f"Precision@{n_slider}", width="small"),
            "Review Score": st.column_config.TextColumn("Review Score", width="small"),
        },
    )

# ---------------------------------------------------------------------------
# Section 2: Precision@N Chart
# ---------------------------------------------------------------------------
st.header("Precision@N over N")

valid_results = [r for r in results if r.precision_curve]

if not valid_results:
    st.info("No scored submissions yet.")
else:
    fig = go.Figure()

    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]

    for i, r in enumerate(valid_results):
        curve = r.precision_curve
        xs = list(range(1, len(curve) + 1))
        color = colors[i % len(colors)]

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=curve,
                mode="lines",
                name=r.candidate_name,
                line=dict(color=color, width=2),
                hovertemplate=(
                    f"<b>{r.candidate_name}</b><br>"
                    "N=%{x:,}<br>"
                    "Precision=%{y:.3f}<extra></extra>"
                ),
            )
        )

        # Mark recommended N
        if r.recommended_n and 1 <= r.recommended_n <= len(curve):
            fig.add_vline(
                x=r.recommended_n,
                line=dict(color=color, width=1, dash="dot"),
                opacity=0.5,
            )

    # Baseline
    if show_baseline:
        fig.add_hline(
            y=baseline,
            line=dict(color="gray", width=1, dash="dash"),
            annotation_text=f"Random baseline ({baseline:.1%})",
            annotation_position="bottom right",
        )

    # Mark the slider N
    fig.add_vline(
        x=n_slider,
        line=dict(color="black", width=2, dash="solid"),
        annotation_text=f"N={n_slider:,}",
        annotation_position="top right",
    )

    fig.update_layout(
        xaxis_title="N (outreach size)",
        yaxis_title="Precision@N",
        yaxis=dict(range=[0, max(baseline * 1.5, 0.5)]),
        legend=dict(orientation="v", x=1.02, y=1),
        hovermode="x unified",
        height=450,
        margin=dict(l=40, r=150, t=40, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 3: Code Review Viewer
# ---------------------------------------------------------------------------
st.header("Code Review")

reviewed = [r for r in results if r.review_result]

if not reviewed:
    st.info("No reviews available yet.")
else:
    candidate_names = [r.candidate_name for r in reviewed]
    selected_name = st.selectbox("Select candidate", options=candidate_names)

    selected = next((r for r in reviewed if r.candidate_name == selected_name), None)
    if selected and selected.review_result:
        review = selected.review_result

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Overall Review Score", f"{review.weighted_score:.0%}")
        with col2:
            p = selected.precision_at_recommended_n
            st.metric(
                f"Precision @ N={selected.recommended_n or '?'}",
                f"{p:.3f}" if p is not None else "—",
            )
        with col3:
            st.metric("Submission Status", selected.status.value)

        st.divider()

        for q in review.questions:
            score_display = ["🔴 0/2", "🟡 1/2", "🟢 2/2"][q.score]
            with st.expander(
                f"{score_display} — **{q.id}** (weight {q.weight:.0f})",
                expanded=q.score < 2,
            ):
                st.markdown(f"**Score:** {q.score}/2 &nbsp; **Weight:** {q.weight}")
                st.markdown(f"**Justification:** {q.justification}")
