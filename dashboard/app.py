"""
WellCo Grader Dashboard

Sections:
  1. Leaderboard — sortable by precision@N (N from slider)
  2. Precision@N Chart — one line per candidate over all N
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

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
    return get_cache().get_all_latest()


results = load_results()
scorer = get_scorer()
baseline = scorer.baseline_precision

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("WellCo Grader")
    st.caption(f"Auto-refreshes every {settings.refresh_interval_seconds}s")
    st.divider()

    if "outreach_n" not in st.session_state:
        st.session_state.outreach_n = 1_000

    def _on_slider():
        st.session_state.outreach_n = st.session_state._n_slider
        st.session_state._n_input = st.session_state._n_slider

    def _on_input():
        val = max(1, min(10_000, int(st.session_state._n_input)))
        st.session_state.outreach_n = val
        st.session_state._n_slider = val

    st.slider(
        "Outreach N",
        min_value=1,
        max_value=10_000,
        value=st.session_state.outreach_n,
        step=50,
        key="_n_slider",
        on_change=_on_slider,
        help="Adjust N to see how leaderboard rankings change",
    )
    st.number_input(
        "Or type N",
        min_value=1,
        max_value=10_000,
        value=st.session_state.outreach_n,
        step=1,
        key="_n_input",
        on_change=_on_input,
    )
    n_slider = st.session_state.outreach_n

    show_baseline = st.checkbox("Show random baseline", value=True)
    show_only_ok = st.checkbox("Show only valid submissions", value=False)

    st.divider()
    if "rec_n_overrides" not in st.session_state:
        st.session_state.rec_n_overrides = {}

    with st.expander("Edit Rec. N per candidate"):
        for r in results:
            default = st.session_state.rec_n_overrides.get(
                r.candidate_name, r.recommended_n or 1_000
            )
            max_n = len(r.precision_curve) if r.precision_curve else 10_000
            st.session_state.rec_n_overrides[r.candidate_name] = st.number_input(
                r.candidate_name,
                min_value=1,
                max_value=max(max_n, 1),
                value=int(default),
                step=1,
                key=f"rec_n_{r.candidate_name}",
            )

    st.divider()
    st.metric("Total candidates", len(results))
    ok_count = sum(1 for r in results if r.status == PredictionStatus.OK)
    st.metric("Valid submissions", ok_count)
    st.metric("Churn rate (baseline)", f"{baseline:.1%}")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_STATUS_ICON = {
    PredictionStatus.OK: "🟢",
    PredictionStatus.DEGENERATE_PREDICTIONS: "🟡",
    PredictionStatus.INVALID_PREDICTIONS: "🔴",
    PredictionStatus.SCHEMA_ERROR: "🔴",
    PredictionStatus.CSV_DOWNLOAD_ERROR: "⚫",
}


def _effective_rec_n(r: CandidateResult) -> int | None:
    return st.session_state.rec_n_overrides.get(r.candidate_name, r.recommended_n)


def _build_leaderboard(results: list[CandidateResult], n: int) -> pd.DataFrame:
    rows = []
    for r in results:
        p_at_n = r.precision_at_n(n)
        rec_n = _effective_rec_n(r)
        p_at_rec = r.precision_at_n(rec_n) if rec_n is not None else None
        rows.append(
            {
                "": _STATUS_ICON.get(r.status, "❓"),
                "Candidate": r.candidate_name,
                f"Precision@{n:,}": f"{p_at_n:.3f}" if p_at_n is not None else "—",
                "_sort": p_at_n if p_at_n is not None else -1,
                f"Precision@Rec.N": f"{p_at_rec:.3f}" if p_at_rec is not None else "—",
                "Rec. N": rec_n,
                "Status": r.status.value,
            }
        )

    df = pd.DataFrame(rows)
    if show_only_ok:
        df = df[df["Status"] == PredictionStatus.OK.value]
    df = df.sort_values("_sort", ascending=False).drop(columns=["_sort"])
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
            "Rec. N": st.column_config.NumberColumn("Rec. N", width="small"),
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
        rec_n = _effective_rec_n(r)
        if rec_n and 1 <= rec_n <= len(curve):
            fig.add_vline(
                x=rec_n,
                line=dict(color=color, width=1, dash="dot"),
                opacity=0.5,
                annotation_text=f"{r.candidate_name} N={rec_n:,}",
                annotation_position="top left",
                annotation_font_size=10,
            )

    if show_baseline:
        fig.add_hline(
            y=baseline,
            line=dict(color="gray", width=1, dash="dash"),
            annotation_text=f"Random baseline ({baseline:.1%})",
            annotation_position="bottom right",
        )

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
        margin=dict(l=40, r=200, t=40, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)
