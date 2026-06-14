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

import requests

from config.settings import Settings
from grader.pipeline import _normalize_url, run_pipeline
from grader.scoring.metrics import random_baseline_precision
from grader.scoring.scorer import Scorer
from grader.storage.cache import ResultCache
from grader.storage.models import CandidateResult, PredictionStatus
from grader.validation import Severity, validate_and_standardize

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
def get_scorer():
    """Return a Scorer, or None if true labels are unavailable."""
    import tempfile, os

    # 1. Local file (dev / local run)
    try:
        return Scorer(settings.true_labels_path)
    except FileNotFoundError:
        pass

    # 2. Streamlit Cloud secret: TRUE_LABELS_CSV (raw CSV text)
    try:
        csv_text = st.secrets.get("TRUE_LABELS_CSV")
        if csv_text:
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False, encoding="utf-8"
            )
            tmp.write(csv_text)
            tmp.close()
            scorer = Scorer(tmp.name)
            os.unlink(tmp.name)
            return scorer
    except Exception:
        pass

    return None


def load_results() -> list[CandidateResult]:
    return get_cache().get_all_latest()


_DEFAULT_BASELINE = 0.2004  # 2004 churners / 10000 members

results = load_results()
scorer = get_scorer()
baseline = scorer.baseline_precision if scorer else _DEFAULT_BASELINE

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("WellCo Grader")
    st.caption(f"Auto-refreshes every {settings.refresh_interval_seconds}s")

    if scorer is not None and st.button("Run Grader", type="primary", use_container_width=True):
        with st.spinner("Fetching candidates and scoring..."):
            try:
                run_results = run_pipeline(settings)
                get_cache.clear()
                st.session_state["last_run"] = [
                    {"name": r.candidate_name, "status": r.status.value, "error": r.error}
                    for r in run_results
                ]
                st.rerun()
            except Exception as e:
                st.session_state["last_run"] = [{"name": "—", "status": "PIPELINE_ERROR", "error": str(e)}]
                st.rerun()

    if "last_run" in st.session_state:
        rows = st.session_state["last_run"]
        n_ok = sum(1 for r in rows if r["status"] == "OK")
        n_total = len(rows)
        st.caption(f"Last run: {n_ok}/{n_total} OK — see leaderboard for details")

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


def _effective_rec_n(r: CandidateResult):
    return st.session_state.rec_n_overrides.get(r.candidate_name, r.recommended_n)


def _status_label(r: CandidateResult) -> str:
    if r.error:
        return r.error
    if r.status == PredictionStatus.DEGENERATE_PREDICTIONS:
        return "All scores identical — ranking unreliable"
    if r.notes:
        return r.notes
    return ""


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
                "Precision@Rec.N": f"{p_at_rec:.3f}" if p_at_rec is not None else "—",
                "Rec. N": rec_n,
                "Status": _status_label(r),
                "_status_code": r.status.value,
            }
        )

    df = pd.DataFrame(rows)
    if show_only_ok:
        df = df[df["_status_code"] == PredictionStatus.OK.value]
    df = df.sort_values("_sort", ascending=False).drop(columns=["_sort", "_status_code"])
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

# ---------------------------------------------------------------------------
# Section 3: Overlap Analysis
# ---------------------------------------------------------------------------
st.header(f"Prediction Overlap @ N={n_slider:,}")

ok_ranked = [r for r in results if r.status == PredictionStatus.OK and r.ranked_member_ids]

if len(ok_ranked) < 2:
    if any(r.status == PredictionStatus.OK and not r.ranked_member_ids for r in results):
        st.info("Overlap data missing — re-run `python -m grader` to populate ranked predictions.")
    else:
        st.info("Need at least 2 valid candidates to show overlap.")
else:
    cand_names = [r.candidate_name for r in ok_ranked]
    selected_names = st.multiselect(
        "Candidates to compare",
        options=cand_names,
        default=cand_names,
    )
    sel = [r for r in ok_ranked if r.candidate_name in selected_names]

    if len(sel) < 2:
        st.info("Select at least 2 candidates.")
    else:
        # Build top-N sets per candidate
        top_sets = {r.candidate_name: set(r.ranked_member_ids[:n_slider]) for r in sel}
        names = [r.candidate_name for r in sel]

        col_heat, col_excl = st.columns([1, 1])

        # --- Pairwise overlap heatmap ---
        with col_heat:
            matrix = []
            text_matrix = []
            for a in names:
                row, trow = [], []
                for b in names:
                    pct = len(top_sets[a] & top_sets[b]) / max(len(top_sets[a]), 1) * 100
                    row.append(round(pct, 1))
                    trow.append(f"{pct:.1f}%")
                matrix.append(row)
                text_matrix.append(trow)

            fig_heat = go.Figure(go.Heatmap(
                z=matrix,
                x=names,
                y=names,
                colorscale="Blues",
                zmin=0,
                zmax=100,
                text=text_matrix,
                texttemplate="%{text}",
                colorbar=dict(title="% overlap"),
            ))
            fig_heat.update_layout(
                title="Pairwise overlap (row's top-N ∩ col's top-N) / N",
                height=120 + 80 * len(names),
                margin=dict(l=120, r=40, t=60, b=80),
                xaxis=dict(tickangle=-30),
            )
            st.plotly_chart(fig_heat, use_container_width=True)

        # --- Exclusivity distribution ---
        with col_excl:
            from collections import Counter
            union_members = set().union(*top_sets.values())
            counts = Counter(
                sum(1 for s in top_sets.values() if m in s)
                for m in union_members
            )
            k_vals = list(range(1, len(sel) + 1))
            y_vals = [counts.get(k, 0) for k in k_vals]
            labels = [
                "Unique to 1" if k == 1
                else f"In all {k}" if k == len(sel)
                else f"In {k}"
                for k in k_vals
            ]

            fig_excl = go.Figure(go.Bar(
                x=labels,
                y=y_vals,
                marker_color=[
                    "#d62728" if k == 1 else "#2ca02c" if k == len(sel) else "#1f77b4"
                    for k in k_vals
                ],
                text=y_vals,
                textposition="outside",
            ))
            fig_excl.update_layout(
                title="How many candidates share each member in top-N",
                xaxis_title="Shared by",
                yaxis_title="Member count",
                height=120 + 80 * len(names),
                margin=dict(l=60, r=40, t=60, b=60),
            )
            st.plotly_chart(fig_excl, use_container_width=True)

        # --- Overlap % over N line chart ---
        pairs = [
            (names[i], names[j])
            for i in range(len(names))
            for j in range(i + 1, len(names))
        ]

        # Build per-pair overlap curves once, cache by selected candidate set
        cache_key = "overlap_curves|" + "|".join(sorted(names))
        if cache_key not in st.session_state:
            ranked = {r.candidate_name: r.ranked_member_ids for r in sel}
            curves = {}
            for a_name, b_name in pairs:
                ra, rb = ranked[a_name], ranked[b_name]
                max_n = min(len(ra), len(rb))
                set_a, set_b = set(), set()
                shared = 0
                pct_curve = []
                for i in range(max_n):
                    ai, bi = ra[i], rb[i]
                    if ai == bi:
                        shared += 1
                    else:
                        if ai in set_b:
                            shared += 1
                        if bi in set_a:
                            shared += 1
                    set_a.add(ai)
                    set_b.add(bi)
                    pct_curve.append(shared / (i + 1) * 100)
                curves[(a_name, b_name)] = pct_curve
            st.session_state[cache_key] = curves
        else:
            curves = st.session_state[cache_key]

        pair_colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
            "#9467bd", "#8c564b", "#e377c2", "#17becf",
        ]
        fig_ov = go.Figure()
        for idx, (a_name, b_name) in enumerate(pairs):
            pct_curve = curves[(a_name, b_name)]
            xs = list(range(1, len(pct_curve) + 1))
            color = pair_colors[idx % len(pair_colors)]
            fig_ov.add_trace(go.Scatter(
                x=xs,
                y=pct_curve,
                mode="lines",
                name=f"{a_name} vs {b_name}",
                line=dict(color=color, width=2),
                hovertemplate=(
                    f"<b>{a_name} vs {b_name}</b><br>"
                    "N=%{x:,}<br>"
                    "Overlap=%{y:.1f}%<extra></extra>"
                ),
            ))

        fig_ov.add_vline(
            x=n_slider,
            line=dict(color="black", width=2, dash="solid"),
            annotation_text=f"N={n_slider:,}",
            annotation_position="top right",
        )
        fig_ov.update_layout(
            title="Pairwise overlap % over N",
            xaxis_title="N (outreach size)",
            yaxis_title="Overlap %",
            yaxis=dict(range=[0, 100]),
            legend=dict(orientation="v", x=1.02, y=1),
            hovermode="x unified",
            height=400,
            margin=dict(l=60, r=200, t=60, b=40),
        )
        st.plotly_chart(fig_ov, use_container_width=True)

        # --- Score distribution ---
        sel_with_scores = [r for r in sel if r.ranked_scores]
        if not sel_with_scores:
            st.info("Score distribution unavailable — re-run `python -m grader` to populate scores.")
        else:
            st.subheader("Score Distribution")
            standardize = st.checkbox(
                "Standardise scores (0–1 scale)",
                value=False,
                help="Apply min-max normalisation per candidate so all curves share the same axis.",
            )

            fig_dist = go.Figure()
            for idx, r in enumerate(sel_with_scores):
                scores = r.ranked_scores
                if standardize:
                    lo, hi = min(scores), max(scores)
                    scores = [(s - lo) / (hi - lo) if hi > lo else 0.5 for s in scores]

                color = pair_colors[idx % len(pair_colors)]
                fig_dist.add_trace(go.Histogram(
                    x=scores,
                    name=r.candidate_name,
                    opacity=0.55,
                    nbinsx=60,
                    histnorm="probability density",
                    marker_color=color,
                    hovertemplate=(
                        f"<b>{r.candidate_name}</b><br>"
                        "Score=%{x:.3f}<br>"
                        "Density=%{y:.4f}<extra></extra>"
                    ),
                ))

            fig_dist.update_layout(
                barmode="overlay",
                xaxis_title="Score (standardised 0–1)" if standardize else "Score",
                yaxis_title="Density",
                legend=dict(orientation="v", x=1.02, y=1),
                height=380,
                margin=dict(l=60, r=180, t=40, b=40),
            )
            st.plotly_chart(fig_dist, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 4: Validate a Submission
# ---------------------------------------------------------------------------
st.header("Validate a Submission")
st.caption("Paste a CSV URL to check it before (or after) adding it to the sheet.")

csv_url_input = st.text_input(
    "CSV URL",
    placeholder="https://github.com/.../predictions.csv",
    label_visibility="collapsed",
)

if st.button("Validate", type="secondary"):
    if not csv_url_input.strip():
        st.warning("Enter a URL first.")
    else:
        with st.spinner("Downloading and validating…"):
            try:
                url = _normalize_url(csv_url_input.strip())
                resp = requests.get(url, timeout=60, allow_redirects=True)
                resp.raise_for_status()
                raw = resp.content
            except Exception as exc:
                st.error(f"Could not download CSV: {exc}")
                raw = None

        if raw is not None:
            vr = validate_and_standardize(
                raw,
                true_member_ids=scorer.true_member_ids,
                min_overlap=settings.min_member_id_overlap,
            )

            if vr.ok:
                st.success(f"Validation passed — {vr.row_count:,} members, overlap {vr.overlap_pct:.1%}")
            else:
                st.error("Validation failed")

            for issue in vr.issues:
                if issue.severity == Severity.ERROR:
                    st.error(f"**{issue.code}** — {issue.message}")
                elif issue.severity == Severity.WARNING:
                    st.warning(f"**{issue.code}** — {issue.message}")
                else:
                    st.info(f"**{issue.code}** — {issue.message}")

            if vr.standardized is not None:
                with st.expander("Preview standardised data (top 10 rows)"):
                    st.dataframe(vr.standardized.head(10), use_container_width=True)
