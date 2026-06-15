"""
WellCo Grader Dashboard

Sections:
  1. Leaderboard — sortable by selected metric@N (N from slider)
  2. Metric Chart  — one line per candidate over all N (metric selected in sidebar)
  3. Candidate Overlap
  4. Validate a Submission
"""
import sys
from pathlib import Path
from typing import Optional

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
# On Streamlit Cloud, GOOGLE_SHEET_ID may not be in env vars — fall back to st.secrets.
if not settings.google_sheet_id:
    try:
        _sid = st.secrets.get("GOOGLE_SHEET_ID")
        if _sid:
            settings.google_sheet_id = _sid
    except Exception:
        pass
st_autorefresh(interval=settings.refresh_interval_seconds * 1000, key="autorefresh")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
@st.cache_resource
def get_cache() -> ResultCache:
    return ResultCache(settings.cache_db_path)


@st.cache_resource
def get_scorer():
    """Return a (Scorer, error_msg) tuple. error_msg is None on success."""
    import base64, os, tempfile

    # 1. Local file
    try:
        return Scorer(settings.true_labels_path), None
    except FileNotFoundError:
        pass

    # 2. Streamlit Cloud secret: TRUE_LABELS_CSV_B64 (base64-encoded CSV bytes)
    try:
        b64 = st.secrets.get("TRUE_LABELS_CSV_B64")
        if b64:
            raw = base64.b64decode(b64)
            tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
            tmp.write(raw)
            tmp.close()
            scorer = Scorer(Path(tmp.name))
            os.unlink(tmp.name)
            return scorer, None
    except Exception as e:
        return None, f"Secret TRUE_LABELS_CSV_B64 found but failed to load: {e}"

    # 3. Streamlit Cloud secret: TRUE_LABELS_CSV (raw CSV text)
    try:
        csv_text = st.secrets.get("TRUE_LABELS_CSV")
        if csv_text:
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False, encoding="utf-8"
            )
            tmp.write(csv_text)
            tmp.close()
            scorer = Scorer(Path(tmp.name))
            os.unlink(tmp.name)
            return scorer, None
    except Exception as e:
        return None, f"Secret TRUE_LABELS_CSV found but failed to load: {e}"

    return None, None


def load_results() -> list[CandidateResult]:
    return get_cache().get_all_latest()


_DEFAULT_BASELINE = 0.2004  # 2004 churners / 10000 members

results = load_results()
scorer, _scorer_error = get_scorer()
baseline = scorer.baseline_precision if scorer else _DEFAULT_BASELINE

# Fill in gain/lift/qini curves for results cached before these metrics existed.
# Done inline here to avoid calling new scorer methods that may not exist on
# older cached scorer versions on Streamlit Cloud.
if scorer:
    _churn_rate = getattr(scorer, '_churn_rate', _DEFAULT_BASELINE)
    _churner_ids = getattr(scorer, '_true_churner_ids', set())
    _total_churners = len(_churner_ids) or int(_DEFAULT_BASELINE * 10_000)
    _qini_data = getattr(scorer, '_qini_data', None)

    for r in results:
        _prec = getattr(r, 'precision_curve', None)
        if _prec is None:
            continue
        _n = len(_prec)
        try:
            if getattr(r, 'gain_curve', None) is None:
                r.gain_curve = [_prec[i] * (i + 1) / _total_churners for i in range(_n)]
        except Exception:
            pass
        try:
            if getattr(r, 'lift_curve', None) is None and _churn_rate > 0:
                r.lift_curve = [p / _churn_rate for p in _prec]
        except Exception:
            pass
        try:
            from grader.scoring.metrics import qini_curve as _qc
            _ids = getattr(r, 'ranked_member_ids', None)
            if getattr(r, 'qini_curve', None) is None and _qini_data and _ids:
                r.qini_curve = _qc(_ids, *_qini_data)
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("WellCo Grader")
    st.caption(f"Auto-refreshes every {settings.refresh_interval_seconds}s")

    if _scorer_error:
        st.error(_scorer_error)
    elif scorer is None:
        st.warning("Scorer not loaded — add `TRUE_LABELS_CSV_B64` to Streamlit secrets to enable grading.")
        if st.button("Retry loading scorer", use_container_width=True):
            get_scorer.clear()
            st.rerun()
    if scorer is not None:
        if st.button("Run Grader", type="primary", use_container_width=True,
                     help="Score new/changed submissions (cached results are reused)"):
            with st.spinner("Fetching candidates and scoring..."):
                try:
                    run_results = run_pipeline(settings, scorer=scorer)
                    get_cache.clear()
                    st.session_state["last_run"] = [
                        {"name": r.candidate_name, "status": r.status.value, "error": r.error}
                        for r in run_results
                    ]
                    st.rerun()
                except Exception as e:
                    st.session_state["last_run"] = [{"name": "—", "status": "PIPELINE_ERROR", "error": str(e)}]
                    st.rerun()

        if st.button("Re-grade All", type="secondary", use_container_width=True,
                     help="Clear cache and re-score every candidate from scratch"):
            with st.spinner("Clearing cache and re-grading all candidates..."):
                try:
                    get_cache().clear_all()
                    run_results = run_pipeline(settings, scorer=scorer)
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
        if n_total == 0:
            st.warning("Last run: no candidates found — check GOOGLE_SHEET_ID secret.")
        elif n_ok == n_total:
            st.success(f"Last run: {n_ok}/{n_total} OK")
        else:
            st.warning(f"Last run: {n_ok}/{n_total} OK")
        with st.expander("Run details"):
            for row in rows:
                icon = "✅" if row["status"] == "OK" else "❌"
                st.write(f"{icon} **{row['name']}** — {row['status']}")
                if row.get("error"):
                    st.caption(row["error"])

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

    st.divider()

    metric_choice = st.selectbox(
        "Leaderboard & chart metric",
        options=["Precision", "Gain", "Lift", "Qini"],
        index=0,
        help=(
            "Precision@N — fraction of top-N that are churners\n"
            "Gain@N — fraction of all churners captured in top-N\n"
            "Lift@N — how many times better than random\n"
            "Qini@N — uplift-aware (outreach-adjusted)"
        ),
    )

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

_METRIC_LABELS = {
    "Precision": "Precision",
    "Gain": "Gain",
    "Lift": "Lift",
    "Qini": "Qini",
}

_METRIC_FMT = {
    "Precision": lambda v: f"{v:.3f}",
    "Gain": lambda v: f"{v:.1%}",
    "Lift": lambda v: f"{v:.2f}x",
    "Qini": lambda v: f"{v:.4f}",
}


def _at_n(curve: Optional[list], n: int) -> Optional[float]:
    """Read index n-1 from a curve list; returns None if unavailable."""
    if curve is None:
        return None
    idx = n - 1
    if 0 <= idx < len(curve):
        return curve[idx]
    return None


def _get_curve(r: CandidateResult, metric: str) -> Optional[list]:
    """Safely retrieve a metric curve from a result, tolerating old cached model versions."""
    if metric == "Precision":
        return getattr(r, "precision_curve", None)
    if metric == "Gain":
        return getattr(r, "gain_curve", None)
    if metric == "Lift":
        return getattr(r, "lift_curve", None)
    if metric == "Qini":
        return getattr(r, "qini_curve", None)
    return None


def _metric_at_n(r: CandidateResult, metric: str, n: int) -> Optional[float]:
    return _at_n(_get_curve(r, metric), n)


def _metric_curve(r: CandidateResult, metric: str) -> Optional[list[float]]:
    return _get_curve(r, metric)


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


def _fmt(value, metric: str) -> str:
    if value is None:
        return "—"
    return _METRIC_FMT[metric](value)


def _build_leaderboard(results: list[CandidateResult], n: int, sort_metric: str) -> pd.DataFrame:
    rows = []
    for r in results:
        rec_n = _effective_rec_n(r)
        prec = _at_n(getattr(r, "precision_curve", None), n)
        gain = _at_n(getattr(r, "gain_curve", None), n)
        lift = _at_n(getattr(r, "lift_curve", None), n)
        qini = _at_n(getattr(r, "qini_curve", None), n)
        sort_val = _metric_at_n(r, sort_metric, n)
        # Scores at each candidate's own recommended N
        prec_rec = _at_n(getattr(r, "precision_curve", None), rec_n) if rec_n else None
        gain_rec = _at_n(getattr(r, "gain_curve", None), rec_n) if rec_n else None
        lift_rec = _at_n(getattr(r, "lift_curve", None), rec_n) if rec_n else None
        qini_rec = _at_n(getattr(r, "qini_curve", None), rec_n) if rec_n else None
        rows.append(
            {
                "": _STATUS_ICON.get(r.status, "❓"),
                "Candidate": r.candidate_name,
                f"Precision@{n:,}": _fmt(prec, "Precision"),
                f"Gain@{n:,}": _fmt(gain, "Gain"),
                f"Lift@{n:,}": _fmt(lift, "Lift"),
                f"Qini@{n:,}": _fmt(qini, "Qini"),
                "Rec. N": rec_n,
                "Precision@Rec.N": _fmt(prec_rec, "Precision"),
                "Gain@Rec.N": _fmt(gain_rec, "Gain"),
                "Lift@Rec.N": _fmt(lift_rec, "Lift"),
                "Qini@Rec.N": _fmt(qini_rec, "Qini"),
                "Status": _status_label(r),
                "_sort": sort_val if sort_val is not None else -999,
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
st.header(f"Leaderboard @ N={n_slider:,}")
st.caption(f"Sorted by {metric_choice}. Gain = cumulative recall · Lift = × random · Qini = uplift-aware")

if not results:
    st.info("No results yet. Run `python -m grader` to process candidates.")
else:
    leaderboard_df = _build_leaderboard(results, n_slider, metric_choice)
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
# Section 2: Metric Chart
# ---------------------------------------------------------------------------

_METRIC_Y_LABEL = {
    "Precision": "Precision@N",
    "Gain": "Gain@N (cumulative recall)",
    "Lift": "Lift@N (× random)",
    "Qini": "Qini@N",
}

_METRIC_Y_RANGE = {
    "Precision": None,   # dynamic
    "Gain": [0, 1.0],
    "Lift": None,        # dynamic
    "Qini": None,        # dynamic — can go negative
}

st.header(f"{metric_choice}@N over N")

valid_results = [r for r in results if _metric_curve(r, metric_choice)]

if not valid_results:
    st.info("No scored submissions yet.")
else:
    fig = go.Figure()

    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]

    all_y_vals = []

    for i, r in enumerate(valid_results):
        curve = _metric_curve(r, metric_choice)
        xs = list(range(1, len(curve) + 1))
        color = colors[i % len(colors)]
        all_y_vals.extend(curve)

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
                    f"{metric_choice}=%{{y:.4f}}<extra></extra>"
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
        total_pop = getattr(scorer, 'total_population', None) or len(getattr(scorer, '_labels_df', None) or range(10_000))

        if metric_choice == "Precision":
            fig.add_hline(
                y=baseline,
                line=dict(color="gray", width=1, dash="dash"),
                annotation_text=f"Random baseline ({baseline:.1%})",
                annotation_position="bottom right",
            )
        elif metric_choice == "Gain":
            # Diagonal: gain_random(N) = N / total_population
            xs_base = list(range(0, total_pop + 1, max(1, total_pop // 200)))
            ys_base = [x / total_pop for x in xs_base]
            fig.add_trace(go.Scatter(
                x=xs_base, y=ys_base,
                mode="lines", name="Random baseline",
                line=dict(color="gray", width=1, dash="dash"),
                hoverinfo="skip",
            ))
        elif metric_choice == "Lift":
            fig.add_hline(
                y=1.0,
                line=dict(color="gray", width=1, dash="dash"),
                annotation_text="Random baseline (lift=1)",
                annotation_position="bottom right",
            )
        elif metric_choice == "Qini":
            fig.add_hline(
                y=0.0,
                line=dict(color="gray", width=1, dash="dash"),
                annotation_text="Random baseline (qini=0)",
                annotation_position="bottom right",
            )

    fig.add_vline(
        x=n_slider,
        line=dict(color="black", width=2, dash="solid"),
        annotation_text=f"N={n_slider:,}",
        annotation_position="top right",
    )

    # Determine y-axis range
    y_range = _METRIC_Y_RANGE[metric_choice]
    if y_range is None and all_y_vals:
        y_min = min(all_y_vals)
        y_max = max(all_y_vals)
        pad = (y_max - y_min) * 0.1 or 0.05
        if metric_choice == "Qini":
            y_range = [min(y_min - pad, -0.02), y_max + pad]
        elif metric_choice == "Lift":
            y_range = [0, y_max * 1.1]
        else:
            y_range = [0, max(baseline * 1.5, y_max * 1.1, 0.3)]

    fig.update_layout(
        xaxis_title="N (outreach size)",
        yaxis_title=_METRIC_Y_LABEL[metric_choice],
        yaxis=dict(range=y_range) if y_range else {},
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
# Section 5: Outreach Impact Analysis
# ---------------------------------------------------------------------------
st.header("Outreach Impact Analysis")
st.caption(
    "How effective was the outreach in the test period, and what value does model-driven targeting provide?"
)

_labels_df_raw = getattr(scorer, '_labels_df', None) if scorer else None

if _labels_df_raw is None or 'outreach' not in _labels_df_raw.columns:
    st.info("Outreach data not available — scorer must be loaded from labels file with an `outreach` column.")
else:
    import math as _math

    _ctrl = _labels_df_raw[_labels_df_raw['outreach'] == 0]
    _trt  = _labels_df_raw[_labels_df_raw['outreach'] == 1]
    _p_c  = float(_ctrl['churn'].mean())
    _p_t  = float(_trt['churn'].mean())
    _n_c, _n_t = len(_ctrl), len(_trt)
    _abs_eff = _p_c - _p_t
    _rel_eff = _abs_eff / _p_c if _p_c > 0 else 0.0
    _se = _math.sqrt(_p_c*(1-_p_c)/_n_c + _p_t*(1-_p_t)/_n_t)
    _z  = _abs_eff / _se if _se > 0 else 0.0
    _p_val = 2 * 0.5 * _math.erfc(abs(_z) / _math.sqrt(2))

    # --- Outreach effectiveness stats ---
    _col1, _col2, _col3, _col4 = st.columns(4)
    _col1.metric("Control group churn", f"{_p_c:.1%}",
                 help=f"Churn rate among the {_n_c:,} non-outreached members")
    _col2.metric("Outreached group churn", f"{_p_t:.1%}",
                 help=f"Churn rate among the {_n_t:,} outreached members")
    _col3.metric("Absolute effect", f"{_abs_eff:+.2%}",
                 help="Reduction in churn rate from outreach (positive = outreach helped)")
    _col4.metric("Relative reduction", f"{_rel_eff:.1%}",
                 help=f"z={_z:.2f}, p={_p_val:.3f}")

    if _p_val >= 0.05:
        st.warning(
            f"**Outreach effect is not statistically significant** (z={_z:.2f}, p={_p_val:.2f}). "
            "The ~0.5 pp churn reduction is likely noise. "
            "This is why Qini scores are near zero — the metric requires measurable outreach lift to show signal. "
            "The candidates' models are still evaluated fairly on their ability to identify churners (Precision, Gain, Lift)."
        )
    else:
        st.success(
            f"Outreach is statistically significant (z={_z:.2f}, p={_p_val:.3f}). "
            f"Outreach reduced churn by ~{_abs_eff:.2%} in absolute terms."
        )

    st.divider()

    # --- Targeting value: extra churners found vs random ---
    _ok_scored_impact = [r for r in results
                         if r.status in (PredictionStatus.OK, PredictionStatus.DEGENERATE_PREDICTIONS)
                         and getattr(r, 'precision_curve', None)]

    if _ok_scored_impact:
        st.subheader(f"Targeting Value @ N={n_slider:,}")
        st.caption(
            "Each model's top-N contains more churners than a random selection of N. "
            "This chart shows how many **additional churners** each model identifies versus random outreach."
        )

        _base = getattr(scorer, '_churn_rate', _DEFAULT_BASELINE)
        _random_churners_at_n = _base * n_slider

        _impact_rows = []
        for r in _ok_scored_impact:
            _prec = _at_n(getattr(r, 'precision_curve', None), n_slider)
            if _prec is None:
                continue
            _extra = (_prec - _base) * n_slider
            _impact_rows.append({
                "Candidate": r.candidate_name,
                "Precision@N": _prec,
                "Churners in top-N (model)": _prec * n_slider,
                "Churners in top-N (random)": _random_churners_at_n,
                "Extra churners found": _extra,
            })

        if _impact_rows:
            _idf = sorted(_impact_rows, key=lambda x: x["Extra churners found"], reverse=True)
            _names_i = [r["Candidate"] for r in _idf]
            _extra_vals = [r["Extra churners found"] for r in _idf]
            _model_vals = [r["Churners in top-N (model)"] for r in _idf]

            _fig_impact = go.Figure()
            # Random baseline bar (base layer)
            _fig_impact.add_trace(go.Bar(
                x=_names_i,
                y=[_random_churners_at_n] * len(_idf),
                name=f"Random baseline ({_base:.1%})",
                marker_color="lightgray",
                hovertemplate="Random: %{y:.0f} churners<extra></extra>",
            ))
            # Extra churners on top
            _fig_impact.add_trace(go.Bar(
                x=_names_i,
                y=[max(0, v) for v in _extra_vals],
                name="Extra churners found",
                marker_color="#2ca02c",
                base=[_random_churners_at_n] * len(_idf),
                hovertemplate="Extra churners: %{y:.0f}<br>Total model: %{customdata:.0f}<extra></extra>",
                customdata=_model_vals,
            ))
            _fig_impact.update_layout(
                barmode="stack",
                xaxis_title="Candidate",
                yaxis_title=f"Churners in top-{n_slider:,}",
                legend=dict(orientation="h", y=1.1),
                height=380,
                margin=dict(l=60, r=40, t=60, b=60),
            )
            st.plotly_chart(_fig_impact, use_container_width=True)

            # Summary table
            _sum_df = pd.DataFrame([{
                "Candidate": r["Candidate"],
                f"Precision@{n_slider:,}": f"{r['Precision@N']:.1%}",
                "Churners (model)": f"{r['Churners in top-N (model)']:.0f}",
                "Churners (random)": f"{_random_churners_at_n:.0f}",
                "Extra churners found": f"{r['Extra churners found']:+.0f}",
            } for r in _idf])
            with st.expander("Show targeting value table"):
                st.dataframe(_sum_df, use_container_width=True, hide_index=True)

        st.divider()

        # --- Simulated savings ---
        st.subheader("Simulated Churn Prevention")
        st.caption(
            "If outreach were effective, better targeting would prevent more churn. "
            "Use the slider to explore what would happen at different outreach effectiveness levels. "
            f"The test data measured **{_rel_eff:.1%} relative reduction** (not statistically significant)."
        )

        _default_save_rate = max(0.1, round(_rel_eff * 100, 1))
        _save_rate_pct = st.slider(
            "Outreach effectiveness — % of contacted churners saved",
            min_value=0.0,
            max_value=50.0,
            value=_default_save_rate,
            step=0.5,
            format="%.1f%%",
            help=(
                "How many of the churners you contact are actually saved by outreach? "
                "Measured from test data: ~2.4% (not significant). "
                "Move the slider to explore optimistic scenarios."
            ),
            key="save_rate_slider",
        )
        _save_rate = _save_rate_pct / 100.0

        if _impact_rows:
            _sim_rows = []
            _random_saves = _random_churners_at_n * _save_rate
            for r in _idf:
                _model_saves = r["Churners in top-N (model)"] * _save_rate
                _extra_saves = _model_saves - _random_saves
                _sim_rows.append({
                    "Candidate": r["Candidate"],
                    "Saves (random outreach)": _random_saves,
                    "Saves (model targeting)": _model_saves,
                    "Extra saves vs random": _extra_saves,
                })

            _snames = [r["Candidate"] for r in _sim_rows]
            _extra_saves_vals = [r["Extra saves vs random"] for r in _sim_rows]

            _fig_sim = go.Figure()
            _fig_sim.add_trace(go.Bar(
                x=_snames,
                y=[_random_saves] * len(_sim_rows),
                name="Saves from random outreach",
                marker_color="lightgray",
                hovertemplate=f"Random outreach saves: {_random_saves:.1f}<extra></extra>",
            ))
            _fig_sim.add_trace(go.Bar(
                x=_snames,
                y=[max(0, v) for v in _extra_saves_vals],
                name="Additional saves from model targeting",
                marker_color="#9467bd",
                base=[_random_saves] * len(_sim_rows),
                hovertemplate="Additional saves: %{y:.1f}<extra></extra>",
            ))
            _fig_sim.update_layout(
                barmode="stack",
                xaxis_title="Candidate",
                yaxis_title=f"Estimated churns prevented (N={n_slider:,})",
                legend=dict(orientation="h", y=1.1),
                height=380,
                margin=dict(l=60, r=40, t=60, b=60),
                annotations=[dict(
                    x=0.5, y=-0.22, xref="paper", yref="paper",
                    text=f"Assumes {_save_rate_pct:.1f}% of contacted churners are saved by outreach",
                    showarrow=False, font=dict(size=11, color="gray"),
                )],
            )
            st.plotly_chart(_fig_sim, use_container_width=True)

            if _save_rate_pct < 1.0:
                st.info(
                    "At near-zero effectiveness the difference is negligible — "
                    "try raising the slider to see what better outreach would mean for each model."
                )
            else:
                best = _sim_rows[0]
                st.success(
                    f"At {_save_rate_pct:.1f}% outreach effectiveness with N={n_slider:,}: "
                    f"**{best['Candidate']}** would prevent **{best['Saves (model targeting)']:.0f} churns** "
                    f"({best['Extra saves vs random']:+.1f} vs random outreach)."
                )

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
                true_member_ids=scorer.true_member_ids if scorer else None,
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
