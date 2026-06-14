"""
Validate and standardise a candidate prediction CSV.

Usage (programmatic):
    from grader.validation import validate_and_standardize
    result = validate_and_standardize(raw_bytes, true_member_ids=..., min_overlap=0.5)
    if result.ok:
        df = result.standardized  # columns: member_id, score — sorted by score desc

Usage (CLI):
    python -m grader validate https://github.com/.../predictions.csv
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from io import BytesIO
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hint lists (priority order: first match wins)
# ---------------------------------------------------------------------------
MEMBER_ID_HINTS = [
    "member_id", "memberid", "member", "id", "user_id", "userid", "user",
    "customer_id", "client_id", "patient_id", "account_id",
]
SCORE_HINTS = [
    # Uplift / causal (preferred for outreach targeting)
    "weighted_uplift", "uplift", "cate", "cate_estimate", "cate_score",
    "benefit_score",
    # Explicit priority / propensity
    "propensity_score", "propensity", "priority_score", "prioritization_score",
    # Churn probability variants
    "churn_score", "churn_prob", "churn_probability",
    "churn_prob_no_outreach", "p_churn_no_outreach", "baseline_churn_proba",
    # Generic
    "score", "probability", "prob", "risk", "risk_score", "pred", "prediction",
    # Rank last — needs inversion (rank 1 = best)
    "rank",
]
RANK_HINTS = {"rank", "ranking", "position"}

# Member IDs in this range are almost certainly from the training set, not the test set
_TRAIN_ID_MAX = 20_000


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class Issue:
    severity: Severity
    code: str
    message: str

    def __str__(self) -> str:
        return f"[{self.severity.value.upper()}] {self.code}: {self.message}"


@dataclass
class ValidationResult:
    ok: bool                              # True = no ERROR-level issues
    issues: list = field(default_factory=list)  # list[Issue]
    standardized: Optional[pd.DataFrame] = None  # member_id + score, sorted desc
    member_col: Optional[str] = None
    score_col: Optional[str] = None
    rank_inverted: bool = False
    row_count: int = 0
    overlap_pct: Optional[float] = None   # fraction of IDs in the test set

    def errors(self) -> list:
        return [i for i in self.issues if i.severity == Severity.ERROR]

    def warnings(self) -> list:
        return [i for i in self.issues if i.severity == Severity.WARNING]

    def summary(self) -> str:
        overlap = f"  overlap={self.overlap_pct:.1%}" if self.overlap_pct is not None else ""
        header = f"ok={self.ok}  rows={self.row_count}{overlap}"
        lines = [header] + [f"  {i}" for i in self.issues]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def validate_and_standardize(
    raw: bytes,
    true_member_ids: Optional[set] = None,
    min_overlap: float = 0.5,
) -> ValidationResult:
    """
    Validate *raw* CSV bytes and return a standardised DataFrame.

    Steps
    -----
    1. Parse CSV
    2. Map columns (exact → substring → dtype fallback)
    3. Normalise types, invert rank columns
    4. Drop NaN rows, deduplicate member IDs
    5. Check score quality (degenerate, low variety)
    6. Validate member ID overlap against the test set (if provided)

    Parameters
    ----------
    raw :             Raw bytes of the submitted CSV.
    true_member_ids : Set of valid integer member IDs from the test labels.
                      When omitted, overlap checks are skipped.
    min_overlap :     Minimum fraction of submitted IDs that must appear in
                      the test labels (default 0.5).

    Returns
    -------
    ValidationResult — check `.ok` first, then `.standardized` for the
    cleaned DataFrame and `.issues` for the full audit trail.
    """
    issues: list = []

    # ── 1. Parse ─────────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(BytesIO(raw))
    except Exception as exc:
        issues.append(Issue(
            Severity.ERROR, "PARSE_ERROR",
            f"Cannot parse CSV: {exc}. "
            "Ensure the file is a valid comma-separated CSV (not Excel or JSON).",
        ))
        return ValidationResult(ok=False, issues=issues)

    if df.empty:
        issues.append(Issue(
            Severity.ERROR, "EMPTY_CSV",
            "The CSV file has no rows. Submit a file with at least your top-N predictions.",
        ))
        return ValidationResult(ok=False, issues=issues)

    # ── 2. Column mapping ────────────────────────────────────────────────────
    mapping = _map_columns(df)
    if mapping is None:
        col_list = ", ".join(f"'{c}'" for c in df.columns)
        issues.append(Issue(
            Severity.ERROR, "COLUMNS_NOT_FOUND",
            f"Could not identify member_id or score columns from: [{col_list}]. "
            f"Supported member_id names include: {', '.join(MEMBER_ID_HINTS[:5])}, ... "
            f"Supported score names include: {', '.join(SCORE_HINTS[:8])}, ... "
            "Rename your columns or contact the grader administrator.",
        ))
        return ValidationResult(ok=False, issues=issues)

    member_col = mapping["member_id_col"]
    score_col = mapping["score_col"]
    invert = mapping["invert_score"]

    _standard_member = member_col.lower().strip() == "member_id"
    _standard_score = score_col.lower().strip() in {
        "score", "churn_score", "churn_prob", "churn_probability", "probability",
    }
    if not (_standard_member and _standard_score):
        issues.append(Issue(
            Severity.INFO, "COLUMN_REMAPPED",
            f"Non-standard column names detected. "
            f"Mapped '{member_col}' → member_id, '{score_col}' → score.",
        ))
    if invert:
        issues.append(Issue(
            Severity.INFO, "RANK_INVERTED",
            f"'{score_col}' is a rank column (lower number = better). "
            "Values were negated so rank 1 is treated as highest priority.",
        ))

    # ── 3. Normalise types ───────────────────────────────────────────────────
    out = pd.DataFrame()
    out["member_id"] = pd.to_numeric(df[member_col], errors="coerce").astype("Int64")
    out["score"] = pd.to_numeric(df[score_col], errors="coerce")
    if invert:
        out["score"] = -out["score"]

    n_before = len(out)
    out = out.dropna(subset=["member_id", "score"])
    n_dropped = n_before - len(out)
    if n_dropped > 0:
        issues.append(Issue(
            Severity.WARNING, "ROWS_DROPPED",
            f"{n_dropped}/{n_before} row(s) had non-numeric member_id or score and were removed. "
            "Ensure both columns contain only numeric values.",
        ))

    if out.empty:
        issues.append(Issue(
            Severity.ERROR, "NO_VALID_ROWS",
            "No valid rows remain after parsing. "
            "Both member_id and score columns must contain numeric values.",
        ))
        return ValidationResult(ok=False, issues=issues,
                                member_col=member_col, score_col=score_col)

    # ── 4. Deduplication ─────────────────────────────────────────────────────
    n_before_dedup = len(out)
    out = out.sort_values("score", ascending=False).drop_duplicates("member_id")
    n_dupes = n_before_dedup - len(out)
    if n_dupes > 0:
        issues.append(Issue(
            Severity.WARNING, "DUPLICATE_IDS",
            f"{n_dupes} duplicate member_id(s) found. "
            "Kept the entry with the highest score for each.",
        ))

    out = out.sort_values("score", ascending=False).reset_index(drop=True)
    row_count = len(out)

    # ── 5. Score quality ─────────────────────────────────────────────────────
    n_unique = out["score"].nunique()
    if n_unique == 1:
        issues.append(Issue(
            Severity.WARNING, "DEGENERATE_SCORES",
            "All scores are identical — the model cannot rank members. "
            "Precision@N will equal the random baseline (~20%) for all N.",
        ))
    elif n_unique < 10:
        issues.append(Issue(
            Severity.WARNING, "LOW_SCORE_VARIETY",
            f"Only {n_unique} distinct score values. "
            "Consider using a continuous score for better discrimination.",
        ))

    # ── 6. Row count ─────────────────────────────────────────────────────────
    if row_count < 10:
        issues.append(Issue(
            Severity.WARNING, "LOW_ROW_COUNT",
            f"Only {row_count} members submitted. "
            "Precision@N is only meaningful up to this count.",
        ))
    elif row_count < 10_000:
        issues.append(Issue(
            Severity.INFO, "PARTIAL_SUBMISSION",
            f"{row_count:,} members submitted (not all 10,000). "
            f"Precision@N is valid for N ≤ {row_count:,}; shown as '—' beyond that.",
        ))

    # ── 7. Member ID overlap ─────────────────────────────────────────────────
    overlap_pct = None
    if true_member_ids is not None:
        candidate_ids = set(out["member_id"].dropna().astype(int))
        overlap_pct = (
            len(candidate_ids & true_member_ids) / len(candidate_ids)
            if candidate_ids else 0.0
        )

        if overlap_pct == 0.0:
            max_id = max(candidate_ids) if candidate_ids else 0
            if max_id < _TRAIN_ID_MAX:
                detail = (
                    f"Your member IDs (max={max_id:,}) are from the TRAINING set. "
                    "The test set uses member IDs 20,001–30,000. "
                    "Re-run your model's prediction step on the test members "
                    "(the ones you did NOT train on)."
                )
            else:
                detail = (
                    "0% of your member IDs match the test set. "
                    "Check that you are predicting on the correct member population "
                    f"(expected IDs in range 20,001–30,000; your max is {max_id:,})."
                )
            issues.append(Issue(Severity.ERROR, "WRONG_DATASET", detail))

        elif overlap_pct < min_overlap:
            issues.append(Issue(
                Severity.ERROR, "LOW_OVERLAP",
                f"Only {overlap_pct:.1%} of your member IDs exist in the test set "
                f"(minimum required: {min_overlap:.0%}). "
                "You may be predicting on a mix of training and test members, "
                "or using member IDs that do not match the official test set.",
            ))

    has_errors = any(i.severity == Severity.ERROR for i in issues)
    standardized = None if has_errors else out[["member_id", "score"]].copy()

    return ValidationResult(
        ok=not has_errors,
        issues=issues,
        standardized=standardized,
        member_col=member_col,
        score_col=score_col,
        rank_inverted=invert,
        row_count=row_count,
        overlap_pct=overlap_pct,
    )


# ---------------------------------------------------------------------------
# Column mapper
# ---------------------------------------------------------------------------
def _map_columns(df: pd.DataFrame) -> Optional[dict]:
    """Three-stage heuristic: exact → substring → dtype fallback."""
    lower = {c.lower().strip(): c for c in df.columns}

    # Stage 1: exact match
    member_col = next((lower[k] for k in MEMBER_ID_HINTS if k in lower), None)
    score_col = next((lower[k] for k in SCORE_HINTS if k in lower), None)

    # Stage 2: substring match (hint ⊆ col_name or col_name ⊆ hint)
    if member_col is None:
        for col_l, col_orig in lower.items():
            if any(hint in col_l or col_l in hint for hint in MEMBER_ID_HINTS):
                member_col = col_orig
                break
    if score_col is None:
        for hint in SCORE_HINTS:  # preserve priority order
            for col_l, col_orig in lower.items():
                if col_orig == member_col:
                    continue
                if hint in col_l or col_l in hint:
                    score_col = col_orig
                    break
            if score_col:
                break

    # Stage 3: two-column dtype fallback
    if member_col is None or score_col is None:
        num_cols = [
            c for c in df.columns
            if pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.8
        ]
        if len(num_cols) >= 2 and member_col is None and score_col is None:
            c1, c2 = num_cols[0], num_cols[1]
            c1_is_prob = pd.to_numeric(df[c1], errors="coerce").between(0, 1).mean() > 0.5
            c2_is_prob = pd.to_numeric(df[c2], errors="coerce").between(0, 1).mean() > 0.5
            if c2_is_prob and not c1_is_prob:
                member_col, score_col = c1, c2
            elif c1_is_prob and not c2_is_prob:
                member_col, score_col = c2, c1
            else:
                member_col, score_col = c1, c2  # best-effort

    if member_col and score_col:
        invert = any(r in score_col.lower() for r in RANK_HINTS)
        return {"member_id_col": member_col, "score_col": score_col, "invert_score": invert}
    return None
