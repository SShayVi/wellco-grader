"""
Scoring metrics for churn prediction submissions.

Metrics (all computed as curves — one value per N from 1 to len(predictions)):
  precision@N  = |top_N ∩ churners| / N
  gain@N       = |top_N ∩ churners| / total_churners   (cumulative recall)
  lift@N       = precision@N / churn_rate              (relative to random)
  qini@N       = control_churners_in_topN/N_C - treated_churners_in_topN/N_T
                 (cumulative uplift; requires outreach labels; baseline 0)
  uplift@N     = churn_rate_control_in_topN - churn_rate_treated_in_topN
                 (conditional treatment effect; requires outreach labels; baseline ≈ ATE)
"""
from typing import Optional


def precision_at_n(
    ranked_member_ids: list[int],
    true_churner_ids: set[int],
    n: int,
) -> float:
    """
    Fraction of the top-N ranked members that are true churners.

    Args:
        ranked_member_ids: Member IDs ordered by model score (highest risk first).
        true_churner_ids:  Set of member IDs that actually churned.
        n:                 Cutoff size.

    Returns:
        precision@N in [0, 1]. Returns 0.0 if n <= 0 or list is empty.
    """
    if n <= 0 or not ranked_member_ids:
        return 0.0
    n = min(n, len(ranked_member_ids))
    top_n = set(ranked_member_ids[:n])
    return len(top_n & true_churner_ids) / n


def precision_curve(
    ranked_member_ids: list[int],
    true_churner_ids: set[int],
) -> list[float]:
    """
    Compute precision@N for every N from 1 to len(ranked_member_ids).

    Returns a list of length len(ranked_member_ids) where index i = precision@(i+1).
    This is O(n) — suitable for pre-computing the full curve at write time.
    """
    if not ranked_member_ids:
        return []

    hits = 0
    curve: list[float] = []
    for i, member_id in enumerate(ranked_member_ids, start=1):
        if member_id in true_churner_ids:
            hits += 1
        curve.append(hits / i)
    return curve


def random_baseline_precision(churn_rate: float) -> float:
    """Expected precision@N for a random ranker (equals the population churn rate)."""
    return churn_rate


def gain_curve(
    ranked_member_ids: list[int],
    true_churner_ids: set[int],
) -> list[float]:
    """
    Cumulative gain@N for every N: fraction of all churners captured in top-N.

    gain@N = |top_N ∩ churners| / total_churners

    Random baseline at N: N / total_population (diagonal line).
    """
    total_churners = len(true_churner_ids)
    if not ranked_member_ids or total_churners == 0:
        return []

    hits = 0
    curve: list[float] = []
    for member_id in ranked_member_ids:
        if member_id in true_churner_ids:
            hits += 1
        curve.append(hits / total_churners)
    return curve


def lift_curve(
    ranked_member_ids: list[int],
    true_churner_ids: set[int],
    total_population: int,
) -> list[float]:
    """
    Lift@N for every N: how many times better than random at the same N.

    lift@N = precision@N / churn_rate = gain@N / (N / total_population)

    Random baseline: 1.0 (constant).
    """
    if not ranked_member_ids or total_population == 0:
        return []
    churn_rate = len(true_churner_ids) / total_population
    if churn_rate == 0:
        return []
    prec = precision_curve(ranked_member_ids, true_churner_ids)
    return [p / churn_rate for p in prec]


def qini_curve(
    ranked_member_ids: list[int],
    treated_churner_ids: set[int],
    control_churner_ids: set[int],
    n_treated: int,
    n_control: int,
) -> list[float]:
    """
    Qini@N for every N: cumulative uplift metric using outreach labels.

    qini@N = control_churners_in_topN / N_C - treated_churners_in_topN / N_T

    where treated = outreach=1, control = outreach=0.
    Positive = model surfaces control churners (persuadables — those who churn
    WITHOUT outreach) disproportionately vs treated churners (lost causes — those
    who churn DESPITE outreach).  Random baseline: 0.0.  Max ≈ 0.16 for this dataset.
    """
    if not ranked_member_ids or n_treated == 0 or n_control == 0:
        return []

    hits_t = 0
    hits_c = 0
    curve: list[float] = []
    for member_id in ranked_member_ids:
        if member_id in treated_churner_ids:
            hits_t += 1
        elif member_id in control_churner_ids:
            hits_c += 1
        curve.append(hits_c / n_control - hits_t / n_treated)
    return curve


def uplift_curve(
    ranked_member_ids: list[int],
    treated_member_ids: set[int],
    treated_churner_ids: set[int],
    control_churner_ids: set[int],
) -> list[float]:
    """
    Uplift@N for every N: conditional average treatment effect within top-N.

    uplift@N = churn_rate_control_in_topN - churn_rate_treated_in_topN
             = (control_churners_in_topN / n_control_in_topN)
             - (treated_churners_in_topN / n_treated_in_topN)

    Uses local counts within top-N as denominators (vs Qini which uses global N_T/N_C).
    Interpretation: within the model's top-N, the control group's churn rate exceeds
    the treated group's churn rate by uplift@N — i.e., outreach reduces churn by that
    amount for the members this model recommends.
    Random baseline ≈ overall ATE (≈ 0.0048 in this dataset, not zero).
    Returns 0.0 until at least one member from each group appears in top-N.
    """
    if not ranked_member_ids:
        return []

    n_t = 0
    n_c = 0
    hits_t = 0
    hits_c = 0
    curve: list[float] = []
    for member_id in ranked_member_ids:
        if member_id in treated_member_ids:
            n_t += 1
            if member_id in treated_churner_ids:
                hits_t += 1
        else:
            n_c += 1
            if member_id in control_churner_ids:
                hits_c += 1
        if n_t > 0 and n_c > 0:
            curve.append(hits_c / n_c - hits_t / n_t)
        else:
            curve.append(0.0)
    return curve
