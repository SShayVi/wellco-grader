"""
Scoring metrics for churn prediction submissions.

Primary metric: precision@N
  precision@N = |top_N_predicted ∩ true_churners| / N

Future metrics (TODO):
  - qini_at_n: uplift-aware metric using the outreach column
  - auuc: Area Under Uplift Curve
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


# ---------------------------------------------------------------------------
# TODO: Uplift-aware metrics
# The outreach column in the true labels records who was actually outreached.
# These metrics will reward targeting members who churn WITHOUT outreach
# and respond to intervention.
#
# def qini_at_n(
#     ranked_member_ids: list[int],
#     labels_df: pd.DataFrame,   # columns: member_id, churn, outreach
#     n: int,
# ) -> float:
#     """
#     Qini coefficient at N: rewards targeting members who respond to outreach.
#     Formula: (true_positives_treated / total_treated) - (true_positives_control / total_control)
#     where treatment = members in top-N who were outreached in the test set.
#     """
#     ...
#
# def uplift_curve(
#     ranked_member_ids: list[int],
#     labels_df: pd.DataFrame,
# ) -> list[float]:
#     """Qini@N for every N from 1 to len(ranked_member_ids)."""
#     ...
# ---------------------------------------------------------------------------
