"""
Scores a candidate's standardized predictions against the true labels.
"""
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from grader.scoring.metrics import (
    gain_curve,
    lift_curve,
    precision_curve,
    qini_curve,
    uplift_curve,
    random_baseline_precision,
)

logger = logging.getLogger(__name__)


class Scorer:
    """
    Loads true labels once and scores any number of prediction DataFrames.

    Parameters
    ----------
    labels_path : Path to test_churn_labels.csv.
    metric      : Ignored (kept for backward compatibility). All metrics are
                  always computed via score_all().
    """

    SUPPORTED_METRICS = {"precision_at_n"}

    def __init__(self, labels_path: Path, metric: str = "precision_at_n") -> None:
        self._metric = metric
        self._labels_df = self._load_labels(labels_path)
        self._true_churner_ids = set(
            self._labels_df.loc[self._labels_df["churn"] == 1, "member_id"].astype(int)
        )
        self._churn_rate = len(self._true_churner_ids) / len(self._labels_df)
        self._qini_data = self._build_qini_data()
        self._uplift_data = self._build_uplift_data()
        logger.info(
            "Scorer loaded: %d members, %d churners (%.1f%%)",
            len(self._labels_df),
            len(self._true_churner_ids),
            self._churn_rate * 100,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def true_member_ids(self) -> set[int]:
        return set(self._labels_df["member_id"].astype(int))

    @property
    def churn_rate(self) -> float:
        return self._churn_rate

    @property
    def total_population(self) -> int:
        return len(self._labels_df)

    @property
    def total_churners(self) -> int:
        return len(self._true_churner_ids)

    @property
    def baseline_precision(self) -> float:
        return random_baseline_precision(self._churn_rate)

    @property
    def overall_ate(self) -> float:
        """Average treatment effect: control_churn_rate − treated_churn_rate."""
        if "outreach" not in self._labels_df.columns:
            return 0.0
        df = self._labels_df
        p_c = df.loc[df["outreach"] == 0, "churn"].mean()
        p_t = df.loc[df["outreach"] == 1, "churn"].mean()
        return float(p_c - p_t)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, predictions_df: pd.DataFrame) -> Optional[list[float]]:
        """Return the precision curve (backward-compatible)."""
        if predictions_df is None or predictions_df.empty:
            return None
        try:
            ranked_ids = (
                predictions_df.sort_values("score", ascending=False)["member_id"]
                .astype(int)
                .tolist()
            )
        except (KeyError, ValueError) as e:
            logger.error("Cannot extract ranked member IDs: %s", e)
            return None
        return precision_curve(ranked_ids, self._true_churner_ids)

    def score_all(self, predictions_df: pd.DataFrame) -> dict:
        """
        Compute all metric curves for a predictions DataFrame.

        Returns a dict with keys: precision, gain, lift, qini.
        Each value is a list[float] of length len(predictions_df), or [] on error.
        """
        if predictions_df is None or predictions_df.empty:
            return {}
        try:
            ranked_ids = (
                predictions_df.sort_values("score", ascending=False)["member_id"]
                .astype(int)
                .tolist()
            )
        except (KeyError, ValueError) as e:
            logger.error("Cannot extract ranked member IDs: %s", e)
            return {}

        curves = {
            "precision": precision_curve(ranked_ids, self._true_churner_ids),
            "gain": gain_curve(ranked_ids, self._true_churner_ids),
            "lift": lift_curve(ranked_ids, self._true_churner_ids, self.total_population),
            "qini": (
                qini_curve(ranked_ids, *self._qini_data)
                if self._qini_data
                else []
            ),
            "uplift": (
                uplift_curve(ranked_ids, *self._uplift_data)
                if self._uplift_data
                else []
            ),
        }
        return curves

    def fill_curves(self, result) -> None:
        """
        Fill in missing gain/lift/qini curves on a CandidateResult in-place.

        Called by the dashboard for results cached before these metrics existed.
        gain and lift are derived analytically from precision_curve;
        qini requires ranked_member_ids and outreach labels.
        """
        if result.precision_curve is None:
            return

        n = len(result.precision_curve)

        if result.gain_curve is None and self.total_churners > 0:
            result.gain_curve = [
                result.precision_curve[i] * (i + 1) / self.total_churners
                for i in range(n)
            ]

        if result.lift_curve is None and self._churn_rate > 0:
            result.lift_curve = [p / self._churn_rate for p in result.precision_curve]

        if result.qini_curve is None and result.ranked_member_ids and self._qini_data:
            result.qini_curve = qini_curve(result.ranked_member_ids, *self._qini_data)

        if result.uplift_curve is None and result.ranked_member_ids and self._uplift_data:
            result.uplift_curve = uplift_curve(result.ranked_member_ids, *self._uplift_data)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_qini_data(self):
        """Pre-compute lookup sets needed for qini_curve(). Returns None if no outreach column."""
        if "outreach" not in self._labels_df.columns:
            return None
        df = self._labels_df
        treated_churners = set(
            df.loc[(df["outreach"] == 1) & (df["churn"] == 1), "member_id"].astype(int)
        )
        control_churners = set(
            df.loc[(df["outreach"] == 0) & (df["churn"] == 1), "member_id"].astype(int)
        )
        n_treated = int((df["outreach"] == 1).sum())
        n_control = int((df["outreach"] == 0).sum())
        return treated_churners, control_churners, n_treated, n_control

    def _build_uplift_data(self):
        """Pre-compute lookup sets needed for uplift_curve(). Returns None if no outreach column."""
        if "outreach" not in self._labels_df.columns:
            return None
        df = self._labels_df
        treated_members = set(
            df.loc[df["outreach"] == 1, "member_id"].astype(int)
        )
        treated_churners = set(
            df.loc[(df["outreach"] == 1) & (df["churn"] == 1), "member_id"].astype(int)
        )
        control_churners = set(
            df.loc[(df["outreach"] == 0) & (df["churn"] == 1), "member_id"].astype(int)
        )
        return treated_members, treated_churners, control_churners

    @staticmethod
    def _load_labels(path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"True labels file not found: {path}")
        df = pd.read_csv(path)
        required = {"member_id", "churn"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Labels file missing columns: {missing}")
        df["member_id"] = df["member_id"].astype(int)
        df["churn"] = df["churn"].astype(int)
        return df
