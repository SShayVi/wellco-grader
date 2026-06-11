"""
Scores a candidate's standardized predictions against the true labels.
"""
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from grader.scoring.metrics import precision_curve, random_baseline_precision

logger = logging.getLogger(__name__)


class Scorer:
    """
    Loads true labels once and scores any number of prediction DataFrames.

    Parameters
    ----------
    labels_path : Path to test_churn_labels.csv.
    metric      : Scoring metric to use. Currently supports 'precision_at_n'.
                  Extend by registering new functions in metrics.py.
    """

    SUPPORTED_METRICS = {"precision_at_n"}

    def __init__(self, labels_path: Path, metric: str = "precision_at_n") -> None:
        if metric not in self.SUPPORTED_METRICS:
            raise ValueError(f"Unknown metric '{metric}'. Supported: {self.SUPPORTED_METRICS}")
        self._metric = metric
        self._labels_df = self._load_labels(labels_path)
        self._true_churner_ids = set(
            self._labels_df.loc[self._labels_df["churn"] == 1, "member_id"].astype(int)
        )
        self._churn_rate = len(self._true_churner_ids) / len(self._labels_df)
        logger.info(
            "Scorer loaded: %d members, %d churners (%.1f%%), metric=%s",
            len(self._labels_df),
            len(self._true_churner_ids),
            self._churn_rate * 100,
            metric,
        )

    @property
    def true_member_ids(self) -> set[int]:
        return set(self._labels_df["member_id"].astype(int))

    @property
    def churn_rate(self) -> float:
        return self._churn_rate

    @property
    def baseline_precision(self) -> float:
        return random_baseline_precision(self._churn_rate)

    def score(self, predictions_df: pd.DataFrame) -> Optional[list[float]]:
        """
        Compute the precision curve for a predictions DataFrame.

        Parameters
        ----------
        predictions_df : DataFrame with columns (member_id, score, rank).
                         Must be sorted by score descending already (done by PredictionAgent).

        Returns
        -------
        List of floats: precision@N for N = 1 .. len(predictions_df).
        Returns None if the DataFrame is empty or malformed.
        """
        if predictions_df is None or predictions_df.empty:
            return None

        try:
            ranked_ids = predictions_df.sort_values("score", ascending=False)["member_id"].astype(int).tolist()
        except (KeyError, ValueError) as e:
            logger.error("Cannot extract ranked member IDs: %s", e)
            return None

        if self._metric == "precision_at_n":
            return precision_curve(ranked_ids, self._true_churner_ids)

        raise NotImplementedError(f"Metric '{self._metric}' not yet implemented")

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
