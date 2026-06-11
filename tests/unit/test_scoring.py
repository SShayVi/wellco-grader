"""
Unit tests for grader/scoring/metrics.py and scorer.py.
All tests use the real true labels — no mocking of labels.
"""
import pytest
import pandas as pd
from pathlib import Path

from grader.scoring.metrics import precision_at_n, precision_curve, random_baseline_precision
from grader.scoring.scorer import Scorer

LABELS_PATH = Path("data/test_churn_labels.csv")


# ---------------------------------------------------------------------------
# precision_at_n
# ---------------------------------------------------------------------------
class TestPrecisionAtN:
    def test_perfect_ranking(self):
        churners = {1, 2, 3, 4, 5}
        ranked = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert precision_at_n(ranked, churners, 5) == 1.0

    def test_zero_hits(self):
        churners = {1, 2, 3}
        ranked = [10, 11, 12, 13]
        assert precision_at_n(ranked, churners, 4) == 0.0

    def test_partial(self):
        churners = {1, 3, 5}
        ranked = [1, 2, 3, 4, 5, 6]
        assert precision_at_n(ranked, churners, 4) == 0.5  # 2 hits in top 4

    def test_n_larger_than_list(self):
        churners = {1, 2}
        ranked = [1, 2, 3]
        # n=10, but only 3 members → precision = 2/3
        assert precision_at_n(ranked, churners, 10) == pytest.approx(2 / 3)

    def test_n_zero(self):
        assert precision_at_n([1, 2, 3], {1}, 0) == 0.0

    def test_empty_ranked(self):
        assert precision_at_n([], {1, 2}, 5) == 0.0

    def test_n_equals_one(self):
        assert precision_at_n([1, 2, 3], {1}, 1) == 1.0
        assert precision_at_n([99, 2, 3], {1}, 1) == 0.0

    def test_duplicate_ids_in_top_n(self):
        # Each member counted once
        churners = {1}
        ranked = [1, 1, 2, 3]
        # top-2 = {1}, hits = 1, precision = 1/2
        assert precision_at_n(ranked, churners, 2) == 0.5


# ---------------------------------------------------------------------------
# precision_curve
# ---------------------------------------------------------------------------
class TestPrecisionCurve:
    def test_length_matches_input(self):
        ranked = list(range(1, 101))
        churners = set(range(1, 21))
        curve = precision_curve(ranked, churners)
        assert len(curve) == 100

    def test_all_churners_first(self):
        ranked = [1, 2, 3, 10, 11, 12]
        churners = {1, 2, 3}
        curve = precision_curve(ranked, churners)
        assert curve[0] == 1.0
        assert curve[1] == 1.0
        assert curve[2] == 1.0
        assert curve[3] == pytest.approx(3 / 4)

    def test_no_churners(self):
        ranked = [1, 2, 3]
        curve = precision_curve(ranked, set())
        assert curve == [0.0, 0.0, 0.0]

    def test_empty_input(self):
        assert precision_curve([], {1, 2}) == []

    def test_monotone_denominator(self):
        """Precision curve need not be monotone, but must use correct denominator."""
        ranked = [1, 99, 2, 99, 3]
        churners = {1, 2, 3}
        curve = precision_curve(ranked, churners)
        assert curve[0] == 1.0      # 1/1
        assert curve[1] == pytest.approx(1 / 2)  # 1/2 (99 is not churner)
        assert curve[2] == pytest.approx(2 / 3)  # 2/3


# ---------------------------------------------------------------------------
# random_baseline_precision
# ---------------------------------------------------------------------------
def test_baseline():
    assert random_baseline_precision(0.2) == pytest.approx(0.2)
    assert random_baseline_precision(0.0) == 0.0
    assert random_baseline_precision(1.0) == 1.0


# ---------------------------------------------------------------------------
# Scorer (uses real labels)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not LABELS_PATH.exists(), reason="True labels not found")
class TestScorer:
    @pytest.fixture(scope="class")
    def scorer(self):
        return Scorer(LABELS_PATH)

    def test_churn_rate(self, scorer):
        assert scorer.churn_rate == pytest.approx(0.2, abs=0.01)

    def test_true_member_count(self, scorer):
        assert len(scorer.true_member_ids) == 10_000

    def test_score_returns_correct_length(self, scorer, true_labels_df, all_member_ids):
        ranked = list(all_member_ids)[:5_000]
        df = pd.DataFrame({"member_id": ranked, "score": range(len(ranked), 0, -1), "rank": range(1, len(ranked) + 1)})
        curve = scorer.score(df)
        assert len(curve) == 5_000

    def test_score_perfect_ranker(self, scorer, true_labels_df):
        """A ranker that puts all churners first should have precision@N=1 at N<=num_churners."""
        churners = list(scorer._true_churner_ids)
        non_churners = list(scorer.true_member_ids - scorer._true_churner_ids)
        ranked = churners + non_churners
        df = pd.DataFrame({
            "member_id": ranked,
            "score": list(range(len(ranked), 0, -1)),
            "rank": list(range(1, len(ranked) + 1)),
        })
        curve = scorer.score(df)
        n_churners = len(churners)
        assert curve[n_churners - 1] == pytest.approx(1.0)

    def test_score_empty_df(self, scorer):
        df = pd.DataFrame(columns=["member_id", "score", "rank"])
        assert scorer.score(df) is None

    def test_labels_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Scorer(tmp_path / "nonexistent.csv")

    def test_labels_missing_columns_raises(self, tmp_path):
        bad = tmp_path / "bad.csv"
        bad.write_text("member_id,outreach\n1,0\n")
        with pytest.raises(ValueError, match="missing columns"):
            Scorer(bad)

    def test_score_fixture_standard(self, scorer):
        """Standard fixture CSV should produce a valid curve."""
        df = pd.read_csv("tests/fixtures/csvs/standard.csv")
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
        curve = scorer.score(df)
        assert len(curve) == len(df)
        # All values must be in [0, 1]
        assert all(0.0 <= v <= 1.0 for v in curve)
        # Curve at N=10000 should be close to the true churn rate
        assert abs(curve[-1] - scorer.churn_rate) < 0.05

    def test_score_fixture_degenerate(self, scorer):
        """Degenerate CSV (all same score) should still produce a curve close to baseline."""
        df = pd.read_csv("tests/fixtures/csvs/all_same_score.csv")
        curve = scorer.score(df)
        assert curve is not None

    def test_precision_at_n_property(self, scorer, all_member_ids):
        from grader.storage.models import CandidateResult, PredictionResult, PredictionStatus
        ranked = list(all_member_ids)[:100]
        df = pd.DataFrame({"member_id": ranked, "score": range(100, 0, -1), "rank": range(1, 101)})
        curve = scorer.score(df)
        result = CandidateResult(
            candidate_name="test",
            repo_url="http://example.com",
            commit_sha="abc",
            precision_curve=curve,
        )
        assert result.precision_at_n(1) == curve[0]
        assert result.precision_at_n(100) == curve[99]
        assert result.precision_at_n(0) is None
        assert result.precision_at_n(101) is None
