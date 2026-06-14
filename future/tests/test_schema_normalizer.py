"""
Unit tests for PredictionAgent schema normalization.
LLM calls are mocked; file I/O uses fixture CSVs.
"""
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch

from grader.agents.prediction_agent import PredictionAgent, _heuristic_map
from grader.storage.models import PredictionStatus

FIXTURES = Path("tests/fixtures/csvs")


# ---------------------------------------------------------------------------
# Heuristic mapper
# ---------------------------------------------------------------------------
class TestHeuristicMap:
    def test_standard_names(self):
        result = _heuristic_map(["member_id", "score", "rank"])
        assert result["member_id_col"] == "member_id"
        assert result["score_col"] == "score"
        assert result["rank_col"] == "rank"

    def test_fuzzy_names(self):
        result = _heuristic_map(["user", "churn_probability", "priority"])
        assert result["member_id_col"] == "user"
        assert result["score_col"] == "churn_probability"
        assert result["rank_col"] == "priority"

    def test_no_rank_column(self):
        result = _heuristic_map(["member_id", "churn_score"])
        assert result["member_id_col"] == "member_id"
        assert result["score_col"] == "churn_score"
        assert result["rank_col"] is None

    def test_missing_score_returns_none(self):
        result = _heuristic_map(["member_id", "some_random_col"])
        assert result is None

    def test_missing_member_id_returns_none(self):
        result = _heuristic_map(["score", "rank"])
        assert result is None

    def test_case_insensitive(self):
        result = _heuristic_map(["MEMBER_ID", "SCORE", "RANK"])
        assert result is not None

    def test_all_hint_variants_for_score(self):
        for col in ["score", "churn_prob", "probability", "risk", "churn_score", "pred", "prediction"]:
            result = _heuristic_map(["member_id", col])
            assert result is not None, f"Should match score column: {col}"


# ---------------------------------------------------------------------------
# PredictionAgent with mocked LLM + GitHub
# ---------------------------------------------------------------------------
@pytest.fixture
def agent_with_mocks(all_member_ids):
    mock_gh = MagicMock()
    agent = PredictionAgent(
        api_key="test-key",
        github_client=mock_gh,
        true_member_ids=all_member_ids,
        min_overlap=0.5,
    )
    # Bypass real LLM calls
    agent._call = MagicMock()
    return agent, mock_gh


def _make_llm_response(member_id_col, score_col, rank_col=None, confidence=0.95):
    """Build a fake Anthropic message response for tool_use."""
    mock_response = MagicMock()
    mock_block = MagicMock()
    mock_block.type = "tool_use"
    mock_block.input = {
        "member_id_col": member_id_col,
        "score_col": score_col,
        "rank_col": rank_col,
        "confidence": confidence,
        "reasoning": "test",
    }
    mock_response.content = [mock_block]
    return mock_response


class TestPredictionAgentNormalization:
    def test_standard_csv(self, agent_with_mocks, all_member_ids):
        agent, mock_gh = agent_with_mocks
        df = pd.read_csv(FIXTURES / "standard.csv")

        mock_gh.list_files.return_value = [MagicMock(path="output.csv", size=len(df) * 30)]
        mock_gh.download_file.return_value = (FIXTURES / "standard.csv").read_bytes()
        agent._call.return_value = _make_llm_response("member_id", "score", "rank")

        mock_repo = MagicMock()
        pred_result, pred_df = agent.run(mock_repo)

        assert pred_result.status == PredictionStatus.OK
        assert pred_df is not None
        assert set(pred_df.columns) >= {"member_id", "score", "rank"}
        assert len(pred_df) == len(df)

    def test_fuzzy_columns(self, agent_with_mocks):
        agent, mock_gh = agent_with_mocks
        raw = (FIXTURES / "fuzzy_columns.csv").read_bytes()

        mock_gh.list_files.return_value = [MagicMock(path="preds.csv", size=len(raw))]
        mock_gh.download_file.return_value = raw
        agent._call.return_value = _make_llm_response("user", "churn_probability", "priority")

        pred_result, pred_df = agent.run(MagicMock())

        assert pred_result.status == PredictionStatus.OK
        assert pred_df is not None
        assert "member_id" in pred_df.columns
        assert "score" in pred_df.columns

    def test_degenerate_predictions_flagged(self, agent_with_mocks, all_member_ids):
        agent, mock_gh = agent_with_mocks
        raw = (FIXTURES / "all_same_score.csv").read_bytes()

        mock_gh.list_files.return_value = [MagicMock(path="output.csv", size=len(raw))]
        mock_gh.download_file.return_value = raw
        agent._call.return_value = _make_llm_response("member_id", "score", "rank")

        pred_result, pred_df = agent.run(MagicMock())

        assert pred_result.status == PredictionStatus.DEGENERATE_PREDICTIONS

    def test_wrong_member_ids_skipped(self, agent_with_mocks):
        agent, mock_gh = agent_with_mocks
        raw = (FIXTURES / "wrong_member_ids.csv").read_bytes()

        mock_gh.list_files.return_value = [MagicMock(path="output.csv", size=len(raw))]
        mock_gh.download_file.return_value = raw
        agent._call.return_value = _make_llm_response("member_id", "score", "rank", confidence=0.95)

        pred_result, pred_df = agent.run(MagicMock())

        # No CSV passes overlap check, so should return SCHEMA_ERROR
        assert pred_result.status == PredictionStatus.SCHEMA_ERROR
        assert pred_df is None

    def test_no_csvs_found(self, agent_with_mocks):
        agent, mock_gh = agent_with_mocks
        mock_gh.list_files.return_value = []

        pred_result, pred_df = agent.run(MagicMock())

        assert pred_result.status == PredictionStatus.MISSING_PREDICTIONS
        assert pred_df is None

    def test_missing_rank_column_still_normalizes(self, agent_with_mocks):
        agent, mock_gh = agent_with_mocks
        raw = (FIXTURES / "missing_rank.csv").read_bytes()

        mock_gh.list_files.return_value = [MagicMock(path="output.csv", size=len(raw))]
        mock_gh.download_file.return_value = raw
        agent._call.return_value = _make_llm_response("member_id", "churn_score", rank_col=None)

        pred_result, pred_df = agent.run(MagicMock())

        # Should succeed — rank is derived from score ordering
        assert pred_result.status in (PredictionStatus.OK, PredictionStatus.DEGENERATE_PREDICTIONS)
        assert pred_df is not None
        assert "rank" in pred_df.columns

    def test_llm_fallback_to_heuristics(self, agent_with_mocks):
        """If LLM returns low confidence, heuristic map should kick in."""
        agent, mock_gh = agent_with_mocks
        raw = (FIXTURES / "standard.csv").read_bytes()

        mock_gh.list_files.return_value = [MagicMock(path="output.csv", size=len(raw))]
        mock_gh.download_file.return_value = raw
        # LLM returns low confidence
        agent._call.return_value = _make_llm_response("member_id", "score", confidence=0.3)

        pred_result, pred_df = agent.run(MagicMock())

        # Heuristics should handle standard column names
        assert pred_result.status in (PredictionStatus.OK, PredictionStatus.DEGENERATE_PREDICTIONS)

    def test_llm_exception_falls_back_to_heuristics(self, agent_with_mocks):
        agent, mock_gh = agent_with_mocks
        raw = (FIXTURES / "standard.csv").read_bytes()

        mock_gh.list_files.return_value = [MagicMock(path="output.csv", size=len(raw))]
        mock_gh.download_file.return_value = raw
        agent._call.side_effect = Exception("API error")

        pred_result, pred_df = agent.run(MagicMock())

        assert pred_result.status in (PredictionStatus.OK, PredictionStatus.DEGENERATE_PREDICTIONS)

    def test_output_sorted_by_score_descending(self, agent_with_mocks):
        agent, mock_gh = agent_with_mocks
        raw = (FIXTURES / "standard.csv").read_bytes()

        mock_gh.list_files.return_value = [MagicMock(path="output.csv", size=len(raw))]
        mock_gh.download_file.return_value = raw
        agent._call.return_value = _make_llm_response("member_id", "score", "rank")

        _, pred_df = agent.run(MagicMock())

        assert pred_df is not None
        scores = pred_df["score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_ranks_are_contiguous_from_one(self, agent_with_mocks):
        agent, mock_gh = agent_with_mocks
        raw = (FIXTURES / "standard.csv").read_bytes()

        mock_gh.list_files.return_value = [MagicMock(path="output.csv", size=len(raw))]
        mock_gh.download_file.return_value = raw
        agent._call.return_value = _make_llm_response("member_id", "score", "rank")

        _, pred_df = agent.run(MagicMock())

        assert pred_df is not None
        assert pred_df["rank"].tolist() == list(range(1, len(pred_df) + 1))
