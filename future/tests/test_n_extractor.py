"""
Unit tests for NExtractorAgent — regex search and LLM extraction.
"""
import pytest
import pandas as pd
from unittest.mock import MagicMock

from grader.agents.n_extractor import NExtractorAgent, _try_flatten_notebook
from grader.storage.models import NSource


@pytest.fixture
def agent():
    mock_gh = MagicMock()
    a = NExtractorAgent(api_key="test-key", github_client=mock_gh)
    a._call = MagicMock()
    return a, mock_gh


def _make_llm_n_response(n: int, confidence: float = 0.9):
    mock_response = MagicMock()
    mock_block = MagicMock()
    mock_block.type = "tool_use"
    mock_block.input = {"n": n, "confidence": confidence, "explanation": "test"}
    mock_response.content = [mock_block]
    return mock_response


# ---------------------------------------------------------------------------
# Regex search
# ---------------------------------------------------------------------------
class TestRegexSearch:
    def test_n_equals_pattern(self, agent):
        a, _ = agent
        assert a._regex_search("We recommend N = 1500 members for outreach.") == 1500

    def test_top_n_pattern(self, agent):
        a, _ = agent
        assert a._regex_search("Contact the top-2000 members first.") == 2000

    def test_recommend_pattern(self, agent):
        a, _ = agent
        assert a._regex_search("We recommend 800 members for targeted outreach.") == 800

    def test_too_small_ignored(self, agent):
        a, _ = agent
        assert a._regex_search("N = 5 epochs") is None

    def test_too_large_ignored(self, agent):
        a, _ = agent
        assert a._regex_search("N = 99999") is None

    def test_no_match_returns_none(self, agent):
        a, _ = agent
        assert a._regex_search("This text has no outreach recommendations.") is None


# ---------------------------------------------------------------------------
# NExtractorAgent.run — various sources
# ---------------------------------------------------------------------------
class TestNExtractorRun:
    def test_n_from_csv_explicit_column(self, agent):
        a, mock_gh = agent
        df = pd.DataFrame({
            "member_id": [1, 2, 3],
            "score": [0.9, 0.8, 0.7],
            "rank": [1, 2, 3],
            "n_recommended": [1200, 1200, 1200],
        })
        result = a.run(MagicMock(), predictions_df=df)
        assert result.n == 1200
        assert result.source == NSource.CSV_EXPLICIT_COLUMN
        assert result.confidence == 1.0

    def test_n_from_readme_regex(self, agent):
        a, mock_gh = agent
        mock_gh.list_files.side_effect = [
            [MagicMock(path="README.md", size=500)],  # docs
            [],   # code
        ]
        mock_gh.download_file.return_value = b"We recommend N = 750 members for outreach."
        result = a.run(MagicMock())
        assert result.n == 750
        assert result.source == NSource.README

    def test_n_from_readme_llm(self, agent):
        a, mock_gh = agent
        mock_gh.list_files.side_effect = [
            [MagicMock(path="README.md", size=500)],
            [],
        ]
        mock_gh.download_file.return_value = b"Based on our analysis we suggest targeting 650 at-risk members."
        a._call.return_value = _make_llm_n_response(650, confidence=0.85)
        result = a.run(MagicMock())
        assert result.n == 650

    def test_fallback_to_row_count(self, agent):
        a, mock_gh = agent
        mock_gh.list_files.return_value = []
        df = pd.DataFrame({"member_id": range(1, 1001), "score": [0.5] * 1000, "rank": range(1, 1001)})
        result = a.run(MagicMock(), predictions_df=df)
        assert result.n == 1000
        assert result.source == NSource.INFERRED

    def test_n_clamped_to_max(self, agent):
        a, mock_gh = agent
        mock_gh.list_files.side_effect = [
            [MagicMock(path="README.md", size=100)],  # text files
            [],  # code files
            [],  # PDF files
        ]
        mock_gh.download_file.return_value = b"We recommend N = 99999 members."
        result = a.run(MagicMock())
        assert result.n <= 10_000

    def test_n_warning_set_for_large_n(self, agent):
        a, _ = agent
        result = a._make_result(6000, NSource.README, 0.9)
        assert result.n_warning is True

    def test_n_warning_clear_for_normal_n(self, agent):
        a, _ = agent
        result = a._make_result(1000, NSource.README, 0.9)
        assert result.n_warning is False

    def test_llm_low_confidence_ignored(self, agent):
        """Low-confidence LLM responses should not be used."""
        a, mock_gh = agent
        mock_gh.list_files.side_effect = [
            [MagicMock(path="README.md", size=100)],  # text files
            [],  # code files
            [],  # PDF files
        ]
        mock_gh.download_file.return_value = b"Some text with no clear N recommendation."
        a._call.return_value = _make_llm_n_response(999, confidence=0.3)
        df = pd.DataFrame({"member_id": range(500), "score": [0.5] * 500, "rank": range(500)})
        result = a.run(MagicMock(), predictions_df=df)
        # Should fall back to inferred from predictions
        assert result.source == NSource.INFERRED


# ---------------------------------------------------------------------------
# Notebook flattening
# ---------------------------------------------------------------------------
class TestFlattenNotebook:
    def test_extracts_code_cells(self):
        nb = b'{"cells": [{"cell_type": "code", "source": ["x = 1\\n", "y = 2"]}, {"cell_type": "markdown", "source": ["# Title"]}]}'
        result = _try_flatten_notebook(nb)
        assert "x = 1" in result
        assert "y = 2" in result

    def test_invalid_json_returns_none(self):
        result = _try_flatten_notebook(b"not json")
        assert result is None

    def test_empty_notebook(self):
        nb = b'{"cells": []}'
        result = _try_flatten_notebook(nb)
        assert result == ""
