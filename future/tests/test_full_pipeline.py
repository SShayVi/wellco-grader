"""
Integration tests — run against real GitHub + Anthropic API.
Gated behind pytest --integration flag.
"""
import pytest
from pathlib import Path

from config.settings import Settings
from grader.agents.code_reviewer import CodeReviewerAgent
from grader.agents.n_extractor import NExtractorAgent
from grader.agents.prediction_agent import PredictionAgent
from grader.scoring.scorer import Scorer
from grader.sources.github import GitHubClient
from grader.storage.models import PredictionStatus

SHAY_REPO = "https://github.com/SShayVi/wellco-churn-prediction-home-assignment"
LABELS_PATH = Path("data/test_churn_labels.csv")
QUESTIONS_PATH = Path("config/review_questions.yaml")


@pytest.fixture(scope="module")
def settings():
    try:
        return Settings()
    except Exception as e:
        pytest.skip(f"Settings not configured: {e}")


@pytest.fixture(scope="module")
def scorer():
    if not LABELS_PATH.exists():
        pytest.skip(f"Labels not found at {LABELS_PATH}")
    return Scorer(LABELS_PATH)


@pytest.fixture(scope="module")
def gh(settings):
    return GitHubClient(token=settings.github_token)


@pytest.fixture(scope="module")
def repo(gh):
    return gh.get_repo(SHAY_REPO)


@pytest.mark.integration
class TestPredictionAgentIntegration:
    def test_finds_and_normalizes_predictions(self, settings, gh, repo, scorer):
        agent = PredictionAgent(
            api_key=settings.anthropic_api_key,
            github_client=gh,
            true_member_ids=scorer.true_member_ids,
            model=settings.anthropic_model,
        )
        pred_result, pred_df = agent.run(repo)

        assert pred_result.status in (
            PredictionStatus.OK,
            PredictionStatus.DEGENERATE_PREDICTIONS,
        ), f"Unexpected status: {pred_result.status}"
        assert pred_df is not None
        assert set(pred_df.columns) >= {"member_id", "score", "rank"}
        assert len(pred_df) > 0
        assert pred_result.member_id_overlap >= 0.5

    def test_predictions_sorted_by_score(self, settings, gh, repo, scorer):
        agent = PredictionAgent(
            api_key=settings.anthropic_api_key,
            github_client=gh,
            true_member_ids=scorer.true_member_ids,
            model=settings.anthropic_model,
        )
        _, pred_df = agent.run(repo)
        if pred_df is not None:
            scores = pred_df["score"].tolist()
            assert scores == sorted(scores, reverse=True)


@pytest.mark.integration
class TestNExtractorIntegration:
    def test_extracts_reasonable_n(self, settings, gh, repo):
        agent = NExtractorAgent(
            api_key=settings.anthropic_api_key,
            github_client=gh,
            model=settings.anthropic_model,
        )
        result = agent.run(repo)
        assert 1 <= result.n <= 10_000
        assert 0.0 <= result.confidence <= 1.0


@pytest.mark.integration
class TestCodeReviewerIntegration:
    def test_reviews_all_questions(self, settings, gh, repo):
        if not QUESTIONS_PATH.exists():
            pytest.skip("review_questions.yaml not found")

        agent = CodeReviewerAgent(
            api_key=settings.anthropic_api_key,
            github_client=gh,
            questions_path=QUESTIONS_PATH,
            model=settings.anthropic_model,
        )
        result = agent.run(repo)

        assert 0.0 <= result.weighted_score <= 1.0
        assert len(result.questions) == 8  # matches review_questions.yaml

        for q in result.questions:
            assert q.score in (0, 1, 2)
            assert len(q.justification) > 0
            assert q.weight > 0


@pytest.mark.integration
class TestFullPipelineIntegration:
    def test_full_pipeline_shay_repo(self, settings, scorer, gh, repo, tmp_db):
        from grader.storage.cache import ResultCache
        from grader.pipeline import _process_candidate

        cache = ResultCache(tmp_db)

        prediction_agent = PredictionAgent(
            api_key=settings.anthropic_api_key,
            github_client=gh,
            true_member_ids=scorer.true_member_ids,
            model=settings.anthropic_model,
        )
        n_agent = NExtractorAgent(
            api_key=settings.anthropic_api_key,
            github_client=gh,
            model=settings.anthropic_model,
        )
        reviewer = CodeReviewerAgent(
            api_key=settings.anthropic_api_key,
            github_client=gh,
            questions_path=QUESTIONS_PATH,
            model=settings.anthropic_model,
        )

        result = _process_candidate(
            name="Shay Shavit",
            url=SHAY_REPO,
            gh=gh,
            cache=cache,
            prediction_agent=prediction_agent,
            n_agent=n_agent,
            reviewer=reviewer,
            scorer=scorer,
        )

        assert result.candidate_name == "Shay Shavit"
        assert result.commit_sha != "unavailable"
        assert result.prediction_result is not None

        # Should be cached now
        cached = cache.get("Shay Shavit", result.commit_sha)
        assert cached is not None

        # Second call should hit cache
        result2 = _process_candidate(
            name="Shay Shavit",
            url=SHAY_REPO,
            gh=gh,
            cache=cache,
            prediction_agent=prediction_agent,
            n_agent=n_agent,
            reviewer=reviewer,
            scorer=scorer,
        )
        assert result2.commit_sha == result.commit_sha
