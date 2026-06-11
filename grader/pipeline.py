"""
Pipeline orchestrator: reads candidates from Google Sheet, runs agents,
scores predictions, and writes results to the SQLite cache.

Core invariant: a repo is processed at most once per commit SHA.
"""
import logging
import sys
from pathlib import Path

from config.settings import Settings
from grader.agents.code_reviewer import CodeReviewerAgent
from grader.agents.n_extractor import NExtractorAgent
from grader.agents.prediction_agent import PredictionAgent
from grader.scoring.scorer import Scorer
from grader.sources.github import GitHubClient, RepoUnavailableError
from grader.sources.google_sheets import fetch_candidates
from grader.storage.cache import ResultCache
from grader.storage.models import CandidateResult, PredictionStatus

logger = logging.getLogger(__name__)


def run_pipeline(settings: Settings) -> list[CandidateResult]:
    """
    Process all candidates in the Google Sheet.

    - Skips candidates whose latest commit SHA is already cached.
    - Continues processing remaining candidates if one fails.
    - Returns the full list of results (cached + newly processed).
    """
    _configure_logging(settings.log_level)

    cache = ResultCache(settings.cache_db_path)
    scorer = Scorer(settings.true_labels_path)
    gh = GitHubClient(token=settings.github_token)

    questions_path = Path("config/review_questions.yaml")
    if not questions_path.exists():
        logger.error("review_questions.yaml not found at %s", questions_path)
        sys.exit(1)

    candidates = fetch_candidates(settings.google_sheet_id)
    if not candidates:
        logger.warning("No candidates found in Google Sheet")
        return []

    prediction_agent = PredictionAgent(
        api_key=settings.anthropic_api_key,
        github_client=gh,
        true_member_ids=scorer.true_member_ids,
        model=settings.anthropic_model,
        max_csv_candidates=settings.max_csv_candidates,
        min_overlap=settings.min_member_id_overlap,
    )
    n_agent = NExtractorAgent(
        api_key=settings.anthropic_api_key,
        github_client=gh,
        model=settings.anthropic_model,
    )
    reviewer = CodeReviewerAgent(
        api_key=settings.anthropic_api_key,
        github_client=gh,
        questions_path=questions_path,
        model=settings.anthropic_model,
        max_chars=settings.max_repo_chars,
    )

    results: list[CandidateResult] = []
    for candidate in candidates:
        name = candidate["candidate_name"]
        url = candidate["repo_url"]
        result = _process_candidate(
            name=name,
            url=url,
            gh=gh,
            cache=cache,
            prediction_agent=prediction_agent,
            n_agent=n_agent,
            reviewer=reviewer,
            scorer=scorer,
        )
        results.append(result)

    logger.info(
        "Pipeline complete: %d total, %d OK, %d errors",
        len(results),
        sum(1 for r in results if r.status == PredictionStatus.OK),
        sum(1 for r in results if r.error),
    )
    return results


def _process_candidate(
    *,
    name: str,
    url: str,
    gh: GitHubClient,
    cache: ResultCache,
    prediction_agent: PredictionAgent,
    n_agent: NExtractorAgent,
    reviewer: CodeReviewerAgent,
    scorer: Scorer,
) -> CandidateResult:
    logger.info("Processing candidate: %s (%s)", name, url)

    # Step 1: get latest commit SHA
    try:
        repo = gh.get_repo(url)
        sha = gh.get_latest_sha(repo)
    except RepoUnavailableError as e:
        logger.warning("Repo unavailable for %s: %s", name, e)
        result = CandidateResult(
            candidate_name=name,
            repo_url=url,
            commit_sha="unavailable",
            prediction_result=None,
            error=str(e),
        )
        # Don't cache unavailable — retry on next run
        return result

    # Step 2: cache check — skip if already processed this SHA
    cached = cache.get(name, sha)
    if cached is not None:
        logger.info("Cache hit for %s @ %s — skipping", name, sha[:7])
        return cached

    # Step 3: run agents
    result = CandidateResult(candidate_name=name, repo_url=url, commit_sha=sha)

    try:
        pred_result, pred_df = prediction_agent.run(repo)
        result.prediction_result = pred_result
        logger.info("%s — predictions: %s", name, pred_result.status)
    except Exception as e:
        logger.error("PredictionAgent failed for %s: %s", name, e, exc_info=True)
        result.error = f"PredictionAgent error: {e}"
        cache.put(result)
        return result

    try:
        n_result = n_agent.run(
            repo,
            predictions_df=pred_df,
            predictions_path=pred_result.csv_path,
        )
        result.n_extraction = n_result
        logger.info("%s — N=%d (source=%s, conf=%.2f)", name, n_result.n, n_result.source, n_result.confidence)
    except Exception as e:
        logger.error("NExtractorAgent failed for %s: %s", name, e, exc_info=True)
        # Non-fatal: continue without N
        result.error = f"NExtractorAgent error: {e}"

    try:
        review = reviewer.run(repo)
        result.review_result = review
        logger.info("%s — review score: %.2f", name, review.weighted_score)
    except Exception as e:
        logger.error("CodeReviewerAgent failed for %s: %s", name, e, exc_info=True)
        # Non-fatal: continue without review

    # Step 4: score predictions
    if pred_df is not None and pred_result.status in (
        PredictionStatus.OK,
        PredictionStatus.DEGENERATE_PREDICTIONS,
    ):
        curve = scorer.score(pred_df)
        result.precision_curve = curve

    # Step 5: persist
    cache.put(result)
    return result


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
