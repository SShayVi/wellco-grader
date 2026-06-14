"""
Seeds the cache with the data we can compute without an LLM:
  - predictions (found + normalized from GitHub)
  - precision curve (scored against true labels)
  - N (from row count / heuristic)
  - review placeholder (shows "pending" in dashboard until API key is added)

Run: python seed_cache.py
"""
import logging
from io import BytesIO
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("seed_cache")

from config.settings import Settings
from grader.sources.github import GitHubClient
from grader.agents.prediction_agent import PredictionAgent
from grader.agents.n_extractor import NExtractorAgent
from grader.scoring.scorer import Scorer
from grader.storage.cache import ResultCache
from grader.storage.models import (
    CandidateResult, NExtractionResult, NSource, PredictionStatus
)
from grader.sources.google_sheets import fetch_candidates


def main():
    settings = Settings()
    gh      = GitHubClient(token=settings.github_token)
    scorer  = Scorer(settings.true_labels_path)
    cache   = ResultCache(settings.cache_db_path)

    candidates = fetch_candidates(settings.google_sheet_id)
    log.info("Processing %d candidate(s) from sheet", len(candidates))

    for c in candidates:
        name, url = c["candidate_name"], c["repo_url"]
        log.info("── %s", name)

        repo = gh.get_repo(url)
        sha  = gh.get_latest_sha(repo)

        if cache.has(name, sha):
            log.info("Already cached — skipping")
            continue

        # --- Predictions (heuristic, no LLM) ---
        pred_agent = PredictionAgent(
            api_key="dummy",
            github_client=gh,
            true_member_ids=scorer.true_member_ids,
            model=settings.anthropic_model,
        )
        def heuristic_identify(df, path):
            from grader.agents.prediction_agent import _heuristic_map
            m = _heuristic_map(list(df.columns))
            if m: m["confidence"] = 0.8
            return m
        pred_agent._identify_columns = heuristic_identify

        pred_result, pred_df = pred_agent.run(repo)
        log.info("Predictions: %s  rows=%s  overlap=%s",
                 pred_result.status.value,
                 len(pred_df) if pred_df is not None else "N/A",
                 f"{pred_result.member_id_overlap:.0%}" if pred_result.member_id_overlap else "N/A")

        # --- N (row count from predictions) ---
        n_result = None
        if pred_df is not None:
            # Also try regex on README without LLM
            n_agent = NExtractorAgent(api_key="dummy", github_client=gh)
            n_agent._llm_extract_n = lambda text, src: None  # disable LLM
            n_result = n_agent.run(repo, predictions_df=pred_df)
            log.info("N=%d source=%s", n_result.n, n_result.source.value)

        # --- Precision curve ---
        curve = None
        if pred_df is not None and pred_result.status in (
            PredictionStatus.OK, PredictionStatus.DEGENERATE_PREDICTIONS
        ):
            curve = scorer.score(pred_df)
            if curve and n_result:
                idx = n_result.n - 1
                p = curve[min(idx, len(curve)-1)]
                log.info("precision@%d = %.3f  (baseline %.1f%%)", n_result.n, p, scorer.baseline_precision*100)

        result = CandidateResult(
            candidate_name=name,
            repo_url=url,
            commit_sha=sha,
            prediction_result=pred_result,
            n_extraction=n_result,
            review_result=None,   # requires API key — will be filled on next full run
            precision_curve=curve,
        )
        cache.put(result)
        log.info("Cached ✓")

    log.info("Done. Run: streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
