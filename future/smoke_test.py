"""
Smoke test — exercises the full pipeline without LLM calls.
LLM steps (schema ID, N extraction, code review) fall back to heuristics
or are skipped. Everything else runs for real: GitHub, scoring, cache.

Run: python smoke_test.py
"""
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("smoke_test")

from config.settings import Settings
from grader.sources.google_sheets import fetch_candidates
from grader.sources.github import GitHubClient
from grader.agents.prediction_agent import PredictionAgent
from grader.agents.n_extractor import NExtractorAgent
from grader.scoring.scorer import Scorer
from grader.storage.cache import ResultCache
from grader.storage.models import PredictionStatus


def make_heuristic_llm_response(df):
    """Return a mock LLM response that just uses heuristic column names."""
    from grader.agents.prediction_agent import _heuristic_map
    mapping = _heuristic_map(list(df.columns))
    if not mapping:
        raise ValueError(f"Heuristic failed for columns: {list(df.columns)}")
    mock_response = MagicMock()
    mock_block = MagicMock()
    mock_block.type = "tool_use"
    mock_block.input = {
        "member_id_col": mapping["member_id_col"],
        "score_col": mapping["score_col"],
        "rank_col": mapping.get("rank_col"),
        "confidence": 0.8,
        "reasoning": "heuristic smoke test",
    }
    mock_response.content = [mock_block]
    return mock_response


def main():
    settings = Settings()
    sep = "=" * 60

    # ── 1. Google Sheet ──────────────────────────────────────────
    print(f"\n{sep}\n1. Reading Google Sheet\n{sep}")
    candidates = fetch_candidates(settings.google_sheet_id)
    for c in candidates:
        print(f"  ✓  {c['candidate_name']}  →  {c['repo_url']}")

    # ── 2. GitHub ────────────────────────────────────────────────
    print(f"\n{sep}\n2. Connecting to GitHub\n{sep}")
    gh = GitHubClient(token=settings.github_token)
    scorer = Scorer(settings.true_labels_path)
    cache = ResultCache(settings.cache_db_path)

    for candidate in candidates:
        name = candidate["candidate_name"]
        url  = candidate["repo_url"]
        print(f"\n── {name} ({url})")

        try:
            repo = gh.get_repo(url)
            sha  = gh.get_latest_sha(repo)
            print(f"  Repo: {repo.full_name}  SHA: {sha[:7]}")
        except Exception as e:
            print(f"  ✗ Repo unavailable: {e}")
            continue

        if cache.has(name, sha):
            print("  ✓ Already cached — skipping agents")
            continue

        # ── 3. File listing ──────────────────────────────────────
        all_files = gh.list_files(repo)
        csv_files = [f for f in all_files if f.path.lower().endswith(".csv")]
        print(f"  Files in repo: {len(all_files)}  |  CSVs: {len(csv_files)}")
        for f in sorted(csv_files, key=lambda x: -x.size)[:5]:
            print(f"    {f.size:>8,}B  {f.path}")

        # ── 4. Prediction agent (heuristic, no LLM) ──────────────
        print("\n  Running PredictionAgent (heuristic mode)…")
        pred_agent = PredictionAgent(
            api_key=settings.anthropic_api_key,
            github_client=gh,
            true_member_ids=scorer.true_member_ids,
            model=settings.anthropic_model,
        )

        # Patch _call to use heuristic response based on whatever columns we see
        _original_identify = pred_agent._identify_columns
        def heuristic_identify(df, path):
            from grader.agents.prediction_agent import _heuristic_map
            mapping = _heuristic_map(list(df.columns))
            if mapping:
                mapping["confidence"] = 0.8
            return mapping
        pred_agent._identify_columns = heuristic_identify

        pred_result, pred_df = pred_agent.run(repo)
        print(f"  Status:  {pred_result.status.value}")

        if pred_df is not None:
            print(f"  Rows:    {len(pred_df):,}")
            print(f"  Overlap: {pred_result.member_id_overlap:.1%}")
            print(f"  Columns: {list(pred_df.columns)}")
            print(f"  Top 3:")
            print(pred_df.head(3).to_string(index=False))

        # ── 5. N extraction (row count, no LLM) ──────────────────
        n = len(pred_df) if pred_df is not None else None
        print(f"\n  N (from row count): {n}")

        # ── 6. Scoring ───────────────────────────────────────────
        if pred_df is not None and pred_result.status in (
            PredictionStatus.OK, PredictionStatus.DEGENERATE_PREDICTIONS
        ):
            curve = scorer.score(pred_df)
            if curve and n:
                p_at_n   = curve[min(n, len(curve)) - 1]
                p_at_500 = curve[499] if len(curve) >= 500 else None
                p_at_1k  = curve[999] if len(curve) >= 1000 else None
                print(f"\n  Scoring results (baseline = {scorer.baseline_precision:.1%}):")
                print(f"    precision@{n:,}  = {p_at_n:.3f}")
                if p_at_500: print(f"    precision@500   = {p_at_500:.3f}")
                if p_at_1k:  print(f"    precision@1000  = {p_at_1k:.3f}")
                print(f"    precision@10k   = {curve[-1]:.3f}  (should ≈ {scorer.baseline_precision:.3f})")
        else:
            print("\n  ✗ Skipping scoring (no valid predictions)")

    print(f"\n{sep}\nSmoke test complete.\n{sep}\n")


if __name__ == "__main__":
    main()
