"""
CLI entry point: python -m grader

Usage:
  python -m grader                                         # reads from Google Sheet
  python -m grader --candidate "Name" https://github.com/x/y  # single repo, no sheet needed
"""
import argparse
import sys

from config.settings import Settings
from grader.pipeline import run_pipeline
from grader.storage.models import PredictionStatus


def main() -> None:
    parser = argparse.ArgumentParser(description="WellCo Grader pipeline")
    parser.add_argument(
        "--candidate",
        nargs=2,
        metavar=("NAME", "REPO_URL"),
        help="Process a single candidate without reading the Google Sheet",
    )
    args = parser.parse_args()

    try:
        settings = Settings()
    except Exception as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        print("Copy .env.example to .env and fill in required values.", file=sys.stderr)
        sys.exit(1)

    override = None
    if args.candidate:
        name, url = args.candidate
        override = [{"candidate_name": name, "repo_url": url}]

    if not override and not settings.google_sheet_id:
        print("Error: GOOGLE_SHEET_ID is required when not using --candidate.", file=sys.stderr)
        sys.exit(1)

    results = run_pipeline(settings, override_candidates=override)

    print(f"\n{'='*60}")
    print(f"Pipeline complete — {len(results)} candidate(s)")
    print(f"{'='*60}")
    for r in results:
        status = r.status.value
        n_str = f"N={r.recommended_n}" if r.recommended_n else "N=?"
        p_str = (
            f"precision@{r.recommended_n}={r.precision_at_recommended_n:.3f}"
            if r.precision_at_recommended_n is not None
            else "no score"
        )
        review_str = (
            f"review={r.review_result.weighted_score:.2f}"
            if r.review_result
            else "review=?"
        )
        print(f"  {r.candidate_name:<25} {status:<25} {n_str:<10} {p_str:<30} {review_str}")
        if r.error:
            print(f"    ERROR: {r.error}")


if __name__ == "__main__":
    main()
