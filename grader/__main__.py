"""
CLI entry point: python -m grader

Usage:
  python -m grader                                                    # reads from Google Sheet
  python -m grader --candidate "Name" https://url/to/file.csv 1000  # single candidate
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
        nargs=3,
        metavar=("NAME", "CSV_URL", "RECOMMENDED_N"),
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
        name, csv_url, n_str = args.candidate
        try:
            recommended_n = int(n_str)
        except ValueError:
            print(f"Error: RECOMMENDED_N must be an integer, got '{n_str}'", file=sys.stderr)
            sys.exit(1)
        override = [{"candidate_name": name, "csv_url": csv_url, "recommended_n": recommended_n}]

    if not override and not settings.google_sheet_id:
        print("Error: GOOGLE_SHEET_ID is required when not using --candidate.", file=sys.stderr)
        sys.exit(1)

    results = run_pipeline(settings, override_candidates=override)

    print(f"\n{'='*60}")
    print(f"Pipeline complete — {len(results)} candidate(s)")
    print(f"{'='*60}")
    for r in results:
        p = r.precision_at_recommended_n
        p_str = f"precision@{r.recommended_n}={p:.3f}" if p is not None else "no score"
        print(f"  {r.candidate_name:<25} {r.status.value:<28} N={r.recommended_n:<6} {p_str}")
        if r.error:
            print(f"    ERROR: {r.error}")


if __name__ == "__main__":
    main()
