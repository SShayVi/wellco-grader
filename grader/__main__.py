"""
CLI entry point: python -m grader

Usage:
  python -m grader                                                    # reads from Google Sheet
  python -m grader --candidate "Name" https://url/to/file.csv 1000  # single candidate
  python -m grader validate https://url/to/file.csv                  # validate a CSV only
"""
import argparse
import sys

import requests

from config.settings import Settings
from grader.pipeline import run_pipeline
from grader.pipeline import _normalize_url
from grader.storage.models import PredictionStatus
from grader.validation import validate_and_standardize


def _run_validate(target: str) -> None:
    if not target:
        print("Usage: python -m grader validate <CSV_URL_or_PATH>", file=sys.stderr)
        sys.exit(1)

    # Load true labels for overlap check (optional — skip if not configured)
    true_member_ids = None
    try:
        settings = Settings()
        import pandas as pd
        labels = pd.read_csv(settings.true_labels_path)
        true_member_ids = set(labels["member_id"].astype(int))
    except Exception:
        print("[INFO] True labels not found — overlap check skipped.\n")

    # Download or read CSV
    try:
        if target.startswith("http"):
            url = _normalize_url(target)
            resp = requests.get(url, timeout=60, allow_redirects=True)
            resp.raise_for_status()
            raw = resp.content
        else:
            with open(target, "rb") as f:
                raw = f.read()
    except Exception as exc:
        print(f"[ERROR] Could not fetch CSV: {exc}", file=sys.stderr)
        sys.exit(1)

    result = validate_and_standardize(raw, true_member_ids=true_member_ids)

    print(result.summary())
    print()
    if result.ok:
        print(f"✓ Validation passed. {result.row_count:,} members, ready to submit.")
        if result.standardized is not None:
            print("\nStandardised preview (first 5 rows):")
            print(result.standardized.head().to_string(index=False))
    else:
        print("✗ Validation failed — fix the errors above before submitting.")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="WellCo Grader pipeline")
    parser.add_argument(
        "--candidate",
        nargs=3,
        metavar=("NAME", "CSV_URL", "RECOMMENDED_N"),
        help="Process a single candidate without reading the Google Sheet",
    )
    parser.add_argument(
        "validate",
        nargs="?",
        metavar="validate",
        help="Subcommand: validate a CSV file or URL",
    )
    parser.add_argument(
        "csv_target",
        nargs="?",
        metavar="CSV_URL_OR_PATH",
        help="URL or local path for the 'validate' subcommand",
    )
    args = parser.parse_args()

    if args.validate == "validate":
        _run_validate(args.csv_target)
        return

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
