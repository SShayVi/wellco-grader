"""
CLI entry point: python -m grader
"""
import sys

from config.settings import Settings
from grader.pipeline import run_pipeline
from grader.storage.models import PredictionStatus


def main() -> None:
    try:
        settings = Settings()
    except Exception as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        print("Copy .env.example to .env and fill in required values.", file=sys.stderr)
        sys.exit(1)

    results = run_pipeline(settings)

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
