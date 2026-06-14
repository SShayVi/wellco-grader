"""
Pipeline orchestrator: reads candidates from Google Sheet, downloads their prediction CSVs,
maps columns, scores against true labels, and writes results to the SQLite cache.

Core invariant: a candidate's CSV is processed at most once per content hash.
"""
import hashlib
import logging
import re
from io import BytesIO
from typing import Optional, List

import pandas as pd
import requests

from config.settings import Settings
from grader.scoring.scorer import Scorer
from grader.sources.google_sheets import fetch_candidates
from grader.storage.cache import ResultCache
from grader.storage.models import CandidateResult, PredictionStatus

logger = logging.getLogger(__name__)

_MEMBER_ID_HINTS = ["member_id", "memberid", "member", "id", "user_id", "userid", "user"]
_SCORE_HINTS = [
    "score", "churn_score", "churn_prob", "churn_probability", "probability",
    "prob", "risk", "risk_score", "pred", "prediction",
]


def run_pipeline(
    settings: Settings,
    override_candidates: Optional[List[dict]] = None,
) -> List[CandidateResult]:
    """
    Process all candidates from the Google Sheet (or override_candidates if provided).

    override_candidates: list of {candidate_name, csv_url, recommended_n} dicts.
    Skips candidates whose CSV content hash is already cached.
    """
    _configure_logging(settings.log_level)

    cache = ResultCache(settings.cache_db_path)
    scorer = Scorer(settings.true_labels_path)

    candidates = override_candidates or fetch_candidates(settings.google_sheet_id)
    if not candidates:
        logger.warning("No candidates found")
        return []

    results: list[CandidateResult] = []
    for c in candidates:
        result = _process_candidate(
            name=c["candidate_name"],
            csv_url=c["csv_url"],
            recommended_n=int(c["recommended_n"]),
            cache=cache,
            scorer=scorer,
            min_overlap=settings.min_member_id_overlap,
        )
        results.append(result)

    ok = sum(1 for r in results if r.status == PredictionStatus.OK)
    logger.info("Pipeline complete: %d total, %d OK, %d errors", len(results), ok, len(results) - ok)
    return results


def _process_candidate(
    *,
    name: str,
    csv_url: str,
    recommended_n: int,
    cache: ResultCache,
    scorer: Scorer,
    min_overlap: float,
) -> CandidateResult:
    logger.info("Processing candidate: %s", name)

    # Step 1: download CSV
    try:
        raw = _download_csv(csv_url)
    except Exception as e:
        logger.error("Download failed for %s: %s", name, e)
        return CandidateResult(
            candidate_name=name,
            csv_url=csv_url,
            recommended_n=recommended_n,
            content_hash="download_error",
            status=PredictionStatus.CSV_DOWNLOAD_ERROR,
            error=str(e),
        )

    content_hash = hashlib.md5(raw).hexdigest()

    # Step 2: cache check
    cached = cache.get(name, content_hash)
    if cached is not None:
        logger.info("Cache hit for %s — skipping", name)
        return cached

    # Step 3: parse CSV
    try:
        df = pd.read_csv(BytesIO(raw))
    except Exception as e:
        result = CandidateResult(
            candidate_name=name, csv_url=csv_url, recommended_n=recommended_n,
            content_hash=content_hash, status=PredictionStatus.SCHEMA_ERROR,
            error=f"Cannot parse CSV: {e}",
        )
        cache.put(result)
        return result

    # Step 4: map columns
    mapping = _map_columns(df)
    if mapping is None:
        cols = list(df.columns)
        result = CandidateResult(
            candidate_name=name, csv_url=csv_url, recommended_n=recommended_n,
            content_hash=content_hash, status=PredictionStatus.SCHEMA_ERROR,
            error=f"Could not identify member_id and score columns. Found: {cols}",
        )
        cache.put(result)
        return result

    # Step 5: normalize
    out = pd.DataFrame()
    out["member_id"] = pd.to_numeric(df[mapping["member_id_col"]], errors="coerce").astype("Int64")
    out["score"] = pd.to_numeric(df[mapping["score_col"]], errors="coerce")
    out = out.dropna(subset=["member_id", "score"])
    out = out.sort_values("score", ascending=False).reset_index(drop=True)

    if out.empty:
        result = CandidateResult(
            candidate_name=name, csv_url=csv_url, recommended_n=recommended_n,
            content_hash=content_hash, status=PredictionStatus.SCHEMA_ERROR,
            error="No valid rows after normalization",
        )
        cache.put(result)
        return result

    # Step 6: overlap check
    candidate_ids = set(out["member_id"].dropna().astype(int))
    true_ids = scorer.true_member_ids
    overlap = len(candidate_ids & true_ids) / len(candidate_ids) if candidate_ids else 0.0

    if overlap < min_overlap:
        result = CandidateResult(
            candidate_name=name, csv_url=csv_url, recommended_n=recommended_n,
            content_hash=content_hash, status=PredictionStatus.INVALID_PREDICTIONS,
            member_id_overlap=overlap,
            error=f"member_id overlap {overlap:.1%} < required {min_overlap:.0%}",
        )
        cache.put(result)
        return result

    # Step 7: score
    status = PredictionStatus.DEGENERATE_PREDICTIONS if out["score"].nunique() == 1 else PredictionStatus.OK
    precision_curve = scorer.score(out)

    result = CandidateResult(
        candidate_name=name, csv_url=csv_url, recommended_n=recommended_n,
        content_hash=content_hash, status=status,
        precision_curve=precision_curve, member_id_overlap=overlap,
    )
    cache.put(result)
    logger.info("%s — %s, precision@%d=%.3f", name, status.value, recommended_n,
                result.precision_at_recommended_n or 0)
    return result


def _download_csv(url: str) -> bytes:
    """Download CSV bytes, handling Google Drive sharing links."""
    url = _normalize_url(url)
    resp = requests.get(url, timeout=60, allow_redirects=True)
    resp.raise_for_status()
    return resp.content


def _normalize_url(url: str) -> str:
    """Convert sharing URLs to direct download URLs."""
    # GitHub blob view → raw
    # https://github.com/{owner}/{repo}/blob/{branch}/{path}
    m = re.match(r"https://github\.com/([^/]+/[^/]+)/blob/(.+)", url)
    if m:
        return f"https://raw.githubusercontent.com/{m.group(1)}/{m.group(2)}"
    # Google Drive: https://drive.google.com/file/d/{ID}/view?...
    m = re.match(r"https://drive\.google\.com/file/d/([^/?]+)", url)
    if m:
        return f"https://drive.google.com/uc?export=download&id={m.group(1)}"
    # Google Drive: https://drive.google.com/open?id={ID}
    m = re.search(r"[?&]id=([^&]+)", url)
    if m and "drive.google.com" in url:
        return f"https://drive.google.com/uc?export=download&id={m.group(1)}"
    return url


def _map_columns(df: pd.DataFrame) -> Optional[dict]:
    """Heuristic column mapper: find member_id and score columns by name."""
    lower = {c.lower().strip(): c for c in df.columns}
    member_col = next((lower[k] for k in _MEMBER_ID_HINTS if k in lower), None)
    score_col = next((lower[k] for k in _SCORE_HINTS if k in lower), None)
    if member_col and score_col:
        logger.debug("Column mapping: member_id=%s, score=%s", member_col, score_col)
        return {"member_id_col": member_col, "score_col": score_col}
    return None


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
