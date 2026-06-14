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

_MEMBER_ID_HINTS = [
    "member_id", "memberid", "member", "id", "user_id", "userid", "user",
    "customer_id", "client_id", "patient_id", "account_id",
]
# Ordered by preference: uplift/CATE → explicit priority → churn prob → generic → rank (inverted)
_SCORE_HINTS = [
    "weighted_uplift", "uplift", "cate", "cate_estimate", "cate_score",
    "benefit_score",
    "propensity_score", "propensity", "priority_score", "prioritization_score",
    "churn_score", "churn_prob", "churn_probability",
    "churn_prob_no_outreach", "p_churn_no_outreach", "baseline_churn_proba",
    "score", "probability", "prob", "risk", "risk_score", "pred", "prediction",
    "rank",
]
# Score columns that represent rank (1 = best) — values are negated before sorting
_RANK_HINTS = {"rank", "ranking", "position"}


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
    if mapping.get("invert_score"):
        out["score"] = -out["score"]  # rank 1 → -1 sorts first when descending
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
    ranked_ids = out["member_id"].astype(int).tolist()

    result = CandidateResult(
        candidate_name=name, csv_url=csv_url, recommended_n=recommended_n,
        content_hash=content_hash, status=status,
        precision_curve=precision_curve, ranked_member_ids=ranked_ids,
        member_id_overlap=overlap,
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
    """Heuristic column mapper: exact match → substring match → 2-column dtype fallback."""
    lower = {c.lower().strip(): c for c in df.columns}

    # 1. Exact match
    member_col = next((lower[k] for k in _MEMBER_ID_HINTS if k in lower), None)
    score_col = next((lower[k] for k in _SCORE_HINTS if k in lower), None)

    # 2. Substring match (hint contained in col name or col name contained in hint)
    if member_col is None:
        for col_l, col_orig in lower.items():
            if any(hint in col_l or col_l in hint for hint in _MEMBER_ID_HINTS):
                member_col = col_orig
                break
    if score_col is None:
        for hint in _SCORE_HINTS:  # preserve priority order
            for col_l, col_orig in lower.items():
                if col_orig == member_col:
                    continue
                if hint in col_l or col_l in hint:
                    score_col = col_orig
                    break
            if score_col:
                break

    # 3. Two-column dtype fallback: one int-like (IDs), one float-like (scores)
    if member_col is None or score_col is None:
        num_cols = [
            c for c in df.columns
            if pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.8
        ]
        if len(num_cols) >= 2 and member_col is None and score_col is None:
            c1, c2 = num_cols[0], num_cols[1]
            c2_01 = pd.to_numeric(df[c2], errors="coerce").between(0, 1).mean() > 0.5
            c1_01 = pd.to_numeric(df[c1], errors="coerce").between(0, 1).mean() > 0.5
            if c2_01 and not c1_01:
                member_col, score_col = c1, c2
            elif c1_01 and not c2_01:
                member_col, score_col = c2, c1
            else:
                member_col, score_col = c1, c2  # best guess

    if member_col and score_col:
        invert = score_col.lower().strip() in _RANK_HINTS or any(
            r in score_col.lower() for r in _RANK_HINTS
        )
        logger.debug("Column mapping: member_id=%s, score=%s, invert=%s", member_col, score_col, invert)
        return {"member_id_col": member_col, "score_col": score_col, "invert_score": invert}
    return None


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
