"""
Pipeline orchestrator: reads candidates from Google Sheet, downloads their prediction CSVs,
maps columns, scores against true labels, and writes results to the SQLite cache.

Core invariant: a candidate's CSV is processed at most once per content hash.
"""
import hashlib
import logging
import re
from typing import List, Optional

import requests

from config.settings import Settings
from grader.scoring.scorer import Scorer
from grader.sources.google_sheets import fetch_candidates
from grader.storage.cache import ResultCache
from grader.storage.models import CandidateResult, PredictionStatus
from grader.validation import Severity, validate_and_standardize

logger = logging.getLogger(__name__)


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
            n_defaulted=bool(c.get("n_defaulted", False)),
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
    n_defaulted: bool = False,
    cache: ResultCache,
    scorer: Scorer,
    min_overlap: float,
) -> CandidateResult:
    logger.info("Processing candidate: %s", name)
    notes = "Rec. N defaulted to 1,000 (not set in sheet)" if n_defaulted else None

    # Step 1: download CSV
    try:
        raw = _download_csv(csv_url)
    except Exception as e:
        logger.error("Download failed for %s: %s", name, e)
        url_hash = "url:" + hashlib.md5(csv_url.encode()).hexdigest()
        result = CandidateResult(
            candidate_name=name,
            csv_url=csv_url,
            recommended_n=recommended_n,
            content_hash=url_hash,
            status=PredictionStatus.CSV_DOWNLOAD_ERROR,
            error=str(e),
            notes=notes,
        )
        cache.put(result)
        return result

    content_hash = hashlib.md5(raw).hexdigest()

    # Step 2: cache check
    cached = cache.get(name, content_hash)
    if cached is not None:
        logger.info("Cache hit for %s — skipping", name)
        return cached

    # Steps 3–6: validate and standardise
    vr = validate_and_standardize(raw, true_member_ids=scorer.true_member_ids,
                                  min_overlap=min_overlap)

    if not vr.ok:
        error_codes = {i.code for i in vr.errors()}
        if error_codes & {"WRONG_DATASET", "LOW_OVERLAP"}:
            status = PredictionStatus.INVALID_PREDICTIONS
        else:
            status = PredictionStatus.SCHEMA_ERROR
        result = CandidateResult(
            candidate_name=name, csv_url=csv_url, recommended_n=recommended_n,
            content_hash=content_hash, status=status,
            member_id_overlap=vr.overlap_pct,
            error="; ".join(i.message for i in vr.errors()),
            notes=notes,
        )
        cache.put(result)
        return result

    out = vr.standardized

    # Step 7: score
    has_degenerate = any(i.code == "DEGENERATE_SCORES" for i in vr.issues)
    status = PredictionStatus.DEGENERATE_PREDICTIONS if has_degenerate else PredictionStatus.OK
    curves = scorer.score_all(out)
    ranked_ids = out["member_id"].astype(int).tolist()
    ranked_scores = out["score"].tolist()

    result = CandidateResult(
        candidate_name=name, csv_url=csv_url, recommended_n=recommended_n,
        content_hash=content_hash, status=status,
        precision_curve=curves.get("precision"),
        gain_curve=curves.get("gain"),
        lift_curve=curves.get("lift"),
        qini_curve=curves.get("qini"),
        ranked_member_ids=ranked_ids,
        ranked_scores=ranked_scores, member_id_overlap=vr.overlap_pct,
        notes=notes,
    )
    cache.put(result)
    logger.info(
        "%s — %s, precision@%d=%.3f, gain@%d=%.3f, lift@%d=%.2f",
        name, status.value,
        recommended_n, result.precision_at_n(recommended_n) or 0,
        recommended_n, result.gain_at_n(recommended_n) or 0,
        recommended_n, result.lift_at_n(recommended_n) or 0,
    )
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


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
