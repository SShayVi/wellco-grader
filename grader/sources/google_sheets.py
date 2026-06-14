import logging
from io import StringIO

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_EXPORT_URL = "https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"


def fetch_candidates(sheet_id: str) -> list[dict]:
    """
    Fetch the candidate list from a public Google Sheet.

    Expected columns: candidate_name, csv_url, recommended_n
    Returns a list of dicts with those keys.
    """
    url = _EXPORT_URL.format(sheet_id=sheet_id)
    logger.info("Fetching candidates from sheet %s", sheet_id)

    response = requests.get(url, timeout=15)
    response.raise_for_status()

    df = pd.read_csv(StringIO(response.text))
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    required = {"candidate_name", "csv_url", "recommended_n"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Google Sheet is missing required columns: {missing}. "
            f"Found: {list(df.columns)}"
        )

    df = df[["candidate_name", "csv_url", "recommended_n"]].dropna(subset=["candidate_name", "csv_url"])
    df["candidate_name"] = df["candidate_name"].str.strip()
    df["csv_url"] = df["csv_url"].str.strip()
    df["recommended_n"] = pd.to_numeric(df["recommended_n"], errors="coerce").fillna(1000).astype(int)

    # Deduplicate by candidate_name, keep last row
    df = df.drop_duplicates(subset=["candidate_name"], keep="last")

    candidates = df.to_dict("records")
    logger.info("Found %d candidates", len(candidates))
    return candidates
