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

    # Flexible column matching — sheet column names may vary
    col_map = _find_columns(df.columns.tolist())
    required_missing = [k for k in ("candidate_name", "csv_url") if col_map.get(k) is None]
    if required_missing:
        raise ValueError(
            f"Google Sheet is missing required columns for: {required_missing}. "
            f"Found: {list(df.columns)}"
        )

    rename = {col_map[k]: k for k in col_map if col_map[k] is not None}
    df = df.rename(columns=rename)

    if "recommended_n" not in df.columns:
        df["recommended_n"] = None  # column absent entirely

    df = df[["candidate_name", "csv_url", "recommended_n"]].dropna(subset=["candidate_name", "csv_url"])
    df["candidate_name"] = df["candidate_name"].str.strip()
    df["csv_url"] = df["csv_url"].str.strip()

    rec_n_raw = pd.to_numeric(df["recommended_n"], errors="coerce")
    df["n_defaulted"] = rec_n_raw.isna()
    df["recommended_n"] = rec_n_raw.fillna(1000).astype(int)

    # Deduplicate by candidate_name, keep last row
    df = df.drop_duplicates(subset=["candidate_name"], keep="last")

    candidates = df.to_dict("records")
    logger.info("Found %d candidates", len(candidates))
    return candidates


def _find_columns(columns: list) -> dict:
    """
    Fuzzy-match sheet columns to the required logical names.
    Handles variations like 'test_csv_url', 'recommended_N', 'Candidate Name', etc.
    """
    lower = {c.lower().replace(" ", "_"): c for c in columns}

    def find(patterns):
        for p in patterns:
            for k, orig in lower.items():
                if p in k:
                    return orig
        return None

    return {
        "candidate_name": find(["candidate_name", "candidate", "name"]),
        "csv_url":        find(["csv_url", "csv", "url", "link"]),
        "recommended_n":  find(["recommended_n", "recommended", "_n", "outreach_n", "n_outreach"]),
    }
