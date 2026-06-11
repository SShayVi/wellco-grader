import logging
from io import StringIO

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_EXPORT_URL = "https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"


def fetch_candidates(sheet_id: str) -> list[dict]:
    """
    Fetch the candidate list from a public Google Sheet.

    Returns a list of dicts with keys 'candidate_name' and 'repo_url'.
    Deduplicates by repo_url (last row wins).
    Raises requests.HTTPError if the sheet is not accessible.
    """
    url = _EXPORT_URL.format(sheet_id=sheet_id)
    logger.info("Fetching candidates from sheet %s", sheet_id)

    response = requests.get(url, timeout=15)
    response.raise_for_status()

    df = pd.read_csv(StringIO(response.text))
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    required = {"candidate_name", "repo_url"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Google Sheet is missing required columns: {missing}. "
            f"Found: {list(df.columns)}"
        )

    df = df[["candidate_name", "repo_url"]].dropna(subset=["repo_url"])
    df["repo_url"] = df["repo_url"].str.strip()
    df["candidate_name"] = df["candidate_name"].str.strip()

    # Deduplicate by repo_url, keep last
    df = df.drop_duplicates(subset=["repo_url"], keep="last")

    candidates = df.to_dict("records")
    logger.info("Found %d candidates", len(candidates))
    return candidates
