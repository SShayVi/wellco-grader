import sqlite3
import logging
from pathlib import Path
from typing import Optional

from grader.storage.models import CandidateResult

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS candidate_runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_name TEXT NOT NULL,
    repo_url       TEXT NOT NULL,
    commit_sha     TEXT NOT NULL,
    result_json    TEXT NOT NULL,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(candidate_name, commit_sha)
);
CREATE INDEX IF NOT EXISTS idx_candidate_sha
    ON candidate_runs(candidate_name, commit_sha);
"""


class ResultCache:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def get(self, candidate_name: str, commit_sha: str) -> Optional[CandidateResult]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT result_json FROM candidate_runs "
                "WHERE candidate_name = ? AND commit_sha = ?",
                (candidate_name, commit_sha),
            ).fetchone()
        if row:
            logger.debug("Cache hit for %s @ %s", candidate_name, commit_sha[:7])
            return CandidateResult.model_validate_json(row["result_json"])
        return None

    def put(self, result: CandidateResult) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO candidate_runs "
                "(candidate_name, repo_url, commit_sha, result_json) "
                "VALUES (?, ?, ?, ?)",
                (
                    result.candidate_name,
                    result.repo_url,
                    result.commit_sha,
                    result.model_dump_json(),
                ),
            )
        logger.debug("Cached result for %s @ %s", result.candidate_name, result.commit_sha[:7])

    def get_all_latest(self) -> list[CandidateResult]:
        """Return the most recent result per candidate, ordered by candidate name."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT result_json FROM candidate_runs "
                "WHERE id IN ("
                "  SELECT MAX(id) FROM candidate_runs GROUP BY candidate_name"
                ") ORDER BY candidate_name"
            ).fetchall()
        return [CandidateResult.model_validate_json(row["result_json"]) for row in rows]

    def has(self, candidate_name: str, commit_sha: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM candidate_runs "
                "WHERE candidate_name = ? AND commit_sha = ?",
                (candidate_name, commit_sha),
            ).fetchone()
        return row is not None
