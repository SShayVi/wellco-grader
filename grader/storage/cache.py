import sqlite3
import logging
from pathlib import Path
from typing import Optional

from grader.storage.models import CandidateResult

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS candidate_runs (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_name TEXT NOT NULL,
    csv_url        TEXT NOT NULL,
    content_hash   TEXT NOT NULL,
    result_json    TEXT NOT NULL,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(candidate_name, content_hash)
);
CREATE INDEX IF NOT EXISTS idx_candidate_hash
    ON candidate_runs(candidate_name, content_hash);
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
            # Migrate if the old schema (commit_sha column) is present
            existing = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='candidate_runs'"
            ).fetchone()
            if existing:
                cols = [row[1] for row in conn.execute("PRAGMA table_info(candidate_runs)").fetchall()]
                if "content_hash" not in cols:
                    logger.info("Migrating cache DB: dropping old schema (commit_sha-based)")
                    conn.execute("DROP TABLE IF EXISTS candidate_runs")
                    conn.execute("DROP INDEX IF EXISTS idx_candidate_sha")
            conn.executescript(_SCHEMA)

    def get(self, candidate_name: str, content_hash: str) -> Optional[CandidateResult]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT result_json FROM candidate_runs "
                "WHERE candidate_name = ? AND content_hash = ?",
                (candidate_name, content_hash),
            ).fetchone()
        if row:
            logger.debug("Cache hit for %s (%s)", candidate_name, content_hash[:7])
            return CandidateResult.model_validate_json(row["result_json"])
        return None

    def put(self, result: CandidateResult) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO candidate_runs "
                "(candidate_name, csv_url, content_hash, result_json) "
                "VALUES (?, ?, ?, ?)",
                (
                    result.candidate_name,
                    result.csv_url,
                    result.content_hash,
                    result.model_dump_json(),
                ),
            )
        logger.debug("Cached result for %s", result.candidate_name)

    def clear_all(self) -> None:
        """Delete all cached results (forces full re-grade on next pipeline run)."""
        with self._connect() as conn:
            conn.execute("DELETE FROM candidate_runs")
        logger.info("Cache cleared")

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
