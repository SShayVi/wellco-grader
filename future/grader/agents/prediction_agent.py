"""
Finds the predictions CSV in a GitHub repo, normalizes its schema,
and returns a standardized DataFrame with columns (member_id, score, rank).
"""
import logging
from io import BytesIO
from typing import Optional

import pandas as pd

from grader.agents.base import BaseAgent
from grader.sources.github import GitHubClient
from github.Repository import Repository
from grader.storage.models import (
    PredictionResult,
    PredictionStatus,
    SchemaMapping,
)

logger = logging.getLogger(__name__)

_MEMBER_ID_HINTS = {"member_id", "memberid", "member", "id", "user_id", "userid", "user"}
_SCORE_HINTS = {"score", "churn_score", "churn_prob", "probability", "prob", "risk",
                "risk_score", "pred", "prediction", "churn_probability"}
_RANK_HINTS = {"rank", "priority", "position", "order", "priority_rank", "churn_rank"}

_IDENTIFY_SCHEMA_TOOL = {
    "name": "identify_schema",
    "description": (
        "Identify which columns in a CSV correspond to member_id, churn score, and rank. "
        "This CSV is a WellCo churn prediction submission with ~10,000 member rows."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "member_id_col": {
                "type": "string",
                "description": "Column containing the member identifier",
            },
            "score_col": {
                "type": "string",
                "description": "Column containing the churn probability / risk score (higher = more likely to churn)",
            },
            "rank_col": {
                "type": ["string", "null"],
                "description": "Column containing the priority rank (1 = highest priority), or null if absent",
            },
            "confidence": {
                "type": "number",
                "description": "Confidence 0.0–1.0 that this is the correct CSV and mapping",
            },
            "reasoning": {
                "type": "string",
                "description": "Brief explanation of the mapping decision",
            },
        },
        "required": ["member_id_col", "score_col", "rank_col", "confidence", "reasoning"],
    },
}


def _heuristic_map(columns: list[str]) -> Optional[dict]:
    """Fast heuristic column mapper as LLM fallback."""
    lower = {c.lower(): c for c in columns}
    member_col = next((lower[k] for k in _MEMBER_ID_HINTS if k in lower), None)
    score_col = next((lower[k] for k in _SCORE_HINTS if k in lower), None)
    rank_col = next((lower[k] for k in _RANK_HINTS if k in lower), None)
    if member_col and score_col:
        return {"member_id_col": member_col, "score_col": score_col, "rank_col": rank_col}
    return None


class PredictionAgent(BaseAgent):
    def __init__(
        self,
        api_key: str,
        github_client: GitHubClient,
        true_member_ids: set,
        model: str = "claude-sonnet-4-6",
        max_csv_candidates: int = 5,
        min_overlap: float = 0.5,
    ) -> None:
        super().__init__(api_key, model)
        self._gh = github_client
        self._true_ids = true_member_ids
        self._max_candidates = max_csv_candidates
        self._min_overlap = min_overlap

    def run(self, repo: Repository) -> tuple[PredictionResult, Optional[pd.DataFrame]]:
        """
        Scan repo for predictions CSV.

        Returns (PredictionResult, DataFrame | None).
        DataFrame has columns (member_id, score, rank) if status is OK.
        """
        csv_files = self._gh.list_files(repo, extensions=[".csv"])
        if not csv_files:
            return PredictionResult(status=PredictionStatus.MISSING_PREDICTIONS), None

        # Sort by size descending — the submission file is likely the largest
        csv_files.sort(key=lambda f: f.size, reverse=True)

        for csv_file in csv_files[: self._max_candidates]:
            logger.info("Trying CSV: %s (%d bytes)", csv_file.path, csv_file.size)
            try:
                raw = self._gh.download_file(repo, csv_file.path)
                df = pd.read_csv(BytesIO(raw))
            except Exception as e:
                logger.warning("Cannot parse %s: %s", csv_file.path, e)
                continue

            if df.empty or len(df.columns) < 2:
                continue

            mapping = self._identify_columns(df, csv_file.path)
            if mapping is None:
                continue

            overlap = self._compute_overlap(df, mapping["member_id_col"])
            if overlap < self._min_overlap:
                logger.info(
                    "Skipping %s — member_id overlap %.1f%% < %.0f%%",
                    csv_file.path, overlap * 100, self._min_overlap * 100,
                )
                continue

            df = self._normalize(df, mapping)
            if df is None:
                continue

            status = PredictionStatus.OK
            if df["score"].nunique() == 1:
                status = PredictionStatus.DEGENERATE_PREDICTIONS
                logger.warning("Degenerate predictions in %s (all scores identical)", csv_file.path)

            schema = SchemaMapping(
                member_id_col=mapping["member_id_col"],
                score_col=mapping["score_col"],
                rank_col=mapping.get("rank_col"),
                confidence=mapping.get("confidence", 1.0),
                csv_path=csv_file.path,
            )
            result = PredictionResult(
                status=status,
                csv_path=csv_file.path,
                schema_mapping=schema,
                member_id_overlap=overlap,
            )
            return result, df

        return PredictionResult(status=PredictionStatus.SCHEMA_ERROR), None

    def _identify_columns(self, df: pd.DataFrame, path: str) -> Optional[dict]:
        columns = list(df.columns)
        sample = df.head(3).to_csv(index=False)

        # Try LLM first
        try:
            response = self._call(
                max_tokens=512,
                tools=[_IDENTIFY_SCHEMA_TOOL],
                tool_choice={"type": "any"},
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"CSV file: {path}\n"
                            f"Columns: {columns}\n\n"
                            f"First 3 rows:\n{sample}\n\n"
                            "This is a WellCo churn prediction submission. "
                            "Identify member_id, churn score, and rank columns."
                        ),
                    }
                ],
            )
            result = self._extract_tool_input(response)
            if result["confidence"] >= 0.7 and result["member_id_col"] in df.columns and result["score_col"] in df.columns:
                logger.debug("LLM mapped %s with confidence %.2f", path, result["confidence"])
                return result
            logger.info("LLM low confidence (%.2f) for %s, falling back to heuristics", result["confidence"], path)
        except Exception as e:
            logger.warning("LLM schema identification failed for %s: %s", path, e)

        # Heuristic fallback
        mapping = _heuristic_map(columns)
        if mapping:
            mapping["confidence"] = 0.6
            logger.info("Heuristic mapping succeeded for %s", path)
        return mapping

    def _compute_overlap(self, df: pd.DataFrame, member_col: str) -> float:
        try:
            candidate_ids = set(df[member_col].dropna().astype(int))
        except (ValueError, TypeError):
            candidate_ids = set(df[member_col].dropna().astype(str))
            true_ids = {str(i) for i in self._true_ids}
            if not candidate_ids:
                return 0.0
            return len(candidate_ids & true_ids) / len(candidate_ids)
        if not candidate_ids:
            return 0.0
        return len(candidate_ids & self._true_ids) / len(candidate_ids)

    def _normalize(self, df: pd.DataFrame, mapping: dict) -> Optional[pd.DataFrame]:
        try:
            out = pd.DataFrame()
            out["member_id"] = pd.to_numeric(df[mapping["member_id_col"]], errors="coerce").astype("Int64")
            out["score"] = pd.to_numeric(df[mapping["score_col"]], errors="coerce")

            rank_col = mapping.get("rank_col")
            if rank_col and rank_col in df.columns:
                out["rank"] = pd.to_numeric(df[rank_col], errors="coerce")
            else:
                out["rank"] = None

            out = out.dropna(subset=["member_id", "score"])
            out = out.sort_values("score", ascending=False)
            out["rank"] = range(1, len(out) + 1)
            out = out.reset_index(drop=True)
            return out
        except Exception as e:
            logger.warning("Normalization failed: %s", e)
            return None
