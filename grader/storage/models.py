from enum import Enum
from typing import Optional
from pydantic import BaseModel, field_validator


class PredictionStatus(str, Enum):
    OK = "OK"
    MISSING_PREDICTIONS = "MISSING_PREDICTIONS"
    SCHEMA_ERROR = "SCHEMA_ERROR"
    INVALID_PREDICTIONS = "INVALID_PREDICTIONS"
    DEGENERATE_PREDICTIONS = "DEGENERATE_PREDICTIONS"
    REPO_UNAVAILABLE = "REPO_UNAVAILABLE"
    GITHUB_ERROR = "GITHUB_ERROR"


class NSource(str, Enum):
    CSV_ROW_COUNT = "csv_row_count"
    CSV_EXPLICIT_COLUMN = "csv_explicit_column"
    README = "readme"
    CODE = "code"
    PDF = "pdf"
    INFERRED = "inferred"


class NExtractionResult(BaseModel):
    n: int
    source: NSource
    confidence: float
    n_warning: bool = False

    @field_validator("n")
    @classmethod
    def clamp_n(cls, v: int) -> int:
        return max(1, min(v, 10_000))


class SchemaMapping(BaseModel):
    member_id_col: str
    score_col: str
    rank_col: Optional[str]
    confidence: float
    csv_path: str


class PredictionResult(BaseModel):
    status: PredictionStatus
    csv_path: Optional[str] = None
    schema_mapping: Optional[SchemaMapping] = None
    member_id_overlap: Optional[float] = None


class ReviewQuestionResult(BaseModel):
    id: str
    score: int  # 0 = not addressed, 1 = partial, 2 = full
    justification: str
    weight: float

    @field_validator("score")
    @classmethod
    def validate_score(cls, v: int) -> int:
        if v not in (0, 1, 2):
            raise ValueError(f"score must be 0, 1, or 2, got {v}")
        return v


class ReviewResult(BaseModel):
    questions: list[ReviewQuestionResult]
    weighted_score: float  # normalized 0–1


class CandidateResult(BaseModel):
    candidate_name: str
    repo_url: str
    commit_sha: str
    prediction_result: Optional[PredictionResult] = None
    n_extraction: Optional[NExtractionResult] = None
    review_result: Optional[ReviewResult] = None
    precision_curve: Optional[list[float]] = None  # precision@N for N=1..len(predictions)
    error: Optional[str] = None

    @property
    def status(self) -> PredictionStatus:
        if self.prediction_result is None:
            return PredictionStatus.MISSING_PREDICTIONS
        return self.prediction_result.status

    @property
    def recommended_n(self) -> Optional[int]:
        return self.n_extraction.n if self.n_extraction else None

    @property
    def precision_at_recommended_n(self) -> Optional[float]:
        if self.precision_curve is None or self.n_extraction is None:
            return None
        idx = self.n_extraction.n - 1
        if 0 <= idx < len(self.precision_curve):
            return self.precision_curve[idx]
        return None

    def precision_at_n(self, n: int) -> Optional[float]:
        if self.precision_curve is None:
            return None
        idx = n - 1
        if 0 <= idx < len(self.precision_curve):
            return self.precision_curve[idx]
        return None
