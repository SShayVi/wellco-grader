from enum import Enum
from typing import Optional
from pydantic import BaseModel


class PredictionStatus(str, Enum):
    OK = "OK"
    CSV_DOWNLOAD_ERROR = "CSV_DOWNLOAD_ERROR"
    SCHEMA_ERROR = "SCHEMA_ERROR"
    INVALID_PREDICTIONS = "INVALID_PREDICTIONS"
    DEGENERATE_PREDICTIONS = "DEGENERATE_PREDICTIONS"


class CandidateResult(BaseModel):
    candidate_name: str
    csv_url: str
    recommended_n: int
    content_hash: str  # MD5 of CSV bytes — cache key alongside candidate_name
    status: PredictionStatus = PredictionStatus.CSV_DOWNLOAD_ERROR
    precision_curve: Optional[list[float]] = None  # precision@N for N=1..len(predictions)
    member_id_overlap: Optional[float] = None
    error: Optional[str] = None

    def precision_at_n(self, n: int) -> Optional[float]:
        if self.precision_curve is None:
            return None
        idx = n - 1
        if 0 <= idx < len(self.precision_curve):
            return self.precision_curve[idx]
        return None

    @property
    def precision_at_recommended_n(self) -> Optional[float]:
        return self.precision_at_n(self.recommended_n)
