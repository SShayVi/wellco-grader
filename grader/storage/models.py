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
    gain_curve: Optional[list[float]] = None       # gain@N = cumulative recall
    lift_curve: Optional[list[float]] = None       # lift@N = precision@N / churn_rate
    qini_curve: Optional[list[float]] = None        # qini@N = cumulative uplift (requires outreach labels)
    uplift_curve: Optional[list[float]] = None      # uplift@N = conditional treatment effect (requires outreach labels)
    ranked_member_ids: Optional[list] = None       # member_ids sorted by score desc
    ranked_scores: Optional[list] = None           # corresponding scores (same order)
    member_id_overlap: Optional[float] = None
    error: Optional[str] = None   # set on any non-OK status
    notes: Optional[str] = None   # informational (e.g. "Rec. N defaulted to 1,000")

    def _curve_at_n(self, curve: Optional[list[float]], n: int) -> Optional[float]:
        if curve is None:
            return None
        idx = n - 1
        if 0 <= idx < len(curve):
            return curve[idx]
        return None

    def precision_at_n(self, n: int) -> Optional[float]:
        return self._curve_at_n(self.precision_curve, n)

    def gain_at_n(self, n: int) -> Optional[float]:
        return self._curve_at_n(self.gain_curve, n)

    def lift_at_n(self, n: int) -> Optional[float]:
        return self._curve_at_n(self.lift_curve, n)

    def qini_at_n(self, n: int) -> Optional[float]:
        return self._curve_at_n(self.qini_curve, n)

    def uplift_at_n(self, n: int) -> Optional[float]:
        return self._curve_at_n(self.uplift_curve, n)

    @property
    def precision_at_recommended_n(self) -> Optional[float]:
        return self.precision_at_n(self.recommended_n)
