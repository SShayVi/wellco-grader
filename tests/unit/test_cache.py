"""
Unit tests for ResultCache (SQLite persistence).
"""
import pytest
from pathlib import Path

from grader.storage.cache import ResultCache
from grader.storage.models import (
    CandidateResult,
    NExtractionResult,
    NSource,
    PredictionResult,
    PredictionStatus,
    ReviewQuestionResult,
    ReviewResult,
)


@pytest.fixture
def cache(tmp_db) -> ResultCache:
    return ResultCache(tmp_db)


def _make_result(name: str = "Alice", sha: str = "abc123") -> CandidateResult:
    return CandidateResult(
        candidate_name=name,
        repo_url=f"https://github.com/test/{name}",
        commit_sha=sha,
        prediction_result=PredictionResult(
            status=PredictionStatus.OK,
            csv_path="output/predictions.csv",
            member_id_overlap=0.95,
        ),
        n_extraction=NExtractionResult(n=800, source=NSource.README, confidence=0.85),
        review_result=ReviewResult(
            questions=[
                ReviewQuestionResult(id="code_quality", score=2, justification="Great", weight=1),
            ],
            weighted_score=1.0,
        ),
        precision_curve=[0.3, 0.32, 0.31, 0.29],
    )


class TestResultCache:
    def test_put_and_get(self, cache):
        result = _make_result()
        cache.put(result)
        retrieved = cache.get("Alice", "abc123")
        assert retrieved is not None
        assert retrieved.candidate_name == "Alice"
        assert retrieved.commit_sha == "abc123"

    def test_get_miss_returns_none(self, cache):
        assert cache.get("nobody", "sha_that_doesnt_exist") is None

    def test_has_returns_true_after_put(self, cache):
        cache.put(_make_result())
        assert cache.has("Alice", "abc123")

    def test_has_returns_false_before_put(self, cache):
        assert not cache.has("Alice", "nonexistent")

    def test_put_overwrites_same_key(self, cache):
        r1 = _make_result()
        cache.put(r1)
        r2 = _make_result()
        r2.precision_curve = [0.99, 0.98]
        cache.put(r2)
        retrieved = cache.get("Alice", "abc123")
        assert retrieved.precision_curve == [0.99, 0.98]

    def test_different_sha_stores_separately(self, cache):
        cache.put(_make_result(sha="sha1"))
        cache.put(_make_result(sha="sha2"))
        assert cache.get("Alice", "sha1") is not None
        assert cache.get("Alice", "sha2") is not None

    def test_get_all_latest_returns_one_per_candidate(self, cache):
        cache.put(_make_result("Alice", "sha1"))
        cache.put(_make_result("Alice", "sha2"))  # newer
        cache.put(_make_result("Bob", "sha3"))
        all_results = cache.get_all_latest()
        names = [r.candidate_name for r in all_results]
        assert names.count("Alice") == 1
        assert "Bob" in names

    def test_get_all_latest_returns_newest_for_candidate(self, cache):
        cache.put(_make_result("Alice", "sha1"))
        r2 = _make_result("Alice", "sha2")
        r2.precision_curve = [0.42]
        cache.put(r2)
        latest = cache.get_all_latest()
        alice = next(r for r in latest if r.candidate_name == "Alice")
        assert alice.precision_curve == [0.42]

    def test_round_trip_preserves_full_model(self, cache):
        original = _make_result()
        cache.put(original)
        retrieved = cache.get("Alice", "abc123")
        assert retrieved.n_extraction.n == 800
        assert retrieved.n_extraction.source == NSource.README
        assert retrieved.review_result.weighted_score == 1.0
        assert retrieved.prediction_result.status == PredictionStatus.OK

    def test_create_parent_dirs(self, tmp_path):
        deep_path = tmp_path / "a" / "b" / "c" / "grader.db"
        cache = ResultCache(deep_path)
        assert deep_path.exists()

    def test_empty_cache_returns_empty_list(self, cache):
        assert cache.get_all_latest() == []

    def test_result_with_error_persists(self, cache):
        r = CandidateResult(
            candidate_name="broken",
            repo_url="https://github.com/x/y",
            commit_sha="deadbeef",
            error="Repo not found",
        )
        cache.put(r)
        retrieved = cache.get("broken", "deadbeef")
        assert retrieved.error == "Repo not found"
