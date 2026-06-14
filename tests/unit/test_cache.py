"""
Unit tests for ResultCache (SQLite persistence).
"""
import pytest
from pathlib import Path

from grader.storage.cache import ResultCache
from grader.storage.models import CandidateResult, PredictionStatus


@pytest.fixture
def cache(tmp_db) -> ResultCache:
    return ResultCache(tmp_db)


def _make_result(
    name: str = "Alice",
    content_hash: str = "abc123",
    status: PredictionStatus = PredictionStatus.OK,
) -> CandidateResult:
    return CandidateResult(
        candidate_name=name,
        csv_url=f"https://example.com/{name}.csv",
        recommended_n=800,
        content_hash=content_hash,
        status=status,
        precision_curve=[0.3, 0.32, 0.31, 0.29],
        member_id_overlap=0.95,
    )


class TestResultCache:
    def test_put_and_get(self, cache):
        result = _make_result()
        cache.put(result)
        retrieved = cache.get("Alice", "abc123")
        assert retrieved is not None
        assert retrieved.candidate_name == "Alice"
        assert retrieved.content_hash == "abc123"

    def test_get_miss_returns_none(self, cache):
        assert cache.get("nobody", "sha_that_doesnt_exist") is None

    def test_put_overwrites_same_key(self, cache):
        cache.put(_make_result())
        r2 = _make_result()
        r2.precision_curve = [0.99, 0.98]
        cache.put(r2)
        retrieved = cache.get("Alice", "abc123")
        assert retrieved.precision_curve == [0.99, 0.98]

    def test_different_hash_stores_separately(self, cache):
        cache.put(_make_result(content_hash="hash1"))
        cache.put(_make_result(content_hash="hash2"))
        assert cache.get("Alice", "hash1") is not None
        assert cache.get("Alice", "hash2") is not None

    def test_get_all_latest_returns_one_per_candidate(self, cache):
        cache.put(_make_result("Alice", "hash1"))
        cache.put(_make_result("Alice", "hash2"))
        cache.put(_make_result("Bob", "hash3"))
        all_results = cache.get_all_latest()
        names = [r.candidate_name for r in all_results]
        assert names.count("Alice") == 1
        assert "Bob" in names

    def test_get_all_latest_returns_newest_for_candidate(self, cache):
        cache.put(_make_result("Alice", "hash1"))
        r2 = _make_result("Alice", "hash2")
        r2.precision_curve = [0.42]
        cache.put(r2)
        latest = cache.get_all_latest()
        alice = next(r for r in latest if r.candidate_name == "Alice")
        assert alice.precision_curve == [0.42]

    def test_round_trip_preserves_fields(self, cache):
        original = _make_result()
        cache.put(original)
        retrieved = cache.get("Alice", "abc123")
        assert retrieved.recommended_n == 800
        assert retrieved.status == PredictionStatus.OK
        assert retrieved.member_id_overlap == pytest.approx(0.95)

    def test_create_parent_dirs(self, tmp_path):
        deep_path = tmp_path / "a" / "b" / "c" / "grader.db"
        ResultCache(deep_path)
        assert deep_path.exists()

    def test_empty_cache_returns_empty_list(self, cache):
        assert cache.get_all_latest() == []

    def test_result_with_error_persists(self, cache):
        r = CandidateResult(
            candidate_name="broken",
            csv_url="https://example.com/broken.csv",
            recommended_n=500,
            content_hash="deadbeef",
            status=PredictionStatus.CSV_DOWNLOAD_ERROR,
            error="Connection timed out",
        )
        cache.put(r)
        retrieved = cache.get("broken", "deadbeef")
        assert retrieved.error == "Connection timed out"
        assert retrieved.status == PredictionStatus.CSV_DOWNLOAD_ERROR
