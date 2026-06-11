import pytest
import pandas as pd
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"
TRUE_LABELS_PATH = Path("data/test_churn_labels.csv")


def pytest_addoption(parser):
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests (requires real GitHub + Anthropic API keys)",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--integration"):
        skip = pytest.mark.skip(reason="Pass --integration to run integration tests")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip)


@pytest.fixture(scope="session")
def true_labels_df() -> pd.DataFrame:
    """Load the real true labels once for all tests."""
    if not TRUE_LABELS_PATH.exists():
        pytest.skip(f"True labels not found at {TRUE_LABELS_PATH}")
    return pd.read_csv(TRUE_LABELS_PATH)


@pytest.fixture(scope="session")
def true_churner_ids(true_labels_df) -> set:
    return set(true_labels_df.loc[true_labels_df["churn"] == 1, "member_id"].astype(int))


@pytest.fixture(scope="session")
def all_member_ids(true_labels_df) -> set:
    return set(true_labels_df["member_id"].astype(int))


@pytest.fixture
def tmp_db(tmp_path) -> Path:
    return tmp_path / "test_grader.db"
