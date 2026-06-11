from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: str
    github_token: str = ""
    google_sheet_id: str = ""  # optional when --candidate flag is used

    true_labels_path: Path = Path("data/test_churn_labels.csv")
    cache_db_path: Path = Path(".cache/grader.db")
    refresh_interval_seconds: int = 60
    log_level: str = "INFO"

    anthropic_model: str = "claude-sonnet-4-6"
    max_csv_candidates: int = 5
    min_member_id_overlap: float = 0.5
    max_repo_chars: int = 150_000
