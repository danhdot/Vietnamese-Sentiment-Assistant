"""Application configuration helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _default_db_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "sentiment_history.sqlite3"


@dataclass(slots=True)
class Settings:
    """Container for runtime configuration values."""

    model_name: str = os.getenv(
        "SENTIMENT_MODEL_NAME",
        "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    )
    min_text_length: int = int(os.getenv("MIN_TEXT_LENGTH", "4"))
    history_soft_limit: int = int(os.getenv("HISTORY_SOFT_LIMIT", "50"))
    sqlite_path: Path = Path(os.getenv("SQLITE_PATH", _default_db_path()))
    allow_origins: tuple[str, ...] = tuple(
        origin.strip()
        for origin in os.getenv("ALLOWED_CORS_ORIGINS", "http://localhost:5173,http://localhost:4173").split(",")
        if origin.strip()
    )


settings = Settings()
