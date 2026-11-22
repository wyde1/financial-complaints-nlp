"""Configuration and optional .env file loading for financial-complaints-nlp."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "rows.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "complaint_classifier.joblib"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "artifacts" / "metrics.json"

NARRATIVE_COLUMN = "Consumer complaint narrative"
PRODUCT_COLUMN = "Product"


def load_dotenv_file(env_path: Path | None = None) -> None:
    """Load key=value pairs from a local .env file without overriding existing env vars."""
    path = env_path or (PROJECT_ROOT / ".env")
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


@dataclass(frozen=True)
class Settings:
    """Runtime paths loaded from environment variables or defaults."""

    data_path: Path = DEFAULT_DATA_PATH
    model_path: Path = DEFAULT_MODEL_PATH

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> Settings:
        load_dotenv_file(env_file)

        raw_data = (os.getenv("DATA_PATH") or "").strip() or "data/rows.csv"
        data_path = Path(raw_data)
        if not data_path.is_absolute():
            data_path = PROJECT_ROOT / data_path

        raw_model = (os.getenv("MODEL_PATH") or "").strip() or "artifacts/complaint_classifier.joblib"
        model_path = Path(raw_model)
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path

        return cls(data_path=data_path, model_path=model_path)
