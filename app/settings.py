"""Configuration from environment variables (or a .env file)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

DEV_API_KEY = "dev-key-change-me"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", protected_namespaces=())

    api_keys: str = DEV_API_KEY  # comma-separated
    model_dir: Path = Path(__file__).resolve().parent.parent / "models"
    app_env: str = "development"  # development | production
    log_level: str = "INFO"

    @property
    def keys(self) -> list[str]:
        return [k.strip() for k in self.api_keys.split(",") if k.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
