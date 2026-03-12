from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Auth0
    auth0_domain: str
    auth0_audience: str
    auth0_algorithms: list[str] = ["RS256"]

    # Redis
    redis_url: str = "redis://localhost:6379"

    # Rate limiting
    rate_limit_requests: int = 60       # requests per window
    rate_limit_window_seconds: int = 60  # 1-minute window

    # Data paths (relative to backend directory)
    data_dir: str = "../../data"
    forecasts_dir: str = "../../forecasts"

    # CORS
    cors_origins: list[str] = ["*"]

    # App
    app_env: str = "development"
    log_level: str = "INFO"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
