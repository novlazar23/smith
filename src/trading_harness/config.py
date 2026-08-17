from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8080
    database_url: str = "postgresql://trading:trading@localhost:5432/trading"
    redis_url: str = "redis://localhost:6379/0"

    llm_base_url: str = "http://localhost:4000/v1"
    llm_api_key: str = "change-me"
    llm_model_fast: str = "local-fast"
    llm_model_main: str = "local-main"
    llm_model_critic: str = "local-critic"

    live_execution_enabled: bool = False
    kill_switch_default: bool = True
    risk_policy_path: str = "config/risk-policy.yaml"
    population_policy_path: str = "config/population-policy.yaml"

    # Phase 5 — Read/Trade API Separation (R5.21–R5.22)
    read_api_key: str = ""
    trade_api_key: str = ""

    # Phase 5 — Network Isolation (R5.15–R5.17)
    network_allowed_patterns: list[str] = []

    # Phase 5 — Credential Management (R5.18–R5.20)
    credential_source: str = "env"  # env | vault | aws_secrets_manager

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
