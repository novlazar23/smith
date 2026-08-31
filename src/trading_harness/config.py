from __future__ import annotations

import json
from functools import lru_cache
from typing import Annotated, Any

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import (
    BaseSettings,
    NoDecode,
    SettingsConfigDict,
)


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
    kill_switch_state_path: str = "data/kill_switch.json"
    execution_log_state_path: str = "data/execution_log.json"
    risk_policy_path: str = "config/risk-policy.yaml"
    population_policy_path: str = "config/population-policy.yaml"

    # Phase 5 — Read/Trade API Separation (R5.21–R5.22)
    read_api_key: str = ""
    trade_api_key: str = ""

    # Phase 5 — Network Isolation (R5.15–R5.17)
    network_allowed_patterns: list[str] = []

    # Phase 5 — Credential Management (R5.18–R5.20)
    credential_source: str = "env"  # env | vault | aws_secrets_manager

    # Shadow-Trading-Epic (Spec §6.2) — default-off, read-only gegenüber der Exchange
    shadow_trading_enabled: bool = False
    shadow_loop_interval_seconds: int = 900
    shadow_max_decisions_per_day: int = 96
    shadow_trading_symbols: Annotated[list[str], NoDecode] = Field(default_factory=list)
    shadow_min_confidence: float = 0.6
    shadow_stop_loss_fraction: float = 0.02
    shadow_min_risk_reward: float = 2.0
    shadow_state_path: str = "data/shadow_trading_state.json"
    shadow_start_equity: float = 100000.0

    @field_validator("shadow_trading_symbols", mode="before")
    @classmethod
    def _parse_symbol_list(cls, value: Any) -> Any:
        """Erlaubt sowohl JSON-Listen als auch Komma-getrennte Strings aus .env."""
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("["):  # JSON-Liste
                try:
                    return json.loads(stripped)
                except json.JSONDecodeError:
                    return stripped
            return [s.strip() for s in stripped.split(",") if s.strip()]
        return value

    # Autonomous shadow runtime + encrypted cross-system state handoff.
    autonomous_shadow_enabled: bool = False
    state_handoff_enabled: bool = False
    state_handoff_password: SecretStr = SecretStr("")
    state_node_id: str = ""
    state_data_dir: str = "data"
    state_bundle_path: str = "handoff/runtime-state.enc.json"
    state_handoff_lease_seconds: int = 300
    state_handoff_sync_seconds: int = 60

    # Quant Platform — InfluxDB (Phase 1)
    influxdb_url: str = "http://localhost:8086"
    influxdb_token: str = ""
    influxdb_org: str = "smith"
    influxdb_bucket: str = "market_data"
    influxdb_enabled: bool = False

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()