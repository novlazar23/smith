"""Tests für die Env-Konfiguration (config_from_env) des Orchestrator-Services."""

from __future__ import annotations

import pytest
from apps.orchestrator_service.service import (
    DEFAULT_AGENT_STATUS,
    OrchestratorServiceConfig,
    config_from_env,
)

ORCHESTRATOR_ENV_KEYS = (
    "ORCHESTRATOR_INTERVAL_SECONDS",
    "ORCHESTRATOR_INSTRUMENTS",
    "ORCHESTRATOR_CANDLE_LIMIT",
    "ORCHESTRATOR_MIN_CANDLES",
    "ORCHESTRATOR_HORIZON",
    "ORCHESTRATOR_AGENT_STATUS",
    "ORCHESTRATOR_HEARTBEAT",
)


@pytest.fixture(autouse=True)
def _clean_orchestrator_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Entfernt alle ORCHESTRATOR_*-Variablen für saubere Defaults."""
    for key in ORCHESTRATOR_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


class TestAgentStatusFromEnv:
    """config_from_env() liest ORCHESTRATOR_AGENT_STATUS mit Default ACTIVE."""

    def test_default_is_active(self) -> None:
        """Ohne Env-Variabe arbeiten die Agenten im Realbetrieb (ACTIVE)."""
        config = config_from_env()

        assert config.agent_status == "ACTIVE"
        assert config.agent_status == DEFAULT_AGENT_STATUS

    def test_shadow_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ORCHESTRATOR_AGENT_STATUS=SHADOW schaltet den Beobachtungsmodus an."""
        monkeypatch.setenv("ORCHESTRATOR_AGENT_STATUS", "shadow")

        config = config_from_env()

        assert config.agent_status == "SHADOW"

    def test_invalid_status_falls_back_to_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unbekannte Werte fallen mit Warnung auf den Default zurück."""
        monkeypatch.setenv("ORCHESTRATOR_AGENT_STATUS", "bogus")

        config = config_from_env()

        assert config.agent_status == DEFAULT_AGENT_STATUS

    def test_config_field_default(self) -> None:
        """Das Config-Dataclass trägt den ACTIVE-Default direkt."""
        config = OrchestratorServiceConfig()

        assert config.agent_status == "ACTIVE"
