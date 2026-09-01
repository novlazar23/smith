"""Tests für die Env-Konfiguration (config_from_env) des Demo-Traders."""

from __future__ import annotations

from pathlib import Path

import pytest
from apps.demo_trader.service import (
    DEFAULT_CANDLE_VENUE,
    DEFAULT_INITIAL_CASH,
    DEFAULT_INTERVAL_SECONDS,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_TRADE_NOTIONAL,
    DemoTraderConfig,
    config_from_env,
)

DEMO_ENV_KEYS = (
    "DEMO_INTERVAL_SECONDS",
    "DEMO_INSTRUMENTS",
    "DEMO_INITIAL_CASH",
    "DEMO_TRADE_NOTIONAL",
    "DEMO_MIN_CONFIDENCE",
    "CANDLE_VENUE",
    "DEMO_HEARTBEAT",
)


@pytest.fixture(autouse=True)
def _clean_demo_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Entfernt alle DEMO_*-/CANDLE_VENUE-Variablen für saubere Defaults."""
    for key in DEMO_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


class TestConfigFromEnv:
    """config_from_env() liest die DEMO_*-Variablen mit korrekten Defaults."""

    def test_defaults(self) -> None:
        """Ohne Env-Variablen gelten die vorgegebenen Defaults."""
        config = config_from_env()
        assert config.interval_seconds == DEFAULT_INTERVAL_SECONDS
        assert config.instruments == ("BTC/USDT", "ETH/USDT")
        assert config.initial_cash == DEFAULT_INITIAL_CASH
        assert config.trade_notional == DEFAULT_TRADE_NOTIONAL
        assert config.min_confidence == DEFAULT_MIN_CONFIDENCE
        assert config.candle_venue == DEFAULT_CANDLE_VENUE
        assert config.account_id == "demo"

    def test_overrides(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Env-Variablen überschreiben die Defaults."""
        monkeypatch.setenv("DEMO_INTERVAL_SECONDS", "60")
        monkeypatch.setenv("DEMO_INSTRUMENTS", "SOL/USDT, BTC/USDT")
        monkeypatch.setenv("DEMO_INITIAL_CASH", "50000")
        monkeypatch.setenv("DEMO_TRADE_NOTIONAL", "500")
        monkeypatch.setenv("DEMO_MIN_CONFIDENCE", "0.7")
        monkeypatch.setenv("CANDLE_VENUE", "DUMMY_EXCHANGE")
        monkeypatch.setenv("DEMO_HEARTBEAT", str(tmp_path / "hb"))

        config = config_from_env()

        assert config.interval_seconds == 60.0
        assert config.instruments == ("SOL/USDT", "BTC/USDT")
        assert config.initial_cash == 50000.0
        assert config.trade_notional == 500.0
        assert config.min_confidence == 0.7
        assert config.candle_venue == "DUMMY_EXCHANGE"
        assert config.heartbeat_path == tmp_path / "hb"

    def test_invalid_interval_falls_back_to_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ungültiges DEMO_INTERVAL_SECONDS → Default statt Absturz."""
        monkeypatch.setenv("DEMO_INTERVAL_SECONDS", "nicht-zahl")

        config = config_from_env()

        assert config.interval_seconds == DEFAULT_INTERVAL_SECONDS

    def test_empty_instruments(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Leeres DEMO_INSTRUMENTS → leeres Tuple (main() exit 2)."""
        monkeypatch.setenv("DEMO_INSTRUMENTS", " , ")

        config = config_from_env()

        assert config.instruments == ()

    def test_return_type(self) -> None:
        """config_from_env liefert ein DemoTraderConfig."""
        assert isinstance(config_from_env(), DemoTraderConfig)
