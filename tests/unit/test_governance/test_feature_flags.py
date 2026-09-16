"""Tests für den LIVE_TRADING_ENABLED-Bootstrap (nur in production wirksam)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from packages.governance.feature_flags import FeatureFlags


@pytest.fixture(autouse=True)
def _fresh_flags(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for key in ("APP_ENV", "ENV", "LIVE_TRADING_ENABLED"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(FeatureFlags, "_instance", None)
    yield
    FeatureFlags._instance = None


class TestLiveTradingBootstrap:
    """LIVE_TRADING_ENABLED=true startet das Flag nur in production."""

    def test_production_with_env_true_starts_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("LIVE_TRADING_ENABLED", "true")
        assert FeatureFlags().is_enabled("live_trading_enabled") is True

    def test_development_ignores_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_ENV", "development")
        monkeypatch.setenv("LIVE_TRADING_ENABLED", "true")
        assert FeatureFlags().is_enabled("live_trading_enabled") is False

    def test_staging_ignores_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_ENV", "staging")
        monkeypatch.setenv("LIVE_TRADING_ENABLED", "true")
        assert FeatureFlags().is_enabled("live_trading_enabled") is False

    def test_production_without_env_starts_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_ENV", "production")
        assert FeatureFlags().is_enabled("live_trading_enabled") is False

    def test_env_value_is_case_and_whitespace_tolerant(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("LIVE_TRADING_ENABLED", "  TRUE ")
        assert FeatureFlags().is_enabled("live_trading_enabled") is True
