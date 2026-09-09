"""Tests for packages.rollout.thresholds — RolloutThresholds defaults and capital ramp."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from packages.rollout import RolloutThresholds


class TestRolloutThresholdsDefaults:
    """Default values must match the documented rollout progression."""

    def test_default_values(self) -> None:
        t = RolloutThresholds()
        assert t.capital_ramp_pct == (0.01, 0.05, 0.25, 0.50, 1.00)
        assert t.shadow_duration_pct == 0.25
        assert t.min_brier_score == 0.30
        assert t.max_drawdown_pct == 0.05
        assert t.max_spread_anomaly_ratio == 2.5
        assert t.max_exchange_error_rate == 0.10
        assert t.exchange_error_window_seconds == 300.0
        assert t.manual_kill_enabled is True

    def test_capital_ramp_property_matches_field(self) -> None:
        t = RolloutThresholds()
        assert t.capital_ramp == t.capital_ramp_pct

    def test_capital_ramp_covers_five_phases(self) -> None:
        t = RolloutThresholds()
        assert len(t.capital_ramp) == 5
        assert t.capital_ramp == tuple(sorted(t.capital_ramp))
        assert t.capital_ramp[0] == 0.01
        assert t.capital_ramp[-1] == 1.00

    def test_capital_ramp_custom_values(self) -> None:
        ramp = (0.02, 0.10, 0.50, 0.90, 1.00)
        t = RolloutThresholds(capital_ramp_pct=ramp)
        assert t.capital_ramp == ramp
        assert t.capital_ramp_pct == ramp

    def test_custom_risk_gates(self) -> None:
        t = RolloutThresholds(
            max_drawdown_pct=0.10,
            max_spread_anomaly_ratio=4.0,
            max_exchange_error_rate=0.20,
        )
        assert t.max_drawdown_pct == 0.10
        assert t.max_spread_anomaly_ratio == 4.0
        assert t.max_exchange_error_rate == 0.20
        assert t.min_brier_score == 0.30

    def test_instance_is_frozen(self) -> None:
        t = RolloutThresholds()
        with pytest.raises(FrozenInstanceError):
            t.min_brier_score = 0.99  # type: ignore[misc]
