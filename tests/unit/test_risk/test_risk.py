"""Tests for packages.risk — position sizing, drawdown, risk-adjusted returns."""

from __future__ import annotations

import math

import numpy as np
import pytest
from packages.risk import (
    ATRPositionSizer,
    DrawdownMonitor,
    KellyPositionSizer,
    RiskAdjustedReturns,
    RiskDecision,
    RiskGateResult,
    RiskGateType,
)


class TestRiskGateType:
    """Teste RiskGateType-Enum."""

    def test_all_gate_types_present(self) -> None:
        assert RiskGateType.DATA_QUALITY == "data_quality"
        assert RiskGateType.EXPOSURE == "exposure"
        assert RiskGateType.DRAWDOWN == "drawdown"
        assert RiskGateType.LIQUIDITY == "liquidity"
        assert RiskGateType.SPREAD == "spread"
        assert RiskGateType.NEGATIVE_EDGE == "negative_edge"
        assert RiskGateType.EXPIRED_SIGNAL == "expired_signal"
        assert RiskGateType.SYSTEM_NOT_READY == "system_not_ready"
        assert RiskGateType.UNCERTAINTY == "uncertainty"
        assert RiskGateType.DISAGREEMENT == "disagreement"
        assert RiskGateType.REGIME_CHANGE == "regime_change"
        assert RiskGateType.NEWS_RISK == "news_risk"


class TestRiskDecision:
    """Teste RiskDecision-veto-Logik."""

    def test_veto_hard_gate(self) -> None:
        gates = [
            RiskGateResult(
                gate_type=RiskGateType.DRAWDOWN,
                passed=False,
                severity="hard",
                blocking_reasons=["max dd exceeded"],
            )
        ]
        decision = RiskDecision(
            risk_version="1.0",
            run_id="test-1",
            instrument="BTC/USD",
            approved=False,
            gates=gates,
        )
        assert decision.veto is True

    def test_no_veto_all_pass(self) -> None:
        gates = [
            RiskGateResult(
                gate_type=RiskGateType.DRAWDOWN,
                passed=True,
                severity="soft",
            )
        ]
        decision = RiskDecision(
            risk_version="1.0",
            run_id="test-2",
            instrument="BTC/USD",
            approved=True,
            gates=gates,
        )
        assert decision.veto is False

    def test_no_veto_soft_only(self) -> None:
        gates = [
            RiskGateResult(
                gate_type=RiskGateType.UNCERTAINTY,
                passed=False,
                severity="soft",
                reduction_factor=0.5,
            )
        ]
        decision = RiskDecision(
            risk_version="1.0",
            run_id="test-3",
            instrument="ETH/USD",
            approved=True,
            reduction_factor=0.5,
            gates=gates,
        )
        assert decision.veto is False

    def test_veto_independent_of_soft_failures(self) -> None:
        """Veto darf nicht von Soft-Gate-Fehlern beeinflusst werden."""
        gates = [
            RiskGateResult(
                gate_type=RiskGateType.UNCERTAINTY,
                passed=False,
                severity="soft",
            ),
            RiskGateResult(
                gate_type=RiskGateType.DISAGREEMENT,
                passed=False,
                severity="soft",
            ),
        ]
        decision = RiskDecision(
            risk_version="1.0",
            run_id="test-4",
            instrument="BTC/USD",
            approved=True,
            gates=gates,
        )
        assert decision.veto is False


class TestKellyPositionSizer:
    """Teste Kelly-Positionsgrössen-Berechner."""

    def test_positive_edge(self) -> None:
        sizer = KellyPositionSizer()
        # win_rate=0.6, avg_win=2.0, avg_loss=1.0 → b=2
        # kelly = (0.6*2 - 0.4)/2 = 0.4 → half-kelly = 0.2
        fraction = sizer.calculate_fraction(win_rate=0.6, avg_win=2.0, avg_loss=1.0)
        assert fraction > 0.0

    def test_negative_edge(self) -> None:
        sizer = KellyPositionSizer()
        # win_rate=0.3, avg_win=1.0, avg_loss=2.0 → b=0.5
        # kelly = (0.3*0.5 - 0.7)/0.5 = -1.1 → half-kelly = -0.55 → 0
        fraction = sizer.calculate_fraction(win_rate=0.3, avg_win=1.0, avg_loss=2.0)
        assert fraction == 0.0

    def test_half_conservation(self) -> None:
        sizer = KellyPositionSizer()
        fraction = sizer.calculate_fraction(win_rate=0.6, avg_win=2.0, avg_loss=1.0)
        # Full Kelly would be 0.4; half Kelly is 0.2
        # But clamped by config base_risk_pct=0.02
        assert fraction <= 0.2

    def test_zero_avg_loss_raises(self) -> None:
        sizer = KellyPositionSizer()
        with pytest.raises(ValueError, match="avg_loss must be non-zero"):
            sizer.calculate_fraction(win_rate=0.5, avg_win=1.0, avg_loss=0.0)

    def test_calculate_size(self) -> None:
        sizer = KellyPositionSizer()
        size = sizer.calculate_size(win_rate=0.6, avg_win=2.0, avg_loss=1.0, account_size=10000.0)
        assert size > 0.0


class TestATRPositionSizer:
    """Teste ATR-basierten Positionsgrössen-Berechner."""

    def test_atr_position_sizing(self) -> None:
        sizer = ATRPositionSizer(max_atr_risk=0.02)
        # position_size = (10000 * 0.02) / 2.0 = 100.0
        size = sizer.calculate_size(atr=2.0, stop_distance_atr=2, account_size=10000.0)
        assert size == 100.0

    def test_atr_zero_raises(self) -> None:
        sizer = ATRPositionSizer()
        with pytest.raises(ValueError, match="ATR must be positive"):
            sizer.calculate_size(atr=0.0)

    def test_atr_negative_raises(self) -> None:
        sizer = ATRPositionSizer()
        with pytest.raises(ValueError, match="ATR must be positive"):
            sizer.calculate_size(atr=-1.0)


class TestDrawdownMonitor:
    """Teste Drawdown-Monitor."""

    def test_update_and_check(self) -> None:
        monitor = DrawdownMonitor(max_drawdown_pct=0.15, warning_drawdown_pct=0.10)
        monitor.update_equity(100.0)
        result = monitor.check_drawdown(90.0)
        # drawdown = (100 - 90) / 100 = 0.10 → at warning threshold
        assert result.gate_type == RiskGateType.DRAWDOWN

    def test_below_warning_passes(self) -> None:
        monitor = DrawdownMonitor(max_drawdown_pct=0.15, warning_drawdown_pct=0.10)
        monitor.update_equity(100.0)
        result = monitor.check_drawdown(99.0)
        dd = (100.0 - 99.0) / 100.0  # 0.01
        assert dd < 0.10
        assert result.passed is True

    def test_above_max_fails(self) -> None:
        monitor = DrawdownMonitor(max_drawdown_pct=0.15, warning_drawdown_pct=0.10)
        monitor.update_equity(100.0)
        result = monitor.check_drawdown(80.0)
        # drawdown = 0.20 > 0.15 → hard fail
        assert result.passed is False
        assert result.severity == "hard"

    def test_warning_reduces(self) -> None:
        monitor = DrawdownMonitor(max_drawdown_pct=0.15, warning_drawdown_pct=0.10)
        monitor.update_equity(100.0)
        result = monitor.check_drawdown(89.0)
        # drawdown = 0.11, in [0.10, 0.15) → soft fail with reduction
        assert result.passed is False
        assert result.severity == "soft"
        assert result.reduction_factor == 0.5

    def test_get_state(self) -> None:
        monitor = DrawdownMonitor()
        monitor.update_equity(100.0)
        state = monitor.get_state()
        assert state["peak_equity"] == 100.0
        assert state["current_equity"] == 100.0
        assert state["drawdown_pct"] == 0.0


class TestRiskAdjustedReturns:
    """Teste risikobereinigte Kennzahlen."""

    def test_sharpe_ratio_positive(self) -> None:
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.01, 252).tolist()
        ratio = RiskAdjustedReturns.sharpe_ratio(returns)
        assert isinstance(ratio, float)
        assert not math.isnan(ratio)

    def test_sharpe_ratio_zero_vol(self) -> None:
        returns = [0.0] * 252
        ratio = RiskAdjustedReturns.sharpe_ratio(returns)
        assert ratio == 0.0

    def test_sortino_ratio(self) -> None:
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.01, 252).tolist()
        ratio = RiskAdjustedReturns.sortino_ratio(returns)
        assert isinstance(ratio, float)
        assert not math.isnan(ratio)

    def test_calmar_ratio(self) -> None:
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.01, 252).tolist()
        dd = 0.10
        ratio = RiskAdjustedReturns.calmar_ratio(returns, dd)
        assert isinstance(ratio, float)
        assert not math.isnan(ratio)

    def test_max_drawdown_from_equity(self) -> None:
        equity = [100.0, 110.0, 105.0, 100.0, 115.0, 110.0, 120.0]
        dd = RiskAdjustedReturns.max_drawdown(equity)
        # Peak at 115, drops to 110 → dd = 5/115 ≈ 0.043
        assert dd > 0.0
        assert dd < 1.0

    def test_max_drawdown_with_large_drop(self) -> None:
        equity = [100.0, 100.0, 80.0, 90.0]
        dd = RiskAdjustedReturns.max_drawdown(equity)
        # Peak = 100, lowest after peak = 80 → dd = 20/100 = 0.20
        assert abs(dd - 0.20) < 1e-6
