"""Tests for packages.risk.risk_adjusted — Sharpe, Sortino, Calmar, MaxDD."""

from __future__ import annotations

import math

import numpy as np
from packages.risk import RiskAdjustedReturns


class TestSharpeRatio:
    """Testet Sharpe-Verhaltnis."""

    def test_positive_returns(self) -> None:
        """Sharpe mit positiven Renditen."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.01, 252).tolist()
        ratio = RiskAdjustedReturns.sharpe_ratio(returns)
        assert isinstance(ratio, float)
        assert not math.isnan(ratio)

    def test_zero_vol_returns_zero(self) -> None:
        """Keine Volatilitat -> Sharpe = 0."""
        returns = [0.0] * 252
        ratio = RiskAdjustedReturns.sharpe_ratio(returns)
        assert ratio == 0.0

    def test_constant_returns(self) -> None:
        """Konstante Renditen -> Sharpe = 0 (ddof std = 0)."""
        returns = [0.001] * 252
        ratio = RiskAdjustedReturns.sharpe_ratio(returns)
        # np.std(ddof=1) von konstantem Array = 0 -> returns 0.0
        assert ratio == 0.0

    def test_risk_free_rate_affects(self) -> None:
        """Verschiedene risk_free_rate ergeben verschiedene Sharpe."""
        returns = [0.002] * 100
        rf_0 = RiskAdjustedReturns.sharpe_ratio(returns, risk_free_rate=0.0)
        rf_2 = RiskAdjustedReturns.sharpe_ratio(returns, risk_free_rate=0.02)
        # Hohere RF -> niedrigerer Sharpe
        assert rf_0 >= rf_2

    def test_negative_returns(self) -> None:
        """Negatives Sharpe mit negativen Renditen."""
        returns = np.random.normal(-0.002, 0.01, 252).tolist()
        ratio = RiskAdjustedReturns.sharpe_ratio(returns)
        assert ratio < 0.0

    def test_sharpe_type(self) -> None:
        """Sharpe gibt immer float zuruck."""
        ratio = RiskAdjustedReturns.sharpe_ratio([0.01, -0.01, 0.02, -0.02])
        assert isinstance(ratio, float)

    def test_small_return_set(self) -> None:
        """Kleine Returns-Menge funktioniert."""
        returns = [0.001, 0.002, -0.001, 0.003, -0.002]
        ratio = RiskAdjustedReturns.sharpe_ratio(returns)
        assert isinstance(ratio, float)


class TestSortinoRatio:
    """Testet Sortino-Verhaltnis."""

    def test_normal_returns(self) -> None:
        """Sortino mit normalen Renditen."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.01, 252).tolist()
        ratio = RiskAdjustedReturns.sortino_ratio(returns)
        assert isinstance(ratio, float)
        assert not math.isnan(ratio)

    def test_no_downside_returns(self) -> None:
        """Keine Downside-Returns -> Sortino = 0."""
        # Alle Renditen sind sehr hoch, keine unter RF/252
        returns = [0.05] * 252  # 5% pro Tag
        ratio = RiskAdjustedReturns.sortino_ratio(returns)
        assert ratio == 0.0

    def test_all_negative_returns(self) -> None:
        """Alle negativen Returns ergeben negatives Sortino."""
        returns = np.random.normal(-0.01, 0.02, 100).tolist()
        ratio = RiskAdjustedReturns.sortino_ratio(returns)
        assert ratio <= 0.0

    def sortino_constant(self) -> None:
        """Konstante Returns -> Sortino = 0 (keine downside)."""
        returns = [0.01] * 50
        ratio = RiskAdjustedReturns.sortino_ratio(returns)
        assert ratio == 0.0

    def test_sortino_type(self) -> None:
        """Sortino gibt immer float zuruck."""
        returns = [0.01, -0.01, 0.02, -0.02]
        ratio = RiskAdjustedReturns.sortino_ratio(returns)
        assert isinstance(ratio, float)

    def test_high_vol_downside(self) -> None:
        """Hohe Downside-Volatilitat -> niedrigeres Sortino."""
        high_vol = np.random.normal(0.001, 0.05, 252).tolist()
        low_vol = np.random.normal(0.001, 0.005, 252).tolist()
        ratio_high = RiskAdjustedReturns.sortino_ratio(high_vol)
        ratio_low = RiskAdjustedReturns.sortino_ratio(low_vol)
        # Bei gleicher Rendite und hoherer Downside -> niedrigeres Sortino
        assert ratio_high <= ratio_low


class TestCalmarRatio:
    """Testet Calmar-Verhaltnis."""

    def test_calmar_positive(self) -> None:
        """Calmar mit positiven Renditen."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.01, 252).tolist()
        ratio = RiskAdjustedReturns.calmar_ratio(returns, -0.10)
        assert isinstance(ratio, float)
        assert not math.isnan(ratio)

    def test_calmar_zero_drawdown(self) -> None:
        """Drawdown = 0 -> Calmar = 0."""
        returns = [0.001] * 252
        ratio = RiskAdjustedReturns.calmar_ratio(returns, 0.0)
        assert ratio == 0.0

    def test_calmar_large_drawdown(self) -> None:
        """GroBer Drawdown -> kleineres Calmar."""
        returns = np.random.normal(0.001, 0.01, 252).tolist()
        ratio_small_dd = RiskAdjustedReturns.calmar_ratio(returns, -0.05)
        ratio_large_dd = RiskAdjustedReturns.calmar_ratio(returns, -0.20)
        assert ratio_small_dd > ratio_large_dd

    def test_calmar_type(self) -> None:
        """Calmar gibt immer float zuruck."""
        ratio = RiskAdjustedReturns.calmar_ratio([0.01, 0.02, -0.01], -0.10)
        assert isinstance(ratio, float)

    def test_calmar_with_high_return(self) -> None:
        """Hohen Renditen -> hohes Calmar (gleicher DD)."""
        high_ret = [0.01] * 252
        low_ret = [0.001] * 252
        ratio_high = RiskAdjustedReturns.calmar_ratio(high_ret, -0.10)
        ratio_low = RiskAdjustedReturns.calmar_ratio(low_ret, -0.10)
        assert ratio_high > ratio_low


class TestMaxDrawdown:
    """Testet Max Drawdown Berechnung."""

    def test_max_drawdown_basic(self) -> None:
        """Einfacher Drawdown."""
        equity = [100.0, 110.0, 105.0, 100.0, 115.0, 110.0, 120.0]
        dd = RiskAdjustedReturns.max_drawdown(equity)
        assert dd > 0.0
        assert dd < 1.0

    def test_max_drawdown_large_drop(self) -> None:
        """GroBer Ruckgang."""
        equity = [100.0, 100.0, 80.0, 90.0]
        dd = RiskAdjustedReturns.max_drawdown(equity)
        assert abs(dd - 0.20) < 1e-6

    def test_max_drawdown_no_drawdown(self) -> None:
        """Monoton steigend -> DD = 0."""
        equity = [100.0, 105.0, 110.0, 120.0]
        dd = RiskAdjustedReturns.max_drawdown(equity)
        assert dd == 0.0

    def test_max_drawdown_constant(self) -> None:
        """Konstant -> DD = 0."""
        equity = [100.0] * 50
        dd = RiskAdjustedReturns.max_drawdown(equity)
        assert dd == 0.0

    def test_max_drawdown_type(self) -> None:
        """MaxDD gibt immer float zuruck."""
        dd = RiskAdjustedReturns.max_drawdown([100.0, 90.0, 95.0])
        assert isinstance(dd, float)
        assert 0.0 <= dd <= 1.0

    def test_max_drawdown_multiple_peaks(self) -> None:
        """Mehrere Peaks -> grobter Drawdown gewinnt."""
        equity = [100.0, 120.0, 100.0, 80.0, 150.0]
        dd = RiskAdjustedReturns.max_drawdown(equity)
        # Peak 120 -> 80 = 33.3%
        # Peak 150 -> 0%
        assert dd > 0.30
        assert dd <= 0.34
