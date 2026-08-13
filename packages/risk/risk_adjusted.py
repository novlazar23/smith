"""Risk-adjusted return metrics — Sharpe, Sortino, Calmar ratios."""

from __future__ import annotations

import math

import numpy as np


class RiskAdjustedReturns:
    """Berechnet risikobereinigte Kennzahlen für Rendite-Zeitreihen."""

    TRADING_DAYS = 252

    @staticmethod
    def sharpe_ratio(returns: list[float], risk_free_rate: float = 0.02) -> float:
        """Berechnet das annualisierte Sharpe-Verhältnis.

        Sharpe = (annualized_return - rf) / annualized_vol
        """
        arr = np.array(returns, dtype=np.float64)
        annualized_return = float(np.mean(arr)) * RiskAdjustedReturns.TRADING_DAYS
        annualized_vol = float(np.std(arr, ddof=1)) * math.sqrt(
            RiskAdjustedReturns.TRADING_DAYS
        )
        if annualized_vol < 1e-10:
            return 0.0
        return (annualized_return - risk_free_rate) / annualized_vol

    @staticmethod
    def sortino_ratio(
        returns: list[float], risk_free_rate: float = 0.02
    ) -> float:
        """Berechnet das annualisierte Sortino-Verhältnis.

        Verwendet nur die Abwärts-Standardabweichung (downside deviation).
        """
        arr = np.array(returns, dtype=np.float64)
        annualized_return = float(np.mean(arr)) * RiskAdjustedReturns.TRADING_DAYS
        rf_per_period = risk_free_rate / RiskAdjustedReturns.TRADING_DAYS

        downside_returns = arr[arr < rf_per_period]
        if len(downside_returns) == 0:
            return 0.0

        downside_deviation = float(np.std(downside_returns, ddof=1))
        downside_annualized = downside_deviation * math.sqrt(
            RiskAdjustedReturns.TRADING_DAYS
        )

        if downside_annualized == 0:
            return 0.0
        return (annualized_return - risk_free_rate) / downside_annualized

    @staticmethod
    def calmar_ratio(returns: list[float], drawdown: float) -> float:
        """Berechnet das Calmar-Verhältnis.

        Calmar = annualized_return / |drawdown|
        """
        arr = np.array(returns, dtype=np.float64)
        annualized_return = float(np.mean(arr)) * RiskAdjustedReturns.TRADING_DAYS
        if drawdown == 0:
            return 0.0
        return annualized_return / abs(drawdown)

    @staticmethod
    def max_drawdown(equity_curve: list[float]) -> float:
        """Berechnet den maximalen Drawdown aus einer Equity-Kurve."""
        peak = equity_curve[0]
        max_dd = 0.0
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        return max_dd
