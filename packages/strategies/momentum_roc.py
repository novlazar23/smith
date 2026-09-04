"""Momentum-ROC-Strategie (Trendfolge)."""

from __future__ import annotations

from typing import ClassVar

from packages.backtesting.core import Candle
from packages.backtesting.strategies import SignalAction, StrategySignal

from . import indicators as ta
from ._common import RuleStrategy


class MomentumRocStrategy(RuleStrategy):
    """Trendfolge über Rate-of-Change (ROC).

    BUY: der ROC über ``period`` Kerzen übersteigt ``buy_threshold``
    (relativer Aufwärts-Momentum). SELL: der ROC fällt unter
    ``sell_threshold`` (negativer Momentum). Conviction skaliert mit dem
    Schwellen-Abstand des ROC.
    """

    strategy_name = "momentum_roc"
    description = "Trendfolge über Rate-of-Change (ROC-Schwelle)."
    param_specs: ClassVar[dict[str, tuple[float, float, float]]] = {
        "period": (20.0, 5.0, 50.0),
        "buy_threshold": (0.02, 0.001, 0.10),
        "sell_threshold": (-0.01, -0.10, 0.0),
    }
    min_bars = 60

    def _evaluate(self, candle: Candle) -> StrategySignal | None:
        period = int(self.params["period"])
        buy_threshold = float(self.params["buy_threshold"])
        sell_threshold = float(self.params["sell_threshold"])
        _, _, _, closes, _ = self._arrays()
        roc = ta.roc(closes, period)
        curr = float(roc[-1])
        if curr > buy_threshold:
            strength = min((curr - buy_threshold) / buy_threshold, 1.0)
            return self._signal(candle, SignalAction.BUY, self._conviction(strength), f"ROC{period} > {buy_threshold:.1%}")
        if curr < sell_threshold:
            strength = min((sell_threshold - curr) / abs(sell_threshold), 1.0)
            return self._signal(candle, SignalAction.SELL, self._conviction(strength), f"ROC{period} < {sell_threshold:.1%}")
        return None
