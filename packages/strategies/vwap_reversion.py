"""VWAP-Reversion-Strategie."""

from __future__ import annotations

from typing import ClassVar

from packages.backtesting.core import Candle
from packages.backtesting.strategies import SignalAction, StrategySignal

from . import indicators as ta
from ._common import RuleStrategy


class VwapReversionStrategy(RuleStrategy):
    """Mean Reversion zum rollierenden VWAP.

    BUY: der Schließkurs steht weiter als ``band`` (relativ) unter dem
    VWAP und kehrt in der aktuellen Bar zur VWAP zurück (Abstand
    schrumpft). SELL: spiegelbildlich über dem VWAP. Conviction skaliert
    mit dem relativen VWAP-Abstand.
    """

    strategy_name = "vwap_reversion"
    description = "Mean Reversion zum rollierenden VWAP."
    param_specs: ClassVar[dict[str, tuple[float, float, float]]] = {
        "period": (50.0, 10.0, 100.0),
        "band": (0.01, 0.001, 0.05),
    }
    min_bars = 105

    def _evaluate(self, candle: Candle) -> StrategySignal | None:
        period = int(self.params["period"])
        band = float(self.params["band"])
        _, highs, lows, closes, volumes = self._arrays()
        if volumes[-1] <= 0 or volumes[-2] <= 0:
            return None
        vwap = ta.rolling_vwap(highs, lows, closes, volumes, period)
        prev_dev = float(closes[-2] / vwap[-2] - 1.0) if vwap[-2] > 0 else 0.0
        curr_dev = float(closes[-1] / vwap[-1] - 1.0) if vwap[-1] > 0 else 0.0
        if prev_dev < -band and curr_dev > prev_dev and curr_dev < 0:
            strength = min(max(-prev_dev / band - 1.0, 0.0), 1.0)
            return self._signal(candle, SignalAction.BUY, self._conviction(strength), f"Reversion zur VWAP von unten (>{band:.1%} unter)")
        if prev_dev > band and curr_dev < prev_dev and curr_dev > 0:
            strength = min(max(prev_dev / band - 1.0, 0.0), 1.0)
            return self._signal(candle, SignalAction.SELL, self._conviction(strength), f"Reversion zur VWAP von oben (>{band:.1%} über)")
        return None
