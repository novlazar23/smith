"""RSI-Mean-Reversion mit Volatilitätsregime-Gate."""

from __future__ import annotations

from typing import ClassVar

from packages.backtesting.core import Candle
from packages.backtesting.strategies import SignalAction, StrategySignal

from . import indicators as ta
from .rsi_mean_reversion import RsiMeanReversionStrategy


class RsiVolGateStrategy(RsiMeanReversionStrategy):
    """RSI-Mean-Reversion, die BUY nur in hohem Volatilitätsregime erlaubt.

    Signallogik identisch zu ``rsi_mean_reversion``; zusätzlich wird ein
    BUY-Signal verworfen, wenn ``ATR(atr_period) / close < vol_min`` —
    die Strategie kauft also nur tiefe Oversold-Ausschläge mit
    entsprechender Volatilität (Crash-/Stress-Phasen) und bleibt in
    ruhigen Seitwärtsmärkten flach (weniger Kostenabrieb).
    """

    strategy_name = "rsi_vol_gate"
    description = "RSI-Mean-Reversion mit ATR-Volatilitäts-Gate (BUY nur bei erhöhter ATR-Relative-Vol)."
    param_specs: ClassVar[dict[str, tuple[float, float, float]]] = {
        "period": (30.0, 5.0, 50.0),
        "buy_below": (30.0, 5.0, 45.0),
        "sell_above": (80.0, 55.0, 95.0),
        "vol_min": (0.008, 0.001, 0.05),
        "atr_period": (14.0, 5.0, 30.0),
    }
    min_bars = 60

    def _evaluate(self, candle: Candle) -> StrategySignal | None:
        sig = super()._evaluate(candle)
        if sig is None or sig.action is not SignalAction.BUY:
            return sig
        _, highs, lows, closes, _ = self._arrays()
        atr = ta._atr_series(highs, lows, closes, int(self.params["atr_period"]))
        a = atr[-1]
        if a is None or candle.close <= 0 or a / candle.close < float(self.params["vol_min"]):
            return None
        return sig
