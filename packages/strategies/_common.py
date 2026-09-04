"""Gemeinsame Grundlage der Regel-Strategien (Fenster, Parameter, Signale).

Jede Strategie ist eine `BaseStrategy`-Subklasse mit:
- begrenztem Kerzen-Fenster (``deque``, maxlen = ``WINDOW``),
- einheitlichem ``__init__``-Vertrag (``instrument``, Kapital-Parameter,
  ``**params`` mit Validierung gegen ``param_specs``),
- zentralem Signal-Aufbau mit dokumentierter Conviction-Semantik.
"""

from __future__ import annotations

from collections import deque
from typing import Any, ClassVar

import numpy as np
from packages.backtesting.core import Candle
from packages.backtesting.strategies import BaseStrategy, SignalAction, StrategySignal

#: Fensterlänge aller Regel-Strategien (1h bei 1m, 1 Tag bei 5m).
WINDOW: int = 300


class StrategyParamError(ValueError):
    """Unbekannter oder ungültiger Strategie-Parameter."""


class RuleStrategy(BaseStrategy):
    """Basis aller Regel-Strategien der Bibliothek.

    Conviction-Semantik (einheitlich): ``0.5`` = Signal knapp über
    Schwelle, ``0.85`` = maximale Überzeugung (gesättigter Indikatorabstand).
    ``position_size`` ist in allen Fällen
    ``trade_notional / initial_capital`` (Flatsize).
    """

    #: Registry-Name (klein, underscore).
    strategy_name: ClassVar[str] = ""
    #: Kurzbeschreibung (eine Zeile).
    description: ClassVar[str] = ""
    #: Parameter-Manifest: Name → (Default, Minimum, Maximum).
    param_specs: ClassVar[dict[str, tuple[float, float, float]]] = {}
    #: Kerzenzahl vor dem ersten Signal (konservativ für alle zulässigen
    #: Parameterwerte gewählt).
    min_bars: ClassVar[int] = WINDOW

    def __init__(
        self,
        instrument: str,
        *,
        initial_capital: float = 100_000.0,
        trade_notional: float = 2_000.0,
        **params: float,
    ) -> None:
        cls = type(self)
        if not cls.strategy_name:
            raise StrategyParamError("Regel-Strategie ohne strategy_name")
        super().__init__(name=cls.strategy_name)
        self.instrument = instrument
        if initial_capital <= 0:
            raise StrategyParamError("initial_capital muss > 0 sein")
        if trade_notional <= 0 or trade_notional >= initial_capital:
            raise StrategyParamError("trade_notional muss in (0, initial_capital) liegen")
        self.initial_capital = float(initial_capital)
        self.trade_notional = float(trade_notional)

        resolved: dict[str, float] = {}
        for key, (default, lo, hi) in cls.param_specs.items():
            value = float(params.pop(key, default))
            if not lo <= value <= hi:
                raise StrategyParamError(f"Parameter {key}={value} außerhalb [{lo}, {hi}]")
            resolved[key] = value
        if params:
            raise StrategyParamError(f"Unbekannte Parameter: {', '.join(sorted(params))}")
        self.params = resolved
        self._check_params(resolved)

        self._window: deque[Candle] = deque(maxlen=WINDOW)
        self.n_buy_signals = 0
        self.n_sell_signals = 0
        # Wird vom Backtest-Runner als warmup_bars gelesen (Runner-Vertrag, s. apps/backtest/runner.py).
        self.candle_limit = WINDOW

    # ── Parameter- und Signal-Hilfen ───────────────────────────────────

    def _check_params(self, resolved: dict[str, float]) -> None:
        """Relationale Parameter-Checks der Subklasse (Default: keine)."""

    @property
    def n_bars_seen(self) -> int:
        """Anzahl der bisher verarbeiteten Kerzen."""
        return len(self._window)

    def _signal(self, candle: Candle, action: SignalAction, confidence: float, reason: str) -> StrategySignal:
        if action is SignalAction.BUY:
            self.n_buy_signals += 1
        else:
            self.n_sell_signals += 1
        return StrategySignal(
            action=action,
            symbol=candle.symbol,
            confidence=round(float(confidence), 4),
            reason=f"{type(self).strategy_name}: {reason}",
            position_size=self.trade_notional / self.initial_capital,
            timestamp=candle.timestamp,
            metadata={"strategy": type(self).strategy_name, "params": dict(self.params)},
        )

    @staticmethod
    def _conviction(strength: float) -> float:
        """Conviction aus normierter Indikatorstärke (0..1) → 0.5..0.85."""
        return 0.5 + 0.35 * float(min(max(strength, 0.0), 1.0))

    # ── Fenster ────────────────────────────────────────────────────────

    def _arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(open, high, low, close, volume) des Fensters als float64."""
        opens = np.array([c.open for c in self._window], dtype=np.float64)
        highs = np.array([c.high for c in self._window], dtype=np.float64)
        lows = np.array([c.low for c in self._window], dtype=np.float64)
        closes = np.array([c.close for c in self._window], dtype=np.float64)
        volumes = np.array([c.volume for c in self._window], dtype=np.float64)
        return opens, highs, lows, closes, volumes

    def on_bar(self, candle: Candle) -> StrategySignal | None:
        self._window.append(candle)
        if len(self._window) < type(self).min_bars:
            return None
        return self._evaluate(candle)

    def _evaluate(self, candle: Candle) -> StrategySignal | None:
        """Regelauswertung der Subklasse (wird erst nach Warmup gerufen)."""
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        """Serialisierter Zustand (Name, Instrument, Parameter, Signalkonten)."""
        return {
            "strategy": type(self).strategy_name,
            "instrument": self.instrument,
            "initial_capital": self.initial_capital,
            "trade_notional": self.trade_notional,
            "params": dict(self.params),
            "n_bars_seen": self.n_bars_seen,
            "n_buy_signals": self.n_buy_signals,
            "n_sell_signals": self.n_sell_signals,
        }
