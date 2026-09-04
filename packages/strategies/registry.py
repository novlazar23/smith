"""Registry der Regel-Strategie-Bibliothek.

Zentrale Anbindungspunkte:
- ``list_strategies()``: alle verfügbaren Strategie-Namen,
- ``describe(name)``: Metadaten (Beschreibung, Parameter-Manifest, Warmup),
- ``create_strategy(name, instrument, params=None, ...)``: fertige
  `BaseStrategy`-Instanz für den Backtest-Runner.
"""

from __future__ import annotations

from typing import Any

from ._common import RuleStrategy, StrategyParamError
from .bollinger_reversion import BollingerReversionStrategy
from .donchian_breakout import DonchianBreakoutStrategy
from .ema_cross import EmaCrossStrategy
from .keltner_breakout import KeltnerBreakoutStrategy
from .macd_cross import MacdCrossStrategy
from .momentum_roc import MomentumRocStrategy
from .rsi_mean_reversion import RsiMeanReversionStrategy
from .stochastics import StochasticsStrategy
from .supertrend import SupertrendStrategy
from .vwap_reversion import VwapReversionStrategy

_STRATEGY_CLASSES: tuple[type[RuleStrategy], ...] = (
    BollingerReversionStrategy,
    DonchianBreakoutStrategy,
    EmaCrossStrategy,
    KeltnerBreakoutStrategy,
    MacdCrossStrategy,
    MomentumRocStrategy,
    RsiMeanReversionStrategy,
    StochasticsStrategy,
    SupertrendStrategy,
    VwapReversionStrategy,
)

STRATEGIES: dict[str, type[RuleStrategy]] = {cls.strategy_name: cls for cls in _STRATEGY_CLASSES}


def _lookup(name: str) -> type[RuleStrategy]:
    try:
        return STRATEGIES[name]
    except KeyError:
        raise StrategyParamError(
            f"Unbekannte Strategie {name!r}; verfügbar: {', '.join(list_strategies())}"
        ) from None


def list_strategies() -> list[str]:
    """Alle registrierten Strategie-Namen (sortiert)."""
    return sorted(STRATEGIES)


def describe(name: str) -> dict[str, Any]:
    """Metadaten einer Strategie: Beschreibung, Parameter-Manifest, Warmup."""
    cls = _lookup(name)
    return {
        "name": name,
        "description": cls.description,
        "params": {
            key: {"default": default, "min": lo, "max": hi}
            for key, (default, lo, hi) in cls.param_specs.items()
        },
        "min_bars": cls.min_bars,
    }


def create_strategy(
    name: str,
    instrument: str,
    params: dict[str, float] | None = None,
    *,
    initial_capital: float = 100_000.0,
    trade_notional: float = 2_000.0,
) -> RuleStrategy:
    """Erzeugt eine Regel-Strategie aus der Registry.

    ``params`` übernimmt/ersetzt Manifest-Werte (wird validiert);
    Kapital-Parameter übernehmen die Werte des Backtest-Runners.
    """
    cls = _lookup(name)
    return cls(
        instrument,
        initial_capital=initial_capital,
        trade_notional=trade_notional,
        **(params or {}),
    )
