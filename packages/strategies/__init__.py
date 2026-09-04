"""Strategie-Bibliothek: deterministische Regel-Strategien für Backtests.

Öffentliche API:
- ``list_strategies()`` / ``describe(name)`` / ``create_strategy(...)``
  aus der Registry,
- ``RuleStrategy`` als Basis für eigene Strategien,
- ``indicators`` als numpy-Indikator-Modul (ohne Lookahead).
"""

from __future__ import annotations

from . import indicators
from ._common import WINDOW, RuleStrategy, StrategyParamError
from .registry import STRATEGIES, create_strategy, describe, list_strategies

__all__ = [
    "STRATEGIES",
    "WINDOW",
    "RuleStrategy",
    "StrategyParamError",
    "create_strategy",
    "describe",
    "indicators",
    "list_strategies",
]
