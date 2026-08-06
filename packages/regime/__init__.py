"""Regime Detection — Marktregime-Klassifizierung.

Erkennt Bull, Bear und Range (Choppy) Märkte.
"""

from __future__ import annotations

from .base import MarketRegime
from .hmm import HiddenMarkovModel
from .rules import RuleBasedRegimeDetector

__all__ = [
    "HiddenMarkovModel",
    "MarketRegime",
    "RuleBasedRegimeDetector",
]
