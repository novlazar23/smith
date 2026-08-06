"""Technical Indicators — Numpy/basierte TA-Lib Ersatz-Implementierung.

Enthält Momentum-, Trend-, Volatilitäts- und Volumen-Indikatoren
ohne externe TA-Bibliotheken.
"""

from __future__ import annotations

from .base import Indicator
from .momentum import MACD, RSI, StochasticOscillator
from .trend import ADX, EMA, SMA, BollingerBands
from .volatility import ATR
from .volume import OBV, VWAP

__all__ = [
    "ADX",
    "ATR",
    "EMA",
    "MACD",
    "OBV",
    "RSI",
    "SMA",
    "VWAP",
    "BollingerBands",
    "Indicator",
    "StochasticOscillator",
]
