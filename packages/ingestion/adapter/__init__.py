"""Ingestion Adapter — pluggable exchange adapters with venue-aware metadata."""

from __future__ import annotations

from .base import (
    ConnectionConfig,
    ExchangeAdapterBase,
    MarketDataType,
    VenueFees,
)
from .binance import BinanceAdapter
from .dummy import DummyAdapter

__all__ = [
    "BinanceAdapter",
    "ConnectionConfig",
    "DummyAdapter",
    "ExchangeAdapterBase",
    "MarketDataType",
    "VenueFees",
]
