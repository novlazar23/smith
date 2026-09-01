"""Ingestion — Exchange-Datenimport für Trading Orchestra.

Enthält Adapter für den Echtzeit-Datenimport von Kryptowährungsbörsen:
- ExchangeAdapterBase: Abstrakte Basisklasse mit Reconnect, Rate-Limit, Heartbeat
- SpotAdapter: Binance Spot Adapter (kline, trade, depth Streams)
- FuturesAdapter: Binance Futures Adapter (funding_rate, open_interest, liquidation)
- DataIngestionService: Redpanda-konformer Daten-Ingestion-Dienst (Consumer)
"""

from .base_adapter import (
    ConnectionConfig,
    ConnectionError,  # noqa: A004  # Etablierter Name, wie in base_adapter
    ExchangeAdapterBase,
    RateLimitError,
)
from .binance_futures import FuturesAdapter
from .binance_spot import SpotAdapter
from .consumer import DataIngestionService, MarketDataProcessor

__all__ = [
    "ConnectionConfig",
    "ConnectionError",
    "DataIngestionService",
    "ExchangeAdapterBase",
    "FuturesAdapter",
    "MarketDataProcessor",
    "RateLimitError",
    "SpotAdapter",
]
