"""Helfer für die Strategie-Bibliotheks-Tests (deterministisch, ohne Netzwerk)."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest
from packages.backtesting.core import Candle

BTC = "BTC/USDT"
BASE_TIME = datetime(2021, 5, 15, 0, 0, 0, tzinfo=UTC)


def make_candles(
    n: int,
    start: datetime = BASE_TIME,
    price0: float = 100.0,
    step: float = 0.0,
    symbol: str = BTC,
) -> list[Candle]:
    """Erzeugt n deterministische 1m-Kerzen (Close = price0 + i*step)."""
    candles: list[Candle] = []
    price = price0
    for i in range(n):
        close = price + step
        candles.append(
            Candle(
                timestamp=start + timedelta(minutes=i),
                symbol=symbol,
                open=price,
                high=max(price, close) + 0.5,
                low=min(price, close) - 0.5,
                close=close,
                volume=1000.0,
            )
        )
        price = close
    return candles


def make_sine_candles(n: int, period: float = 60.0, amplitude: float = 5.0, price0: float = 100.0) -> list[Candle]:
    """Oszillierende Kerzen: Close = price0 + amplitude * sin(i / period * 2pi)."""
    candles: list[Candle] = []
    for i in range(n):
        close = price0 + amplitude * math.sin(i / period * 2.0 * math.pi)
        prev = price0 + amplitude * math.sin((i - 1) / period * 2.0 * math.pi) if i > 0 else price0
        candles.append(
            Candle(
                timestamp=BASE_TIME + timedelta(minutes=i),
                symbol=BTC,
                open=prev,
                high=max(prev, close) + 0.5,
                low=min(prev, close) - 0.5,
                close=close,
                volume=1000.0,
            )
        )
    return candles


@pytest.fixture
def candles_uptrend() -> list[Candle]:
    return make_candles(400, step=0.5)


@pytest.fixture
def candles_downtrend() -> list[Candle]:
    return make_candles(400, step=-0.5)


@pytest.fixture
def candles_flat() -> list[Candle]:
    return make_candles(400, step=0.0)


@pytest.fixture
def candles_sine() -> list[Candle]:
    return make_sine_candles(400)
