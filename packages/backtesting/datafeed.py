"""Data feeds for backtesting.

Provides abstraction over different data sources:
- CSV files
- Parquet files
- In-memory list of Candles

All feed classes implement the DataFeed protocol.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd

from .core import Candle

# ─── Protocol ───────────────────────────────────────────────────────────────


class DataFeed:
    """Protocol for backtest data sources."""

    def iter_candles(self, symbol: str | None = None) -> Iterator[Candle]:
        """Yield candles in chronological order."""
        raise NotImplementedError

    def get_candles(self, symbol: str | None = None) -> list[Candle]:
        """Return all candles as a list."""
        return list(self.iter_candles(symbol))

    @property
    def symbol(self) -> str | None:
        """The symbol this feed provides data for."""
        return None


# ─── CSV Feed ───────────────────────────────────────────────────────────────


class CsvDataFeed(DataFeed):
    """Load OHLCV data from a CSV file.

    Expected columns (case-insensitive):
        timestamp, open, high, low, close, volume

    The timestamp column is parsed as ISO 8601 or standard date format.
    """

    def __init__(
        self,
        filepath: str,
        symbol: str = "BTC/USDT",
        timestamp_col: str = "timestamp",
        open_col: str = "open",
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        volume_col: str = "volume",
        date_col: str | None = None,  # legacy date column
    ) -> None:
        self.filepath = Path(filepath)
        self._symbol = symbol
        self.timestamp_col = timestamp_col
        self.open_col = open_col
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col
        self.volume_col = volume_col
        self.date_col = date_col

        if not self.filepath.exists():
            raise FileNotFoundError(f"CSV not found: {filepath}")

        self._candles: list[Candle] = self._load()

    @property
    def symbol(self) -> str:
        return self._symbol

    def _load(self) -> list[Candle]:
        """Load and parse CSV data."""
        candles: list[Candle] = []

        with open(self.filepath, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                ts_str = row.get(self.timestamp_col, row.get(self.date_col, ""))
                try:
                    ts = pd.Timestamp(ts_str).to_pydatetime()
                except (ValueError, TypeError):
                    continue

                try:
                    o = float(row[self.open_col])
                    h = float(row[self.high_col])
                    l = float(row[self.low_col])
                    c = float(row[self.close_col])
                    v = float(row.get(self.volume_col, 0))
                except (ValueError, KeyError):
                    continue

                candles.append(
                    Candle(
                        timestamp=ts,
                        symbol=self.symbol,
                        open=o,
                        high=h,
                        low=l,
                        close=c,
                        volume=v,
                    )
                )

        candles.sort(key=lambda c: c.timestamp)
        return candles

    def iter_candles(self, symbol: str | None = None) -> Iterator[Candle]:
        if symbol and symbol != self.symbol:
            return
        yield from self._candles


# ─── In-Memory Feed ─────────────────────────────────────────────────────────


class MemoryDataFeed(DataFeed):
    """Load candles from a list (useful for testing)."""

    def __init__(self, candles: list[Candle] | None = None) -> None:
        self._candles: list[Candle] = sorted(candles or [], key=lambda c: c.timestamp)
        self._symbol = self._candles[0].symbol if self._candles else None

    @property
    def symbol(self) -> str | None:
        return self._symbol

    def add_candles(self, candles: list[Candle]) -> None:
        self._candles.extend(candles)
        self._candles.sort(key=lambda c: c.timestamp)

    def iter_candles(self, symbol: str | None = None) -> Iterator[Candle]:
        if symbol:
            for c in self._candles:
                if c.symbol == symbol:
                    yield c
        else:
            for c in self._candles:
                yield c


# ─── Parquet Feed ───────────────────────────────────────────────────────────


class ParquetDataFeed(DataFeed):
    """Load OHLCV data from a Parquet file."""

    def __init__(
        self,
        filepath: str,
        symbol: str = "BTC/USDT",
    ) -> None:
        self.filepath = Path(filepath)
        self._symbol = symbol

        if not self.filepath.exists():
            raise FileNotFoundError(f"Parquet not found: {filepath}")

        self._candles: list[Candle] = self._load()

    @property
    def symbol(self) -> str:
        return self._symbol

    def _load(self) -> list[Candle]:
        df = pd.read_parquet(self.filepath)

        # Normalize column names
        col_map = {}
        for col in df.columns:
            cl = col.lower()
            if cl in ("timestamp", "date", "time", "datetime"):
                col_map[col] = "timestamp"
            elif cl == "open":
                col_map[col] = "open"
            elif cl == "high":
                col_map[col] = "high"
            elif cl == "low":
                col_map[col] = "low"
            elif cl == "close":
                col_map[col] = "close"
            elif cl == "volume":
                col_map[col] = "volume"

        df = df.rename(columns=col_map)

        if "timestamp" not in df.columns:
            raise ValueError("Parquet must have a timestamp column")

        candles: list[Candle] = []
        for _, row in df.iterrows():
            try:
                ts = pd.Timestamp(row["timestamp"]).to_pydatetime()
            except (ValueError, TypeError, KeyError):
                continue

            candles.append(
                Candle(
                    timestamp=ts,
                    symbol=self.symbol,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row.get("volume", 0)),
                )
            )

        candles.sort(key=lambda c: c.timestamp)
        return candles

    def iter_candles(self, symbol: str | None = None) -> Iterator[Candle]:
        if symbol and symbol != self.symbol:
            return
        yield from self._candles


# ─── Factory ────────────────────────────────────────────────────────────────


class DataFeedFactory:
    """Create data feeds from configuration."""

    @staticmethod
    def create(config: dict[str, Any]) -> DataFeed:
        """Create a DataFeed from a config dict.

        Args:
            config: Must contain:
                - provider: "csv", "parquet", or "memory"
                - filepath: path to data file
                - symbol: trading symbol

        Returns:
            A DataFeed instance.

        Raises:
            ValueError: If provider is unknown or required fields missing.
        """
        provider = config.get("provider", "csv")
        filepath = config.get("filepath", "")
        symbol = config.get("symbol", "BTC/USDT")

        if not filepath:
            raise ValueError("filepath is required for data feeds")

        if provider == "csv":
            return CsvDataFeed(filepath=filepath, symbol=symbol)
        elif provider == "parquet":
            return ParquetDataFeed(filepath=filepath, symbol=symbol)
        elif provider == "memory":
            return MemoryDataFeed()
        else:
            raise ValueError(f"Unknown data provider: {provider}")


# ─── Indicator Helpers ──────────────────────────────────────────────────────


def compute_indicators(candles: list[Candle]) -> list[Candle]:
    """Compute common technical indicators and attach to candles.

    Computes:
    - SMA(20), SMA(50)
    - EMA(12), EMA(26)
    - RSI(14)
    - ATR(14)
    - MACD(12, 26, 9)

    Note: First `warmup_bars` candles will have None values for slow indicators.
    """
    closes = [c.close for c in candles]
    highs = [c.high for c in candles]
    lows = [c.low for c in candles]

    result: list[Candle] = []

    for i, candle in enumerate(candles):
        # SMA
        sma_20 = _sma(closes, i, 20)
        sma_50 = _sma(closes, i, 50)

        # EMA
        ema_12 = _ema(closes, i, 12)
        ema_26 = _ema(closes, i, 26)

        # RSI
        rsi_14 = _rsi(closes, i, 14)

        # ATR
        atr_14 = _atr(highs, lows, closes, i, 14)

        # MACD
        macd_line, macd_signal, macd_histogram = _macd(closes, i)

        # Create a new Candle with indicators attached (copy with extra attrs)
        new_candle = Candle(
            timestamp=candle.timestamp,
            symbol=candle.symbol,
            open=candle.open,
            high=candle.high,
            low=candle.low,
            close=candle.close,
            volume=candle.volume,
        )
        object.__setattr__(new_candle, "sma_20", sma_20)
        object.__setattr__(new_candle, "sma_50", sma_50)
        object.__setattr__(new_candle, "ema_12", ema_12)
        object.__setattr__(new_candle, "ema_26", ema_26)
        object.__setattr__(new_candle, "rsi_14", rsi_14)
        object.__setattr__(new_candle, "atr_14", atr_14)
        object.__setattr__(new_candle, "macd_line", macd_line)
        object.__setattr__(new_candle, "macd_signal", macd_signal)
        object.__setattr__(new_candle, "macd_histogram", macd_histogram)

        result.append(new_candle)

    return result


def _sma(values: list[float], idx: int, period: int) -> float | None:
    if idx < period - 1:
        return None
    return sum(values[idx - period + 1 : idx + 1]) / period


def _ema(values: list[float], idx: int, period: int) -> float | None:
    if idx < period - 1:
        return None
    if idx == 0:
        return values[0]

    k = 2.0 / (period + 1)
    # Compute iteratively
    result = None
    for i in range(idx + 1):
        if i < period - 1:
            result = (result if result is not None else 0.0) + values[i]
            if i == period - 2:
                result /= period
        else:
            result = values[i] if result is None else values[i] * k + result * (1 - k)
    return result


def _rsi(values: list[float], idx: int, period: int) -> float | None:
    if idx < period:
        return None

    gains = []
    losses = []
    for i in range(idx - period + 1, idx + 1):
        change = values[i] - values[i - 1]
        gains.append(max(0, change))
        losses.append(max(0, -change))

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    idx: int,
    period: int,
) -> float | None:
    if idx < period:
        return None

    trs = []
    for i in range(idx - period + 1, idx + 1):
        if i == 0:
            tr = highs[i] - lows[i]
        else:
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
        trs.append(tr)

    return sum(trs) / period


def _macd(
    values: list[float],
    idx: int,
) -> tuple[float | None, float | None, float | None]:
    """Simple MACD (12, 26, 9)."""
    if idx < 33:  # Need at least 33 bars for reliable MACD
        return None, None, None

    # Compute EMA(12)
    ema12 = _ema(values, idx, 12)
    ema26 = _ema(values, idx, 26)

    if ema12 is None or ema26 is None:
        return None, None, None

    macd_line = ema12 - ema26

    # Approximate signal line (9-period EMA of MACD)
    # For simplicity, use a rolling average
    macd_values = []
    for j in range(max(0, idx - 33), idx + 1):
        e12 = _ema(values, j, 12)
        e26 = _ema(values, j, 26)
        if e12 is not None and e26 is not None:
            macd_values.append(e12 - e26)

    if len(macd_values) < 9:
        return macd_line, None, macd_line

    signal = sum(macd_values[-9:]) / min(9, len(macd_values))
    return macd_line, signal, macd_line - signal
