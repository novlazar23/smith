"""Tests für ClickHouseCandleProvider (gestubte ClickHouse-Engine)."""

from __future__ import annotations

import numpy as np
from apps.orchestrator_service.service import ClickHouseCandleProvider

NAMES = ["open", "high", "low", "close", "volume"]


class FakeCHEngine:
    """Stellvertreter für ClickHouseEngine (rekordiert Queries, liefert Zeilen)."""

    def __init__(self, names: list[str], rows: list[list[str]]) -> None:
        self._names = names
        self._rows = rows
        self.queries: list[str] = []

    def query(self, sql: str) -> tuple[list[str], list[list[str]]]:
        self.queries.append(sql)
        return (self._names, self._rows)


def _descending_rows() -> list[list[str]]:
    """Drei Kerzen in absteigender open_time-Reihenfolge (neueste zuerst)."""
    return [
        ["102.9", "104.5", "102.5", "103.0", "1500.0"],
        ["101.9", "103.5", "101.5", "102.0", "1100.0"],
        ["100.9", "102.5", "100.5", "101.0", "1000.0"],
    ]


class TestClickHouseCandleProvider:
    """Verhalten des Kerzen-Providers."""

    def test_returns_ascending_window(self) -> None:
        """DESC-Query-Ergebnis wird in aufsteigende OHLCV-Arrays umgewandelt."""
        engine = FakeCHEngine(NAMES, _descending_rows())
        provider = ClickHouseCandleProvider(engine)

        window = provider.fetch_candles("BTC/USDT", 200)

        assert window is not None
        np.testing.assert_array_equal(window.close, np.array([101.0, 102.0, 103.0]))
        np.testing.assert_array_equal(window.open, np.array([100.9, 101.9, 102.9]))
        np.testing.assert_array_equal(window.high, np.array([102.5, 103.5, 104.5]))
        np.testing.assert_array_equal(window.low, np.array([100.5, 101.5, 102.5]))
        np.testing.assert_array_equal(window.volume, np.array([1000.0, 1100.0, 1500.0]))
        assert window.close.dtype == np.float64

    def test_builds_expected_query(self) -> None:
        """Die Query filtert nach Instrument, sortiert DESC und limitiert."""
        engine = FakeCHEngine(NAMES, _descending_rows())
        provider = ClickHouseCandleProvider(engine)

        provider.fetch_candles("BTC/USDT", 200)

        sql = engine.queries[0]
        assert "FROM candles" in sql
        assert "WHERE instrument = 'BTC/USDT'" in sql
        assert "ORDER BY open_time DESC" in sql
        assert "LIMIT 200" in sql

    def test_escapes_instrument(self) -> None:
        """Apostrophe im Instrument werden für das SQL-Literal escappt."""
        engine = FakeCHEngine(NAMES, [])
        provider = ClickHouseCandleProvider(engine)

        provider.fetch_candles("A'B", 10)

        assert "WHERE instrument = 'A\\'B'" in engine.queries[0]

    def test_returns_none_without_rows(self) -> None:
        """Leeres Query-Ergebnis → None (Instrument wird übersprungen)."""
        engine = FakeCHEngine(NAMES, [])
        provider = ClickHouseCandleProvider(engine)

        assert provider.fetch_candles("DOGE/USDT", 200) is None

    def test_maps_columns_by_name(self) -> None:
        """Spalten werden per Name gemappt, nicht nach Position."""
        names = ["volume", "close", "low", "high", "open"]
        rows = [["1000.0", "101.0", "100.5", "102.5", "100.9"]]
        engine = FakeCHEngine(names, rows)
        provider = ClickHouseCandleProvider(engine)

        window = provider.fetch_candles("BTC/USDT", 1)

        assert window is not None
        np.testing.assert_array_equal(window.close, np.array([101.0]))
        np.testing.assert_array_equal(window.volume, np.array([1000.0]))
