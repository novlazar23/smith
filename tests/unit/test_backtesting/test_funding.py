"""Tests für die 8h-Funding-Abrechnung (Perpetual-Futures) der Backtest-Engine.

Settlement-Bars sind 00:00/08:00/16:00 UTC; positiv = Long zahlt
(Cash-Debit), negativ = Long wird gutgeschrieben. ``funding_rate=0.0``
reproduziert die Funding-freie Legacy-Abrechnung exakt.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from apps.backtest.ch_feed import load_funding_rates
from packages.backtesting.core import BacktestConfig, BacktestResult, Candle
from packages.backtesting.datafeed import MemoryDataFeed
from packages.backtesting.engine import BacktestEngine
from packages.backtesting.strategies import BaseStrategy, SignalAction, StrategySignal

CLOSE = 100.0


def _candles(start: datetime, n: int, close: float = CLOSE) -> list[Candle]:
    """1m-Kerzen mit konstantem Close ab ``start`` (aware oder naive, wie übergeben)."""
    return [
        Candle(
            timestamp=start + timedelta(minutes=i),
            symbol="BTC/USD",
            open=close,
            high=close,
            low=close,
            close=close,
            volume=10.0,
        )
        for i in range(n)
    ]


class ScriptedStrategy(BaseStrategy):
    """Ergibt BUY/SELL-Signale an festgelegten Bar-Indizes (Funding-Tests)."""

    def __init__(self, buys: list[int], sells: list[int], size: float = 0.10) -> None:
        super().__init__(name="scripted")
        self.buys = set(buys)
        self.sells = set(sells)
        self.size = size
        self._seen = 0

    def on_bar(self, candle: Candle) -> StrategySignal | None:
        index = self._seen
        self._seen += 1
        if index in self.buys:
            return StrategySignal(
                action=SignalAction.BUY,
                symbol=candle.symbol,
                confidence=1.0,
                reason="scripted buy",
                position_size=self.size,
                timestamp=candle.timestamp,
            )
        if index in self.sells:
            return StrategySignal(
                action=SignalAction.SELL,
                symbol=candle.symbol,
                confidence=1.0,
                reason="scripted sell",
                position_size=0.0,
                timestamp=candle.timestamp,
            )
        return None


def _run(
    candles: list[Candle],
    buys: list[int],
    funding_rate: float,
    *,
    sells: list[int] | None = None,
    allow_pyramiding: bool = True,
    funding_rates: dict[datetime, float] | None = None,
) -> BacktestResult:
    """Engine-Run mit warmup=0 und konstantem Close (deterministische Fills)."""
    config = BacktestConfig(
        symbol="BTC/USD",
        warmup_bars=0,
        funding_rate=funding_rate,
        allow_pyramiding=allow_pyramiding,
    )
    engine = BacktestEngine(config=config)
    return engine.run(
        MemoryDataFeed(candles=candles),
        ScriptedStrategy(buys=buys, sells=sells or []),
        warmup_bars=0,
        funding_rates=funding_rates,
    )


def _open_quantity(result: BacktestResult) -> float:
    return result.metadata["open_positions"][0]["quantity"]


class TestSettlementDetection:
    """Nur 00:00/08:00/16:00 UTC buchen; naive Timestamps gelten als UTC."""

    def test_books_at_0800_boundary_only(self) -> None:
        # 07:58..08:01: nur die 08:00-Bar ist Settlement-Bar
        start = datetime(2024, 1, 1, 7, 58, tzinfo=UTC)
        result = _run(_candles(start, 4), buys=[0], funding_rate=0.001)
        assert result.metadata["total_funding"] == pytest.approx(
            _open_quantity(result) * CLOSE * 0.001
        )

    def test_no_booking_off_boundary(self) -> None:
        # 08:01..08:04: keine 8h-Grenze im Fenster → kein Funding
        start = datetime(2024, 1, 1, 8, 1, tzinfo=UTC)
        result = _run(_candles(start, 4), buys=[0], funding_rate=0.001)
        assert len(result.metadata["open_positions"]) == 1
        assert result.metadata["total_funding"] == 0.0

    def test_naive_timestamps_treated_as_utc(self) -> None:
        start = datetime(2024, 1, 1, 7, 58)  # naive → als UTC interpretiert
        result = _run(_candles(start, 4), buys=[0], funding_rate=0.001)
        assert result.metadata["total_funding"] == pytest.approx(
            _open_quantity(result) * CLOSE * 0.001
        )


class TestFundingSign:
    """Positiv = Long zahlt (Equity sinkt), negativ = Long wird gutgeschrieben."""

    @pytest.fixture
    def day_candles(self) -> list[Candle]:
        # 00:00..16:00 → Settlements bei 08:00 (Bar 480) und 16:00 (Bar 960)
        return _candles(datetime(2024, 1, 1, tzinfo=UTC), 961)

    def test_positive_rate_debits_long(self, day_candles: list[Candle]) -> None:
        baseline = _run(day_candles, buys=[10], funding_rate=0.0)
        funded = _run(day_candles, buys=[10], funding_rate=0.001)
        expected = _open_quantity(funded) * CLOSE * 0.001 * 2
        assert funded.metadata["total_funding"] == pytest.approx(expected)
        assert (
            baseline.metadata["final_equity"] - funded.metadata["final_equity"]
            == pytest.approx(expected, rel=1e-9)
        )

    def test_negative_rate_credits_long(self, day_candles: list[Candle]) -> None:
        baseline = _run(day_candles, buys=[10], funding_rate=0.0)
        funded = _run(day_candles, buys=[10], funding_rate=-0.001)
        expected = _open_quantity(funded) * CLOSE * 0.001 * 2
        assert funded.metadata["total_funding"] == pytest.approx(-expected)
        assert (
            funded.metadata["final_equity"] - baseline.metadata["final_equity"]
            == pytest.approx(expected, rel=1e-9)
        )


class TestPositionGating:
    """Kein Funding ohne offene Long; Entry auf der Settlement-Bar zahlt sofort."""

    def test_no_position_no_booking(self) -> None:
        # Kauf erst bei 08:01, nach der 08:00-Settlement
        start = datetime(2024, 1, 1, 7, 59, tzinfo=UTC)  # 07:59, 08:00, 08:01, 08:02
        result = _run(_candles(start, 4), buys=[2], funding_rate=0.001)
        assert len(result.metadata["open_positions"]) == 1
        assert result.metadata["total_funding"] == 0.0

    def test_entry_on_settlement_bar_pays_same_bar(self) -> None:
        # BUY auf der 08:00-Bar: Booking läuft nach dem Fill derselben Bar
        start = datetime(2024, 1, 1, 7, 59, tzinfo=UTC)  # 07:59, 08:00, 08:01, 08:02
        result = _run(_candles(start, 4), buys=[1], funding_rate=0.001)
        assert result.metadata["total_funding"] == pytest.approx(
            _open_quantity(result) * CLOSE * 0.001
        )


class TestFundingRateOverride:
    """``funding_rates``-Mapping überschreibt ``config.funding_rate`` pro Settlement."""

    def test_mapping_key_overrides_config_default(self) -> None:
        start = datetime(2024, 1, 1, 7, 58, tzinfo=UTC)  # Settlement 08:00
        settlement = datetime(2024, 1, 1, 8, tzinfo=UTC)
        result = _run(
            _candles(start, 4), buys=[0], funding_rate=0.0001,
            funding_rates={settlement: 0.01},
        )
        assert result.metadata["total_funding"] == pytest.approx(
            _open_quantity(result) * CLOSE * 0.01
        )

    def test_missing_mapping_key_falls_back_to_config(self) -> None:
        start = datetime(2024, 1, 1, 7, 58, tzinfo=UTC)  # Settlement 08:00
        other = datetime(2023, 1, 1, 8, tzinfo=UTC)
        result = _run(
            _candles(start, 4), buys=[0], funding_rate=0.0001,
            funding_rates={other: 0.01},
        )
        assert result.metadata["total_funding"] == pytest.approx(
            _open_quantity(result) * CLOSE * 0.0001
        )


class TestTotalFundingAccumulation:
    """``total_funding`` summiert über mehrere Settlements (inkl. 00:00/16:00)."""

    def test_full_day_books_all_four_settlements(self) -> None:
        # 00:00..24:00: Settlements 00:00 (Bar 0 = Entry-Bar), 08:00, 16:00, 00:00
        candles = _candles(datetime(2024, 1, 1, tzinfo=UTC), 1441)
        result = _run(candles, buys=[0], funding_rate=0.001)
        assert result.metadata["total_funding"] == pytest.approx(
            _open_quantity(result) * CLOSE * 0.001 * 4
        )


class TestPyramidingFunding:
    """Pyramiding: Funding greift auf die summierte Positions-Quantity."""

    def test_funding_applies_to_summed_quantity(self) -> None:
        # BUYs bei Bar 10/100 (Aufstapeln), Settlements 08:00 + 16:00
        candles = _candles(datetime(2024, 1, 1, tzinfo=UTC), 961)
        result = _run(candles, buys=[10, 100], funding_rate=0.001)
        single = _run(candles, buys=[10], funding_rate=0.0)
        assert len(result.metadata["open_positions"]) == 1
        # Zweiter BUY wurde auf die bestehende Position gestapelt
        assert _open_quantity(result) > _open_quantity(single)
        expected = _open_quantity(result) * CLOSE * 0.001 * 2
        assert result.metadata["total_funding"] == pytest.approx(expected)


class TestFundingDisabled:
    """``funding_rate=0.0`` reproduziert die Legacy-Abrechnung exakt."""

    def test_zero_rate_reproduces_legacy_accounting(self) -> None:
        with_settlement = _candles(datetime(2024, 1, 1, tzinfo=UTC), 961)  # 08:00/16:00 drin
        # Gleiches Fenster ohne 8h-Grenzen = Referenz ohne Settlement-Bars
        without_settlement = _candles(datetime(2024, 1, 1, 1, tzinfo=UTC), 961)
        zero = _run(with_settlement, buys=[10], funding_rate=0.0)
        legacy = _run(without_settlement, buys=[10], funding_rate=0.0)
        # Guard: auf denselben Kerzen wird bei rate>0 tatsächlich gebucht
        assert _run(with_settlement, buys=[10], funding_rate=0.001).metadata["total_funding"] > 0
        assert zero.metadata["total_funding"] == 0.0
        assert zero.metadata["equity_curve"] == legacy.metadata["equity_curve"]
        assert zero.metadata["final_equity"] == legacy.metadata["final_equity"]


class _FakeChEngine:
    """Duck-Typ-Ersatz für die ClickHouse-Engine (rekordiert SQL)."""

    def __init__(
        self, rows: list[list[str]] | None = None, names: list[str] | None = None
    ) -> None:
        self.names = names or ["funding_time", "funding_rate"]
        self.rows = rows or []
        self.queries: list[str] = []

    def query(self, sql: str) -> tuple[list[str], list[list[str]]]:
        self.queries.append(sql)
        return self.names, self.rows


class _RaisingChEngine:
    """Simuliert eine fehlende funding_rates-Tabelle."""

    def query(self, sql: str) -> tuple[list[str], list[list[str]]]:
        raise RuntimeError("Table trading_events.funding_rates doesn't exist")


class TestLoadFundingRates:
    """``load_funding_rates``: Parsing, Zeitfenster, Fail-Soft."""

    def test_parses_rows_to_aware_utc_keys(self) -> None:
        engine = _FakeChEngine(
            rows=[
                ["2024-01-01 00:00:00", "0.0001"],
                ["2024-01-01 08:00:00", "-0.0002"],
            ]
        )
        rates = load_funding_rates(engine, "BTC/USDT", "BINANCE_FUTURES", None, None)
        assert rates == {
            datetime(2024, 1, 1, tzinfo=UTC): 0.0001,
            datetime(2024, 1, 1, 8, tzinfo=UTC): -0.0002,
        }

    def test_missing_table_returns_empty_dict(self) -> None:
        result = load_funding_rates(
            _RaisingChEngine(), "BTC/USDT", "BINANCE_FUTURES", None, None
        )
        assert result == {}

    def test_time_window_appears_in_sql(self) -> None:
        engine = _FakeChEngine()
        load_funding_rates(
            engine,
            "BTC/USDT",
            "BINANCE_FUTURES",
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 31, tzinfo=UTC),
        )
        sql = engine.queries[0]
        assert "FROM funding_rates" in sql
        assert "funding_time BETWEEN '2024-01-01 00:00:00' AND '2024-01-31 00:00:00'" in sql
        assert "instrument = 'BTC/USDT'" in sql
        assert "venue = 'BINANCE_FUTURES'" in sql
