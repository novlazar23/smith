"""Tests für die Backfill-Orchestrierung (apps/backfill/service.py).

Fake-Client (keine HTTP) + Fake-Engine (kein ClickHouse): Idempotenz
(nur fehlende Lücken laden), Dry-Run (keine Downloads, keine
Schreibzugriffe) sowie die reinen Lücken-/Fenster-/Monats-Berechnungen.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from apps.backfill.client import BackfillCandle
from apps.backfill.service import (
    BackfillConfig,
    BackfillService,
    compute_missing_ranges,
    estimate_requests,
    months_ago,
)
from packages.persistence.clickhouse.engine import ClickHouseConfig

ONE_MINUTE = timedelta(minutes=1)
T0 = datetime(2025, 1, 1, tzinfo=UTC)
T1 = datetime(2025, 6, 1, tzinfo=UTC)
NOW = datetime(2026, 9, 2, 8, 15, tzinfo=UTC)
_DT = "%Y-%m-%d %H:%M:%S"


class FakeClient:
    """Simulierter KlineClient: liefert pro Range eine feste Kerzenanzahl."""

    def __init__(self, candles_per_range: int = 123) -> None:
        self.candles_per_range = candles_per_range
        self.raise_for: frozenset[str] = frozenset()
        self.calls: list[tuple[str, datetime, datetime]] = []

    def fetch_range(
        self,
        instrument: str,
        start: datetime,
        end: datetime,
        on_chunk: Callable[[datetime, datetime, int], None] | None = None,
    ) -> list[BackfillCandle]:
        del on_chunk
        if instrument in self.raise_for:
            raise RuntimeError(f"simulierter Download-Fehler für {instrument}")
        self.calls.append((instrument, start, end))
        return [
            BackfillCandle(
                instrument=instrument,
                venue="BINANCE_FUTURES",
                open_time=start + timedelta(minutes=i),
                open=1.0,
                high=2.0,
                low=0.5,
                close=1.5,
                volume=1.0,
            )
            for i in range(self.candles_per_range)
        ]


class FakeEngine:
    """Fake-ClickHouse-Engine mit aufzeichnenden ``query()``/``_execute()``."""

    def __init__(
        self,
        existing: tuple[datetime, datetime] | None = None,
        total: int = 700_000,
    ) -> None:
        self.config = ClickHouseConfig(database="trading_events")
        self.statements: list[tuple[str, str]] = []
        self._existing = existing
        self._total = total

    def query(self, sql: str) -> tuple[list[str], list[list[str]]]:
        self.statements.append(("query", sql))
        if "min(open_time)" in sql:
            if self._existing is None:
                return ["min_open_time", "max_open_time"], [["", ""]]
            return (
                ["min_open_time", "max_open_time"],
                [
                    [
                        self._existing[0].strftime(_DT),
                        self._existing[1].strftime(_DT),
                    ]
                ],
            )
        if "count()" in sql:
            return ["total"], [[str(self._total)]]
        return [], []

    def _execute(self, sql: str) -> None:
        self.statements.append(("execute", sql))


def _service(
    existing: tuple[datetime, datetime] | None = None,
    *,
    dry_run: bool = False,
    client: FakeClient | None = None,
    months: int = 12,
    start: datetime | None = None,
    end: datetime | None = None,
) -> tuple[BackfillService, FakeClient, FakeEngine]:
    fake_client = client or FakeClient()
    engine = FakeEngine(existing)
    config = BackfillConfig(
        months=months,
        instruments=("BTC/USDT",),
        start=start,
        end=end,
        dry_run=dry_run,
    )
    return BackfillService(config, fake_client, engine, now=NOW), fake_client, engine


def _executes(engine: FakeEngine) -> list[str]:
    return [sql for kind, sql in engine.statements if kind == "execute"]


class TestIdempotency:
    def test_only_missing_ranges_are_fetched(self) -> None:
        """Vorhanden (T0..T1), gewünscht (T0-30d..T1+10d) → nur die
        vorderen 30 Tage und die hinteren 10 Tage werden geladen."""
        service, client, engine = _service(
            existing=(T0, T1),
            start=T0 - timedelta(days=30),
            end=T1 + timedelta(days=10),
        )
        result = service.run()

        assert client.calls == [
            ("BTC/USDT", T0 - timedelta(days=30), T0 - ONE_MINUTE),
            ("BTC/USDT", T1 + ONE_MINUTE, T1 + timedelta(days=10)),
        ]
        summary = result.summaries[0]
        assert summary.ranges == (
            (T0 - timedelta(days=30), T0 - ONE_MINUTE),
            (T1 + ONE_MINUTE, T1 + timedelta(days=10)),
        )
        assert summary.fetched_candles == 2 * 123
        assert summary.total_after == 700_000
        assert summary.estimated_requests == 44 + 15
        assert result.failures == ()
        # je Lücke genau ein INSERT-Batch (123 Kerzen < 5000)
        assert len(_executes(engine)) == 2

    def test_fully_covered_window_fetches_nothing(self) -> None:
        service, client, engine = _service(existing=(T0, T1), start=T0, end=T1)
        result = service.run()

        assert client.calls == []
        assert result.summaries[0].ranges == ()
        assert result.summaries[0].fetched_candles == 0
        assert _executes(engine) == []

    def test_empty_table_backfills_whole_window(self) -> None:
        service, client, _ = _service(existing=None, start=T0, end=T1)
        result = service.run()

        assert client.calls == [("BTC/USDT", T0, T1)]
        assert result.summaries[0].ranges == ((T0, T1),)

    def test_instrument_failure_does_not_stop_run(self) -> None:
        client = FakeClient()
        client.raise_for = frozenset({"ETH/USDT"})
        engine = FakeEngine(existing=None)
        config = BackfillConfig(
            months=12, instruments=("BTC/USDT", "ETH/USDT"), start=T0, end=T1
        )
        service = BackfillService(config, client, engine, now=NOW)
        result = service.run()

        assert [summary.instrument for summary in result.summaries] == ["BTC/USDT"]
        assert [name for name, _ in result.failures] == ["ETH/USDT"]
        assert "simulierter Download-Fehler" in result.failures[0][1]


class TestDryRun:
    def test_performs_no_downloads_or_writes(self) -> None:
        """Dry-Run: kein Client-Aufruf, keine INSERT-/DELETE-Statements;
        der Lesezugriff auf das vorhandene Fenster bleibt erlaubt."""
        service, client, engine = _service(
            existing=(T0, T1),
            dry_run=True,
            start=T0 - timedelta(days=30),
            end=T1 + timedelta(days=10),
        )
        result = service.run()

        assert client.calls == []
        assert "execute" not in {kind for kind, _ in engine.statements}
        assert all("delete" not in sql.lower() for _, sql in engine.statements)
        assert any("min(open_time)" in sql for _, sql in engine.statements)

        summary = result.summaries[0]
        assert summary.ranges == (
            (T0 - timedelta(days=30), T0 - ONE_MINUTE),
            (T1 + ONE_MINUTE, T1 + timedelta(days=10)),
        )
        assert summary.estimated_requests == 44 + 15
        assert summary.fetched_candles == 0
        assert summary.total_after is None
        assert result.failures == ()


class TestWindowComputation:
    def test_window_defaults_to_months_back(self) -> None:
        service, _, _ = _service(existing=None, months=12, dry_run=True)
        result = service.run()
        summary = result.summaries[0]
        assert (summary.start, summary.end) == (
            datetime(2025, 9, 2, 8, 15, tzinfo=UTC),
            NOW,
        )


class TestMonthsAgo:
    def test_full_year_back(self) -> None:
        assert months_ago(NOW, 12) == datetime(2025, 9, 2, 8, 15, tzinfo=UTC)

    def test_day_clamped_to_shorter_month(self) -> None:
        moment = datetime(2026, 3, 31, 12, 0, tzinfo=UTC)
        assert months_ago(moment, 1) == datetime(2026, 2, 28, 12, 0, tzinfo=UTC)

    def test_crosses_year_boundary(self) -> None:
        moment = datetime(2026, 1, 15, tzinfo=UTC)
        assert months_ago(moment, 2) == datetime(2025, 11, 15, tzinfo=UTC)


class TestComputeMissingRanges:
    def test_no_existing_full_range(self) -> None:
        assert compute_missing_ranges(T0, T1, None) == [(T0, T1)]

    def test_fully_covered(self) -> None:
        assert compute_missing_ranges(T0, T1, (T0 - ONE_MINUTE, T1 + ONE_MINUTE)) == []

    def test_partial_overlap_both_sides(self) -> None:
        start, end = T0 - timedelta(days=30), T1 + timedelta(days=10)
        assert compute_missing_ranges(start, end, (T0, T1)) == [
            (start, T0 - ONE_MINUTE),
            (T1 + ONE_MINUTE, end),
        ]

    def test_existing_entirely_before_window(self) -> None:
        existing = (T0 - timedelta(days=2), T0 - timedelta(days=1))
        assert compute_missing_ranges(T0, T1, existing) == [(T0, T1)]

    def test_existing_entirely_after_window(self) -> None:
        existing = (T1 + ONE_MINUTE, T1 + timedelta(days=1))
        assert compute_missing_ranges(T0, T1, existing) == [(T0, T1)]

    def test_degenerate_window(self) -> None:
        assert compute_missing_ranges(T1, T0, None) == []
        assert compute_missing_ranges(T0, T0, None) == []


class TestEstimateRequests:
    def test_full_page(self) -> None:
        assert estimate_requests([(T0, T0 + timedelta(minutes=999))]) == 1

    def test_page_plus_one_candle(self) -> None:
        assert estimate_requests([(T0, T0 + timedelta(minutes=1000))]) == 2

    def test_sums_over_ranges(self) -> None:
        ranges = [(T0, T0 + timedelta(days=30)), (T1, T1 + timedelta(days=10))]
        assert estimate_requests(ranges) == 44 + 15

    def test_empty(self) -> None:
        assert estimate_requests([]) == 0
