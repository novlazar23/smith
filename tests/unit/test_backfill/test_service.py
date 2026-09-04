"""Tests für die Backfill-Orchestrierung (apps/backfill/service.py).

Fake-Client (keine HTTP) + Fake-Engine (kein ClickHouse): Idempotenz
(nur fehlende Lücken laden), Dry-Run (keine Downloads, keine
Schreibzugriffe) sowie die reinen Lücken-/Fenster-/Monats-Berechnungen.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from apps.backfill.client import BackfillCandle
from apps.backfill.service import (
    BackfillConfig,
    BackfillService,
    DayCoverage,
    compute_missing_intervals,
    day_is_complete,
    day_window,
    estimate_requests,
    minutes_to_intervals,
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
    """Fake-ClickHouse-Engine mit aufzeichnenden ``query()``/``_execute()``.

    Modelliert die vorhandene Abdeckung als das kontinuierliche
    Intervall ``existing`` und leitet daraus die pro-Tage-Abdeckung
    sowie die Minuten-Details für Teil-Tage ab.
    """

    def __init__(
        self,
        existing: tuple[datetime, datetime] | None = None,
        total: int = 700_000,
        holes: tuple[tuple[datetime, datetime], ...] = (),
    ) -> None:
        self.config = ClickHouseConfig(database="trading_events")
        self.statements: list[tuple[str, str]] = []
        self._existing = existing
        self._total = total
        self._holes = holes

    @staticmethod
    def _between(sql: str) -> tuple[datetime, datetime]:
        matches = re.findall(r"BETWEEN '([^']+)' AND '([^']+)'", sql)
        start = datetime.strptime(matches[-1][0], _DT).replace(tzinfo=UTC)
        end = datetime.strptime(matches[-1][1], _DT).replace(tzinfo=UTC)
        return start, end

    def _coverage_intervals(self) -> list[tuple[datetime, datetime]]:
        if self._existing is None:
            return []
        intervals: list[list[datetime]] = [list(self._existing)]
        for hole_start, hole_end in self._holes:
            remaining: list[list[datetime]] = []
            for cov_start, cov_end in intervals:
                if hole_end < cov_start or hole_start > cov_end:
                    remaining.append([cov_start, cov_end])
                    continue
                if hole_start > cov_start:
                    remaining.append([cov_start, hole_start - ONE_MINUTE])
                if hole_end < cov_end:
                    remaining.append([hole_end + ONE_MINUTE, cov_end])
            intervals = remaining
        return [tuple(interval) for interval in intervals]

    def _covered_minutes(self, start: datetime, end: datetime) -> list[datetime]:
        minutes: list[datetime] = []
        for cov_start, cov_end in self._coverage_intervals():
            cov_start = max(start, cov_start)
            cov_end = min(end, cov_end)
            moment = cov_start
            while moment <= cov_end:
                minutes.append(moment)
                moment += ONE_MINUTE
        return minutes

    def _day_rows(self, start: datetime, end: datetime) -> list[list[str]]:
        rows: list[list[str]] = []
        day = start.replace(hour=0, minute=0, second=0, microsecond=0)
        while day <= end:
            day_start = max(day, start)
            day_end = min(day + timedelta(days=1) - ONE_MINUTE, end)
            covered = self._covered_minutes(day_start, day_end)
            if covered:
                first, last = covered[0], covered[-1]
                rows.append(
                    [
                        day.strftime(_DT),
                        first.strftime(_DT),
                        last.strftime(_DT),
                        str(len(covered)),
                    ]
                )
            day += timedelta(days=1)
        return rows

    def _minute_rows(self, start: datetime, end: datetime) -> list[list[str]]:
        return [[moment.strftime(_DT)] for moment in self._covered_minutes(start, end)]

    def query(self, sql: str) -> tuple[list[str], list[list[str]]]:
        self.statements.append(("query", sql))
        if "toStartOfDay(open_time)" in sql:
            start, end = self._between(sql)
            return ["d", "mn", "mx", "n"], self._day_rows(start, end)
        if "SELECT open_time" in sql:
            start, end = self._between(sql)
            return ["open_time"], self._minute_rows(start, end)
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
    holes: tuple[tuple[datetime, datetime], ...] = (),
) -> tuple[BackfillService, FakeClient, FakeEngine]:
    fake_client = client or FakeClient()
    engine = FakeEngine(existing, holes=holes)
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

    def test_internal_gap_is_fetched(self) -> None:
        # Regression (2022-08 → 2026-02-Vorfall): Daten vorhanden an
        # beiden Enden, komplette Lücke in der Mitte — die alte
        # min/max-Betrachtung hatte sie nie erkannt.
        hole_start = (T0 + timedelta(days=31)).replace(hour=0, minute=0)
        hole_end = (T1 - timedelta(days=31)).replace(hour=0, minute=0) - ONE_MINUTE
        service, client, _ = _service(
            existing=(T0, T1),
            start=T0,
            end=T1,
            holes=((hole_start, hole_end),),
        )
        result = service.run()

        assert client.calls == [("BTC/USDT", hole_start, hole_end)]
        assert result.summaries[0].ranges == ((hole_start, hole_end),)

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
        assert any("toStartOfDay(open_time)" in sql for _, sql in engine.statements)

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


class TestComputeMissingIntervals:
    def test_no_coverage_full_range(self) -> None:
        assert compute_missing_intervals(T0, T1, []) == [(T0, T1)]

    def test_fully_covered(self) -> None:
        assert compute_missing_intervals(T0, T1, [(T0 - ONE_MINUTE, T1 + ONE_MINUTE)]) == []

    def test_gap_both_sides(self) -> None:
        start, end = T0 - timedelta(days=30), T1 + timedelta(days=10)
        assert compute_missing_intervals(start, end, [(T0, T1)]) == [
            (start, T0 - ONE_MINUTE),
            (T1 + ONE_MINUTE, end),
        ]

    def test_internal_gap_detected(self) -> None:
        # Regression: min/max-Betrachtung hätte die Lücke verfehlt
        first = (T0, T0 + timedelta(days=1))
        second = (T0 + timedelta(days=3), T1)
        assert compute_missing_intervals(T0, T1, [first, second]) == [
            (T0 + timedelta(days=1) + ONE_MINUTE, T0 + timedelta(days=3) - ONE_MINUTE)
        ]

    def test_overlapping_covered_intervals_are_merged(self) -> None:
        covered = [
            (T0, T0 + timedelta(hours=1)),
            (T0 + timedelta(minutes=30), T0 + timedelta(hours=2)),
            (T1 - ONE_MINUTE, T1),
        ]
        assert compute_missing_intervals(T0, T1, covered) == [
            (T0 + timedelta(hours=2) + ONE_MINUTE, T1 - ONE_MINUTE - ONE_MINUTE)
        ]

    def test_covered_outside_window_is_clipped(self) -> None:
        assert compute_missing_intervals(T0, T1, [(T0 - timedelta(days=1), T1 + timedelta(days=1))]) == []

    def test_degenerate_window(self) -> None:
        assert compute_missing_intervals(T1, T0, []) == []
        assert compute_missing_intervals(T0, T0, []) == []


class TestDayWindow:
    def test_full_day_inside_window(self) -> None:
        day = T0
        assert day_window(T0 - ONE_MINUTE, T0 + timedelta(days=2), day) == (
            T0,
            T0 + timedelta(days=1) - ONE_MINUTE,
        )

    def test_clipped_to_window_edges(self) -> None:
        start, end = T0 + timedelta(hours=3), T0 + timedelta(hours=5)
        assert day_window(start, end, T0) == (start, end)


class TestDayIsComplete:
    def test_full_day_is_complete(self) -> None:
        day = T0
        coverage = DayCoverage(day, day, day + timedelta(days=1) - ONE_MINUTE, 1440)
        assert day_is_complete(day, day + timedelta(days=2), coverage) is True

    def test_missing_minute_is_partial(self) -> None:
        day = T0
        coverage = DayCoverage(day, day, day + timedelta(days=1) - ONE_MINUTE, 1439)
        assert day_is_complete(day, day + timedelta(days=2), coverage) is False

    def test_window_edge_day_uses_clipped_expectation(self) -> None:
        day = T0
        end = T0 + timedelta(hours=2)  # Tag zählt nur 121 Minuten
        coverage = DayCoverage(day, day, end, 121)
        assert day_is_complete(day, end, coverage) is True
        assert day_is_complete(day, end, DayCoverage(day, day, end, 120)) is False


class TestMinutesToIntervals:
    def test_consecutive_minutes_merge(self) -> None:
        minutes = [T0, T0 + ONE_MINUTE, T0 + 2 * ONE_MINUTE]
        assert minutes_to_intervals(minutes) == [(T0, T0 + 2 * ONE_MINUTE)]

    def test_gap_splits_intervals(self) -> None:
        minutes = [T0, T0 + ONE_MINUTE, T0 + 3 * ONE_MINUTE]
        assert minutes_to_intervals(minutes) == [
            (T0, T0 + ONE_MINUTE),
            (T0 + 3 * ONE_MINUTE, T0 + 3 * ONE_MINUTE),
        ]

    def test_unsorted_input(self) -> None:
        minutes = [T0 + 2 * ONE_MINUTE, T0, T0 + ONE_MINUTE]
        assert minutes_to_intervals(minutes) == [(T0, T0 + 2 * ONE_MINUTE)]

    def test_empty(self) -> None:
        assert minutes_to_intervals([]) == []


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
