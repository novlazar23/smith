"""Tests für die ClickHouse-Schreibseite (apps/backfill/storage.py).

Die ClickHouse-Engine wird durch ein Fake-Objekt mit ``query()`` und
``_execute()`` ersetzt, das alle SQL-Statements aufzeichnet und
vorgefertigte ``(names, rows)``-Ergebnisse liefert.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from apps.backfill import storage
from apps.backfill.client import BackfillCandle
from packages.persistence.clickhouse.engine import ClickHouseConfig

START = datetime(2025, 1, 1, tzinfo=UTC)


class FakeEngine:
    """Fake-ClickHouse-Engine: zeichnet SQL auf, liefert Canned-Rows."""

    def __init__(self, canned: dict[str, tuple[list[str], list[list[str]]]] | None = None) -> None:
        self.config = ClickHouseConfig(database="trading_events")
        self.canned = canned or {}
        self.statements: list[tuple[str, str]] = []

    def query(self, sql: str) -> tuple[list[str], list[list[str]]]:
        self.statements.append(("query", sql))
        for key, result in self.canned.items():
            if key in sql:
                return result
        return [], []

    def _execute(self, sql: str) -> None:
        self.statements.append(("execute", sql))


def _candles(count: int, start: datetime) -> list[BackfillCandle]:
    """Baut ``count`` Kerzen ab ``start`` (1m-Schritte, feste Werte)."""
    return [
        BackfillCandle(
            instrument="BTC/USDT",
            venue="BINANCE_FUTURES",
            open_time=start + timedelta(minutes=i),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=10.0,
        )
        for i in range(count)
    ]


def _executes(engine: FakeEngine) -> list[str]:
    return [sql for kind, sql in engine.statements if kind == "execute"]


class TestExistingRange:
    def test_parses_min_max_as_utc_datetimes(self) -> None:
        engine = FakeEngine(
            {
                "min(open_time)": (
                    ["min_open_time", "max_open_time"],
                    [["2025-01-01 00:00:00", "2025-06-01 12:34:56"]],
                )
            }
        )
        result = storage.existing_range(engine, "BTC/USDT", "BINANCE_FUTURES")
        assert result == (
            datetime(2025, 1, 1, tzinfo=UTC),
            datetime(2025, 6, 1, 12, 34, 56, tzinfo=UTC),
        )
        sql = engine.statements[0][1]
        assert "min(open_time)" in sql
        assert "max(open_time)" in sql
        assert "instrument = 'BTC/USDT'" in sql
        assert "venue = 'BINANCE_FUTURES'" in sql

    def test_empty_string_aggregates_yield_none(self) -> None:
        engine = FakeEngine(
            {"min(open_time)": (["min_open_time", "max_open_time"], [["", ""]])}
        )
        assert storage.existing_range(engine, "BTC/USDT", "BINANCE_FUTURES") is None

    def test_no_rows_yield_none(self) -> None:
        engine = FakeEngine()
        assert storage.existing_range(engine, "BTC/USDT", "BINANCE_FUTURES") is None


class TestExistingDayCoverage:
    def test_parses_day_rows(self) -> None:
        engine = FakeEngine(
            {
                "toStartOfDay(open_time)": (
                    ["d", "mn", "mx", "n"],
                    [
                        ["2025-01-01 00:00:00", "2025-01-01 00:00:00", "2025-01-01 23:59:00", "1440"],
                        ["2025-01-02 00:00:00", "2025-01-02 00:00:00", "2025-01-02 12:00:00", "721"],
                    ],
                )
            }
        )
        result = storage.existing_day_coverage(
            engine, "BTC/USDT", "BINANCE_FUTURES", START, START + timedelta(days=2)
        )
        assert result == [
            (
                datetime(2025, 1, 1, tzinfo=UTC),
                datetime(2025, 1, 1, tzinfo=UTC),
                datetime(2025, 1, 1, 23, 59, tzinfo=UTC),
                1440,
            ),
            (
                datetime(2025, 1, 2, tzinfo=UTC),
                datetime(2025, 1, 2, tzinfo=UTC),
                datetime(2025, 1, 2, 12, 0, tzinfo=UTC),
                721,
            ),
        ]
        sql = engine.statements[0][1]
        assert "toStartOfDay(open_time)" in sql
        assert "uniqExact(open_time)" in sql

    def test_no_rows_yield_empty_list(self) -> None:
        engine = FakeEngine()
        result = storage.existing_day_coverage(
            engine, "BTC/USDT", "BINANCE_FUTURES", START, START + timedelta(days=2)
        )
        assert result == []


class TestExistingMinutes:
    def test_parses_minute_rows(self) -> None:
        engine = FakeEngine(
            {
                "SELECT open_time": (
                    ["open_time"],
                    [
                        ["2025-01-01 00:00:00"],
                        ["2025-01-01 00:02:00"],
                    ],
                )
            }
        )
        result = storage.existing_minutes(
            engine, "BTC/USDT", "BINANCE_FUTURES", START, START + timedelta(hours=1)
        )
        assert result == [
            datetime(2025, 1, 1, tzinfo=UTC),
            datetime(2025, 1, 1, 0, 2, tzinfo=UTC),
        ]

    def test_no_rows_yield_empty_list(self) -> None:
        engine = FakeEngine()
        result = storage.existing_minutes(
            engine, "BTC/USDT", "BINANCE_FUTURES", START, START + timedelta(hours=1)
        )
        assert result == []


class TestDeleteRange:
    def test_no_delete_issued_for_replacing_merge_tree(self) -> None:
        """Dokumentierte Dedup-Entscheidung: ``candles`` ist
        ``ReplacingMergeTree(ingestion_time)`` mit ORDER BY
        ``(instrument, venue, timeframe, open_time)`` → Neu-INSERTs
        ersetzen alte Zeiten beim Merge, ein DELETE ist nicht nötig."""
        engine = FakeEngine()
        storage.delete_range(
            engine, "BTC/USDT", "BINANCE_FUTURES", START, START + timedelta(days=1)
        )
        assert engine.statements == []


class TestCountCandles:
    def test_returns_count(self) -> None:
        engine = FakeEngine({"count()": (["total"], [["700000"]])})
        assert storage.count_candles(engine, "BTC/USDT", "BINANCE_FUTURES") == 700000

    def test_empty_result_is_zero(self) -> None:
        engine = FakeEngine()
        assert storage.count_candles(engine, "BTC/USDT", "BINANCE_FUTURES") == 0


class TestInsertCandles:
    def test_batches_at_5000_rows(self) -> None:
        engine = FakeEngine()
        inserted = storage.insert_candles(engine, _candles(12_003, START))
        executes = _executes(engine)
        assert inserted == 12_003
        assert len(executes) == 3
        assert [sql.count("('BTC/USDT'") for sql in executes] == [5000, 5000, 2003]

    def test_exact_batch_boundary_single_insert(self) -> None:
        engine = FakeEngine()
        storage.insert_candles(engine, _candles(5000, START))
        assert len(_executes(engine)) == 1

    def test_one_row_over_boundary_two_inserts(self) -> None:
        engine = FakeEngine()
        storage.insert_candles(engine, _candles(5001, START))
        executes = _executes(engine)
        assert len(executes) == 2
        assert [sql.count("('BTC/USDT'") for sql in executes] == [5000, 1]

    def test_empty_rows_no_insert(self) -> None:
        engine = FakeEngine()
        assert storage.insert_candles(engine, []) == 0
        assert engine.statements == []

    def test_row_format_matches_ingestion_sink_shape(self) -> None:
        engine = FakeEngine()
        storage.insert_candles(engine, _candles(1, START))
        sql = _executes(engine)[0]
        assert sql.startswith("INSERT INTO trading_events.candles_history (")
        for expected in (
            "instrument, venue, timeframe, open_time, close_time",
            "'BTC/USDT'",
            "'BINANCE_FUTURES'",
            "'1m'",
            "'2025-01-01 00:00:00'",  # open_time (DateTime als String, UTC)
            "'2025-01-01 00:00:59'",  # close_time = open_time + 59 s
            "'backfill'",  # source
        ):
            assert expected in sql
        assert "100.0, 101.0, 99.0, 100.5, 10.0" in sql


class TestEnsureTable:
    """candles_history-DDL: idempotent, ohne TTL (Szenario-Daten > 5 Jahre)."""

    def _executes(self, engine: FakeEngine) -> list[str]:
        return [sql for kind, sql in engine.statements if kind == "execute"]

    def test_creates_table_with_replacing_merge_tree(self) -> None:
        engine = FakeEngine()
        storage.ensure_table(engine)
        ddl = self._executes(engine)[0]
        assert ddl.startswith("CREATE TABLE IF NOT EXISTS trading_events.candles_history")
        assert "ENGINE = ReplacingMergeTree(ingestion_time)" in ddl
        assert "ORDER BY (instrument, venue, timeframe, open_time)" in ddl

    def test_no_ttl_in_ddl(self) -> None:
        """Keine TTL — sonst entfernt ClickHouse 2021er-Szenario-Daten
        während des Backfills (open_time + 5 Jahre ist bereits abgelaufen)."""
        engine = FakeEngine()
        storage.ensure_table(engine)
        assert "TTL" not in self._executes(engine)[0]

    def test_removes_legacy_ttl_from_existing_table(self) -> None:
        """Bestehende Tabelle mit TTL → genau ein ``REMOVE TTL``."""
        engine = FakeEngine(
            canned={
                "system.tables": (
                    ["create_table_query"],
                    [["CREATE TABLE ... TTL open_time + toIntervalYear(5)"]],
                )
            }
        )
        storage.ensure_table(engine)
        alters = [sql for sql in self._executes(engine) if sql.startswith("ALTER TABLE")]
        assert alters == ["ALTER TABLE trading_events.candles_history REMOVE TTL"]

    def test_no_alter_when_table_has_no_ttl(self) -> None:
        """``REMOVE TTL`` ohne bestehende TTL ist ein ClickHouse-Hard-Error
        (HTTP 400, Code 36) → nur bei vorhandener TTL ausführen."""
        engine = FakeEngine(
            canned={"system.tables": (["create_table_query"], [["CREATE TABLE ..."]])}
        )
        storage.ensure_table(engine)
        assert not any(sql.startswith("ALTER TABLE") for sql in self._executes(engine))
