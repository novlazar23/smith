"""Tests für die CLI (``apps/backfill/__main__.py``) — Datums- und Arg-Validierung.

Der Bug, gegen den ``test_day_end_parsing`` regressionsgesichert ist:
``--end`` wurde mit ``second=86340`` in den ``datetime``-Konstruktor
gereicht (muss 0..59 sein) und crashte jeden Lauf mit ``--end``.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from apps.backfill.__main__ import _parse_day, main


class TestParseDay:
    """Tagsgrenzen: Start=00:00:00, Ende=23:59:00 (letzter 1m-Tick), UTC."""

    def test_start_of_day(self) -> None:
        assert _parse_day("2021-05-15") == datetime(2021, 5, 15, 0, 0, 0, tzinfo=UTC)

    def test_day_end_parsing(self) -> None:
        assert _parse_day("2021-05-15", at_day_end=True) == datetime(2021, 5, 15, 23, 59, 0, tzinfo=UTC)

    def test_day_end_before_next_day_start(self) -> None:
        assert _parse_day("2021-05-15", at_day_end=True) < _parse_day("2021-05-16")

    def test_invalid_date_raises(self) -> None:
        with pytest.raises(ValueError):
            _parse_day("15.05.2021")


class TestMainValidation:
    """Arg-Validierung (Exit-Code 2) — ohne ClickHouse-Zugriff."""

    def test_start_before_end(self) -> None:
        assert main(["--start", "2021-05-25", "--end", "2021-05-15"]) == 2

    def test_same_day_start_end_is_valid_window(self) -> None:
        # 00:00:00 → 23:59:00 desselben Tages ist ein gültiges 1m-Fenster
        # (CH-Zugriff folgt erst danach — hier wird vor dem CH-Setup
        # abgefangen, weil der Tag gültig ist; Exit 1 = CH nicht erreichbar,
        # was in der Test-Umgebung der Fall ist)
        assert main(["--start", "2021-05-15", "--end", "2021-05-15", "--ch-host", "invalid-host-xyz"]) == 1

    def test_months_must_be_positive(self) -> None:
        assert main(["--months", "0"]) == 2

    def test_invalid_date_returns_2(self) -> None:
        assert main(["--start", "15.05.2021"]) == 2

    def test_empty_instruments_return_2(self) -> None:
        assert main(["--instruments", ""]) == 2
