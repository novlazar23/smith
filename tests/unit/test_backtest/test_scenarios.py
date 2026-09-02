"""Tests für die Szenario-Presets und das Parsing."""

from __future__ import annotations

from datetime import date

import pytest
from apps.backtest.scenarios import SCENARIOS, parse_scenarios


class TestPresetRanges:
    """Die Preset-Datumsfenster sind korrekt definiert."""

    def test_crash_2021_05(self) -> None:
        assert parse_scenarios("crash-2021-05") == [
            ("crash-2021-05", date(2021, 5, 15), date(2021, 5, 25))
        ]

    def test_pump_2021_11(self) -> None:
        assert parse_scenarios("pump-2021-11") == [
            ("pump-2021-11", date(2021, 11, 1), date(2021, 11, 10))
        ]

    def test_crash_2022_06(self) -> None:
        assert parse_scenarios("crash-2022-06") == [
            ("crash-2022-06", date(2022, 6, 15), date(2022, 6, 25))
        ]

    def test_range_2022_03(self) -> None:
        assert parse_scenarios("range-2022-03") == [
            ("range-2022-03", date(2022, 3, 1), date(2022, 3, 31))
        ]

    def test_full_has_no_bounds(self) -> None:
        assert parse_scenarios("full") == [("full", None, None)]

    def test_presets_have_descriptions(self) -> None:
        for preset in SCENARIOS.values():
            assert preset.label
            assert preset.description
        assert set(SCENARIOS) == {
            "crash-2021-05",
            "pump-2021-11",
            "crash-2022-06",
            "range-2022-03",
            "full",
        }


class TestParsing:
    """Kommagetrennte Listen, Whitespace, Fehler."""

    def test_comma_separated_list(self) -> None:
        raw = "crash-2021-05,pump-2021-11,crash-2022-06,range-2022-03"
        scenarios = parse_scenarios(raw)
        assert [label for label, _, _ in scenarios] == [
            "crash-2021-05",
            "pump-2021-11",
            "crash-2022-06",
            "range-2022-03",
        ]
        assert all(start is not None and end is not None for _, start, end in scenarios)

    def test_whitespace_ignored(self) -> None:
        scenarios = parse_scenarios(" full , crash-2021-05 ")
        assert scenarios == [("full", None, None), ("crash-2021-05", date(2021, 5, 15), date(2021, 5, 25))]

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(ValueError, match="Unbekanntes Szenario"):
            parse_scenarios("krach-1999")

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_scenarios("  , ")
