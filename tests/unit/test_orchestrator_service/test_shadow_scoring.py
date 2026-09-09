"""Tests für das operative Shadow-Scoring (Brier/Calibration).

Deckt die reinen Helfer aus ``packages.governance.shadow_integration`` sowie
den ``score_due_shadow_decisions``-Flow mit FakeConnection + StubProvider ab.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from packages.governance import shadow_integration
from packages.governance.shadow_integration import (
    average_report_probabilities,
    direction_from_return,
    horizon_to_seconds,
    normalize_probabilities,
    score_due_shadow_decisions,
)

from .conftest import FakeConnection, StubProvider, make_ohlcv

NOW = datetime(2026, 9, 9, 12, 0, 0, tzinfo=UTC)
BTC = "BTC/USDT"


class _Report:
    """AgentReport-Stellvertreter mit ``probabilities``-Attribut."""

    def __init__(self, probabilities: dict[str, float] | None) -> None:
        self.probabilities = probabilities


class TestDirectionFromReturn:
    """direction_from_return: up/down/range inkl. Threshold-Grenzen."""

    def test_up(self) -> None:
        assert direction_from_return(0.002, 0.001) == "up"

    def test_down(self) -> None:
        assert direction_from_return(-0.002, 0.001) == "down"

    def test_range(self) -> None:
        assert direction_from_return(0.0, 0.001) == "range"

    def test_threshold_boundaries_are_range(self) -> None:
        assert direction_from_return(0.001, 0.001) == "range"
        assert direction_from_return(-0.001, 0.001) == "range"

    def test_custom_threshold(self) -> None:
        assert direction_from_return(0.005, 0.01) == "range"
        assert direction_from_return(0.015, 0.01) == "up"
        assert direction_from_return(-0.015, 0.01) == "down"


class TestHorizonToSeconds:
    """horizon_to_seconds: s/m/h/d, ValueError bei unklaren Werten."""

    @pytest.mark.parametrize(
        ("horizon", "expected"),
        [
            ("15m", 900.0),
            ("1h", 3600.0),
            ("1d", 86400.0),
            ("90s", 90.0),
            ("2h", 7200.0),
            (" 15M ", 900.0),
        ],
    )
    def test_valid_horizons(self, horizon: str, expected: float) -> None:
        assert horizon_to_seconds(horizon) == expected

    @pytest.mark.parametrize(
        "horizon",
        ["", "  ", "abc", "15x", "1.5", "-5m", "0m", "m", "15 min", "nan"],
    )
    def test_invalid_horizons_raise(self, horizon: str) -> None:
        with pytest.raises(ValueError):
            horizon_to_seconds(horizon)


class TestNormalizeProbabilities:
    """normalize_probabilities: Fallunabhängigkeit, Fremdschlüssel, Summe 1.0."""

    def test_lower_case_passthrough(self) -> None:
        assert normalize_probabilities({"up": 0.5, "down": 0.3, "range": 0.2}) == {
            "up": 0.5,
            "down": 0.3,
            "range": 0.2,
        }

    def test_upper_and_mixed_case(self) -> None:
        assert normalize_probabilities({"UP": 0.5, "Down": 0.3, "RANGE": 0.2}) == {
            "up": 0.5,
            "down": 0.3,
            "range": 0.2,
        }

    def test_ignores_foreign_keys(self) -> None:
        assert normalize_probabilities(
            {"up": 0.5, "down": 0.3, "range": 0.2, "side": 9.0, "note": "x"}
        ) == {"up": 0.5, "down": 0.3, "range": 0.2}

    def test_normalizes_to_unit_sum(self) -> None:
        result = normalize_probabilities({"up": 1.0, "down": 1.0, "range": 1.0})
        assert result == pytest.approx({"up": 1 / 3, "down": 1 / 3, "range": 1 / 3})

    @pytest.mark.parametrize(
        "value",
        [
            None,
            "up",
            42,
            [],
            {},
            {"up": 0.5},
            {"down": 0.3, "range": 0.2},
            {"up": "0.5", "down": 0.3, "range": 0.2},
            {"up": -0.1, "down": 0.55, "range": 0.55},
            {"up": 0.0, "down": 0.0, "range": 0.0},
            {"up": float("inf"), "down": 0.0, "range": 0.0},
            {"up": True, "down": 0.3, "range": 0.7},
        ],
    )
    def test_invalid_returns_none(self, value: object) -> None:
        assert normalize_probabilities(value) is None


class TestAverageReportProbabilities:
    """average_report_probabilities: robust gegen Stubs und ungültige Reports."""

    def test_averages_valid_reports(self) -> None:
        reports = [
            _Report({"up": 0.8, "down": 0.1, "range": 0.1}),
            _Report({"up": 0.2, "down": 0.4, "range": 0.4}),
        ]
        assert average_report_probabilities(reports) == pytest.approx(
            {"up": 0.5, "down": 0.25, "range": 0.25}
        )

    def test_ignores_object_stubs(self) -> None:
        reports = [
            _Report({"up": 0.7, "down": 0.2, "range": 0.1}),
            object(),
            object(),
        ]
        assert average_report_probabilities(reports) == pytest.approx(
            {"up": 0.7, "down": 0.2, "range": 0.1}
        )

    def test_ignores_invalid_reports(self) -> None:
        reports = [
            _Report({"up": 0.5}),
            _Report(None),
            _Report({"up": 0.4, "down": 0.4, "range": 0.2}),
        ]
        assert average_report_probabilities(reports) == pytest.approx(
            {"up": 0.4, "down": 0.4, "range": 0.2}
        )

    @pytest.mark.parametrize(
        "reports", [None, [], [object(), object()], "not-a-list", 42]
    )
    def test_no_valid_reports_returns_none(self, reports: object) -> None:
        assert average_report_probabilities(reports) is None


def _due_row(**overrides: Any) -> dict[str, Any]:
    """Eine fällige shadow_decisions-Zeile (SELECT-Ergebnis)."""
    row: dict[str, Any] = {
        "run_id": "orch-20260901T000000Z-BTC/USDT",
        "instrument": BTC,
        "created_at": NOW - timedelta(hours=1),
        "probabilities": {"up": 0.7, "down": 0.2, "range": 0.1},
        "base_close": 100.0,
        "confidence": 0.7,
    }
    row.update(overrides)
    return row


def _score(
    conn: FakeConnection,
    provider: StubProvider,
    horizon_seconds: float = 900.0,
    range_threshold: float = 0.001,
    now: datetime = NOW,
    limit: int = 50,
) -> int:
    return score_due_shadow_decisions(
        conn=conn,
        provider=provider,
        horizon_seconds=horizon_seconds,
        range_threshold=range_threshold,
        now=now,
        limit=limit,
    )


def _update_params(conn: FakeConnection) -> list[dict[str, Any]]:
    return [
        p
        for (s, p) in conn.executed
        if s is shadow_integration.UPDATE_SCORED_SHADOW_DECISION
    ]


class TestScoreDueShadowDecisions:
    """score_due_shadow_decisions gegen FakeConnection + StubProvider."""

    def test_scores_due_row_with_brier_and_calibration(self) -> None:
        conn = FakeConnection()
        conn.queued_results = [[_due_row()]]
        # Close 110 vs Base 100 → +10 % → "up"
        provider = StubProvider({BTC: make_ohlcv(1, start_price=110.0)})

        assert _score(conn, provider) == 1

        params = _update_params(conn)
        assert len(params) == 1
        assert params[0]["actual_direction"] == "up"
        # (0.7-1)^2 + 0.2^2 + 0.1^2 = 0.14
        assert params[0]["brier_score"] == pytest.approx(0.14)
        assert params[0]["calibration_correct"] is True
        assert params[0]["scored_at"] == NOW
        assert params[0]["run_id"] == "orch-20260901T000000Z-BTC/USDT"
        assert conn.commits == 1
        selects = [
            p
            for (s, p) in conn.executed
            if s is shadow_integration.SELECT_DUE_SHADOW_DECISIONS
        ]
        assert selects == [{"cutoff": NOW - timedelta(seconds=900), "limit": 50}]

    def test_skips_when_provider_returns_none(self) -> None:
        conn = FakeConnection()
        conn.queued_results = [[_due_row()]]
        provider = StubProvider({})

        assert _score(conn, provider) == 0
        assert _update_params(conn) == []
        assert conn.commits == 0

    def test_skips_when_base_close_not_positive(self) -> None:
        conn = FakeConnection()
        conn.queued_results = [[_due_row(base_close=0.0)]]
        provider = StubProvider({BTC: make_ohlcv(1, start_price=110.0)})

        assert _score(conn, provider) == 0
        assert conn.commits == 0

    def test_no_due_rows_returns_zero(self) -> None:
        # Kein queued Result → execute liefert None → keine Zeilen
        conn = FakeConnection()
        provider = StubProvider({})

        assert _score(conn, provider) == 0
        assert conn.commits == 0
        assert len(conn.executed) == 1  # nur der SELECT

    def test_range_threshold_maps_to_range(self) -> None:
        conn = FakeConnection()
        conn.queued_results = [
            [
                _due_row(
                    probabilities={"up": 0.4, "down": 0.4, "range": 0.2},
                    confidence=0.4,
                )
            ]
        ]
        # Close 100.05 vs Base 100 → +0.05 % < Threshold 0.1 % → "range"
        provider = StubProvider({BTC: make_ohlcv(1, start_price=100.05)})

        assert _score(conn, provider) == 1

        params = _update_params(conn)
        assert params[0]["actual_direction"] == "range"
        # 0.4^2 + 0.4^2 + 0.8^2 = 0.96
        assert params[0]["brier_score"] == pytest.approx(0.96)
        # Argmax-Tie (up == down) löst auf "up" → ≠ "range"
        assert params[0]["calibration_correct"] is False

    def test_missing_confidence_sets_calibration_none(self) -> None:
        conn = FakeConnection()
        conn.queued_results = [[_due_row(confidence=None)]]
        provider = StubProvider({BTC: make_ohlcv(1, start_price=110.0)})

        assert _score(conn, provider) == 1
        assert _update_params(conn)[0]["calibration_correct"] is None

    def test_probabilities_as_json_string(self) -> None:
        conn = FakeConnection()
        conn.queued_results = [
            [_due_row(probabilities=json.dumps({"up": 0.7, "down": 0.2, "range": 0.1}))]
        ]
        provider = StubProvider({BTC: make_ohlcv(1, start_price=110.0)})

        assert _score(conn, provider) == 1
        assert _update_params(conn)[0]["brier_score"] == pytest.approx(0.14)

    def test_invalid_probabilities_are_skipped(self) -> None:
        conn = FakeConnection()
        conn.queued_results = [[_due_row(probabilities={"up": 0.5})]]
        provider = StubProvider({BTC: make_ohlcv(1, start_price=110.0)})

        assert _score(conn, provider) == 0
        assert conn.commits == 0

    def test_scores_multiple_rows(self) -> None:
        conn = FakeConnection()
        conn.queued_results = [
            [
                _due_row(),
                _due_row(
                    run_id="orch-20260901T000000Z-ETH/USDT",
                    instrument="ETH/USDT",
                    probabilities={"up": 0.1, "down": 0.8, "range": 0.1},
                ),
            ]
        ]
        # Close 90 vs Base 100 → -10 % → "down" für beide Zeilen
        provider = StubProvider(
            {BTC: make_ohlcv(1, start_price=90.0), "ETH/USDT": make_ohlcv(1, start_price=90.0)}
        )

        assert _score(conn, provider) == 2
        assert conn.commits == 2
        assert [p["actual_direction"] for p in _update_params(conn)] == ["down", "down"]
        # (0.1-0)^2 + (0.8-1)^2 + 0.1^2 = 0.06, Argmax "down" == "down"
        assert _update_params(conn)[1]["brier_score"] == pytest.approx(0.06)
        assert _update_params(conn)[1]["calibration_correct"] is True

    def test_limit_is_forwarded_to_select(self) -> None:
        conn = FakeConnection()
        provider = StubProvider({})

        _score(conn, provider, limit=7)

        selects = [
            p
            for (s, p) in conn.executed
            if s is shadow_integration.SELECT_DUE_SHADOW_DECISIONS
        ]
        assert selects[0]["limit"] == 7
