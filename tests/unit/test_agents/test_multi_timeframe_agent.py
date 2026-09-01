"""Tests for the multi-timeframe agent."""

from __future__ import annotations

import pytest
from packages.agents.multi_timeframe.agent import MultiTimeframeAgent
from packages.agents.multi_timeframe.models import (
    MultiTimeframeReport,
    TimeframeSignal,
)
from packages.consensus.base import VoteDirection


def _make_timeframe_data() -> dict[str, dict[str, float]]:
    return {
        "1m": {"up": 0.55, "down": 0.25, "range": 0.2},
        "5m": {"up": 0.6, "down": 0.2, "range": 0.2},
        "15m": {"up": 0.5, "down": 0.3, "range": 0.2},
        "1h": {"up": 0.65, "down": 0.2, "range": 0.15},
        "4h": {"up": 0.6, "down": 0.25, "range": 0.15},
        "1d": {"up": 0.55, "down": 0.3, "range": 0.15},
    }


class TestMultiTimeframeAgentBasic:
    """Tests for basic multi-timeframe functionality."""

    def test_analyzes_all_timeframes(self):
        agent = MultiTimeframeAgent()
        data = _make_timeframe_data()
        report, _ = agent.analyze(data)

        assert len(report.timeframe_signals) == 6
        assert all(isinstance(s, TimeframeSignal) for s in report.timeframe_signals)

    def test_detects_long_agreement(self):
        agent = MultiTimeframeAgent()
        data = _make_timeframe_data()
        report, _ = agent.analyze(data)

        assert report.direction == VoteDirection.LONG
        assert report.overall_agreement > 0.5

    def test_empty_data_raises(self):
        agent = MultiTimeframeAgent()
        with pytest.raises(ValueError, match="must not be empty"):
            agent.analyze({})

    def test_partial_timeframes(self):
        agent = MultiTimeframeAgent()
        data = {
            "1h": {"up": 0.7, "down": 0.2, "range": 0.1},
            "4h": {"up": 0.6, "down": 0.3, "range": 0.1},
        }
        report, _ = agent.analyze(data)

        assert len(report.timeframe_signals) == 2
        assert report.direction == VoteDirection.LONG


class TestTimeframeConflicts:
    """Tests for conflict detection."""

    def test_detects_long_short_conflict(self):
        agent = MultiTimeframeAgent()
        data = {
            "1m": {"up": 0.7, "down": 0.2, "range": 0.1},
            "1h": {"up": 0.2, "down": 0.7, "range": 0.1},
        }
        report, _ = agent.analyze(data)

        assert len(report.conflicts) > 0
        assert any("SHORT" in c and "LONG" in c for c in report.conflicts)

    def test_no_conflicts(self):
        agent = MultiTimeframeAgent()
        data = {
            "1m": {"up": 0.7, "down": 0.2, "range": 0.1},
            "1h": {"up": 0.7, "down": 0.2, "range": 0.1},
        }
        report, _ = agent.analyze(data)

        assert len(report.conflicts) == 0
        assert report.overall_agreement == 1.0


class TestMultiTimeframeReport:
    """Tests for report structure."""

    def test_report_has_all_fields(self):
        agent = MultiTimeframeAgent()
        data = _make_timeframe_data()
        report, _ = agent.analyze(data)

        assert isinstance(report, MultiTimeframeReport)
        assert isinstance(report.direction, str)
        assert 0.0 <= report.confidence <= 1.0
        assert isinstance(report.timeframe_signals, list)
        assert isinstance(report.conflicts, list)
        assert 0.0 <= report.overall_agreement <= 1.0

    def test_weighted_short_majority(self):
        agent = MultiTimeframeAgent()
        data = {
            "1h": {"up": 0.15, "down": 0.7, "range": 0.15},
            "4h": {"up": 0.2, "down": 0.65, "range": 0.15},
            "1d": {"up": 0.3, "down": 0.6, "range": 0.1},
        }
        report, _ = agent.analyze(data)

        assert report.direction == VoteDirection.SHORT
