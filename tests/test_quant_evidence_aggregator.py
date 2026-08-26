"""Tests für Evidence Aggregator."""
from __future__ import annotations

import pytest

from trading_harness.quant.evidence_aggregator import (
    AggregatedEvidence,
    EvidenceAggregator,
)


class TestEvidenceAggregator:
    def test_add_entry(self):
        agg = EvidenceAggregator()
        agg.add_entry("features", {"rsi": 65.0})
        assert agg.has_source("features")
        assert agg.entry_count == 1

    def test_aggregate_single_source(self):
        agg = EvidenceAggregator()
        agg.add_entry("features", {"rsi": 65.0})
        result = agg.aggregate("BTCUSDT", "1m")
        assert isinstance(result, AggregatedEvidence)
        assert result.symbol == "BTCUSDT"
        assert result.total_confidence > 0

    def test_aggregate_multiple_sources(self):
        agg = EvidenceAggregator()
        agg.add_entry("features", {"rsi": 65.0})
        agg.add_entry("regime", {"regime": "trending"})
        agg.add_entry("anomalies", {"count": 2})
        result = agg.aggregate("BTCUSDT", "1m")
        assert len(result.entries) == 3

    def test_summary_includes_regime(self):
        agg = EvidenceAggregator()
        agg.add_entry("regime", {"regime": "trending"})
        result = agg.aggregate("BTCUSDT", "1m")
        assert result.summary["current_regime"] == "trending"

    def test_summary_includes_anomaly_count(self):
        agg = EvidenceAggregator()
        agg.add_entry("anomalies", {"count": 5})
        result = agg.aggregate("BTCUSDT", "1m")
        assert result.summary["anomaly_count"] == 5

    def test_high_priority_count(self):
        agg = EvidenceAggregator()
        agg.add_entry("anomalies", {"count": 1})  # priority 10
        agg.add_entry("regime", {"regime": "x"})  # priority 8
        agg.add_entry("features", {"rsi": 65})  # priority 4
        result = agg.aggregate("BTCUSDT", "1m")
        assert result.high_priority_count == 2

    def test_empty_aggregate(self):
        agg = EvidenceAggregator()
        result = agg.aggregate("BTCUSDT", "1m")
        assert len(result.entries) == 0
        assert result.total_confidence == 0.0

    def test_clear(self):
        agg = EvidenceAggregator()
        agg.add_entry("features", {"rsi": 65.0})
        agg.clear()
        assert agg.entry_count == 0

    def test_get_entry(self):
        agg = EvidenceAggregator()
        agg.add_entry("features", {"rsi": 65.0})
        entry = agg.get_entry("features")
        assert entry is not None
        assert entry.data["rsi"] == 65.0

    def test_custom_confidence(self):
        agg = EvidenceAggregator()
        agg.add_entry("features", {"rsi": 65.0}, confidence=0.8)
        result = agg.aggregate("BTCUSDT", "1m")
        assert result.total_confidence == pytest.approx(0.8)

    def test_sources_property(self):
        agg = EvidenceAggregator()
        agg.add_entry("features", {})
        agg.add_entry("regime", {})
        assert set(agg.sources) == {"features", "regime"}

    def test_deterministic(self):
        agg = EvidenceAggregator()
        agg.add_entry("features", {"rsi": 65.0})
        r1 = agg.aggregate("BTCUSDT", "1m")
        r2 = agg.aggregate("BTCUSDT", "1m")
        assert r1.summary == r2.summary
