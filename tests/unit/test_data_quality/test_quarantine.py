"""Tests für Quarantine Manager."""

from __future__ import annotations

from packages.domain.data_quality.quarantine import (
    QuarantineManager,
)


class TestQuarantineManager:
    def test_high_quality_passed(self) -> None:
        mgr = QuarantineManager()
        event = {"type": "candle", "instrument": "BTC/USDT", "venue": "BINANCE"}
        result = mgr.evaluate_and_quarantine(event, quality_score=0.95, issues=[])
        assert result.is_quarantined is False

    def test_low_quality_quarantined(self) -> None:
        mgr = QuarantineManager()
        event = {"type": "candle", "instrument": "BTC/USDT", "venue": "BINANCE"}
        result = mgr.evaluate_and_quarantine(
            event, quality_score=0.3,
            issues=[{"field": "price", "message": "bad", "severity": "error"}],
        )
        assert result.is_quarantined is True
        assert result.entry is not None
        assert result.entry.severity == "critical"

    def test_critical_quarantined(self) -> None:
        mgr = QuarantineManager()
        event = {"type": "trade", "instrument": "X", "venue": "Y"}
        result = mgr.evaluate_and_quarantine(
            event, quality_score=0.7,  # between thresholds
            issues=[{"field": "price", "message": "zero", "severity": "critical"}],
        )
        assert result.is_quarantined is True

    def test_degraded_passed(self) -> None:
        mgr = QuarantineManager()
        event = {"type": "candle", "instrument": "BTC/USDT", "venue": "BINANCE"}
        result = mgr.evaluate_and_quarantine(
            event, quality_score=0.7,
            issues=[{"field": "volume", "message": "low", "severity": "warning"}],
        )
        assert result.is_quarantined is False

    def test_release(self) -> None:
        mgr = QuarantineManager()
        event = {"type": "candle", "instrument": "X", "venue": "Y"}
        mgr.evaluate_and_quarantine(event, 0.3, [])
        entries = mgr.get_quarantined_events()
        assert len(entries) == 1

        released = mgr.release(entries[0].event_hash)
        assert released is True
        assert len(mgr.get_quarantined_events()) == 0

    def test_release_nonexistent(self) -> None:
        mgr = QuarantineManager()
        assert mgr.release("nonexistent") is False

    def test_discard(self) -> None:
        mgr = QuarantineManager()
        event = {"type": "trade", "instrument": "X", "venue": "Y"}
        mgr.evaluate_and_quarantine(event, 0.3, [])
        entries = mgr.get_quarantined_events()
        discarded = mgr.discard(entries[0].event_hash)
        assert discarded is True

    def test_get_stats(self) -> None:
        mgr = QuarantineManager()
        assert mgr.get_stats() == {
            "total_quarantined": 0,
            "currently_quarantined": 0,
            "released": 0,
            "discarded": 0,
        }

        event = {"type": "candle", "instrument": "X", "venue": "Y"}
        mgr.evaluate_and_quarantine(event, 0.3, [])
        stats = mgr.get_stats()
        assert stats["total_quarantined"] == 1
        assert stats["currently_quarantined"] == 1

    def test_export_import(self) -> None:
        mgr = QuarantineManager()
        event1 = {"type": "candle", "instrument": "X", "venue": "Y"}
        event2 = {"type": "trade", "instrument": "Z", "venue": "W"}
        mgr.evaluate_and_quarantine(event1, 0.2, [])
        mgr.evaluate_and_quarantine(event2, 0.4, [])

        exported = mgr.export_quarantine()
        assert len(exported) > 0

        mgr2 = QuarantineManager()
        count = mgr2.import_quarantine(exported)
        assert count == 2

    def test_filter_quarantined(self) -> None:
        mgr = QuarantineManager()
        mgr.evaluate_and_quarantine({"type": "candle", "instrument": "BTC", "venue": "BINANCE"}, 0.2, [])
        mgr.evaluate_and_quarantine({"type": "trade", "instrument": "ETH", "venue": "BINANCE"}, 0.3, [])

        btc_only = mgr.get_quarantined_events(instrument="BTC")
        assert len(btc_only) == 1
        assert btc_only[0].instrument == "BTC"

        trades_only = mgr.get_quarantined_events(event_type="trade")
        assert len(trades_only) == 1

    def test_clear(self) -> None:
        mgr = QuarantineManager()
        mgr.evaluate_and_quarantine({"type": "candle", "instrument": "X", "venue": "Y"}, 0.2, [])
        assert len(mgr.get_quarantined_events()) == 1
        mgr.clear()
        assert len(mgr.get_quarantined_events()) == 0

    def test_hash_consistency(self) -> None:
        mgr = QuarantineManager()
        event = {"type": "candle", "a": 1, "b": 2}
        h1 = mgr._hash_event(event)
        h2 = mgr._hash_event(event)
        assert h1 == h2
        assert len(h1) == 64  # SHA256 hex
