"""Tests für Market Data Validator & Gap Detector."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from packages.domain.data_quality.validator import (
    GapDetector,
    MarketDataValidator,
    Severity,
    ValidationResult,
)


class TestMarketDataValidator:
    def _make_candle(self, **overrides: float) -> dict:
        base = {
            "type": "candle",
            "instrument": "BTC/USDT",
            "venue": "BINANCE",
            "open": 100.0,
            "high": 105.0,
            "low": 98.0,
            "close": 102.0,
            "volume": 50.0,
            "open_time": datetime.now(),
            "close_time": datetime.now(),
        }
        base.update(overrides)
        return base

    def _make_trade(self, **overrides: float) -> dict:
        base = {
            "type": "trade",
            "trade_id": "t1",
            "instrument": "BTC/USDT",
            "venue": "BINANCE",
            "price": 100.0,
            "quantity": 1.0,
        }
        base.update(overrides)
        return base

    def _make_orderbook(self, **overrides: Any) -> dict:
        base = {
            "type": "orderbook_snapshot",
            "instrument": "BTC/USDT",
            "venue": "BINANCE",
            "sequence": 100,
            "bids": [[99.0, 1.0]],
            "asks": [[101.0, 1.0]],
        }
        base.update(overrides)
        return base

    def valid_candle(self) -> dict:
        return self._make_candle()

    def test_valid_candle(self) -> None:
        result = MarketDataValidator().validate(self.valid_candle())
        assert result.is_valid
        assert result.quality_score == 1.0
        assert len(result.issues) == 0

    def test_inverted_high_low(self) -> None:
        result = MarketDataValidator().validate(
            self._make_candle(high=95.0, low=100.0)
        )
        assert not result.is_valid
        assert result.quality_score < 1.0
        assert any(
            i.severity == Severity.CRITICAL and "high_low" in i.field
            for i in result.issues
        )

    def test_invalid_volume(self) -> None:
        result = MarketDataValidator().validate(
            self._make_candle(volume=-1.0)
        )
        assert not result.is_valid

    def test_valid_trade(self) -> None:
        result = MarketDataValidator().validate(self._make_trade())
        assert result.is_valid

    def test_invalid_trade(self) -> None:
        result = MarketDataValidator().validate(
            self._make_trade(price=0.0, quantity=-1.0)
        )
        assert not result.is_valid

    def test_valid_orderbook(self) -> None:
        result = MarketDataValidator().validate(self._make_orderbook())
        assert result.is_valid

    def test_crossing_orderbook(self) -> None:
        result = MarketDataValidator().validate(
            self._make_orderbook(bids=[[102.0, 1.0]], asks=[[101.0, 1.0]])
        )
        assert not result.is_valid
        assert any(
            i.severity == Severity.CRITICAL and "spread" in i.field
            for i in result.issues
        )

    def test_unknown_event_type(self) -> None:
        result = MarketDataValidator().validate({"type": "unknown"})
        assert not result.is_valid
        assert any(
            i.severity == Severity.WARNING and "event_type" in i.field
            for i in result.issues
        )

    def test_funding_rate(self) -> None:
        fr = {
            "type": "funding_rate",
            "instrument": "BTC-PERP",
            "venue": "BINANCE",
            "funding_rate": 0.0001,
            "mark_price": 50000.0,
        }
        result = MarketDataValidator().validate(fr)
        assert result.is_valid

    def test_invalid_funding_rate(self) -> None:
        fr = {
            "type": "funding_rate",
            "instrument": "BTC-PERP",
            "venue": "BINANCE",
            "funding_rate": 5.0,  # outside [-1, 1]
            "mark_price": 50000.0,
        }
        result = MarketDataValidator().validate(fr)
        assert not result.is_valid

    def test_liquidation(self) -> None:
        liq = {
            "type": "liquidation",
            "instrument": "BTC-PERP",
            "venue": "BINANCE",
            "quantity": 1.0,
            "price": 40000.0,
        }
        result = MarketDataValidator().validate(liq)
        assert result.is_valid

    def test_batch_validate(self) -> None:
        events = [
            self._make_candle(),
            self._make_trade(),
            self._make_candle(high=90.0, low=100.0),
        ]
        results = MarketDataValidator().batch_validate(events)
        assert len(results) == 3
        assert results[0].is_valid
        assert results[1].is_valid
        assert not results[2].is_valid

    def test_classify_quality_pass(self) -> None:
        validator = MarketDataValidator()
        result = ValidationResult(event_type="candle", is_valid=True, quality_score=0.95)
        assert validator.classify_quality(result) == "pass"

    def test_classify_quality_degraded(self) -> None:
        validator = MarketDataValidator()
        result = ValidationResult(event_type="candle", is_valid=True, quality_score=0.7)
        assert validator.classify_quality(result) == "degraded"

    def test_classify_quality_quarantine(self) -> None:
        validator = MarketDataValidator()
        result = ValidationResult(event_type="candle", is_valid=False, quality_score=0.3)
        assert validator.classify_quality(result) == "quarantine"


class TestGapDetector:
    def test_no_gap(self) -> None:
        detector = GapDetector()
        result = detector.check_gap(
            sequence=100,
            key_fields={"instrument": "BTC", "venue": "BINANCE", "event_type": "candle"},
        )
        assert result.is_valid

        result = detector.check_gap(
            sequence=101,
            key_fields={"instrument": "BTC", "venue": "BINANCE", "event_type": "candle"},
        )
        assert result.is_valid

    def test_gap_detected(self) -> None:
        detector = GapDetector()
        detector.check_gap(
            sequence=100,
            key_fields={"instrument": "X", "venue": "Y", "event_type": "candle"},
        )
        result = detector.check_gap(
            sequence=105,
            key_fields={"instrument": "X", "venue": "Y", "event_type": "candle"},
        )
        assert not result.is_valid
        assert any(
            i.severity == Severity.WARNING and "Gap" in i.message
            for i in result.issues
        )

    def test_sequence_regression(self) -> None:
        detector = GapDetector()
        detector.check_gap(
            sequence=100,
            key_fields={"instrument": "X", "venue": "Y", "event_type": "trade"},
        )
        result = detector.check_gap(
            sequence=90,
            key_fields={"instrument": "X", "venue": "Y", "event_type": "trade"},
        )
        assert not result.is_valid
        assert result.quality_score == 0.0
        assert any(
            i.severity == Severity.ERROR and "regression" in i.message.lower()
            for i in result.issues
        )

    def test_reset(self) -> None:
        detector = GapDetector()
        detector.check_gap(
            sequence=100,
            key_fields={"instrument": "X", "venue": "Y", "event_type": "candle"},
        )
        detector.reset("X:Y:candle")
        result = detector.check_gap(
            sequence=50,
            key_fields={"instrument": "X", "venue": "Y", "event_type": "candle"},
        )
        assert result.is_valid  # Reset → first event, no gap
