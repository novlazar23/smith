"""Tests für Input Validation."""
from __future__ import annotations

from trading_harness.quant.validation import Validator


class TestValidator:
    def test_valid_candle(self):
        v = Validator()
        candle = {"time": "2026-01-01T00:00:00Z", "open": 100, "high": 105,
                  "low": 95, "close": 102, "volume": 1000}
        result = v.validate_candle(candle)
        assert result.valid

    def test_invalid_candle_missing_field(self):
        v = Validator()
        candle = {"time": "2026-01-01T00:00:00Z", "open": 100}
        result = v.validate_candle(candle)
        assert not result.valid
        assert any("Missing" in e for e in result.errors)

    def test_invalid_candle_high_low(self):
        v = Validator()
        candle = {"time": "t", "open": 100, "high": 90, "low": 110,
                  "close": 100, "volume": 100}
        result = v.validate_candle(candle)
        assert not result.valid

    def test_negative_volume(self):
        v = Validator()
        candle = {"time": "t", "open": 100, "high": 100, "low": 100,
                  "close": 100, "volume": -1}
        result = v.validate_candle(candle)
        assert not result.valid

    def test_valid_features(self):
        v = Validator()
        result = v.validate_features({"rsi": 65.0, "macd": 1.2})
        assert result.valid

    def test_nan_feature(self):
        v = Validator()
        result = v.validate_features({"rsi": float("nan")})
        assert not result.valid

    def test_inf_feature(self):
        v = Validator()
        result = v.validate_features({"rsi": float("inf")})
        assert not result.valid

    def test_valid_symbol(self):
        v = Validator()
        assert v.validate_symbol("BTCUSDT").valid
        assert v.validate_symbol("ETH/USDT").valid

    def test_invalid_symbol_empty(self):
        v = Validator()
        assert not v.validate_symbol("").valid

    def test_valid_timeframe(self):
        v = Validator()
        assert v.validate_timeframe("1m").valid
        assert v.validate_timeframe("1h").valid
        assert v.validate_timeframe("1d").valid

    def test_invalid_timeframe(self):
        v = Validator()
        assert not v.validate_timeframe("2m").valid
        assert not v.validate_timeframe("invalid").valid

    def test_empty_candles_valid(self):
        v = Validator()
        result = v.validate_candles([])
        assert result.valid
        assert len(result.warnings) > 0

    def test_batch_size_limit(self):
        v = Validator()
        candles = [{"time": "t", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}] * 100
        result = v.validate_ohlcv_batch(candles, max_size=50)
        assert not result.valid

    def test_out_of_range_feature(self):
        v = Validator()
        result = v.validate_features({"extreme": 1e15})
        assert not result.valid
