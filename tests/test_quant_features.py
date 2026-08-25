"""Tests für die Feature-Berechnung (Phase 2, P2-1).

Verifiziert die Indikatoren des ``FeatureEngine`` gegen von Hand berechnete
Referenzwerte (RSI, MACD, Bollinger, ATR, Volatilität, VWAP) sowie das
Aggregations- und Determinismus-Verhalten von ``compute``.
"""

from __future__ import annotations

import math
import statistics

import pytest

from trading_harness.quant.features import FeatureEngine


def _candle(
    high: float, low: float, close: float, volume: float = 100.0, open_: float | None = None
) -> dict:
    """Erzeugt eine OHLCV-Kerze im Ingestions-Format."""
    return {
        "open": open_ if open_ is not None else close,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }


# ----------------------------------------------------------------------
# RSI
# ----------------------------------------------------------------------


def test_rsi_all_gains_returns_100() -> None:
    engine = FeatureEngine(rsi_period=14)
    closes = [100.0 + i for i in range(20)]
    assert engine._rsi(closes) == 100.0


def test_rsi_all_losses_returns_0() -> None:
    engine = FeatureEngine(rsi_period=14)
    closes = [200.0 - i for i in range(20)]
    assert engine._rsi(closes) == 0.0


def test_rsi_insufficient_data_returns_none() -> None:
    engine = FeatureEngine(rsi_period=14)
    assert engine._rsi([100.0] * 14) is None  # benötigt 15 Werte


# ----------------------------------------------------------------------
# MACD
# ----------------------------------------------------------------------


def test_macd_returns_dict_with_three_keys() -> None:
    engine = FeatureEngine()
    closes = [100.0 + i * 0.5 + 10.0 * math.sin(i / 3.0) for i in range(40)]
    result = engine._macd(closes)
    assert isinstance(result, dict)
    assert set(result) == {"macd", "signal", "histogram"}
    for value in result.values():
        assert isinstance(value, float)
    assert result["histogram"] == pytest.approx(result["macd"] - result["signal"])


def test_macd_insufficient_data_returns_none() -> None:
    engine = FeatureEngine()  # benötigt 26 + 9 = 34 Werte
    assert engine._macd([100.0] * 33) is None


# ----------------------------------------------------------------------
# Bollinger
# ----------------------------------------------------------------------


def test_bollinger_constant_prices_bandwidth_zero() -> None:
    engine = FeatureEngine(bb_period=20)
    result = engine._bollinger([50.0] * 25)
    assert result is not None
    assert result["middle"] == 50.0
    assert result["upper"] == 50.0
    assert result["lower"] == 50.0
    assert result["std_dev"] == 0.0
    assert result["bandwidth"] == 0.0


def test_bollinger_insufficient_data_returns_none() -> None:
    engine = FeatureEngine(bb_period=20)
    assert engine._bollinger([50.0] * 19) is None


# ----------------------------------------------------------------------
# ATR
# ----------------------------------------------------------------------


def test_atr_known_values() -> None:
    # TRs: [10, 20, 10, 5, 5, ..., 5] (15 Kerzen → 15 TRs)
    # ATR0 = (10 + 20 + 10 + 11*5) / 14 = 95/14
    # ATR1 = (95/14 * 13 + 5) / 14 = 1305/196
    engine = FeatureEngine(atr_period=14)
    candles = [
        _candle(105.0, 95.0, 100.0, open_=100.0),   # TR 10 (H-L)
        _candle(110.0, 90.0, 105.0, open_=100.0),   # TR 20 (H-L)
        _candle(108.0, 98.0, 103.0, open_=105.0),   # TR 10 (H-L)
    ]
    candles += [_candle(106.0, 101.0, 103.0, open_=103.0) for _ in range(12)]  # je TR 5
    assert engine._atr(candles) == pytest.approx(1305 / 196)


def test_atr_insufficient_data_returns_none() -> None:
    engine = FeatureEngine(atr_period=14)
    candles = [_candle(101.0, 99.0, 100.0) for _ in range(14)]
    assert engine._atr(candles) is None  # benötigt 15 Kerzen


# ----------------------------------------------------------------------
# Volatility
# ----------------------------------------------------------------------


def test_volatility_known_values() -> None:
    engine = FeatureEngine(vol_period=3)
    closes = [100.0, 110.0, 121.0, 100.0, 120.0]
    # Log-Renditen: [ln 1.1, ln 1.1, ln(100/121), ln 1.2]; letztes 3er-Fenster:
    expected = statistics.pstdev([math.log(1.1), math.log(100.0 / 121.0), math.log(1.2)])
    assert engine._volatility(closes) == pytest.approx(expected)

    # Konstante Kurse → null Volatilität
    default_engine = FeatureEngine(vol_period=20)
    assert default_engine._volatility([7.0] * 25) == 0.0


# ----------------------------------------------------------------------
# VWAP
# ----------------------------------------------------------------------


def test_vwap_known_values() -> None:
    engine = FeatureEngine()
    candles = [
        _candle(110.0, 90.0, 100.0, volume=10.0),   # tp 100 → 1000
        _candle(120.0, 100.0, 110.0, volume=30.0),  # tp 110 → 3300
        _candle(130.0, 110.0, 120.0, volume=60.0),  # tp 120 → 7200
    ]
    assert engine._vwap(candles) == pytest.approx((1000.0 + 3300.0 + 7200.0) / 100.0)
    assert engine._vwap([]) is None


# ----------------------------------------------------------------------
# compute() — Aggregation
# ----------------------------------------------------------------------


def test_compute_empty_candles() -> None:
    result = FeatureEngine().compute([])
    for key in ("rsi", "macd", "bollinger", "atr", "volatility", "vwap"):
        assert result[key] is None
    assert result["feature_count"] == 0
    assert result["computation_time_ms"] >= 0


def test_compute_30_candles_all_features_present() -> None:
    engine = FeatureEngine(
        rsi_period=3,
        macd_fast=2,
        macd_slow=4,
        macd_signal=3,
        bb_period=5,
        atr_period=3,
        vol_period=5,
    )
    candles = []
    for i in range(30):
        base = 100.0 + i
        candles.append(_candle(base + 2.0, base - 2.0, base + 1.0, volume=100.0 + i, open_=base))
    result = engine.compute(candles)
    for key in ("rsi", "macd", "bollinger", "atr", "volatility", "vwap"):
        assert result[key] is not None, key
    assert result["feature_count"] == 6
    assert result["computation_time_ms"] >= 0


def test_compute_deterministic() -> None:
    engine = FeatureEngine()
    candles = []
    for i in range(40):
        base = 100.0 + i
        candles.append(_candle(base + 2.0, base - 2.0, base + 1.0, volume=100.0 + i, open_=base))
    first = engine.compute(candles)
    second = engine.compute(candles)
    first.pop("computation_time_ms")
    second.pop("computation_time_ms")
    assert first == second
