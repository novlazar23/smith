"""Tests für die InfluxDB Schema-Definitionen der Quant-Plattform."""

from __future__ import annotations

import re

from trading_harness.quant.schema import (
    ANOMALY_FIELDS,
    ANOMALY_MEASUREMENT,
    ANOMALY_TAGS,
    DERIVATIVES_FIELDS,
    DERIVATIVES_MEASUREMENT,
    DERIVATIVES_TAGS,
    FEATURE_MEASUREMENT,
    FEATURE_META_FIELDS,
    FEATURE_TAGS,
    FEATURE_VERSION,
    OHLCV_FIELDS,
    OHLCV_MEASUREMENT,
    OHLCV_TAGS,
    ORDERBOOK_FIELDS,
    ORDERBOOK_MEASUREMENT,
    ORDERBOOK_TAGS,
    REGIME_FIELDS,
    REGIME_MEASUREMENT,
    REGIME_TAGS,
    SUPPORTED_TIMEFRAMES,
    TRADES_FIELDS,
    TRADES_MEASUREMENT,
    TRADES_TAGS,
    get_measurement_info,
    validate_fields,
    validate_measurement_name,
    validate_tags,
)


def test_ohlcv_constants() -> None:
    assert OHLCV_MEASUREMENT == "ohlcv"
    assert OHLCV_TAGS == ("symbol", "exchange", "timeframe")
    assert OHLCV_FIELDS == ("open", "high", "low", "close", "volume")


def test_trades_constants() -> None:
    assert TRADES_MEASUREMENT == "trades"
    assert TRADES_TAGS == ("symbol", "exchange")
    assert TRADES_FIELDS == ("price", "size", "side")


def test_orderbook_constants() -> None:
    assert ORDERBOOK_MEASUREMENT == "orderbook"
    assert ORDERBOOK_TAGS == ("symbol", "exchange")
    assert ORDERBOOK_FIELDS == (
        "best_bid",
        "best_ask",
        "spread",
        "bid_depth",
        "ask_depth",
        "imbalance",
    )


def test_derivatives_constants() -> None:
    assert DERIVATIVES_MEASUREMENT == "derivatives"
    assert DERIVATIVES_TAGS == ("symbol", "exchange")
    assert DERIVATIVES_FIELDS == (
        "funding_rate",
        "open_interest",
        "open_interest_change",
        "liquidations_long",
        "liquidations_short",
        "basis",
    )


def test_anomaly_constants() -> None:
    assert ANOMALY_MEASUREMENT == "anomalies"
    assert ANOMALY_TAGS == ("symbol", "exchange", "anomaly_type")
    assert ANOMALY_FIELDS == ("anomaly_score", "severity", "feature")


def test_supported_timeframes() -> None:
    assert len(SUPPORTED_TIMEFRAMES) == 6
    assert SUPPORTED_TIMEFRAMES == ("1m", "5m", "15m", "1h", "4h", "1d")
    assert len(set(SUPPORTED_TIMEFRAMES)) == 6


def test_validate_measurement_name_valid() -> None:
    assert validate_measurement_name("ohlcv") is True
    assert validate_measurement_name("my_measurement") is True
    assert validate_measurement_name("my.measurement") is True
    assert validate_measurement_name("ohlcv123") is True


def test_validate_measurement_name_invalid() -> None:
    assert validate_measurement_name("") is False
    assert validate_measurement_name("has space") is False
    assert validate_measurement_name("has-dash") is False
    assert validate_measurement_name("has#hash") is False
    assert validate_measurement_name("has,comma") is False


def test_validate_tags_missing() -> None:
    missing = validate_tags({"symbol": "BTC-USD"}, ("symbol", "exchange", "timeframe"))
    assert missing == ["exchange", "timeframe"]


def test_validate_tags_complete() -> None:
    assert validate_tags(
        {"symbol": "BTC-USD", "exchange": "binance", "timeframe": "1h"},
        ("symbol", "exchange", "timeframe"),
    ) == []


def test_validate_fields_missing() -> None:
    missing = validate_fields(
        {"open": 1.0, "close": 2.0},
        ("open", "high", "low", "close", "volume"),
    )
    assert missing == ["high", "low", "volume"]


def test_get_measurement_info() -> None:
    info = get_measurement_info(OHLCV_MEASUREMENT)
    assert info is not None
    assert info["tags"] == OHLCV_TAGS
    assert info["fields"] == OHLCV_FIELDS

    derivatives = get_measurement_info(DERIVATIVES_MEASUREMENT)
    assert derivatives is not None
    assert derivatives["tags"] == DERIVATIVES_TAGS
    assert derivatives["fields"] == DERIVATIVES_FIELDS

    features = get_measurement_info(FEATURE_MEASUREMENT)
    assert features is not None
    assert features["tags"] == FEATURE_TAGS
    assert features["fields"] == FEATURE_META_FIELDS

    regime = get_measurement_info(REGIME_MEASUREMENT)
    assert regime is not None
    assert regime["tags"] == REGIME_TAGS
    assert regime["fields"] == REGIME_FIELDS


def test_get_measurement_info_unknown() -> None:
    assert get_measurement_info("does_not_exist") is None
    assert get_measurement_info("") is None


def test_feature_version_semver() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", FEATURE_VERSION)
