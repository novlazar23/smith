"""InfluxDB Schema-Definitionen für die Quant-Plattform.

Measurements, Tags und Fields für OHLCV, Trades, Orderbook und Derivate.
Features versionieren (FEATURE_VERSION), damit Änderungen reproduzierbar bleiben.
"""

from __future__ import annotations

# Schema-Version — bei Änderungen inkrementieren
FEATURE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# OHLCV
# ---------------------------------------------------------------------------
OHLCV_MEASUREMENT: str = "ohlcv"
OHLCV_TAGS: tuple[str, ...] = ("symbol", "exchange", "timeframe")
OHLCV_FIELDS: tuple[str, ...] = ("open", "high", "low", "close", "volume")

# Timeframes in InfluxDB-Notation
SUPPORTED_TIMEFRAMES: tuple[str, ...] = ("1m", "5m", "15m", "1h", "4h", "1d")

# ---------------------------------------------------------------------------
# Trades (Tick-Daten)
# ---------------------------------------------------------------------------
TRADES_MEASUREMENT: str = "trades"
TRADES_TAGS: tuple[str, ...] = ("symbol", "exchange")
TRADES_FIELDS: tuple[str, ...] = ("price", "size", "side")

# ---------------------------------------------------------------------------
# Orderbook
# ---------------------------------------------------------------------------
ORDERBOOK_MEASUREMENT: str = "orderbook"
ORDERBOOK_TAGS: tuple[str, ...] = ("symbol", "exchange")
ORDERBOOK_FIELDS: tuple[str, ...] = (
    "best_bid", "best_ask", "spread",
    "bid_depth", "ask_depth", "imbalance",
)

# ---------------------------------------------------------------------------
# Derivate (Crypto)
# ---------------------------------------------------------------------------
DERIVATIVES_MEASUREMENT: str = "derivatives"
DERIVATIVES_TAGS: tuple[str, ...] = ("symbol", "exchange")
DERIVATIVES_FIELDS: tuple[str, ...] = (
    "funding_rate", "open_interest", "open_interest_change",
    "liquidations_long", "liquidations_short", "basis",
)

# ---------------------------------------------------------------------------
# Anomalien (Phase 3)
# ---------------------------------------------------------------------------
ANOMALY_MEASUREMENT: str = "anomalies"
ANOMALY_TAGS: tuple[str, ...] = ("symbol", "exchange", "anomaly_type")
ANOMALY_FIELDS: tuple[str, ...] = ("anomaly_score", "severity", "feature")

# ---------------------------------------------------------------------------
# Features (Phase 2)
# ---------------------------------------------------------------------------
FEATURE_MEASUREMENT: str = "features"
FEATURE_TAGS: tuple[str, ...] = ("symbol", "exchange", "feature_version")
# Features sind dynamisch — hier nur Metadaten
FEATURE_META_FIELDS: tuple[str, ...] = ("feature_count", "computation_time_ms")

# ---------------------------------------------------------------------------
# Regime (Phase 4)
# ---------------------------------------------------------------------------
REGIME_MEASUREMENT: str = "regime"
REGIME_TAGS: tuple[str, ...] = ("symbol", "exchange")
REGIME_FIELDS: tuple[str, ...] = ("regime_name", "regime_confidence", "regime_duration")

# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def validate_measurement_name(name: str) -> bool:
    """Prüft ob ein Measurement-Name InfluxDB-konform ist."""
    return bool(name) and name.replace("_", "").replace(".", "").isalnum()


def validate_tags(tags: dict[str, str], expected: tuple[str, ...]) -> list[str]:
    """Prüft ob alle erwarteten Tags vorhanden sind. Gibt fehlende zurück."""
    return [t for t in expected if t not in tags]


def validate_fields(fields: dict[str, float | int | str | bool], expected: tuple[str, ...]) -> list[str]:
    """Prüft ob alle erwarteten Fields vorhanden sind. Gibt fehlende zurück."""
    return [f for f in expected if f not in fields]


def get_measurement_info(measurement: str) -> dict[str, tuple[str, ...]] | None:
    """Gibt Tags und Fields für ein Measurement zurück."""
    _MAP = {
        OHLCV_MEASUREMENT: (OHLCV_TAGS, OHLCV_FIELDS),
        TRADES_MEASUREMENT: (TRADES_TAGS, TRADES_FIELDS),
        ORDERBOOK_MEASUREMENT: (ORDERBOOK_TAGS, ORDERBOOK_FIELDS),
        DERIVATIVES_MEASUREMENT: (DERIVATIVES_TAGS, DERIVATIVES_FIELDS),
        ANOMALY_MEASUREMENT: (ANOMALY_TAGS, ANOMALY_FIELDS),
        FEATURE_MEASUREMENT: (FEATURE_TAGS, FEATURE_META_FIELDS),
        REGIME_MEASUREMENT: (REGIME_TAGS, REGIME_FIELDS),
    }
    result = _MAP.get(measurement)
    if result is None:
        return None
    return {"tags": result[0], "fields": result[1]}
