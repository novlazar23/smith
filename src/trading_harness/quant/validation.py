"""Input Validation (Phase 11).

Validierungs-Funktionen für Candle-Daten, Features und Konfiguration.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar


@dataclass
class ValidationResult:
    """Ergebnis einer Validierung."""
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class Validator:
    """Validiert Eingabedaten für die Quant-Plattform."""

    VALID_TIMEFRAMES: ClassVar[frozenset[str]] = frozenset(
        {"1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"}
    )
    MAX_CANDLE_GAP_SECONDS = 86400 * 7  # 7 days
    MAX_FEATURE_VALUE = 1e10
    MIN_FEATURE_VALUE = -1e10

    def validate_candle(self, candle: dict) -> ValidationResult:
        """Validiert eine einzelne Kerze."""
        errors: list[str] = []
        warnings: list[str] = []

        required = ["time", "open", "high", "low", "close", "volume"]
        for field_name in required:
            if field_name not in candle:
                errors.append(f"Missing required field: {field_name}")

        if errors:
            return ValidationResult(valid=False, errors=errors)

        # Price consistency: high >= max(open, close), low <= min(open, close)
        o, h, l, c = candle["open"], candle["high"], candle["low"], candle["close"]
        if h < max(o, c):
            errors.append(f"High ({h}) < max(open={o}, close={c})")
        if l > min(o, c):
            errors.append(f"Low ({l}) > min(open={o}, close={c})")

        # Non-negative volume
        if candle["volume"] < 0:
            errors.append(f"Negative volume: {candle['volume']}")

        # Positive prices
        for p_name in ["open", "high", "low", "close"]:
            if candle[p_name] <= 0:
                errors.append(f"Non-positive {p_name}: {candle[p_name]}")

        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)

    def validate_candles(self, candles: list[dict]) -> ValidationResult:
        """Validiert eine Kerzenreihe."""
        errors: list[str] = []
        warnings: list[str] = []

        if not candles:
            return ValidationResult(valid=True, warnings=["Empty candle list"])

        for i, candle in enumerate(candles):
            result = self.validate_candle(candle)
            if not result.valid:
                for e in result.errors:
                    errors.append(f"Candle {i}: {e}")

        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)

    def validate_feature(self, name: str, value: float) -> ValidationResult:
        """Validiert einen einzelnen Feature-Wert."""
        errors: list[str] = []

        if math.isnan(value):
            errors.append(f"Feature '{name}' is NaN")
        elif math.isinf(value):
            errors.append(f"Feature '{name}' is Inf")
        elif abs(value) > self.MAX_FEATURE_VALUE:
            errors.append(f"Feature '{name}' out of range: {value}")

        return ValidationResult(valid=len(errors) == 0, errors=errors)

    def validate_features(self, features: dict[str, float]) -> ValidationResult:
        """Validiert ein Feature-Dictionary."""
        errors: list[str] = []
        warnings: list[str] = []

        for name, value in features.items():
            result = self.validate_feature(name, value)
            if not result.valid:
                errors.extend(result.errors)

        if not features:
            warnings.append("Empty features dict")

        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)

    def validate_symbol(self, symbol: str) -> ValidationResult:
        """Validiert einen Symbol-Namen."""
        errors: list[str] = []

        if not symbol:
            errors.append("Empty symbol")
        elif len(symbol) > 20:
            errors.append(f"Symbol too long: {len(symbol)} chars")
        elif not symbol.replace("/", "").replace("-", "").replace("_", "").isalnum():
            errors.append(f"Invalid characters in symbol: {symbol}")

        return ValidationResult(valid=len(errors) == 0, errors=errors)

    def validate_timeframe(self, timeframe: str) -> ValidationResult:
        """Validiert einen Timeframe."""
        errors: list[str] = []

        if timeframe not in self.VALID_TIMEFRAMES:
            errors.append(f"Invalid timeframe: {timeframe}. Valid: {self.VALID_TIMEFRAMES}")

        return ValidationResult(valid=len(errors) == 0, errors=errors)

    def validate_ohlcv_batch(self, candles: list[dict], max_size: int = 10000) -> ValidationResult:
        """Validiert einen Batch von Kerzen."""
        errors: list[str] = []
        warnings: list[str] = []

        if len(candles) > max_size:
            errors.append(f"Batch too large: {len(candles)} > {max_size}")

        result = self.validate_candles(candles)
        if not result.valid:
            errors.extend(result.errors)

        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)
