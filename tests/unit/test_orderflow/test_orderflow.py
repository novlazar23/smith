"""Tests für Order Flow Analysis — Footprint, Absorption, Iceberg."""

from __future__ import annotations

import numpy as np
import pytest
from packages.orderflow import (
    AbsorptionDetector,
    FootprintAnalyzer,
    IcebergDetector,
    OrderFlowResult,
    OrderFlowSignal,
    Side,
)


@pytest.fixture
def sample_data() -> dict[str, np.ndarray]:
    """Erstellt synthetische Marktdaten für Tests."""
    np.random.seed(42)
    n = 50
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    volume = np.abs(np.random.randn(n) * 1000) + 500
    return {
        "open": close - np.random.randn(n) * 0.2,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }


class TestSide:
    """Testet Side-Enum."""

    def test_enum_values(self) -> None:
        assert Side.BID == "bid"
        assert Side.ASK == "ask"


class TestOrderFlowSignal:
    """Testet OrderFlowSignal-Enum."""

    def test_all_signal_values(self) -> None:
        assert OrderFlowSignal.AGGRESSIVE_BUY == "aggressive_buy"
        assert OrderFlowSignal.AGGRESSIVE_SELL == "aggressive_sell"
        assert OrderFlowSignal.ABSORPTION == "absorption"
        assert OrderFlowSignal.ICEBERG == "iceberg"
        assert OrderFlowSignal.IMBALANCE == "imbalance"
        assert OrderFlowSignal.NONE == "none"


class TestFootprintAnalyzer:
    """Testet FootprintAnalyzer."""

    def test_cumulative_delta_positive(self, sample_data: dict[str, np.ndarray]) -> None:
        """Uptrend sollte positives kumulatives Delta ergeben."""
        n = 60
        close = np.array([100.0 + i * 1.5 for i in range(n)])
        high = close + 2.0
        low = close - 1.0
        volume = np.ones(n) * 1000.0
        data = {
            "open": close - 0.5,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
        analyzer = FootprintAnalyzer()
        result = analyzer.analyze(data)
        assert result.cumulative_delta > 0

    def test_cumulative_delta_negative(self, sample_data: dict[str, np.ndarray]) -> None:
        """Downtrend sollte negatives kumulatives Delta ergeben."""
        n = 60
        close = np.array([100.0 - i * 1.5 for i in range(n)])
        high = close + 1.0
        low = close - 2.0
        volume = np.ones(n) * 1000.0
        data = {
            "open": close + 0.5,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
        analyzer = FootprintAnalyzer()
        result = analyzer.analyze(data)
        assert result.cumulative_delta < 0

    def test_imbalance_detected(self) -> None:
        """Klare Imbalance in einseitigen Daten sollte erkannt werden."""
        n = 30
        close = np.ones(n) * 100.0
        open_vals = np.ones(n) * 100.0
        # Setze extreme body: fast alles bullish mit viel body
        open_vals = np.full(n, 95.0)
        close = np.full(n, 105.0)
        high = np.full(n, 106.0)
        low = np.full(n, 94.0)
        volume = np.full(n, 1000.0)
        data = {
            "open": open_vals,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
        analyzer = FootprintAnalyzer()
        result = analyzer.analyze(data)
        assert OrderFlowSignal.IMBALANCE in result.signals

    def test_missing_keys_raises(self) -> None:
        """ValueError bei fehlenden Schlüsseln."""
        analyzer = FootprintAnalyzer()
        with pytest.raises(ValueError, match="Missing required data keys"):
            analyzer.analyze({"close": np.array([1.0])})

    def test_analyze_all_bars(self, sample_data: dict[str, np.ndarray]) -> None:
        """Alle Balken werden verarbeitet ohne Index-Fehler."""
        analyzer = FootprintAnalyzer()
        result = analyzer.analyze(sample_data)
        assert len(result.metadata["delta_per_bar"]) == len(sample_data["volume"])
    def test_zero_range_bar(self) -> None:
        """Bars mit Range=0 (Doji/Spinning Top) sollen kein NaN erzeugen."""
        n = 10
        open_vals = np.full(n, 100.0)
        close = np.full(n, 100.0)
        high = np.full(n, 100.0)
        low = np.full(n, 100.0)
        volume = np.ones(n) * 500.0
        data = {
            "open": open_vals,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
        analyzer = FootprintAnalyzer()
        result = analyzer.analyze(data)
        assert not np.any(np.isnan(result.metadata["delta_per_bar"]))
        assert result.cumulative_delta == 0.0

    def test_zero_volume_delta(self) -> None:
        """Volume=0 erzeugt Delta=0."""
        n = 10
        open_vals = np.full(n, 95.0)
        close = np.full(n, 105.0)
        high = np.full(n, 110.0)
        low = np.full(n, 90.0)
        volume = np.zeros(n)
        data = {
            "open": open_vals,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
        analyzer = FootprintAnalyzer()
        result = analyzer.analyze(data)
        assert result.cumulative_delta == 0.0


class TestAbsorptionDetector:
    """Testet AbsorptionDetector."""

    def test_detects_repeated_contact(self) -> None:
        """Wiederholte Kontakte am selben Level → Absorption."""
        n = 20
        # Alle Bars drücken nach unten, berühren aber den selben Tiefpunkt
        close = np.linspace(100.0, 95.0, n)
        high = np.full(n, 101.0)
        low = np.full(n, 96.0)  # Immer gleicher Low-Punkt
        volume = np.ones(n) * 2000.0
        open_vals = np.linspace(100.5, 96.5, n)
        data = {
            "open": open_vals,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
        detector = AbsorptionDetector(min_touches=3)
        result = detector.detect_absorption(data)
        assert OrderFlowSignal.ABSORPTION in result.signals

    def test_no_absorption(self) -> None:
        """Keine wiederholten Kontakte → kein Signal."""
        n = 20
        close = np.linspace(100.0, 80.0, n)
        high = close + 2.0
        low = close - 2.0  # Immer anderer Low-Punkt
        volume = np.ones(n) * 1000.0
        data = {
            "open": close - 0.5,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
        detector = AbsorptionDetector(min_touches=3)
        result = detector.detect_absorption(data)
        assert OrderFlowSignal.ABSORPTION not in result.signals

    def test_min_touches_filter(self) -> None:
        """Weniger Kontakte als min_touches → kein Signal."""
        n = 5
        close = np.linspace(100.0, 98.0, n)
        high = np.full(n, 101.0)
        low = np.full(n, 99.0)
        volume = np.ones(n) * 1000.0
        data = {
            "open": np.full(n, 100.5),
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
        detector = AbsorptionDetector(min_touches=10)
        result = detector.detect_absorption(data)
        assert OrderFlowSignal.ABSORPTION not in result.signals

    def test_contact_price_direction(self) -> None:
        """Kontakt-Preis folgt der Balken-Richtung."""
        n = 4
        opens = np.array([100.0, 100.0, 100.0, 100.0])
        highs = np.array([103.0, 97.0, 103.0, 97.0])
        lows = np.array([97.0, 103.0, 97.0, 103.0])
        closes = np.array([102.0, 98.0, 102.0, 98.0])
        volume = np.ones(n) * 1000.0
        data = {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volume,
        }
        detector = AbsorptionDetector()
        result = detector.detect_absorption(data)
        # Bei Bullish-Bars (0, 2) sollte low der Kontakt sein
        # Bei Bearish-Bars (1, 3) sollte high der Kontakt sein
        # Beide sollten am selben Level sein (97-103 Range)
        metadata = result.metadata
        assert "contact_levels" in metadata

    def test_missing_keys_raises(self) -> None:
        """ValueError bei fehlenden Schlüsseln."""
        detector = AbsorptionDetector()
        with pytest.raises(ValueError, match="Missing required data keys"):
            detector.detect_absorption({"close": np.array([1.0])})

    def test_absorption_with_no_open(self) -> None:
        """Absorption funktioniert auch ohne 'open' — geht von close <= open als bearish aus."""
        n = 10
        close = np.full(n, 95.0)
        high = np.full(n, 101.0)
        low = np.full(n, 96.0)
        volume = np.ones(n) * 2000.0
        data = {
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
        detector = AbsorptionDetector(min_touches=3)
        result = detector.detect_absorption(data)
        assert OrderFlowSignal.ABSORPTION in result.signals


class TestIcebergDetector:
    """Testet IcebergDetector."""

    def test_detects_accumulation(self) -> None:
        """Viele kleine Balken mit Kauf-Bias → Iceberg."""
        np.random.seed(123)
        n = 40
        # Viele sehr kleine Bullish-Bars (Akkumulation)
        opens = np.linspace(100.0, 100.5, n)
        closes = opens + 0.01  # Sehr kleines Body
        highs = closes + 0.005
        lows = opens - 0.005
        volume = np.full(n, 500.0)
        data = {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volume,
        }
        detector = IcebergDetector(min_subtle_threshold=5, lookback=20)
        result = detector.detect_iceberg(data)
        assert OrderFlowSignal.ICEBERG in result.signals

    def test_detects_distribution(self) -> None:
        """Viele kleine Balken mit Verkauf-Bias → Iceberg."""
        np.random.seed(124)
        n = 40
        opens = np.linspace(100.5, 100.0, n)
        closes = opens - 0.01  # Sehr kleines Body nach unten
        highs = opens + 0.005
        lows = closes - 0.005
        volume = np.full(n, 500.0)
        data = {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volume,
        }
        detector = IcebergDetector(min_subtle_threshold=5, lookback=20)
        result = detector.detect_iceberg(data)
        assert OrderFlowSignal.ICEBERG in result.signals

    def test_no_pattern(self) -> None:
        """Gemischte Balken → kein Signal."""
        np.random.seed(456)
        n = 40
        rng = np.random.RandomState(456)
        opens = 100.0 + rng.randn(n) * 2.0
        closes = opens + rng.randn(n) * 3.0  # Große, ungerichtete Schwankungen
        highs = np.maximum(opens, closes) + 1.0
        lows = np.minimum(opens, closes) - 1.0
        volume = np.abs(rng.randn(n) * 1000) + 100
        data = {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volume,
        }
        detector = IcebergDetector(min_subtle_threshold=5, lookback=20)
        result = detector.detect_iceberg(data)
        assert OrderFlowSignal.ICEBERG not in result.signals

    def test_insufficient_data_raises(self) -> None:
        """Zu wenige Daten → ValueError."""
        detector = IcebergDetector(lookback=20)
        n = 10
        data = {
            "open": np.ones(n),
            "high": np.ones(n) * 1.01,
            "low": np.ones(n) * 0.99,
            "close": np.ones(n),
            "volume": np.ones(n),
        }
        with pytest.raises(ValueError, match="Insufficient data"):
            detector.detect_iceberg(data)


class TestOrderFlowResult:
    """Testet OrderFlowResult."""

    def test_defaults(self) -> None:
        """Standardwerte sind sinnvoll."""
        result = OrderFlowResult()
        assert result.signals == []
        assert result.scores == {}
        assert result.metadata == {}
        assert result.cumulative_delta == 0.0
