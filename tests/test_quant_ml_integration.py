"""ML-Features-Integration (Quant-Plattform, Phase 7, P7-4).

End-to-End-Kette Feature-Building → Feature-Importance mit ausschließlich
Mocks: ``MLFeatureBuilder`` (P7-3) ← gemockte Quant-Modul-Ausgaben (``MagicMock``)
→ ``FeatureImportanceEngine`` (P7-2). Kein Netzwerk, keine echten Marktdaten,
kein Store — nur deterministische Synthetik-Zeitreihen.

Prüfte Integration:

1. Modul-Outputs → ``build_from_components`` (Gruppen-Gewichtung, saubere
   Feature-Namen) → Feature-Matrix über mehrere Ticks → ``compute`` liefert
   korrekt gerankte Importance mit dem erwarteten Top-Feature.
2. ``select_features`` respektiert ``max_features`` und die Importance-Reihenfolge.
3. ``feature_groups`` enthält pro Gruppe den Durchschnitt der Importance.
4. Leere Eingaben (kein Features, kein Target, keine Komponenten) → leere
   Resultate ohne Exceptions.
5. Z-Score-Normalisierung + NaN-Füllung des Builders bleiben konsistent
   (Mittelwert ≈ 0, NaN → ``fill_nan``, konstante Features → 0.0).
"""

from __future__ import annotations

import math
from statistics import correlation
from unittest.mock import MagicMock

import pytest

from trading_harness.quant.feature_importance import FeatureImportanceEngine
from trading_harness.quant.ml_features import MLFeatureBuilder, MLFeatures

SYMBOL = "BTCUSDT"
TIMEFRAME = "1h"
N_BARS = 10


# ----------------------------------------------------------------------
# Test-Helfer
# ----------------------------------------------------------------------


def make_module_output(name: str, **features: float) -> MagicMock:
    """Gemocktes Quant-Modul, das eine Feature-Gruppe als ``.features`` ausgibt."""
    module = MagicMock(name=name)
    module.features = dict(features)
    return module


def build_feature_matrix(rows: list[MLFeatures]) -> dict[str, list[float]]:
    """Wandelt eine Reihe von ``MLFeatures`` (einer pro Tick) in Feature-Zeitreihen."""
    names = rows[0].feature_names
    return {name: [row.features[name] for row in rows] for name in names}


# ----------------------------------------------------------------------
# 1. Builder → Importance: vollständige Pipeline
# ----------------------------------------------------------------------


def test_feature_builder_to_importance_pipeline() -> None:
    """Modul-Outputs → Builder (Gewichtung) → Feature-Matrix → Importance-Ranking.

    Momentum ist perfekt linear mit dem Target (|corr| = 1.0), Price und
    Volume alternieren (|corr| ≈ 0.17) → nur Momentum überschreitet den
    Threshold und wird Top-Feature.
    """
    builder = MLFeatureBuilder(normalize=False)
    engine = FeatureImportanceEngine(threshold=0.5)

    rows: list[MLFeatures] = []
    for t in range(N_BARS):
        momentum = make_module_output("momentum_module", rsi=30.0 + 4.0 * t)
        price = make_module_output("price_module", close=100.0 + 5.0 * (-1) ** t)
        volume = make_module_output("volume_module", avg=1000.0 + 100.0 * (t % 2))
        rows.append(
            builder.build_from_components(
                SYMBOL,
                TIMEFRAME,
                momentum_features=momentum.features,
                price_features=price.features,
                volume_features=volume.features,
            )
        )

    # Builder: Gruppen-Gewichtung (momentum ×1.2, price ×1.0, volume ×0.8)
    # und sortierte, gruppierte Feature-Namen.
    assert rows[0].symbol == SYMBOL
    assert rows[0].timeframe == TIMEFRAME
    assert rows[0].feature_names == ["momentum_rsi", "price_close", "volume_avg"]
    assert rows[0].features["momentum_rsi"] == pytest.approx(36.0)  # 30.0 × 1.2
    assert rows[0].features["price_close"] == pytest.approx(105.0)  # 105.0 × 1.0
    assert rows[0].features["volume_avg"] == pytest.approx(800.0)  # 1000.0 × 0.8
    # feature_vector ist konsistent mit den sortierten Namen.
    assert builder.feature_vector(rows[0].features) == [
        rows[0].features[name] for name in rows[0].feature_names
    ]

    # Importance: Momentum dominiert, die schwachen Features fallen raus.
    matrix = build_feature_matrix(rows)
    target = [float(t) for t in range(N_BARS)]
    result = engine.compute(matrix, target)

    assert result.top_features == ["momentum_rsi"]
    assert result.features[0].name == "momentum_rsi"
    assert result.features[0].importance == pytest.approx(1.0)
    assert result.features[0].correlation == pytest.approx(1.0)
    assert result.features[0].rank == 1
    for feature in result.features:
        assert feature.name in {"momentum_rsi", "price_close", "volume_avg"}
        assert 0.0 <= feature.importance <= 1.0
    assert set(result.feature_groups) == {"momentum", "price", "volume"}
    assert result.feature_groups["momentum"] == pytest.approx(1.0)
    assert result.feature_groups["price"] < 0.5
    assert result.feature_groups["volume"] < 0.5


# ----------------------------------------------------------------------
# 2. select_features wählt die Top-N
# ----------------------------------------------------------------------


def test_importance_selects_top_features() -> None:
    """``select_features`` liefert die |corr|-stärksten Features, begrenzt auf N."""
    target = [float(t) for t in range(N_BARS)]
    features = {
        "price_close": [float(t) for t in range(N_BARS)],  # corr = +1.0
        "momentum_rsi": [float(N_BARS - 1 - t) for t in range(N_BARS)],  # corr = -1.0
        "anomaly_score": [1.0, 0.0] * 5,  # alternierend, |corr| ≈ 0.17
        "volatility_atr": [2.0] * N_BARS,  # konstant → corr = 0.0
    }
    engine = FeatureImportanceEngine(threshold=0.1)

    top_two = engine.select_features(features, target, max_features=2)
    assert len(top_two) == 2
    assert set(top_two) == {"price_close", "momentum_rsi"}

    top_one = engine.select_features(features, target, max_features=1)
    assert len(top_one) == 1
    assert top_one[0] in {"price_close", "momentum_rsi"}

    # Alle ≥ Threshold (Konstante fällt bei 0.0 raus), Max-Cap nicht erreicht.
    top_all = engine.select_features(features, target, max_features=10)
    assert set(top_all) == {"price_close", "momentum_rsi", "anomaly_score"}
    # Reihenfolge: Importance absteigend, die perfekten Korrelationen zuerst.
    assert set(top_all[:2]) == {"price_close", "momentum_rsi"}


# ----------------------------------------------------------------------
# 3. Feature-Gruppen werden aggregiert
# ----------------------------------------------------------------------


def test_feature_groups_aggregated() -> None:
    """``feature_groups`` = Durchschnitt der Importance je Gruppen-Präfix."""
    target = [float(t) for t in range(N_BARS)]
    alternating = [1.0, 0.0] * 5
    features = {
        "price_close": [float(t) for t in range(N_BARS)],  # |corr| = 1.0
        "price_sma": [2.0 * t for t in range(N_BARS)],  # linear → |corr| = 1.0
        "momentum_rsi": alternating,  # schwache Korrelation
        "plain_constant": [3.0] * N_BARS,  # konstant → 0.0
        "flat": [7.0] * N_BARS,  # ohne Präfix → Gruppe "other", konstant → 0.0
    }
    engine = FeatureImportanceEngine(threshold=0.0)

    result = engine.compute(features, target)

    expected_momentum = abs(correlation(alternating, target))
    assert set(result.feature_groups) == {"price", "momentum", "plain", "other"}
    # Zwei Mitglieder (close, sma) mit je 1.0 → Durchschnitt 1.0.
    assert result.feature_groups["price"] == pytest.approx(1.0)
    assert result.feature_groups["momentum"] == pytest.approx(expected_momentum)
    assert result.feature_groups["plain"] == pytest.approx(0.0)
    assert result.feature_groups["other"] == pytest.approx(0.0)


# ----------------------------------------------------------------------
# 4. Leere Eingaben werden sauber behandelt
# ----------------------------------------------------------------------


def test_empty_features_handled() -> None:
    """Keine Features / kein Target / keine Komponenten → leere Resultate, keine Exception."""
    engine = FeatureImportanceEngine()
    builder = MLFeatureBuilder()

    # Engine: leere Features oder leeres Target → leeres Result.
    for features, target in [
        ({}, [1.0, 2.0]),
        ({"price_close": [1.0, 2.0]}, []),
        ({}, []),
        ({"price_close": []}, [1.0]),  # zu wenige Beobachtungen → verworfen
    ]:
        result = engine.compute(features, target)
        assert result.features == []
        assert result.top_features == []
        assert result.feature_groups == {}
        assert engine.select_features(features, target) == []

    # Builder: leere Rohe-Features bzw. keine Komponenten → leere Vektoren.
    empty = builder.build(SYMBOL, TIMEFRAME, {})
    assert empty.features == {}
    assert empty.feature_names == []
    assert empty.symbol == SYMBOL

    empty_components = builder.build_from_components(SYMBOL, TIMEFRAME)
    assert empty_components.features == {}
    assert empty_components.feature_names == []
    assert builder.feature_vector(empty_components.features) == []


# ----------------------------------------------------------------------
# 5. Normalisierung + NaN-Füllung bleiben deterministisch
# ----------------------------------------------------------------------


def test_builder_normalization_and_nan_fill() -> None:
    """Z-Score (Mittelwert ≈ 0), NaN → ``fill_nan``, konstante Features → 0.0."""
    builder = MLFeatureBuilder(normalize=True, fill_nan=-1.0)

    result = builder.build(SYMBOL, TIMEFRAME, {"a_x": 1.0, "a_y": 3.0})
    # Mean 2.0, std 1.0 → z-Scores -1.0 / +1.0.
    assert result.features["a_x"] == pytest.approx(-1.0)
    assert result.features["a_y"] == pytest.approx(1.0)
    assert sum(result.features.values()) == pytest.approx(0.0)

    nan_result = builder.build(SYMBOL, TIMEFRAME, {"a_x": float("nan"), "a_y": 4.0})
    # Nach Füllung: mean 1.5, std 2.5 → z-Scores -1.0 / +1.0.
    assert nan_result.features["a_x"] == pytest.approx(-1.0)  # NaN → -1.0 (fill_nan)
    assert nan_result.features["a_y"] == pytest.approx(1.0)
    assert math.isfinite(nan_result.features["a_x"])

    flat = builder.build(SYMBOL, TIMEFRAME, {"a_x": 5.0, "a_y": 5.0})
    assert flat.features == {"a_x": 0.0, "a_y": 0.0}  # std 0 → Division durch 1.0
