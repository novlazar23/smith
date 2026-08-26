"""ML Feature Vector Builder (Phase 7).

Kombiniert Features aus allen Quant-Modulen zu einheitlichem Vektor
mit Normalisierung und NaN-Handling.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, ClassVar


@dataclass
class MLFeatures:
    """Einheitlicher ML-Feature-Vektor für ein Symbol."""
    symbol: str
    timeframe: str
    features: dict[str, float]
    feature_names: list[str]
    timestamp: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class MLFeatureBuilder:
    """Baut einheitliche ML-Feature-Vectoren aus Quant-Modul-Ausgaben."""

    # Gewichtung der verschiedenen Feature-Gruppen
    GROUP_WEIGHTS: ClassVar[dict[str, float]] = {
        "price": 1.0,
        "volume": 0.8,
        "momentum": 1.2,
        "volatility": 1.0,
        "regime": 0.9,
        "anomaly": 0.7,
        "similarity": 0.6,
    }

    def __init__(self, normalize: bool = True, fill_nan: float = 0.0) -> None:
        self.normalize = normalize
        self.fill_nan = fill_nan

    def build(
        self,
        symbol: str,
        timeframe: str,
        features: dict[str, float],
        metadata: dict[str, Any] | None = None,
    ) -> MLFeatures:
        """Baut einen normalisierten ML-Feature-Vektor.

        Args:
            symbol: Symbol-Name
            timeframe: Timeframe
            features: Rohe Features (Key → Wert)
            metadata: Optionale Metadaten

        Returns:
            MLFeatures mit normalisierten Features
        """
        cleaned = self._clean_features(features)
        if self.normalize:
            cleaned = self._normalize(cleaned)
        return MLFeatures(
            symbol=symbol,
            timeframe=timeframe,
            features=cleaned,
            feature_names=sorted(cleaned.keys()),
            metadata=metadata or {},
        )

    def build_from_components(
        self,
        symbol: str,
        timeframe: str,
        price_features: dict[str, float] | None = None,
        volume_features: dict[str, float] | None = None,
        momentum_features: dict[str, float] | None = None,
        volatility_features: dict[str, float] | None = None,
        regime_features: dict[str, float] | None = None,
        anomaly_features: dict[str, float] | None = None,
        similarity_features: dict[str, float] | None = None,
    ) -> MLFeatures:
        """Baut Features aus separaten Modul-Ausgaben."""
        all_features: dict[str, float] = {}
        components = {
            "price": price_features or {},
            "volume": volume_features or {},
            "momentum": momentum_features or {},
            "volatility": volatility_features or {},
            "regime": regime_features or {},
            "anomaly": anomaly_features or {},
            "similarity": similarity_features or {},
        }
        for group, feats in components.items():
            weight = self.GROUP_WEIGHTS.get(group, 1.0)
            for k, v in feats.items():
                all_features[f"{group}_{k}"] = v * weight
        return self.build(symbol, timeframe, all_features)

    def _clean_features(self, features: dict[str, float]) -> dict[str, float]:
        """Entfernt NaN/Inf, füllt fehlende Werte."""
        cleaned: dict[str, float] = {}
        for k, v in features.items():
            if math.isnan(v) or math.isinf(v):
                cleaned[k] = self.fill_nan
            else:
                cleaned[k] = v
        return cleaned

    def _normalize(self, features: dict[str, float]) -> dict[str, float]:
        """Z-Score Normalisierung."""
        if not features:
            return features
        values = list(features.values())
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = math.sqrt(variance) if variance > 0 else 1.0
        return {k: (v - mean) / std for k, v in features.items()}

    def feature_names(self, features: dict[str, float]) -> list[str]:
        """Sortierte Feature-Namen."""
        return sorted(features.keys())

    def feature_vector(self, features: dict[str, float]) -> list[float]:
        """Geordneter Feature-Vektor."""
        return [features[k] for k in sorted(features.keys())]

    def merge(self, *feature_dicts: dict[str, float]) -> dict[str, float]:
        """Mehere Feature-Dicts zusammenführen (letzter gewinnt)."""
        result: dict[str, float] = {}
        for d in feature_dicts:
            result.update(d)
        return result
