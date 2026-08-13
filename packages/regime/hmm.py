"""GMM-basierter Regime-Detektor.

Nutzt Gaussian Mixture Model (scikit-learn) zur probabilistischen
Regime-Klassifizierung in Bull, Bear und Choppy.

Da sklearn kein echtes HMM bietet, nutzen wir GaussianMixture
als pragmatischen Ersatz — die 3 Cluster repräsentieren die
3 Regime.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.mixture import GaussianMixture

from .base import BaseRegimeDetector, MarketRegime, RegimeResult


class HiddenMarkovModel(BaseRegimeDetector):
    """GMM-basierter Regime-Detektor.

    Nutzt GaussianMixture mit 3 Komponenten für Regime-Erkennung.
    Der Cluster mit der höchsten RSI wird als Bull, niedrigste als
    Bear und mittlere als Choppy interpretiert.
    """

    name = "hmm"

    def __init__(
        self,
        n_components: int = 3,
        n_init: int = 3,
        covariance_type: str = "full",
    ) -> None:
        self.n_components = n_components
        self.n_init = n_init
        self.covariance_type = covariance_type
        self._model: GaussianMixture | None = None

    def _build_features(self, data: dict[str, NDArray[np.float64]]) -> NDArray[np.float64]:
        """Berechnet Feature-Vektoren (RSI, SMA-Delta, TrueRange) aus Rohdaten."""
        close = data["close"]
        high = data.get("high", close)
        low = data.get("low", close)
        n = len(close)

        # RSI — compute relative strength index over 14 periods
        delta = np.diff(close)  # len = n-1
        gain = np.where(delta > 0, delta, 0.0)  # len = n-1
        loss = np.where(delta < 0, -delta, 0.0)  # len = n-1
        avg_gain = np.zeros(n)
        avg_loss = np.zeros(n)
        if n > 14:
            avg_gain[13] = float(np.mean(gain[:14]))
            avg_loss[13] = float(np.mean(loss[:14]))
            for i in range(14, n):
                avg_gain[i] = (avg_gain[i - 1] * 13 + gain[i - 1]) / 14
                avg_loss[i] = (avg_loss[i - 1] * 13 + loss[i - 1]) / 14
        rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
        rsi = 100.0 - (100.0 / (1.0 + rs))

        # SMA-Delta (20 vs 50)
        sma20 = np.convolve(close, np.ones(20) / 20, mode="same")
        sma50 = np.convolve(close, np.ones(50) / 50, mode="same")
        sma_delta = (sma20 - sma50) / np.where(sma50 == 0, 1e-10, sma50)

        # True Range
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1]),
            ),
        )
        tr = np.concatenate([[tr[0]], tr])
        true_range = np.convolve(tr, np.ones(14) / 14, mode="same")

        # Features normalisieren
        features = np.column_stack([rsi, sma_delta, true_range])
        mean = np.mean(features[~np.isnan(features).any(axis=1)], axis=0)
        std = np.std(features[~np.isnan(features).any(axis=1)], axis=0) + 1e-10
        features = (features - mean) / std
        features = np.nan_to_num(features, nan=0.0)
        return features

    def detect(self, data: dict[str, NDArray[np.float64]]) -> RegimeResult:
        """Erkennt Regime via Gaussian Mixture Model.

        Trainiert ein GMM online und weist jedem Zeitpunkt die
        wahrscheinlichste Komponente zu.
        """
        features = self._build_features(data)

        # GMM trainieren
        model = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_init=self.n_init,
        )
        model.fit(features)

        # Vorhersage für jeden Zeitpunkt
        assignments = model.predict(features)

        # Letzte 20 Zeitpunkte aggregieren
        window = min(20, len(assignments))
        last_assignments = assignments[-window:]

        # Häufigkeiten der Cluster zählen
        counts = np.bincount(last_assignments, minlength=self.n_components)
        probs = counts / window

        self._model = model

        # Cluster-RSI-Mittelwerte berechnen → Zuordnung Bull/Bear/Choppy
        cluster_rsi_means = []
        for c in range(self.n_components):
            mask = last_assignments == c
            if np.any(mask):
                rsi_vals = features[-window:][mask, 0]
                cluster_rsi_means.append(float(np.mean(rsi_vals)))
            else:
                cluster_rsi_means.append(0.0)

        sorted_indices = np.argsort(cluster_rsi_means)
        bull_cluster = sorted_indices[-1]
        bear_cluster = sorted_indices[0]
        choppy_cluster = sorted_indices[1]

        scores = {
            MarketRegime.BULL: float(probs[bull_cluster]),
            MarketRegime.BEAR: float(probs[bear_cluster]),
            MarketRegime.CHOPPY: float(probs[choppy_cluster]),
        }

        best = max(scores, key=scores.get)
        return RegimeResult(
            regime=best,
            confidence=scores[best],
            scores=scores,
            metadata={
                "n_components": int(self.n_components),
                "cluster_rsi_means": cluster_rsi_means,
                "cluster_probs": {
                    str(i): float(p) for i, p in enumerate(probs)
                },
            },
        )
