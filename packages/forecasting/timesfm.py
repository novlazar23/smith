"""Forecast-Provider-Abstraktion für den TimesFM-Research-Spike.

Die Evolutions-Pipeline bekommt nur einen ``ForecastProvider`` injiziert;
Tests und Smoke-Läufe nutzen ``FakeProvider``. Der echte TimesFM-Adapter
ist optional und lädt ``timesfm`` erst zur Laufzeit.

Pinning:
- Package: ``timesfm[torch]==3.0.1``
- Modellgewichte: ``google/timesfm-2.5-200m-pytorch`` (Apache-2.0)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

DEFAULT_TIMESFM_MODEL = "google/timesfm-2.5-200m-pytorch"
_TIMESFM_MIN_CONTEXT = 32


class TimesFMUnavailableError(RuntimeError):
    """Echter TimesFM-Pfad ist in der aktuellen Phase nicht verfügbar."""


@dataclass(frozen=True)
class QuantileForecast:
    """Quantil-Prognosen für die nächsten ``horizon`` Bars (Preisskala)."""

    median: np.ndarray
    q10: np.ndarray
    q90: np.ndarray


class ForecastProvider(Protocol):
    """Minimaler Provider-Vertrag für den Spike (keine Default-Dependency)."""

    def predict(self, context: np.ndarray, horizon: int) -> QuantileForecast:
        """Prognostiziert ``horizon`` Schritte ab dem letzten Context-Kurs."""
        ...


def provider_label(provider: ForecastProvider) -> str:
    """Stabiles Provider-Label für Reports/Cache-Keys."""
    label = getattr(provider, "label", None)
    if label is None:
        return type(provider).__name__
    return str(label)


@dataclass(frozen=True)
class FakeProvider:
    """Deterministischer Fake-Provider (Tests/Smoke, kein echtes Modell)."""

    label: str = "fake"
    drift_per_bar: float = 0.0002
    spread_per_bar: float = 0.0005

    def predict(self, context: np.ndarray, horizon: int) -> QuantileForecast:
        if horizon < 1:
            raise ValueError("horizon muss >= 1 sein")
        if context.size < 2:
            raise ValueError("Context braucht mindestens zwei Kurse")
        base = float(context[-1])
        if base <= 0.0:
            raise ValueError("letzter Context-Kurs muss positiv sein")
        steps = np.arange(1, horizon + 1, dtype=np.float64)
        drift = self.drift_per_bar * steps
        spread = self.spread_per_bar * steps
        return QuantileForecast(
            median=base * (1.0 + drift),
            q10=base * (1.0 + drift - spread),
            q90=base * (1.0 + drift + spread),
        )


class TimesFMProvider:
    """Optionaler TimesFM-Adapter (Lazy Import, keine Default-Dependency)."""

    label: str

    def __init__(
        self,
        model: str | None = None,
        *,
        max_context: int = 1024,
        max_horizon: int = 288,
        cache_dir: str | Path | None = None,
    ) -> None:
        if max_context < _TIMESFM_MIN_CONTEXT:
            raise ValueError(f"max_context muss >= {_TIMESFM_MIN_CONTEXT} sein")
        if max_horizon < 1:
            raise ValueError("max_horizon muss >= 1 sein")
        self.model = model or DEFAULT_TIMESFM_MODEL
        self.max_context = max_context
        self.max_horizon = max_horizon
        self.cache_dir = str(cache_dir) if cache_dir is not None else None
        self.label = f"timesfm:{self.model}"
        self._model: object | None = None

    def _ensure_model(self) -> object:
        if self._model is not None:
            return self._model
        try:
            import timesfm
        except ImportError as exc:
            raise TimesFMUnavailableError(
                "timesfm ist nicht installiert. Optional: uv pip install 'timesfm[torch]==3.0.1'"
            ) from exc
        model_factory = getattr(timesfm, "TimesFM_2p5_200M_torch")  # noqa: B009
        config_cls = getattr(timesfm, "ForecastConfig")  # noqa: B009
        model = model_factory.from_pretrained(
            self.model,
            cache_dir=self.cache_dir,
        )
        model.compile(
            config_cls(
                max_context=self.max_context,
                max_horizon=self.max_horizon,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )
        self._model = model
        return model

    def predict(self, context: np.ndarray, horizon: int) -> QuantileForecast:
        if horizon < 1:
            raise ValueError("horizon muss >= 1 sein")
        if horizon > self.max_horizon:
            raise ValueError(f"horizon muss <= max_horizon ({self.max_horizon}) sein")
        series = np.asarray(context, dtype=np.float32)
        if series.size < _TIMESFM_MIN_CONTEXT:
            raise ValueError(f"Context braucht mindestens {_TIMESFM_MIN_CONTEXT} Kurse")
        if float(series[-1]) <= 0.0:
            raise ValueError("letzter Context-Kurs muss positiv sein")
        if series.size > self.max_context:
            series = series[-self.max_context :]
        model = self._ensure_model()
        point, quantiles = getattr(model, "forecast")(horizon=horizon, inputs=[series])  # noqa: B009
        return QuantileForecast(
            median=np.asarray(point[0], dtype=np.float64),
            q10=np.asarray(quantiles[0, :, 1], dtype=np.float64),
            q90=np.asarray(quantiles[0, :, 9], dtype=np.float64),
        )


def load_timesfm_provider(
    model: str | None = None,
    *,
    max_context: int = 1024,
    max_horizon: int = 288,
    cache_dir: str | Path | None = None,
) -> ForecastProvider:
    """Liefert den konfigurierten TimesFM-Provider (Default: TimesFM 2.5)."""
    return TimesFMProvider(
        model,
        max_context=max_context,
        max_horizon=max_horizon,
        cache_dir=cache_dir,
    )
