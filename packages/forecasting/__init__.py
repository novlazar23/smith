"""Optionale Forecast-Provider für die Evolutions-Pipeline."""

from packages.forecasting.timesfm import (
    FakeProvider,
    ForecastProvider,
    QuantileForecast,
    TimesFMProvider,
    TimesFMUnavailableError,
    load_timesfm_provider,
    provider_label,
)

__all__ = [
    "FakeProvider",
    "ForecastProvider",
    "QuantileForecast",
    "TimesFMProvider",
    "TimesFMUnavailableError",
    "load_timesfm_provider",
    "provider_label",
]
