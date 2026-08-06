"""Observability — Metriken, strukturiertes Logging, Tracing und MLflow-Integration.

Stellt eine einheitliche Schicht für Prometheus-Metriken, structlog-Logging,
OpenTelemetry-Tracing und MLflow-Experiment-Tracking bereit. Alle Komponenten
sind konfigurierbar und testbar (mit Fakes).
"""

from __future__ import annotations

from .config import ObservabilityConfig
from .logging_ import setup_logging
from .metrics import MetricsRegistry
from .tracing import TracingConfig, TracingRegistry

__all__ = [
    "MetricsRegistry",
    "ObservabilityConfig",
    "TracingConfig",
    "TracingRegistry",
    "setup_logging",
]
