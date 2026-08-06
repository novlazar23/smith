"""Configuration — Pydantic-basierte Konfiguration für Observability.

Zentraler Konfigurationspunkt für Prometheus, structlog, OpenTelemetry
und MLflow. Liest aus Umgebungsvariablen und unterstützt Defaults.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PrometheusConfig(BaseModel):
    """Prometheus-Metriken-Konfiguration."""

    enabled: bool = Field(default=True, description="Metriken aktivieren/deaktivieren.")
    port: int = Field(default=9090, description="HTTP-Port für Metrics-Endpoint.")
    path: str = Field(default="/metrics", description="Pfad für Metrics-Endpoint.")
    namespace: str = Field(default="trading", description="Prometheus-Namespace.")


class StructlogConfig(BaseModel):
    """Structlog-Logging-Konfiguration."""

    level: str = Field(default="INFO", description="Log-Level.")
    json: bool = Field(default=True, description="JSON-Formatierung aktivieren.")
    timestamp_key: str = Field(default="timestamp", description="Schlüssel für Zeitstempel.")


class TracingConfig(BaseModel):
    """OpenTelemetry-Tracing-Konfiguration."""

    enabled: bool = Field(default=True, description="Tracing aktivieren/deaktivieren.")
    service_name: str = Field(default="trading-orchestra", description="Service-Name für Spans.")
    otlp_endpoint: str = Field(default="http://localhost:4317", description="OTLP-Endpoint.")
    sampling_rate: float = Field(default=1.0, ge=0.0, le=1.0, description="Sampling-Rate (0.0-1.0).")


class MLflowConfig(BaseModel):
    """MLflow-Experiment-Tracking-Konfiguration."""

    enabled: bool = Field(default=True, description="MLflow aktivieren/deaktivieren.")
    tracking_uri: str = Field(default="http://localhost:5000", description="MLflow-Server-URI.")
    experiment_name: str = Field(default="trading-orchestra", description="Standard-Experiment-Name.")
    tracking_backend: str = Field(default="sqlite", description="Tracking-Backend (sqlite, postgres).")


class ObservabilityConfig(BaseModel):
    """Gesamte Observability-Konfiguration.

    Bündelt Prometheus, Logging, Tracing und MLflow in einem Objekt.
    """

    prometheus: PrometheusConfig = Field(default_factory=PrometheusConfig)
    logging: StructlogConfig = Field(default_factory=StructlogConfig)
    tracing: TracingConfig = Field(default_factory=TracingConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)

    @classmethod
    def from_env(cls) -> ObservabilityConfig:
        """Erstellt Konfiguration aus Umgebungsvariablen."""
        import os

        return cls(
            prometheus=PrometheusConfig(
                enabled=os.getenv("OBSERVABILITY_PROMETHEUS_ENABLED", "true").lower() == "true",
                port=int(os.getenv("PROMETHEUS_METRICS_PORT", "9090")),
                path=os.getenv("PROMETHEUS_METRICS_PATH", "/metrics"),
                namespace=os.getenv("PROMETHEUS_NAMESPACE", "trading"),
            ),
            logging=StructlogConfig(
                level=os.getenv("LOG_LEVEL", "INFO").upper(),
                json=os.getenv("LOG_JSON", "true").lower() == "true",
                timestamp_key=os.getenv("LOG_TIMESTAMP_KEY", "timestamp"),
            ),
            tracing=TracingConfig(
                enabled=os.getenv("OBSERVABILITY_TRACING_ENABLED", "true").lower() == "true",
                service_name=os.getenv("OTEL_SERVICE_NAME", "trading-orchestra"),
                otlp_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"),
                sampling_rate=float(os.getenv("OTEL_SAMPLING_RATE", "1.0")),
            ),
            mlflow=MLflowConfig(
                enabled=os.getenv("OBSERVABILITY_MLFLOW_ENABLED", "true").lower() == "true",
                tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
                experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "trading-orchestra"),
                tracking_backend=os.getenv("MLFLOW_BACKEND_STORE_URI", "sqlite"),
            ),
        )
