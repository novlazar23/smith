"""Observability Configuration — App-level Monitoring-Settings.

Bündelt Prometheus, Logging, Tracing und MLflow auf Application-Ebene.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ObservabilityConfig(BaseModel):
    """Application-level Observability-Konfiguration."""

    enabled: bool = Field(
        default=True,
        description="Observability insgesamt aktivieren/deaktivieren."
    )
    prometheus_port: int = Field(
        default=9090, ge=1024, le=65535, description="Prometheus Metrics-Port."
    )
    prometheus_path: str = Field(
        default="/metrics", description="Metrics-Endpoint-Pfad."
    )
    prometheus_namespace: str = Field(
        default="trading", description="Prometheus-Namespace."
    )
    log_level: str = Field(
        default="INFO", description="Globales Log-Level."
    )
    log_json: bool = Field(
        default=True, description="JSON-Log-Format."
    )
    tracing_enabled: bool = Field(
        default=True, description="OpenTelemetry-Tracing aktivieren."
    )
    tracing_service_name: str = Field(
        default="trading-orchestra", description="Service-Name für Traces."
    )
    tracing_otlp_endpoint: str = Field(
        default="http://localhost:4317", description="OTLP-Endpoint."
    )
    tracing_sampling_rate: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Tracing-Sampling-Rate."
    )
    mlflow_enabled: bool = Field(
        default=True, description="MLflow-Experiment-Tracking aktivieren."
    )
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000", description="MLflow-Server-URI."
    )
    mlflow_experiment_name: str = Field(
        default="trading-orchestra", description="MLflow-Experiment-Name."
    )

    def get_prometheus_address(self) -> str:
        """Prometheus-Adresse für Scrape-Config."""
        return f"http://localhost:{self.prometheus_port}{self.prometheus_path}"
