"""Tracing — OpenTelemetry-Tracing für Trading-Orchestra.

Bietet span-basierte Verfolgung von:
- Analyse-Pipelines
- Order-Ausführung
- Stream-Operationen
- Datenbankabfragen
"""

from __future__ import annotations

import os
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import SpanKind, StatusCode


class TracingConfig:
    """OpenTelemetry-Tracing-Konfiguration."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        service_name: str = "trading-orchestra",
        otlp_endpoint: str = "http://localhost:4317",
        sampling_rate: float = 1.0,
    ) -> None:
        """Initialisiert die Tracing-Konfiguration.

        Args:
            enabled: Tracing aktivieren/deaktivieren.
            service_name: Service-Name für Spans.
            otlp_endpoint: OTLP-Exporter-Endpoint.
            sampling_rate: Sampling-Rate (0.0-1.0).
        """
        self.enabled = enabled
        self.service_name = service_name
        self.otlp_endpoint = otlp_endpoint
        self.sampling_rate = sampling_rate


class TracingRegistry:
    """OpenTelemetry-Tracing-Registry für Trading-Orchestra.

    Verwaltet den TracerProvider und stellt Hilfsmethoden zum Erzeugen
    und Verwalten von Spans bereit.
    """

    def __init__(self, config: TracingConfig | None = None) -> None:
        """Initialisiert die Tracing-Schicht.

        Args:
            config: Tracing-Konfiguration. Wird aus Env erstellt wenn None.
        """
        self._config = config or self._load_config()
        self._tracer_provider: TracerProvider | None = None
        self._tracer: trace.Tracer | None = None
        self._initialized = False

    @staticmethod
    def _load_config() -> TracingConfig:
        """Lädt Tracing-Konfiguration aus Umgebungsvariablen."""
        return TracingConfig(
            enabled=os.getenv("OBSERVABILITY_TRACING_ENABLED", "true").lower() == "true",
            service_name=os.getenv("OTEL_SERVICE_NAME", "trading-orchestra"),
            otlp_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"),
            sampling_rate=float(os.getenv("OTEL_SAMPLING_RATE", "1.0")),
        )

    def initialize(self) -> None:
        """Initialisiert den TracerProvider mit dem OTLP-Exporter.

        Muss vor der ersten Verwendung aufgerufen werden.
        """
        if self._initialized or not self._config.enabled:
            return

        resource = Resource.create({
            "service.name": self._config.service_name,
            "service.version": "0.1.0",
        })

        self._tracer_provider = TracerProvider(resource=resource)

        # Sampler basierend auf Konfiguration
        if self._config.sampling_rate < 1.0:
            from opentelemetry.trace.sampling import ProbabilitySampler
            self._tracer_provider.sampler = ProbabilitySampler(self._config.sampling_rate)

        # OTLP-Exporter hinzufügen (wenn Endpunkt konfiguriert)
        if self._config.otlp_endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
                exporter = OTLPSpanExporter(endpoint=self._config.otlp_endpoint)
                processor = BatchSpanProcessor(exporter)
                self._tracer_provider.add_span_processor(processor)
            except ImportError:
                # Graceful degradation: kein Export wenn OTLP nicht verfügbar
                pass

        trace.set_tracer_provider(self._tracer_provider)
        self._tracer = trace.get_tracer(self._config.service_name)
        self._initialized = True

    @property
    def tracer(self) -> trace.Tracer | None:
        """Gibt den aktiven Tracer zurück."""
        if not self._initialized:
            self.initialize()
        return self._tracer

    @property
    def config(self) -> TracingConfig:
        """Gibt die Tracing-Konfiguration zurück."""
        return self._config

    @property
    def is_enabled(self) -> bool:
        """Prüft ob Tracing aktiviert ist."""
        return self._config.enabled

    def create_span(self, name: str, kind: SpanKind = SpanKind.INTERNAL, attributes: dict[str, Any] | None = None) -> trace.Span:
        """Erstellt einen neuen Span.

        Args:
            name: Span-Name.
            kind: Span-Typ (server, client, producer, consumer, internal).
            attributes: Attribute für den Span.

        Returns:
            Der erstellte Span.
        """
        if not self._tracer:
            # Fallback: NoOp-Tracer wenn nicht initialisiert
            return trace.get_current_span()

        return self._tracer.start_span(name, kind=kind, attributes=attributes or {})

    def record_exception(self, span: trace.Span, exception: Exception) -> None:
        """Recordet eine Exception in einem Span.

        Args:
            span: Der Span in dem die Exception auftritt.
            exception: Die aufgetretene Exception.
        """
        span.record_exception(exception)
        span.set_status(StatusCode.ERROR, str(exception))

    def set_span_attribute(self, span: trace.Span, key: str, value: str | int | float | bool) -> None:
        """Setzt ein Attribut auf einen Span.

        Args:
            span: Der Span zu aktualisieren.
            key: Attribut-Schlüssel.
            value: Attribut-Wert.
        """
        span.set_attribute(key, str(value) if not isinstance(value, (int, float, bool)) else value)

    def close(self) -> None:
        """Schließt den TracerProvider und flushed alle Spans."""
        if self._tracer_provider:
            self._tracer_provider.force_flush(timeout_millis=5000)
            self._tracer_provider.shutdown(timeout_millis=5000)
            self._initialized = False

    # ── EPIC-12: Context-Helper ──────────────────────────────────────

    def set_run_context(self, span: trace.Span, *, run_id: str, request_id: str | None = None) -> None:
        """Setzt Run- und Request-Kontext-Attribute auf einen Span.

        Args:
            span: Der zu aktualisierende Span.
            run_id: Eindeutige ID des Analyse-Laufs.
            request_id: Optionale Request-ID.
        """
        span.set_attribute("run_id", run_id)
        if request_id:
            span.set_attribute("request_id", request_id)

    def set_agent_context(self, span: trace.Span, agent_id: str) -> None:
        """Setzt Agent-Kontext-Attribute auf einen Span.

        Args:
            span: Der zu aktualisierende Span.
            agent_id: ID des ausführenden Agents.
        """
        span.set_attribute("agent_id", agent_id)

    def set_report_context(self, span: trace.Span, report_id: str) -> None:
        """Setzt Report-Kontext-Attribute auf einen Span.

        Args:
            span: Der zu aktualisierende Span.
            report_id: ID des Reports.
        """
        span.set_attribute("report_id", report_id)

    def set_snapshot_context(self, span: trace.Span, snapshot_id: str) -> None:
        """Setzt Snapshot-Kontext-Attribute auf einen Span.

        Args:
            span: Der zu aktualisierende Span.
            snapshot_id: ID des Market-Snapshots.
        """
        span.set_attribute("snapshot_id", snapshot_id)

    def create_analysis_span(
        self,
        analysis_type: str,
        run_id: str,
        attributes: dict[str, Any] | None = None,
    ) -> trace.Span:
        """Erstellt einen Span für eine Analyse mit vollem Kontext.

        Args:
            analysis_type: Typ der Analyse.
            run_id: Eindeutige ID des Analyse-Laufs.
            attributes: Zusätzliche Attribute.

        Returns:
            Der erstellte Span mit run_id und analysis_type.
        """
        attrs = {"analysis_type": analysis_type, "run_id": run_id, **(attributes or {})}
        return self.create_span(f"analysis.{analysis_type}", attributes=attrs)

    def create_agent_span(
        self,
        agent_id: str,
        run_id: str,
        attributes: dict[str, Any] | None = None,
    ) -> trace.Span:
        """Erstellt einen Span für eine Agentenausführung mit vollem Kontext.

        Args:
            agent_id: ID des Agents.
            run_id: ID des übergeordneten Analyse-Laufs.
            attributes: Zusätzliche Attribute.

        Returns:
            Der erstellte Span mit agent_id und run_id.
        """
        attrs = {"agent_id": agent_id, "run_id": run_id, **(attributes or {})}
        return self.create_span(f"agent.{agent_id}", attributes=attrs)

    def create_decision_span(
        self,
        run_id: str,
        request_id: str,
        report_id: str | None = None,
        snapshot_id: str | None = None,
    ) -> trace.Span:
        """Erstellt einen Span für eine Handelsentscheidung mit vollem Kontext.

        Args:
            run_id: ID des Analyse-Laufs.
            request_id: ID der Anfrage.
            report_id: Optionale ID des Reports.
            snapshot_id: Optionale ID des Market-Snapshots.

        Returns:
            Der erstellte Span mit allen Kontext-Attributen.
        """
        attrs = {"run_id": run_id, "request_id": request_id}
        if report_id:
            attrs["report_id"] = report_id
        if snapshot_id:
            attrs["snapshot_id"] = snapshot_id
        return self.create_span("trading.decision", attributes=attrs)
