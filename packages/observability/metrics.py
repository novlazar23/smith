"""Metrics — Prometheus-Metriken für Trading-Orchestra.

Bietet eine strukturierte Metrik-Schicht mit:
- Handelsmetriken (Order, Trade, PnL)
- Stream-Metriken (Producer, Consumer, DLQ)
- Persistenzmetriken (DB-Queries, Cache)
- Gesundheitsmetriken (Health, Latency)
"""

from __future__ import annotations

from typing import Any

from prometheus_client import (
    CollectorRegistry as Registry,
)
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)


class MetricsRegistry:
    """Zentraler Prometheus-Metriken-Registry für Trading-Orchestra.

    Bündelt alle Metriken unter dem "trading"-Namespace und stellt
    Hilfsfunktionen zum Inkrementieren, Setzen und Zählen bereit.
    """

    def __init__(self, namespace: str = "trading") -> None:
        """Initialisiert den Registry mit allen Handelsmetriken.

        Args:
            namespace: Prometheus-Namespace für alle Metriken.
        """
        self._registry = Registry()

        # ── Handelsmetriken ──────────────────────────────────────────
        self.orders_total = Counter(
            f"{namespace}_orders_total",
            "Gesamtanzahl geordneter Orders.",
            ["side", "order_type", "status"],
            registry=self._registry,
        )

        self.trades_total = Counter(
            f"{namespace}_trades_total",
            "Gesamtanzahl ausgeführter Trades.",
            ["side", "instrument", "venue"],
            registry=self._registry,
        )

        self.pnl_gauge = Gauge(
            f"{namespace}_pnl",
            "Unrealized und realized PnL.",
            ["pnl_type"],
            registry=self._registry,
        )

        self.portfolio_value_gauge = Gauge(
            f"{namespace}_portfolio_value",
            "Aktueller Portfolio-Wert.",
            registry=self._registry,
        )

        self.position_count_gauge = Gauge(
            f"{namespace}_position_count",
            "Anzahl offener Positionen.",
            ["side"],
            registry=self._registry,
        )

        # ── Stream-Metriken ──────────────────────────────────────────
        self.events_published = Counter(
            f"{namespace}_events_published_total",
            "Gesamtanzahl publizerter Events.",
            ["topic", "event_type"],
            registry=self._registry,
        )

        self.events_consumed = Counter(
            f"{namespace}_events_consumed_total",
            "Gesamtanzahl konsumierter Events.",
            ["topic", "consumer_group"],
            registry=self._registry,
        )

        self.dead_letter_total = Counter(
            f"{namespace}_dlq_events_total",
            "Events in der Dead Letter Queue.",
            ["topic", "error_type"],
            registry=self._registry,
        )

        self.stream_latency = Histogram(
            f"{namespace}_stream_latency_seconds",
            "Latenz von Producer/Consumer-Operationen.",
            ["operation"],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0),
            registry=self._registry,
        )

        self.consumer_lag_gauge = Gauge(
            f"{namespace}_consumer_lag",
            "Consumer-Lag in Messages.",
            ["topic", "consumer_group", "partition"],
            registry=self._registry,
        )

        # ── Persistenzmetriken ───────────────────────────────────────
        self.db_queries_total = Counter(
            f"{namespace}_db_queries_total",
            "Gesamtanzahl Datenbankabfragen.",
            ["backend", "operation"],
            registry=self._registry,
        )

        self.db_query_latency = Histogram(
            f"{namespace}_db_query_latency_seconds",
            "Latenz von Datenbankabfragen.",
            ["backend", "operation"],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0),
            registry=self._registry,
        )

        self.cache_hits_total = Counter(
            f"{namespace}_cache_hits_total",
            "Cache-Treffer.",
            ["backend"],
            registry=self._registry,
        )

        self.cache_misses_total = Counter(
            f"{namespace}_cache_misses_total",
            "Cache-Fehler.",
            ["backend"],
            registry=self._registry,
        )

        # ── Gesundheitsmetriken ──────────────────────────────────────
        self.health_status_gauge = Gauge(
            f"{namespace}_health_status",
            "Gesundheitsstatus des Services (1=healthy, 0=unhealthy).",
            ["component"],
            registry=self._registry,
        )

        self.up_gauge = Gauge(
            f"{namespace}_up",
            "1 wenn der Service läuft, sonst 0.",
            registry=self._registry,
        )
        self.up_gauge.set(1)

    @property
    def registry(self) -> Registry:
        """Gibt den zugrunde liegenden Prometheus-Registry zurück."""
        return self._registry

    def get_metrics_text(self) -> bytes:
        """Gibt alle Metriken im Prometheus-Textformat zurück.

        Returns:
            Prometheus-Exposition-format als bytes.
        """
        return generate_latest(self._registry)

    # ── Handelsmetrik-Helper ────────────────────────────────────────

    def record_order(self, side: str, order_type: str, status: str) -> None:
        """Recordet eine Order.

        Args:
            side: "buy" oder "sell".
            order_type: "limit", "market", "stop".
            status: "pending", "filled", "cancelled", "rejected".
        """
        self.orders_total.labels(side=side, order_type=order_type, status=status).inc()

    def record_trade(self, side: str, instrument: str, venue: str) -> None:
        """Recordet einen ausgeführten Trade.

        Args:
            side: "buy" oder "sell".
            instrument: Handelsinstrument (z. B. "BTC/USDT").
            venue: Handelsplatz (z. B. "binance").
        """
        self.trades_total.labels(side=side, instrument=instrument, venue=venue).inc()

    def set_pnl(self, pnl_type: str, value: float) -> None:
        """Setzt den PnL-Wert.

        Args:
            pnl_type: "unrealized" oder "realized".
            value: PnL-Wert.
        """
        self.pnl_gauge.labels(pnl_type=pnl_type).set(value)

    def set_portfolio_value(self, value: float) -> None:
        """Setzt den aktuellen Portfolio-Wert.

        Args:
            value: Portfolio-Wert.
        """
        self.portfolio_value_gauge.set(value)

    def set_position_count(self, side: str, count: int) -> None:
        """Setzt die Anzahl der Positionen pro Seite.

        Args:
            side: "long", "short" oder "flat".
            count: Anzahl der Positionen.
        """
        self.position_count_gauge.labels(side=side).set(count)

    # ── Stream-Metrik-Helper ────────────────────────────────────────

    def record_event_published(self, topic: str, event_type: str) -> None:
        """Recordet ein publiziertes Event.

        Args:
            topic: Topic-Name.
            event_type: Event-Typ (z. B. "MarketEvent").
        """
        self.events_published.labels(topic=topic, event_type=event_type).inc()

    def record_event_consumed(self, topic: str, consumer_group: str) -> None:
        """Recordet ein konsumiertes Event.

        Args:
            topic: Topic-Name.
            consumer_group: Consumer-Group-Name.
        """
        self.events_consumed.labels(topic=topic, consumer_group=consumer_group).inc()

    def record_dead_letter(self, topic: str, error_type: str) -> None:
        """Recordet ein DLQ-Event.

        Args:
            topic: Topic-Name.
            error_type: Fehlerart (z. B. "DeserializationError").
        """
        self.dead_letter_total.labels(topic=topic, error_type=error_type).inc()

    def observe_stream_latency(self, operation: str, duration: float) -> None:
        """Observed die Latenz einer Stream-Operation.

        Args:
            operation: "send", "send_batch", "poll", "commit".
            duration: Dauer in Sekunden.
        """
        self.stream_latency.labels(operation=operation).observe(duration)

    def set_consumer_lag(self, topic: str, consumer_group: str, partition: int, lag: int) -> None:
        """Setzt den Consumer-Lag für ein Topic-Partition.

        Args:
            topic: Topic-Name.
            consumer_group: Consumer-Group-Name.
            partition: Partitionsnummer.
            lag: Anzahl der verzögerten Messages.
        """
        self.consumer_lag_gauge.labels(topic=topic, consumer_group=consumer_group, partition=str(partition)).set(lag)

    # ── Persistenz-Metrik-Helper ────────────────────────────────────

    def record_db_query(self, backend: str, operation: str, duration: float) -> None:
        """Recordet eine Datenbankabfrage.

        Args:
            backend: "postgresql", "clickhouse", "redis".
            operation: "select", "insert", "update", "delete".
            duration: Dauer in Sekunden.
        """
        self.db_queries_total.labels(backend=backend, operation=operation).inc()
        self.db_query_latency.labels(backend=backend, operation=operation).observe(duration)

    def record_cache_hit(self, backend: str) -> None:
        """Recordet einen Cache-Treffer.

        Args:
            backend: Cache-Backend-Name.
        """
        self.cache_hits_total.labels(backend=backend).inc()

    def record_cache_miss(self, backend: str) -> None:
        """Recordet einen Cache-Fehler.

        Args:
            backend: Cache-Backend-Name.
        """
        self.cache_misses_total.labels(backend=backend).inc()

    # ── Gesundheitsmetrik-Helper ────────────────────────────────────

    def set_health(self, *, component: str, healthy: bool) -> None:
        """Setzt den Gesundheitsstatus einer Komponente.

        Args:
            component: Komponentenname (z. B. "redpanda", "postgres").
            healthy: True wenn gesund, sonst False.
        """
        self.health_status_gauge.labels(component=component).set(1.0 if healthy else 0.0)

    def mark_unhealthy(self) -> None:
        """Markiert den Service als ungesund."""
        self.up_gauge.set(0)

    def get_status(self) -> dict[str, Any]:
        """Gibt einen Status-Report aller Metriken zurück.

        Returns:
            Dict mit Metrik-Zuständen.
        """
        health_components: dict[str, float] = {}
        if self.health_status_gauge._metrics:
            label_names = self.health_status_gauge._labelnames
            for label_values_tuple, metric_obj in self.health_status_gauge._metrics.items():
                label_dict = dict(zip(label_names, label_values_tuple, strict=False))
                health_components[str(label_dict)] = metric_obj._value.get()
        return {
            "up": bool(self.up_gauge._value.get()),
            "health_components": health_components,
        }
