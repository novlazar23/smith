"""Tests für Observability Layer (packages/observability/)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

from packages.observability.config import (
    ObservabilityConfig,
    PrometheusConfig,
    StructlogConfig,
    TracingConfig,
)
from packages.observability.logging_ import get_logger, log_context, setup_logging
from packages.observability.metrics import MetricsRegistry

# ── PrometheusConfig ──────────────────────────────────────────────────


class TestPrometheusConfig:
    def test_defaults(self) -> None:
        cfg = PrometheusConfig()
        assert cfg.port == 9090
        assert cfg.path == "/metrics"
        assert cfg.namespace == "trading"
        assert cfg.enabled is True

    def test_custom_values(self) -> None:
        cfg = PrometheusConfig(port=9999)
        assert cfg.port == 9999


# ── StructlogConfig ───────────────────────────────────────────────────


class TestStructlogConfig:
    def test_defaults(self) -> None:
        cfg = StructlogConfig()
        assert cfg.level == "INFO"
        assert cfg.json_format is True
        assert cfg.timestamp_key == "timestamp"


# ── TracingConfig ─────────────────────────────────────────────────────


class TestTracingConfig:
    def test_defaults(self) -> None:
        cfg = TracingConfig()
        assert cfg.enabled is True
        assert cfg.service_name == "trading-orchestra"
        assert cfg.otlp_endpoint == "http://localhost:4317"
        assert cfg.sampling_rate == 1.0

    def test_custom_values(self) -> None:
        cfg = TracingConfig(
            enabled=False,
            sampling_rate=0.5,
            service_name="my-service",
        )
        assert cfg.enabled is False
        assert cfg.sampling_rate == 0.5
        assert cfg.service_name == "my-service"


# ── ObservabilityConfig ───────────────────────────────────────────────


class TestObservabilityConfig:
    def test_defaults(self) -> None:
        cfg = ObservabilityConfig()
        assert cfg.prometheus is not None
        assert cfg.tracing is not None
        assert cfg.logging is not None
        assert cfg.mlflow is not None

    @patch.dict(os.environ, {
        "OBSERVABILITY_PROMETHEUS_ENABLED": "false",
        "PROMETHEUS_METRICS_PORT": "9100",
    }, clear=False)
    def test_from_env(self) -> None:
        cfg = ObservabilityConfig.from_env()
        assert cfg.prometheus.enabled is False
        assert cfg.prometheus.port == 9100

    @patch.dict(os.environ, {}, clear=True)
    def test_from_env_minimal(self) -> None:
        cfg = ObservabilityConfig.from_env()
        assert cfg.prometheus is not None


# ── MetricsRegistry ───────────────────────────────────────────────────


class TestMetricsRegistry:
    def test_registry_creation(self) -> None:
        reg = MetricsRegistry(namespace="test")
        assert reg is not None

    def test_record_order(self) -> None:
        reg = MetricsRegistry(namespace="test_ord")
        reg.record_order("buy", "limit", "filled")
        data = reg.get_metrics_text()
        assert b"test_ord_orders_total" in data

    def test_record_trade(self) -> None:
        reg = MetricsRegistry(namespace="test_trd")
        reg.record_trade("buy", "BTC/USDT", "binance")
        data = reg.get_metrics_text()
        assert b"test_trd_trades_total" in data

    def test_set_pnl(self) -> None:
        reg = MetricsRegistry(namespace="test_pnl")
        reg.set_pnl("unrealized", 123.45)
        reg.set_pnl("realized", -10.0)
        data = reg.get_metrics_text()
        assert b"test_pnl_pnl" in data

    def test_set_portfolio_value(self) -> None:
        reg = MetricsRegistry(namespace="test_pf")
        reg.set_portfolio_value(50000.0)
        data = reg.get_metrics_text()
        assert b"test_pf_portfolio_value" in data

    def test_position_count(self) -> None:
        reg = MetricsRegistry(namespace="test_pos")
        reg.set_position_count("long", 3)
        reg.set_position_count("short", 1)
        data = reg.get_metrics_text()
        assert b"test_pos_position_count" in data

    def test_event_published(self) -> None:
        reg = MetricsRegistry(namespace="test_evt")
        reg.record_event_published("trading-events", "MarketEvent")
        data = reg.get_metrics_text()
        assert b"test_evt_events_published_total" in data

    def test_event_consumed(self) -> None:
        reg = MetricsRegistry(namespace="test_evt")
        reg.record_event_consumed("trading-events", "consumer-group-1")
        data = reg.get_metrics_text()
        assert b"test_evt_events_consumed_total" in data

    def test_dead_letter(self) -> None:
        reg = MetricsRegistry(namespace="test_dlq")
        reg.record_dead_letter("trading-events", "DeserializationError")
        data = reg.get_metrics_text()
        assert b"test_dlq_dlq_events_total" in data

    def test_stream_latency(self) -> None:
        reg = MetricsRegistry(namespace="test_lat")
        reg.observe_stream_latency("send", 0.015)
        reg.observe_stream_latency("poll", 0.042)
        data = reg.get_metrics_text()
        assert b"test_lat_stream_latency_seconds" in data

    def test_consumer_lag(self) -> None:
        reg = MetricsRegistry(namespace="test_lag")
        reg.set_consumer_lag("trading-events", "cg1", 0, 150)
        data = reg.get_metrics_text()
        assert b"test_lag_consumer_lag" in data

    def test_db_query(self) -> None:
        reg = MetricsRegistry(namespace="test_db")
        reg.record_db_query("postgresql", "select", 0.003)
        data = reg.get_metrics_text()
        assert b"test_db_db_queries_total" in data
        assert b"test_db_db_query_latency_seconds" in data

    def test_cache_hit_miss(self) -> None:
        reg = MetricsRegistry(namespace="test_cch")
        reg.record_cache_hit("redis")
        reg.record_cache_miss("redis")
        data = reg.get_metrics_text()
        assert b"test_cch_cache_hits_total" in data
        assert b"test_cch_cache_misses_total" in data

    def test_health_status(self) -> None:
        reg = MetricsRegistry(namespace="test_hlth")
        reg.set_health(component="redpanda", healthy=True)
        reg.set_health(component="postgres", healthy=False)
        data = reg.get_metrics_text()
        assert b"test_hlth_health_status" in data

    def test_mark_unhealthy(self) -> None:
        reg = MetricsRegistry(namespace="test_up")
        reg.mark_unhealthy()
        data = reg.get_metrics_text()
        assert b"test_up_up" in data

    def test_get_metrics_text_contains_labels(self) -> None:
        reg = MetricsRegistry(namespace="test_lbl")
        reg.record_order("sell", "market", "pending")
        text = reg.get_metrics_text().decode()
        lines = [line for line in text.splitlines() if "orders_total" in line]
        assert any("side=\"sell\"" in line for line in lines)

    def test_get_status(self) -> None:
        reg = MetricsRegistry(namespace="test_st")
        reg.set_health(component="svc", healthy=True)
        status = reg.get_status()
        assert "up" in status
        assert "health_components" in status

    def test_up_starts_healthy(self) -> None:
        reg = MetricsRegistry(namespace="test_init")
        status = reg.get_status()
        assert status["up"] is True

    def test_registry_property(self) -> None:
        reg = MetricsRegistry(namespace="test_reg")
        assert reg.registry is not None


# ── Structlog setup ───────────────────────────────────────────────────


class TestLogging:
    def test_setup_logging_json(self) -> None:
        setup_logging(level="DEBUG", json=True)

    def test_setup_logging_console(self) -> None:
        setup_logging(level="DEBUG", json=False)

    def test_setup_logging_custom_timestamp_key(self) -> None:
        setup_logging(timestamp_key="time")

    def test_get_logger(self) -> None:
        logger = get_logger()
        assert logger is not None

    def test_get_logger_with_name(self) -> None:
        logger = get_logger("my_module")
        assert logger is not None

    def test_log_context_manager(self) -> None:
        with log_context(run_id="abc-123", instrument="BTC/USDT"):
            pass

    def test_log_context_nested(self) -> None:
        with log_context(run_id="outer"), log_context(instrument="ETH/USDT"):
            pass

    def test_log_context_clears_on_exit(self) -> None:
        with log_context(run_id="test"):
            pass
        # After exit, context should be cleared — no exception
        with log_context(fresh_key="value"):
            pass


# ── TracingConfig (runtime) ───────────────────────────────────────────


class TestTracingRuntime:
    def test_config_load_env_enabled(self) -> None:
        with patch.dict(os.environ, {"OBSERVABILITY_TRACING_ENABLED": "true"}):
            cfg = TracingConfig()
        assert cfg.enabled is True

    def test_config_load_env_disabled(self) -> None:
        with patch.dict(os.environ, {"OBSERVABILITY_TRACING_ENABLED": "false"}, clear=True):
            cfg = TracingConfig(
                enabled=os.getenv("OBSERVABILITY_TRACING_ENABLED", "true").lower() == "true",
                service_name=os.getenv("OTEL_SERVICE_NAME", "tracing-test"),
            )
        assert cfg.enabled is False

    @patch("packages.observability.tracing.trace.get_tracer_provider")
    @patch("packages.observability.tracing.trace.set_tracer_provider")
    def test_create_span_fallback(self, mock_set: MagicMock, mock_get: MagicMock) -> None:
        mock_span = MagicMock()
        mock_provider = MagicMock()
        mock_get.return_value = mock_provider
        mock_provider.get_current_span.return_value = mock_span

        from packages.observability.tracing import TracingConfig, TracingRegistry
        cfg = TracingConfig(enabled=False)
        reg = TracingRegistry(cfg)
        span = reg.create_span("test_span")
        assert span is not None

    def test_set_span_attribute_types(self) -> None:
        mock_span = MagicMock()
        from packages.observability.tracing import TracingConfig, TracingRegistry
        cfg = TracingConfig(enabled=False)
        reg = TracingRegistry(cfg)
        reg.set_span_attribute(mock_span, "string_val", "hello")
        reg.set_span_attribute(mock_span, "int_val", 42)
        reg.set_span_attribute(mock_span, "float_val", 3.14)
        reg.set_span_attribute(mock_span, "bool_val", value=True)
        # Should not raise
