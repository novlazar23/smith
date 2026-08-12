"""Tests - Konfigurationssystem.

Testet ConfigLoader, AppConfig, DatabaseConfig, TradingConfig etc.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from packages.config import (
    AppConfig,
    ClickHouseConfig,
    ConfigLoader,
    PostgreSQLConfig,
    RedisConfig,
    RiskConfig,
    TradingConfig,
)
from pydantic import ValidationError


class TestConfigLoader:
    """Testet ConfigLoader (YAML/JSON + ENV-Override)."""

    def test_load_empty_dir(self) -> None:
        """Laden aus leerem Verzeichnis gibt empty dict."""
        with tempfile.TemporaryDirectory() as tmp:
            loader = ConfigLoader(config_dir=tmp)
            assert loader.load_yaml("nonexistent.yaml") == {}
            assert loader.load_json("nonexistent.json") == {}
            assert loader.load() == {}

    def test_load_yaml(self) -> None:
        """YAML-Datei wird korrekt geladen."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.yaml"
            path.write_text("key: value\nnested:\n  a: 1\n")
            loader = ConfigLoader(config_dir=tmp)
            data = loader.load_yaml("test.yaml")
            assert data == {"key": "value", "nested": {"a": 1}}

    def test_load_json(self) -> None:
        """JSON-Datei wird korrekt geladen."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.json"
            path.write_text('{"key": "value", "nested": {"a": 1}}')
            loader = ConfigLoader(config_dir=tmp)
            data = loader.load_json("test.json")
            assert data == {"key": "value", "nested": {"a": 1}}

    def test_multiple_load(self) -> None:
        """Mehrere Dateien werden vereinigt."""
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "a.yaml").write_text("x: 1\n")
            Path(tmp, "b.yaml").write_text("y: 2\n")
            loader = ConfigLoader(config_dir=tmp)
            data = loader.load("a.yaml", "b.yaml")
            assert data == {"x": 1, "y": 2}

    def test_env_override_string(self) -> None:
        """ENV-Variablen überschreiben YAML-Werte."""
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "test.yaml").write_text("database:\n  host: yamlhost\n")
            loader = ConfigLoader(config_dir=tmp, env_prefix="TEST")
            with pytest.MonkeyPatch.context() as mp:
                mp.setenv("TEST_DATABASE_HOST", "envhost")
                data = loader.load_yaml("test.yaml")
                data = loader.apply_env_override(data)
                assert data["database"]["host"] == "envhost"

    def test_env_override_bool(self) -> None:
        """ENV-Bool-Werte werden korrekt geparsed."""
        loader = ConfigLoader()
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("APP_DEBUG", "true")
            mp.setenv("APP_VERBOSE", "false")
            result = loader.apply_env_override({})
            assert result["debug"] is True
            assert result["verbose"] is False

    def test_env_override_int(self) -> None:
        """ENV-Int-Werte werden korrekt geparsed."""
        loader = ConfigLoader()
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("APP_PORT", "8080")
            result = loader.apply_env_override({})
            assert result["port"] == 8080

    def test_env_override_float(self) -> None:
        """ENV-Float-Werte werden korrekt geparsed."""
        loader = ConfigLoader()
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("APP_RATE", "0.55")
            result = loader.apply_env_override({})
            assert result["rate"] == 0.55

    def test_env_override_null(self) -> None:
        """ENV-null wird als None geparst."""
        loader = ConfigLoader()
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("APP_VALUE", "null")
            result = loader.apply_env_override({})
            assert result["value"] is None


class TestPostgreSQLConfig:
    """Testet PostgreSQL-Verbindungskonfiguration."""

    def test_defaults(self) -> None:
        cfg = PostgreSQLConfig()
        assert cfg.host == "localhost"
        assert cfg.port == 5432
        assert cfg.database == "trading"
        assert cfg.user == "trading"
        assert cfg.password == ""
        assert cfg.pool_size == 10
        assert cfg.echo is False

    def test_url_no_password(self) -> None:
        cfg = PostgreSQLConfig()
        assert "trading@" in cfg.url
        assert ":5432/trading" in cfg.url

    def test_url_with_password(self) -> None:
        cfg = PostgreSQLConfig(password="secret")
        assert "secret@" in cfg.url

    def test_invalid_port_low(self) -> None:
        with pytest.raises(ValidationError):
            PostgreSQLConfig(port=0)

    def test_invalid_port_high(self) -> None:
        with pytest.raises(ValidationError):
            PostgreSQLConfig(port=99999)


class TestClickHouseConfig:
    """Testet ClickHouse-Verbindungskonfiguration."""

    def test_defaults(self) -> None:
        cfg = ClickHouseConfig()
        assert cfg.host == "localhost"
        assert cfg.port == 8123
        assert cfg.secure is False

    def test_http_url(self) -> None:
        cfg = ClickHouseConfig()
        assert cfg.url.startswith("http://")

    def test_https_url(self) -> None:
        cfg = ClickHouseConfig(secure=True)
        assert cfg.url.startswith("https://")


class TestRedisConfig:
    """Testet Redis-Verbindungskonfiguration."""

    def test_defaults(self) -> None:
        cfg = RedisConfig()
        assert cfg.host == "localhost"
        assert cfg.port == 6379
        assert cfg.db == 0

    def test_url_no_password(self) -> None:
        cfg = RedisConfig()
        assert "redis://" in cfg.url
        assert cfg.url.endswith("/0")

    def test_url_with_password(self) -> None:
        cfg = RedisConfig(password="secret")
        assert "secret@" in cfg.url


class TestRiskConfig:
    """Testet Risikokonfiguration."""

    def test_defaults(self) -> None:
        cfg = RiskConfig()
        assert cfg.max_drawdown == 0.10
        assert cfg.max_drawdown_pct == 10.0
        assert cfg.max_position_size == 0.25
        assert cfg.max_open_positions == 10
        assert cfg.max_single_trade_risk == 0.02
        assert cfg.max_single_trade_risk_pct == 2.0
        assert cfg.kelly_fraction == 0.25
        assert cfg.min_confidence_threshold == 0.55

    def test_custom_values(self) -> None:
        cfg = RiskConfig(max_drawdown=0.05, max_open_positions=5)
        assert cfg.max_drawdown == 0.05
        assert cfg.max_open_positions == 5

    def test_invalid_drawdown(self) -> None:
        with pytest.raises(ValidationError):
            RiskConfig(max_drawdown=1.5)


class TestTradingConfig:
    """Testet Trading-Konfiguration."""

    def test_defaults(self) -> None:
        cfg = TradingConfig()
        assert cfg.mode == "paper"
        assert cfg.validate_mode() is True
        assert cfg.default_timeframes == ["15m", "1h", "4h", "1d"]
        assert cfg.min_candles == 100

    def test_mode_validation(self) -> None:
        cfg = TradingConfig(mode="live")
        assert cfg.validate_mode() is True

        cfg = TradingConfig(mode="research")
        assert cfg.validate_mode() is True

        cfg = TradingConfig(mode="invalid")
        assert cfg.validate_mode() is False


class TestStreamingConfig:
    """Testet Streaming-Konfiguration."""

    def test_defaults(self) -> None:
        from packages.config import StreamingConfig

        cfg = StreamingConfig()
        assert cfg.market_data_topic == "market.data"
        assert cfg.analysis_request_topic == "analysis.requests"
        assert cfg.agent_report_topic == "agents.reports"
        assert cfg.final_decision_topic == "trading.decisions"
        assert cfg.order_topic == "trading.orders"
        assert cfg.dead_letter_topic == "dead.letter"

    def test_redpanda_defaults(self) -> None:
        from packages.config import StreamingConfig

        cfg = StreamingConfig()
        assert cfg.redpanda.bootstrap_servers == "localhost:9092"
        assert cfg.redpanda.acks == "all"
        assert cfg.redpanda.enable_idempotence is True

    def test_consumer_defaults(self) -> None:
        from packages.config import StreamingConfig

        cfg = StreamingConfig()
        assert cfg.consumer.group_id == "trading-consumer-group"
        assert cfg.consumer.auto_offset_reset == "earliest"


class TestObservabilityConfig:
    """Testet Application-Level Observability-Konfiguration."""

    def test_defaults(self) -> None:
        from packages.config.observability import ObservabilityConfig

        cfg = ObservabilityConfig()
        assert cfg.enabled is True
        assert cfg.prometheus_port == 9090
        assert cfg.prometheus_path == "/metrics"
        assert cfg.log_level == "INFO"
        assert cfg.log_json is True
        assert cfg.tracing_enabled is True
        assert cfg.mlflow_enabled is True

    def test_get_prometheus_address(self) -> None:
        from packages.config.observability import ObservabilityConfig

        cfg = ObservabilityConfig()
        assert cfg.get_prometheus_address() == "http://localhost:9090/metrics"

    def test_custom_port(self) -> None:
        from packages.config.observability import ObservabilityConfig

        cfg = ObservabilityConfig(prometheus_port=9999)
        assert cfg.get_prometheus_address() == "http://localhost:9999/metrics"

    def test_invalid_port(self) -> None:
        from packages.config.observability import ObservabilityConfig

        with pytest.raises(ValidationError):
            ObservabilityConfig(prometheus_port=80)  # Below 1024


class TestAppConfig:
    """Testet zentrales AppConfig-Objekt."""

    def test_default(self) -> None:
        cfg = AppConfig.default()
        assert cfg.app_name == "trading-orchestra"
        assert cfg.version == "0.1.0"
        assert cfg.environment == "development"
        assert cfg.is_production() is False
        assert cfg.is_live_trading() is False

    def test_is_production(self) -> None:
        cfg = AppConfig(environment="production")
        assert cfg.is_production() is True
        assert cfg.is_live_trading() is False

    def test_subconfig_access(self) -> None:
        cfg = AppConfig.default()
        # Database
        assert cfg.database.postgresql.host == "localhost"
        assert cfg.database.clickhouse.host == "localhost"
        assert cfg.database.redis.host == "localhost"
        # Trading
        assert cfg.trading.mode == "paper"
        assert cfg.trading.risk.max_drawdown == 0.10
        # Streaming
        assert cfg.streaming.market_data_topic == "market.data"
        # Observability
        assert cfg.observability.prometheus_port == 9090

    def test_from_file_with_yaml(self) -> None:
        """Testet Laden aus configs/trading.yaml."""
        cfg = AppConfig.from_file(config_dir="configs")
        assert cfg.app_name == "trading-orchestra"
        assert cfg.environment == "production"

    def test_env_override_appconfig(self) -> None:
        """ENV überschreibt AppConfig-Werte."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("APP_ENVIRONMENT", "staging")
            mp.setenv("APP_TRADING_MODE", "live")
            cfg = AppConfig.from_file(config_dir="configs")
            assert cfg.environment == "staging"
