"""Tests für Data Persistence Layer (packages/persistence/)."""

from __future__ import annotations

import pytest
from packages.persistence.base import PaginationParams, QueryFilter
from packages.persistence.sqlalchemy.engine import DatabaseConfig, SQLAlchemyEngine
from packages.persistence.sqlalchemy.models import (
    FinalDecisionModel,
    RiskDecisionModel,
    TradingGraphStateModel,
)


class TestDatabaseConfig:
    def test_default_url(self) -> None:
        config = DatabaseConfig()
        assert config.url == "postgresql://trading_user:trading_password@localhost:5432/trading_orchestra"

    def test_custom_config(self) -> None:
        config = DatabaseConfig(host="db.example.com", port=5433, database="test_db")
        assert config.url == "postgresql://trading_user:trading_password@db.example.com:5433/test_db"


class TestSQLAlchemyEngine:
    def test_is_not_connected_initially(self) -> None:
        config = DatabaseConfig(echo=False)
        engine = SQLAlchemyEngine(config)
        assert engine.is_connected() is False

    @pytest.mark.skip(reason="requires psycopg2 for engine creation")
    def test_create_session(self) -> None:
        config = DatabaseConfig(echo=False)
        engine = SQLAlchemyEngine(config)
        session = engine.get_session()
        assert session is not None

    @pytest.mark.asyncio
    async def test_health_check_disconnected(self) -> None:
        config = DatabaseConfig()
        engine = SQLAlchemyEngine(config)
        result = await engine.health_check()
        assert result["backend"] == "postgresql"
        assert result["connected"] is False
        assert "not connected" in result.get("message", "")

    def test_get_engine_singleton(self) -> None:
        from packages.persistence.sqlalchemy.engine import get_engine

        config = DatabaseConfig()
        e1 = get_engine(config)
        e2 = get_engine(config)
        assert e1 is e2

    def test_create_engine_factory(self) -> None:
        from packages.persistence.sqlalchemy.engine import create_engine

        config = DatabaseConfig()
        e1 = create_engine(config)
        e2 = create_engine(config)
        assert e1 is not e2


class TestQueryFilter:
    def test_valid_filter(self) -> None:
        filt = QueryFilter(field="instrument", operator="=", value="BTC")
        assert filt.field == "instrument"
        assert filt.operator == "="
        assert filt.value == "BTC"

    def test_invalid_operator(self) -> None:
        with pytest.raises(ValueError, match="Unsupported operator"):
            QueryFilter(field="id", operator="invalid", value="123")

    def test_all_operators(self) -> None:
        for op in ("=", "!=", ">", ">=", "<", "<=", "in", "like"):
            filt = QueryFilter(field="price", operator=op, value=100)
            assert filt.operator == op


class TestPaginationParams:
    def test_default_pagination(self) -> None:
        p = PaginationParams()
        assert p.offset == 0
        assert p.limit == 100
        assert p.order_by == "created_at"
        assert p.order_direction == "desc"

    def test_custom_pagination(self) -> None:
        p = PaginationParams(offset=10, limit=50, order_by="analysis_time", order_direction="asc")
        assert p.offset == 10
        assert p.limit == 50
        assert p.order_by == "analysis_time"
        assert p.order_direction == "asc"


class TestModels:
    def test_trading_graph_state_model_columns(self) -> None:
        """Prüft, dass das Model die erwarteten Spalten hat."""
        cols = {c.key for c in TradingGraphStateModel.__table__.columns}
        assert "id" in cols
        assert "request_id" in cols
        assert "instrument" in cols
        assert "status" in cols
        assert "current_stage" in cols
        assert "graph_state" in cols
        assert "errors" in cols
        assert "warnings" in cols

    def test_final_decision_model_columns(self) -> None:
        """Prüft, dass das Model die erwarteten Spalten hat."""
        cols = {c.key for c in FinalDecisionModel.__table__.columns}
        assert "id" in cols
        assert "run_id" in cols
        assert "decision" in cols
        assert "reason" in cols
        assert "blocking_reasons" in cols
        assert "forecast" in cols
        assert "risk" in cols

    def test_risk_decision_model_columns(self) -> None:
        """Prüft, dass das Model die erwarteten Spalten hat."""
        cols = {c.key for c in RiskDecisionModel.__table__.columns}
        assert "id" in cols
        assert "risk_version" in cols
        assert "approved" in cols
        assert "reduction_factor" in cols
        assert "gates" in cols
