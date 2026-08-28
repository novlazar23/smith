"""Tests für Shadow-Trading-Settings und -Modelle (Epic WI-ST-01).

Spezifikations-Quellen: `specs/shadow-trading.md` §6.1 (Modelle), §6.2 (Settings),
ST.2 (Flag), ST.7 (Records, E8-Kette), ST.8 (Session-State, F5).

Namenskollision bewusst: `ShadowTradingRecord` (hier) ist nicht das bestehende
`ShadowTradeRecord` aus `services/shadow_mode_logger.py` (Phase 5, Spec-Z3).
"""

from __future__ import annotations

import inspect
from enum import StrEnum

from pydantic import BaseModel

from trading_harness.config import Settings
from trading_harness.models import (
    ShadowEvaluationSummary,
    ShadowLoopStatus,
    ShadowSessionState,
    ShadowTradingRecord,
    ShadowTradingStatus,
    SignalAggregation,
)


class TestShadowTradingSettings:
    """Shadow-Trading-Settings (Spec §6.2) — Feature ist default-off.

    Alle Default-Tests nutzen ``_env_file=None``, damit sie gegen eine lokale
    ``.env`` mit aktiviertem Shadow-Trading immun sind (echte Defaults).
    """

    def test_shadow_trading_disabled_by_default(self):
        """S1/ST.2: `shadow_trading_enabled` Default ist False."""
        settings = Settings(_env_file=None)
        assert settings.shadow_trading_enabled is False

    def test_all_shadow_settings_defaults(self):
        """Alle 9 shadow_*-Keys mit exakten Spec-Default-Werten."""
        settings = Settings(_env_file=None)
        assert settings.shadow_trading_enabled is False
        assert settings.shadow_loop_interval_seconds == 900
        assert settings.shadow_max_decisions_per_day == 96
        assert settings.shadow_trading_symbols == []
        assert settings.shadow_min_confidence == 0.6
        assert settings.shadow_stop_loss_fraction == 0.02
        assert settings.shadow_min_risk_reward == 2.0
        assert settings.shadow_state_path == "data/shadow_trading_state.json"
        assert settings.shadow_start_equity == 100000.0

    def test_shadow_symbols_from_env_json(self, monkeypatch):
        """shadow_trading_symbols wird aus Env per Pydantic-Settings-Parser geladen (JSON-Liste)."""
        monkeypatch.setenv("SHADOW_TRADING_SYMBOLS", '["BTCUSDT", "ETHUSDT"]')
        settings = Settings(_env_file=None)
        assert settings.shadow_trading_symbols == ["BTCUSDT", "ETHUSDT"]

    def test_shadow_settings_overridable_by_env(self, monkeypatch):
        """Alle shadow_*-Keys sind via Env überschreibbar (M1: Defaults, kein Lock)."""
        monkeypatch.setenv("SHADOW_TRADING_ENABLED", "true")
        monkeypatch.setenv("SHADOW_LOOP_INTERVAL_SECONDS", "60")
        monkeypatch.setenv("SHADOW_MAX_DECISIONS_PER_DAY", "12")
        monkeypatch.setenv("SHADOW_MIN_CONFIDENCE", "0.7")
        monkeypatch.setenv("SHADOW_STOP_LOSS_FRACTION", "0.01")
        monkeypatch.setenv("SHADOW_MIN_RISK_REWARD", "3.0")
        monkeypatch.setenv("SHADOW_STATE_PATH", "data/custom_shadow_state.json")
        monkeypatch.setenv("SHADOW_START_EQUITY", "50000.0")
        settings = Settings(_env_file=None)
        assert settings.shadow_trading_enabled is True
        assert settings.shadow_loop_interval_seconds == 60
        assert settings.shadow_max_decisions_per_day == 12
        assert settings.shadow_min_confidence == 0.7
        assert settings.shadow_stop_loss_fraction == 0.01
        assert settings.shadow_min_risk_reward == 3.0
        assert settings.shadow_state_path == "data/custom_shadow_state.json"
        assert settings.shadow_start_equity == 50000.0


class TestShadowTradingEnums:
    """ShadowLoopStatus / ShadowTradingStatus — StrEnum-Mitglieder laut Spec §6.1."""

    def test_shadow_loop_status_members(self):
        assert {member.value for member in ShadowLoopStatus} == {
            "RUNNING",
            "STOPPING",
            "STOPPED",
        }
        assert ShadowLoopStatus.RUNNING is ShadowLoopStatus("RUNNING")

    def test_shadow_trading_status_members(self):
        assert {member.value for member in ShadowTradingStatus} == {
            "NO_TRADE",
            "REJECTED",
            "FILLED",
        }
        assert ShadowTradingStatus.FILLED is ShadowTradingStatus("FILLED")

    def test_enums_are_strenum(self):
        assert issubclass(ShadowLoopStatus, StrEnum)
        assert issubclass(ShadowTradingStatus, StrEnum)


class TestShadowSessionState:
    """ShadowSessionState (ST.8) — Session-State inkl. state_checksum (F5)."""

    def test_default_construction(self):
        """Instanziierbar ohne Argumente; alle Defaults laut Spec §6.1."""
        state = ShadowSessionState()
        assert state.session_id.startswith("shadow-")
        assert state.status is ShadowLoopStatus.STOPPED
        assert state.symbols == []
        assert state.interval_seconds == 900
        assert state.started_at is None
        assert state.stopped_at is None
        assert state.last_iteration_at is None
        assert state.iteration_count == 0
        assert state.decisions_today == 0
        assert state.budget_date == ""
        assert state.start_equity == 100000.0
        assert state.current_equity == 100000.0
        assert state.open_positions == 0
        assert state.error_count == 0
        assert state.last_error is None
        assert state.restart_required is False
        assert state.state_checksum == ""

    def test_session_ids_unique(self):
        """Jede Session erhält eine eindeutige session_id."""
        a = ShadowSessionState()
        b = ShadowSessionState()
        assert a.session_id != b.session_id

    def test_state_checksum_semantics_dokumentiert(self):
        """F5: Semantik von `state_checksum` ist im Source dokumentiert.

        Erwartete Semantik: SHA-256 (hex) über das kanonisierte JSON des
        Session-States, berechnet OHNE das `state_checksum`-Feld selbst.
        Die Berechnungs-/Verifikationshilfe folgt in WI-ST-02
        (`ShadowTradingStateStore`); hier nur Feld + Typ + Doku.
        """
        source = inspect.getsource(ShadowSessionState)
        assert "state_checksum" in source
        assert "SHA-256" in source

    def test_json_roundtrip(self):
        state = ShadowSessionState(symbols=["BTCUSDT"], iteration_count=3, status=ShadowLoopStatus.RUNNING)
        loaded = ShadowSessionState.model_validate_json(state.model_dump_json())
        assert loaded == state


class TestShadowTradingRecord:
    """ShadowTradingRecord (ST.7) — ein Record pro Loop-Decision (alle drei Status)."""

    def _make_record(self, **overrides: object) -> ShadowTradingRecord:
        base: dict[str, object] = {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "direction": "LONG",
            "status": ShadowTradingStatus.FILLED,
            "decision_id": "dec-1",
            "run_id": "run-1",
            "snapshot_id": "snap-1",
        }
        base.update(overrides)
        return ShadowTradingRecord(**base)

    def test_id_default_and_numeric_field_defaults(self):
        record = self._make_record()
        assert record.id.startswith("shadow-rec-")
        assert record.timestamp.tzinfo is not None  # UTC (utcnow)
        assert record.risk_reason == ""
        assert record.agent_ids == []
        assert record.quantity == 0.0
        assert record.entry_price == 0.0
        assert record.mark_price == 0.0
        assert record.slippage == 0.0
        assert record.commission == 0.0
        assert record.pnl_estimate == 0.0
        assert record.slippage_rate == 0.0
        assert record.commission_rate == 0.0
        assert record.config_version == ""

    def test_trade_id_none_except_filled(self):
        """E8: `trade_id` (Paper-Trade-ID) ist None, außer Status FILLED.

        Die Audit-Kette `snapshot_id -> run_id -> decision_id -> record_id
        -> trade_id` schließt sich damit erst beim Fill.
        """
        filled = self._make_record(status=ShadowTradingStatus.FILLED, trade_id="paper-trade-1")
        assert filled.trade_id == "paper-trade-1"
        for status in (ShadowTradingStatus.NO_TRADE, ShadowTradingStatus.REJECTED):
            record = self._make_record(status=status)
            assert record.trade_id is None
            assert record.pnl_estimate == 0.0

    def test_rejected_record_carries_risk_reason(self):
        record = self._make_record(
            status=ShadowTradingStatus.REJECTED,
            side="NONE",
            direction="NO_TRADE",
            risk_reason="MAX_POSITION_SIZE_EXCEEDED",
        )
        assert record.status is ShadowTradingStatus.REJECTED
        assert record.risk_reason == "MAX_POSITION_SIZE_EXCEEDED"
        assert record.trade_id is None

    def test_agent_ids_and_config_version(self):
        record = self._make_record(agent_ids=["agent-a", "agent-b"], config_version="sha256:abc")
        assert record.agent_ids == ["agent-a", "agent-b"]
        assert record.config_version == "sha256:abc"

    def test_json_roundtrip(self):
        record = self._make_record(trade_id="paper-trade-1", pnl_estimate=12.5)
        loaded = ShadowTradingRecord.model_validate_json(record.model_dump_json())
        assert loaded == record


class TestSignalAggregation:
    """SignalAggregation (ST.6) — deterministische Aggregation von Agenten-Signalen."""

    def test_instantiation_and_typed_fields(self):
        aggregation = SignalAggregation(
            direction="LONG",
            confidence=0.75,
            signal_count=2,
            no_trade_count=1,
            agent_ids=["agent-a", "agent-b", "agent-c"],
        )
        assert aggregation.direction == "LONG"
        assert aggregation.confidence == 0.75
        assert aggregation.signal_count == 2
        assert aggregation.no_trade_count == 1
        assert aggregation.agent_ids == ["agent-a", "agent-b", "agent-c"]

    def test_no_trade_aggregation(self):
        aggregation = SignalAggregation(
            direction="NO_TRADE",
            confidence=0.0,
            signal_count=0,
            no_trade_count=2,
            agent_ids=["agent-a", "agent-b"],
        )
        assert aggregation.direction == "NO_TRADE"
        assert aggregation.signal_count == 0


class TestShadowEvaluationSummary:
    """ShadowEvaluationSummary (ST.12/O4) — Phase-2-Metriken auf Shadow-Records."""

    def test_instantiation_and_typed_fields(self):
        summary = ShadowEvaluationSummary(
            brier_score=0.23,
            directional_accuracy=0.8,
            expectancy=0.05,
            max_drawdown=0.12,
            sample_size=12,
        )
        assert summary.brier_score == 0.23
        assert summary.directional_accuracy == 0.8
        assert summary.expectancy == 0.05
        assert summary.max_drawdown == 0.12
        assert summary.sample_size == 12


class TestModelRegistry:
    """Alle 6 neuen Modelle sind Pydantic-Modelle/Enums mit voller Typisierung."""

    def test_models_are_pydantic_models(self):
        for model in (ShadowSessionState, ShadowTradingRecord, SignalAggregation, ShadowEvaluationSummary):
            assert issubclass(model, BaseModel)

    def test_models_have_no_untyped_fields(self):
        for model in (ShadowSessionState, ShadowTradingRecord, SignalAggregation, ShadowEvaluationSummary):
            for name, field_info in model.model_fields.items():
                assert field_info.annotation is not None, f"{model.__name__}.{name} ohne Annotation"
