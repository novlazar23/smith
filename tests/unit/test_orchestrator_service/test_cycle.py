"""Tests für die Zyklus-Logik des Orchestrator-Services (stubbed Pipeline/Provider)."""

from __future__ import annotations

import re
import time
from datetime import UTC, datetime

import numpy as np
import pytest
from apps.orchestrator_service import service as service_module
from apps.orchestrator_service.service import (
    OrchestratorService,
    OrchestratorServiceConfig,
    make_run_id,
)

from .conftest import (
    FakeConnection,
    FakeDB,
    StubPipeline,
    StubProvider,
    make_ohlcv,
    make_result,
)

BTC = "BTC/USDT"
ETH = "ETH/USDT"
RUN_ID_PATTERN = re.compile(r"^orch-\d{8}T\d{6}Z-BTC/USDT$")


def _service_with(
    config: OrchestratorServiceConfig,
    provider: StubProvider,
    db: FakeDB,
    pipeline: StubPipeline,
) -> OrchestratorService:
    """Baut einen Service mit injizierten Stubs."""
    return OrchestratorService(
        config=config,
        provider=provider,
        db=db,
        pipeline_factory=lambda: pipeline,
    )


class _FakeFlags:
    """Stellvertreter für die Feature-Flags-Singleton."""

    def __init__(self, *, live_enabled: bool) -> None:
        self._live_enabled = live_enabled

    def is_enabled(self, flag: str) -> bool:
        return flag == "live_trading_enabled" and self._live_enabled


class _FailingConnectionOnce(FakeConnection):
    """Fehlert nur beim ersten Commit (simuliert einen transienten DB-Fehler)."""

    def __init__(self) -> None:
        super().__init__()
        self._first = True

    def commit(self) -> None:
        if self._first:
            self._first = False
            raise RuntimeError("transient db error")
        super().commit()


class TestRunCycle:
    """Verhalten von run_cycle() mit gestubten Abhängigkeiten."""

    def test_persists_one_row_per_instrument(
        self,
        config: OrchestratorServiceConfig,
        stub_provider: StubProvider,
        fake_conn: FakeConnection,
        stub_pipeline: StubPipeline,
    ) -> None:
        """Pro Instrument wird genau eine shadow_decisions-Zeile persistiert."""
        stub_provider.windows = {BTC: make_ohlcv(200), ETH: make_ohlcv(200)}
        stub_pipeline.results = {
            BTC: make_result(confidence=0.0, first=3, second=3),
            ETH: make_result(confidence=0.0, first=3, second=3),
        }
        service = _service_with(config, stub_provider, FakeDB(fake_conn), stub_pipeline)

        persisted = service.run_cycle()

        assert persisted == 2
        assert len(fake_conn.executed) == 2
        assert fake_conn.commits == 2
        params = [p for (_, p) in fake_conn.executed]
        assert [p["instrument"] for p in params] == [BTC, ETH]
        for p in params:
            assert p["decision"] == "NO_TRADE"
            assert p["first_round_count"] == 3
            assert p["second_round_count"] == 3
            assert p["latency_ms"] >= 0.0
            assert p["confidence"] == 0.0
        assert RUN_ID_PATTERN.match(params[0]["run_id"]) is not None

    def test_run_id_format(self) -> None:
        """Run-ID folgt dem Muster orch-<UTC-Stempel>-<Instrument>."""
        moment = datetime(2026, 1, 1, 12, 30, 5, tzinfo=UTC)
        assert make_run_id(BTC, moment) == f"orch-20260101T123005Z-{BTC}"

    def test_pipeline_receives_market_data_and_shadow_ensemble(
        self,
        config: OrchestratorServiceConfig,
        stub_provider: StubProvider,
        fake_conn: FakeConnection,
        stub_pipeline: StubPipeline,
    ) -> None:
        """Die Pipeline erhält OHLCV-Daten und ein frisches Shadow-Ensemble."""
        stub_provider.windows = {BTC: make_ohlcv(200)}
        stub_pipeline.results = {BTC: make_result()}
        service = _service_with(config, stub_provider, FakeDB(fake_conn), stub_pipeline)

        service.run_cycle()

        assert len(stub_pipeline.calls) == 1
        call = stub_pipeline.calls[0]
        market_data = call["market_data"]
        assert set(market_data) == {"open", "high", "low", "close", "volume"}
        for key in market_data:
            assert len(market_data[key]) == 200
            assert market_data[key].dtype == np.float64
        agents = call["agents"]
        assert len(agents) == 4
        assert {agent.agent_id for agent in agents} == {
            "trend",
            "mean_reversion",
            "volatility_regime",
            "volume_conviction",
        }

    def test_writes_heartbeat_after_cycle(
        self,
        config: OrchestratorServiceConfig,
        stub_provider: StubProvider,
        fake_conn: FakeConnection,
        stub_pipeline: StubPipeline,
    ) -> None:
        """Nach dem Zyklus enthält die Heartbeat-Datei die Epoche in Sekunden."""
        stub_provider.windows = {BTC: make_ohlcv(200), ETH: make_ohlcv(200)}
        stub_pipeline.results = {BTC: make_result(), ETH: make_result()}
        service = _service_with(config, stub_provider, FakeDB(fake_conn), stub_pipeline)
        before = int(time.time())

        service.run_cycle()

        assert config.heartbeat_path.exists()
        written = int(config.heartbeat_path.read_text(encoding="utf-8").strip())
        assert before - 5 <= written <= before + 5
        assert stub_provider.requests == [(BTC, 200), (ETH, 200)]

    def test_logs_feature_flag_state(
        self,
        config: OrchestratorServiceConfig,
        stub_provider: StubProvider,
        fake_conn: FakeConnection,
        stub_pipeline: StubPipeline,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Der Zustand von live_trading_enabled wird pro Zyklus geloggt."""
        stub_provider.windows = {BTC: make_ohlcv(200), ETH: make_ohlcv(200)}
        stub_pipeline.results = {BTC: make_result(), ETH: make_result()}
        service = _service_with(config, stub_provider, FakeDB(fake_conn), stub_pipeline)
        monkeypatch.setattr(service_module, "feature_flags", _FakeFlags(live_enabled=False))

        with caplog.at_level("INFO"):
            service.run_cycle()

        assert "live_trading_enabled=False" in caplog.text

    def test_logs_shadow_cycle_line(
        self,
        config: OrchestratorServiceConfig,
        stub_provider: StubProvider,
        fake_conn: FakeConnection,
        stub_pipeline: StubPipeline,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Die Zeile 'Shadow-Cyklus fertig' erscheint pro Instrument."""
        stub_provider.windows = {BTC: make_ohlcv(200)}
        stub_pipeline.results = {BTC: make_result(confidence=0.42, reason="No active agents (all shadow)", first=4, second=4)}
        service = _service_with(config, stub_provider, FakeDB(fake_conn), stub_pipeline)

        with caplog.at_level("INFO"):
            service.run_cycle()

        assert re.search(
            r"Shadow-Cyklus fertig: BTC/USDT -> NO_TRADE \(confidence=0\.4200, 4/4 Agenten, \d+ ms\)",
            caplog.text,
        ) is not None


class TestSkipOnInsufficientData:
    """Zu wenige Kerzen → Instrument wird übersprungen."""

    def test_skips_when_too_few_candles(
        self,
        config: OrchestratorServiceConfig,
        stub_provider: StubProvider,
        fake_conn: FakeConnection,
        stub_pipeline: StubPipeline,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Weniger als min_candles Kerzen → Skip ohne Persistenz."""
        stub_provider.windows = {BTC: make_ohlcv(10), ETH: make_ohlcv(200)}
        stub_pipeline.results = {ETH: make_result()}
        service = _service_with(config, stub_provider, FakeDB(fake_conn), stub_pipeline)

        with caplog.at_level("WARNING"):
            persisted = service.run_cycle()

        assert persisted == 1
        assert len(fake_conn.executed) == 1
        assert fake_conn.executed[0][1]["instrument"] == ETH
        assert "BTC/USDT" in caplog.text
        assert "übersprungen" in caplog.text

    def test_skips_when_provider_returns_none(
        self,
        config: OrchestratorServiceConfig,
        stub_provider: StubProvider,
        fake_conn: FakeConnection,
        stub_pipeline: StubPipeline,
    ) -> None:
        """Keine Kerzen abrufbar (None) → Skip ohne Persistenz."""
        stub_provider.windows = {BTC: None, ETH: None}
        service = _service_with(config, stub_provider, FakeDB(fake_conn), stub_pipeline)

        assert service.run_cycle() == 0
        assert fake_conn.executed == []


class TestErrorIsolation:
    """Ein Fehler bei einem Instrument stoppt den Zyklus nicht."""

    def test_next_instrument_still_processed(
        self,
        config: OrchestratorServiceConfig,
        stub_provider: StubProvider,
        fake_conn: FakeConnection,
        stub_pipeline: StubPipeline,
    ) -> None:
        """Pipeline-Fehler bei BTC → Fehlerzeile, ETH wird trotzdem persistiert."""
        stub_provider.windows = {BTC: make_ohlcv(200), ETH: make_ohlcv(200)}
        # BTC fehlt im Stub → StubPipeline wirft für BTC eine RuntimeError
        stub_pipeline.results = {ETH: make_result()}
        service = _service_with(config, stub_provider, FakeDB(fake_conn), stub_pipeline)

        persisted = service.run_cycle()

        assert persisted == 2
        params = [p for (_, p) in fake_conn.executed]
        assert params[0]["instrument"] == BTC
        assert params[0]["decision"] == "error"
        assert "RuntimeError" in params[0]["reason"]
        assert params[0]["errors"] is not None
        assert "RuntimeError" in params[0]["errors"]
        assert params[1]["instrument"] == ETH
        assert params[1]["decision"] == "NO_TRADE"

    def test_error_row_not_fatal_when_persistence_fails(
        self,
        config: OrchestratorServiceConfig,
        stub_provider: StubProvider,
        fake_conn: FakeConnection,
        stub_pipeline: StubPipeline,
    ) -> None:
        """Scheitert auch die Fehlerzeile-Persistenz, überlebt der Zyklus."""
        stub_provider.windows = {BTC: make_ohlcv(200), ETH: make_ohlcv(200)}
        stub_pipeline.results = {}
        fake_conn.fail_commit = True
        service = _service_with(config, stub_provider, FakeDB(fake_conn), stub_pipeline)

        persisted = service.run_cycle()

        assert persisted == 0
        # Pro Instrument wurde eine Fehlerzeile versucht (execute), kein Commit erfolgreich
        assert len(fake_conn.executed) == 2
        assert fake_conn.commits == 0

    def test_persist_failure_on_normal_row_persists_error_row(
        self,
        config: OrchestratorServiceConfig,
        stub_provider: StubProvider,
        stub_pipeline: StubPipeline,
    ) -> None:
        """Commit-Fehler bei der normalen Zeile → danach wird eine Fehlerzeile persistiert."""
        failing = _FailingConnectionOnce()
        stub_provider.windows = {BTC: make_ohlcv(200)}
        stub_pipeline.results = {BTC: make_result()}
        service = _service_with(config, stub_provider, FakeDB(failing), stub_pipeline)

        persisted = service.run_cycle()

        assert persisted == 1
        assert len(failing.executed) == 2
        assert failing.executed[0][1]["decision"] == "NO_TRADE"
        assert failing.executed[1][1]["decision"] == "error"
