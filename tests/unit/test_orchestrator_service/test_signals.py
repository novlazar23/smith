"""Tests für Signal-Handling und den Zyklus-Loop (run_service)."""

from __future__ import annotations

import signal
import threading
import time
from pathlib import Path

from apps.orchestrator_service import service as service_module
from apps.orchestrator_service.service import (
    OrchestratorServiceConfig,
    build_ensemble,
    build_market_data,
    run_service,
)
from packages.orchestrator.pipeline import OrchestratorPipeline

from .conftest import FakeConnection, FakeDB, StubProvider, make_ohlcv

TEST_HEARTBEAT = Path("/tmp/orch-test-heartbeat")


class _CountingService:
    """Minimaler Service-Ersatz, der durchgeführte Zyklen zählt."""

    def __init__(
        self,
        interval_seconds: float = 0.05,
        *,
        fail_cycles: tuple[int, ...] = (),
    ) -> None:
        self.config = OrchestratorServiceConfig(
            interval_seconds=interval_seconds,
            instruments=("BTC/USDT",),
            heartbeat_path=TEST_HEARTBEAT,
        )
        self.cycle_count = 0
        self._fail_cycles = fail_cycles

    def run_cycle(self) -> int:
        self.cycle_count += 1
        if self.cycle_count in self._fail_cycles:
            raise RuntimeError("boom")
        return 0


def _restore_signal_handlers(old_term: object, old_int: object) -> None:
    """Stellt die ursprünglichen Signal-Handler wieder her."""
    signal.signal(signal.SIGTERM, old_term)  # type: ignore[arg-type]
    signal.signal(signal.SIGINT, old_int)  # type: ignore[arg-type]


class TestSignalHandling:
    """SIGTERM/SIGINT setzen den Stop-Event (graceful Shutdown)."""

    def test_sigterm_handler_sets_stop_event(self) -> None:
        """Der installierte SIGTERM-Handler setzt den Stop-Event."""
        stop = threading.Event()
        old_term = signal.getsignal(signal.SIGTERM)
        old_int = signal.getsignal(signal.SIGINT)
        try:
            service_module.install_signal_handlers(stop)
            handler = signal.getsignal(signal.SIGTERM)
            assert callable(handler)
            handler(signal.SIGTERM, None)  # type: ignore[operator]
            assert stop.is_set()
        finally:
            _restore_signal_handlers(old_term, old_int)

    def test_sigint_handler_sets_stop_event(self) -> None:
        """Der installierte SIGINT-Handler setzt denselben Stop-Event."""
        stop = threading.Event()
        old_term = signal.getsignal(signal.SIGTERM)
        old_int = signal.getsignal(signal.SIGINT)
        try:
            service_module.install_signal_handlers(stop)
            handler = signal.getsignal(signal.SIGINT)
            handler(signal.SIGINT, None)  # type: ignore[operator]
            assert stop.is_set()
        finally:
            _restore_signal_handlers(old_term, old_int)


class TestRunLoop:
    """run_service(): erster Zyklus sofort, dann im Intervall, Stop wird beachtet."""

    def test_first_cycle_runs_immediately_and_stops(self) -> None:
        """Vorab gesetztes Stop-Flag → genau ein Zyklus, sofortiger Exit."""
        service = _CountingService(interval_seconds=60.0)
        stop = threading.Event()
        stop.set()

        start = time.monotonic()
        run_service(service, stop.is_set)
        elapsed = time.monotonic() - start

        assert service.cycle_count == 1
        assert elapsed < 5.0

    def test_loop_survives_cycle_exception(self) -> None:
        """Ein fehlgeschlagener Zyklus beendet den Loop nicht."""
        service = _CountingService(interval_seconds=0.02, fail_cycles=(1,))
        stop = threading.Event()
        threading.Timer(0.15, stop.set).start()

        run_service(service, stop.is_set)

        assert service.cycle_count >= 2

    def test_loop_stops_after_interval_cycles(self) -> None:
        """Ohne Stop läuft der Loop mehrere Zyklen im konfigurierten Abstand."""
        service = _CountingService(interval_seconds=0.02)
        stop = threading.Event()
        threading.Timer(0.12, stop.set).start()

        run_service(service, stop.is_set)

        assert service.cycle_count >= 3


class TestWiring:
    """Aufbau von Service und Ensemble (ohne Netz/DB)."""

    def test_build_service_with_injected_deps(self) -> None:
        """Injizierte Provider/DB werden verwendet, keine Env-Nachschüsse."""
        config = OrchestratorServiceConfig(
            interval_seconds=900.0,
            instruments=("BTC/USDT",),
            heartbeat_path=TEST_HEARTBEAT,
        )
        provider = StubProvider({})
        db = FakeDB(FakeConnection())

        service = service_module.build_service(config=config, provider=provider, db=db)

        assert service.config is config

    def test_real_pipeline_runs_full_shadow_cycle(self) -> None:
        """Echter Pipeline-Durchlauf (echte Agenten, keine Stubs) bleibt fehlerfrei.

        Alle Agenten sind SHADOW → der Konsens liefert deterministisch
        NO_TRADE ('No active agents (all shadow)'); beide Runden laufen
        vollständig durch (First Round, Seal, Second Round, Konsens).
        """
        window = make_ohlcv(200)
        agents = build_ensemble("BTC/USDT", "15m")
        result = OrchestratorPipeline().run(
            run_id="orch-20260101T000000Z-BTC/USDT",
            instrument="BTC/USDT",
            agents=agents,
            market_data=build_market_data(window),
        )

        assert result.errors == []
        assert result.decision == "NO_TRADE"
        assert result.consensus is not None
        assert result.consensus.reason == "No active agents (all shadow)"
        assert len(result.first_round_reports) == 3
        assert len(result.second_round_reports) == 3
        assert len(result.seal_records) == 3
