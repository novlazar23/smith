"""Tests für Signal-Handling und den Zyklus-Loop (run_service)."""

from __future__ import annotations

import signal
import threading
import time

from apps.demo_trader import service as service_module
from apps.demo_trader.service import DemoTraderConfig, run_service


class _CountingTrader:
    """Minimaler DemoTrader-Ersatz, der durchgeführte Zyklen zählt."""

    def __init__(
        self,
        interval_seconds: float = 0.05,
        fail_cycles: tuple[int, ...] = (),
    ) -> None:
        self.config = DemoTraderConfig(
            interval_seconds=interval_seconds,
            instruments=("BTC/USDT",),
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
            assert callable(handler)
            handler(signal.SIGINT, None)  # type: ignore[operator]
            assert stop.is_set()
        finally:
            _restore_signal_handlers(old_term, old_int)


class TestRunLoop:
    """run_service(): erster Zyklus sofort, dann im Intervall, Stop wird beachtet."""

    def test_first_cycle_runs_immediately_and_stops(self) -> None:
        """Vorab gesetztes Stop-Flag → genau ein Zyklus, sofortiger Exit."""
        trader = _CountingTrader(interval_seconds=60.0)
        stop = threading.Event()
        stop.set()

        start = time.monotonic()
        run_service(trader, stop.is_set)
        elapsed = time.monotonic() - start

        assert trader.cycle_count == 1
        assert elapsed < 5.0

    def test_loop_survives_cycle_exception(self) -> None:
        """Ein fehlgeschlagener Zyklus beendet den Loop nicht."""
        trader = _CountingTrader(interval_seconds=0.02, fail_cycles=(1,))
        stop = threading.Event()
        threading.Timer(0.15, stop.set).start()

        run_service(trader, stop.is_set)

        assert trader.cycle_count >= 2

    def test_loop_stops_after_interval_cycles(self) -> None:
        """Ohne Stop läuft der Loop mehrere Zyklen im konfigurierten Abstand."""
        trader = _CountingTrader(interval_seconds=0.02)
        stop = threading.Event()
        threading.Timer(0.12, stop.set).start()

        run_service(trader, stop.is_set)

        assert trader.cycle_count >= 3
