"""Tests für packages/live_data — Health, Reconnect, Quality, Failover, Gap (EPIC-16 WIP)."""

from __future__ import annotations

from typing import Any

import pytest
from packages.live_data import (
    AutoReconnector,
    ConnectionPool,
    ConnectionType,
    FailoverManager,
    FailoverState,
    FreshnessGate,
    GapDetectionGate,
    GapDetector,
    GapRecoveryEngine,
    GapReport,
    GapType,
    HealthMonitor,
    PriceSanityGate,
    QualityGateEvaluator,
    ReconnectConfig,
    ReconnectState,
    RecoveryState,
    VenueConfig,
    VolumeSanityGate,
)


class TestHealthMonitor:
    """HealthMonitor: Zustand, Latenz, Timeout-Schwelle, Pool."""

    def test_venue_state_registered(self) -> None:
        monitor = HealthMonitor(venues=["binance"])
        state = monitor.get_state("binance")
        assert state.venue == "binance"
        assert state.overall_healthy is True
        assert state.total_checks == 0
        assert monitor.venues == ["binance"]

    def test_unknown_venue_raises(self) -> None:
        monitor = HealthMonitor(venues=["binance"])
        with pytest.raises(ValueError, match="Unknown venue"):
            monitor.get_state("nope")

    async def test_check_venue_healthy_below_threshold(self) -> None:
        monitor = HealthMonitor(venues=["binance"])
        state = await monitor.check_venue("binance", ConnectionType.REST, latency_ms=10.0)
        assert state.overall_healthy is True
        assert state.total_checks == 1
        assert state.total_failures == 0
        assert state.failure_rate == 0.0

    async def test_check_venue_unhealthy_above_threshold(self) -> None:
        monitor = HealthMonitor(venues=["binance"])
        # REST-Timeout ist 5 s -> 4000 ms ist gesund, 6000 ms nicht
        assert (await monitor.check_venue("binance", ConnectionType.REST, latency_ms=4000.0)).overall_healthy is True
        state = await monitor.check_venue("binance", ConnectionType.REST, latency_ms=6000.0)
        assert state.overall_healthy is False
        assert state.total_failures == 1
        assert state.failure_rate == 0.5

    async def test_check_venue_uses_configured_timeouts(self) -> None:
        strict = HealthMonitor(venues=["binance"], ws_timeout=2.0, rest_timeout=1.0)
        assert (await strict.check_venue("binance", ConnectionType.WEBSOCKET, latency_ms=2500.0)).overall_healthy is False
        assert (await strict.check_venue("binance", ConnectionType.REST, latency_ms=1500.0)).overall_healthy is False

        relaxed = HealthMonitor(venues=["binance"], ws_timeout=10.0, rest_timeout=5.0)
        assert (await relaxed.check_venue("binance", ConnectionType.WEBSOCKET, latency_ms=2500.0)).overall_healthy is True
        assert (await relaxed.check_venue("binance", ConnectionType.REST, latency_ms=1500.0)).overall_healthy is True

    async def test_check_venue_none_latency_is_unhealthy(self) -> None:
        monitor = HealthMonitor(venues=["binance"])
        state = await monitor.check_venue("binance", ConnectionType.WEBSOCKET)
        assert state.overall_healthy is False

    async def test_check_venue_rejects_unknown_venue(self) -> None:
        monitor = HealthMonitor(venues=["binance"])
        with pytest.raises(ValueError, match="Unknown venue"):
            await monitor.check_venue("nope", ConnectionType.REST, latency_ms=10.0)

    def test_latency_tracker_percentiles(self) -> None:
        monitor = HealthMonitor(venues=["binance"])
        tracker = monitor.get_latency("binance", ConnectionType.REST)
        for value in (10.0, 20.0, 30.0, 40.0):
            tracker.record(value)
        assert tracker.latest() == 40.0
        assert tracker.average() == pytest.approx(25.0)
        # idx = int(4 * 50 / 100) = 2 -> 30.0
        assert tracker.percentile(50) == 30.0
        assert tracker.percentile(100) == 40.0

    def test_latency_tracker_empty_defaults(self) -> None:
        monitor = HealthMonitor(venues=["binance"])
        tracker = monitor.get_latency("binance", ConnectionType.REST)
        assert tracker.latest() == 0.0
        assert tracker.average() == 0.0
        assert tracker.percentile(50) == 0.0

    async def test_connection_pool_lifecycle(self) -> None:
        monitor = HealthMonitor(venues=["binance"], pool_max_size=2)
        pool = monitor.get_pool("binance")
        assert pool.max_size == 2
        await pool.create_connection(ConnectionType.REST, conn_id="c0")
        assert pool.size == 1
        assert pool.active_count == 1
        await pool.create_connection(ConnectionType.REST, conn_id="c1")
        await pool.create_connection(ConnectionType.REST, conn_id="c2")
        assert pool.size == 2  # älteste Verbindung evictet

    async def test_connection_pool_deactivate_and_evict(self) -> None:
        pool = ConnectionPool(venue="binance")
        await pool.create_connection(ConnectionType.WEBSOCKET, conn_id="w0")
        assert pool.active_count == 1
        await pool.mark_inactive("w0")
        assert pool.active_count == 0
        await pool.evict("w0")
        assert pool.size == 0


class TestAutoReconnector:
    """AutoReconnector: Backoff-Übergänge, State, Max-Attempts."""

    @staticmethod
    def _recording_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
        delays: list[float] = []

        async def fake_sleep(delay: float) -> None:
            delays.append(delay)

        monkeypatch.setattr("asyncio.sleep", fake_sleep)
        return delays

    async def test_backoff_state_transitions_on_repeated_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        delays = self._recording_sleep(monkeypatch)
        states: list[ReconnectState] = []
        config = ReconnectConfig(
            max_attempts=4,
            base_delay=1.0,
            max_delay=100.0,
            jitter_factor=0.0,
            on_state_change=lambda state, venue: states.append(state),
        )
        rec = AutoReconnector(
            venue="binance",
            config=config,
            connect_hook=lambda: False,  # scheitert immer
        )
        result = await rec.run()

        assert result is False
        assert rec.state is ReconnectState.FAILED
        assert rec.is_connected is False
        assert rec.failure_count == 4
        assert rec.event_count == 4
        # Exponentieller Backoff: 1, 2, 4, 8
        assert delays == [1.0, 2.0, 4.0, 8.0]
        # State-Übergänge: CONNECTING -> RECONNECTING (x4) -> FAILED
        assert states[0] is ReconnectState.CONNECTING
        assert states.count(ReconnectState.RECONNECTING) == 4
        assert states[-1] is ReconnectState.FAILED

    async def test_backoff_delay_capped_at_max(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        delays = self._recording_sleep(monkeypatch)
        config = ReconnectConfig(
            max_attempts=3,
            base_delay=10.0,
            max_delay=15.0,
            jitter_factor=0.0,
        )
        rec = AutoReconnector(venue="v", config=config, connect_hook=lambda: False)
        await rec.run()
        # 10, dann 20 -> cap 15, dann 40 -> cap 15
        assert delays == [10.0, 15.0, 15.0]

    async def test_success_sets_connected_state(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._recording_sleep(monkeypatch)
        calls = {"n": 0}

        def flaky_hook() -> bool:
            calls["n"] += 1
            return calls["n"] >= 2  # zweiter Versuch erfolgreich

        config = ReconnectConfig(
            max_attempts=5, base_delay=1.0, max_delay=100.0, jitter_factor=0.0
        )
        rec = AutoReconnector(venue="v", config=config, connect_hook=flaky_hook)
        result = await rec.run()

        assert result is True
        assert rec.state is ReconnectState.CONNECTED
        assert rec.is_connected is True
        # Der erfolgreiche Versuch wurde mit success=True aufgezeichnet
        assert any(e.success for e in rec.events)
        assert rec.last_delay == pytest.approx(2.0)

    async def test_success_records_event_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._recording_sleep(monkeypatch)
        rec = AutoReconnector(
            venue="v",
            config=ReconnectConfig(max_attempts=1, base_delay=1.0, jitter_factor=0.0),
            connect_hook=lambda: True,
        )
        result = await rec.run()
        assert result is True
        assert rec.event_count == 1
        assert rec.success_count == 1
        assert rec.failure_count == 0
        assert rec.events[0].success is True

    async def test_reset_returns_to_idle(self) -> None:
        rec = AutoReconnector(
            venue="v",
            config=ReconnectConfig(max_attempts=1, base_delay=1.0),
            connect_hook=lambda: False,
        )
        assert rec.state is ReconnectState.IDLE
        await rec.reset()
        assert rec.state is ReconnectState.IDLE
        assert rec.event_count == 0
        assert rec.is_connected is False


class TestQualityGates:
    """Einzelne Quality-Gates (synchronous check())."""

    def test_freshness_gate(self) -> None:
        gate = FreshnessGate(max_age_seconds=30.0)
        assert gate.check(100.0, 90.0, "BTC", "binance") == []
        violations = gate.check(100.0, 60.0, "BTC", "binance")
        assert len(violations) == 1
        assert violations[0].gate_name == "freshness"
        # age=40s > 30s aber < 60s (2x) -> warn
        assert violations[0].severity == "warn"
        # age=95s > 60s (2x) -> fail
        assert gate.check(100.0, 5.0, "BTC", "binance")[0].severity == "fail"

    def test_gap_detection_gate(self) -> None:
        gate = GapDetectionGate(max_gap_size=5)
        assert gate.check(100, 100, "BTC", "binance") == []
        # kleine Lücke (2 <= 5) -> warn
        small = gate.check(98, 100, "BTC", "binance")
        assert small[0].severity == "warn"
        # große Lücke (10 > 5) -> fail
        big = gate.check(90, 100, "BTC", "binance")
        assert big[0].severity == "fail"
        assert big[0].details["gap_size"] == 10

    def test_gap_detection_gate_ignores_overshoot_for_now(self) -> None:
        gate = GapDetectionGate(max_gap_size=5)
        assert gate.check(110, 100, "BTC", "binance") == []

    def test_price_sanity_gate(self) -> None:
        gate = PriceSanityGate(deviation_pct=25.0, min_price=0.01)
        assert gate.check(100.0, 99.0, "BTC", "binance") == []
        # unter Minimum -> fail
        assert gate.check(0.001, 1.0, "BTC", "binance")[0].severity == "fail"
        # 30% Abweichung > 25% -> warn
        assert gate.check(130.0, 100.0, "BTC", "binance")[0].severity == "warn"
        # 60% Abweichung > 50% (2x) -> fail
        assert gate.check(160.0, 100.0, "BTC", "binance")[0].severity == "fail"
        # kein Prev-Preis -> nur Min-Preis-Check
        assert gate.check(100.0, None, "BTC", "binance") == []

    def test_volume_sanity_gate(self) -> None:
        gate = VolumeSanityGate(spike_factor=10.0, min_volume=1.0)
        assert gate.check(10.0, 10.0, "BTC", "binance") == []
        # unter Minimum -> warn
        assert gate.check(0.5, 10.0, "BTC", "binance")[0].severity == "warn"
        # 15x > 10x -> warn
        assert gate.check(150.0, 10.0, "BTC", "binance")[0].severity == "warn"
        # 25x > 20x (2x) -> fail
        assert gate.check(250.0, 10.0, "BTC", "binance")[0].severity == "fail"

    async def test_evaluator_passes_clean_data(self) -> None:
        evaluator = QualityGateEvaluator(
            freshness_window=30.0, max_gap_size=5,
            price_deviation_pct=25.0, volume_spike_factor=10.0,
        )
        result = await evaluator.evaluate(
            instrument="BTC/USDT", venue="binance",
            current_time=100.0, last_event_time=99.0,
            sequence=100, expected_sequence=100,
            price=100.0, prev_price=99.5,
            volume=10.0, avg_volume=10.0,
        )
        assert result.passed is True
        assert result.violations == []
        assert result.fail_count == 0
        assert result.warn_count == 0

    async def test_evaluator_fails_on_stale_data(self) -> None:
        evaluator = QualityGateEvaluator(freshness_window=30.0)
        result = await evaluator.evaluate(
            instrument="BTC/USDT", venue="binance",
            current_time=100.0, last_event_time=5.0,
        )
        assert result.passed is False
        assert any(v.gate_name == "freshness" for v in result.violations)

    async def test_evaluator_warn_only_still_passes(self) -> None:
        evaluator = QualityGateEvaluator(freshness_window=30.0)
        # age=40s -> warn (nicht fail)
        result = await evaluator.evaluate(
            instrument="BTC/USDT", venue="binance",
            current_time=100.0, last_event_time=60.0,
        )
        assert result.passed is True
        assert result.warn_count == 1
        assert result.fail_count == 0


class TestFailoverManager:
    """FailoverManager: Primär/Backup-Entscheidungen."""

    @staticmethod
    def _manager() -> FailoverManager:
        return FailoverManager(
            venues=[
                VenueConfig(name="binance", role="primary", priority=1),
                VenueConfig(name="okx", role="backup", priority=2),
            ],
            health_threshold=0.5,
            failover_cooldown=0.0,
        )

    def test_venue_config_rejects_invalid_role(self) -> None:
        with pytest.raises(ValueError, match="role"):
            VenueConfig(name="x", role="middle")

    def test_venue_config_roles(self) -> None:
        assert VenueConfig(name="a", role="primary").is_primary
        assert VenueConfig(name="b", role="backup").is_backup

    async def test_healthy_primary_stays_normal(self) -> None:
        manager = self._manager()
        decision = await manager.evaluate()
        assert decision.state is FailoverState.NORMAL
        assert decision.active_venue == "binance"
        assert decision.new_venue is None
        assert manager.state is FailoverState.NORMAL

    async def test_no_primary_is_error(self) -> None:
        manager = FailoverManager(
            venues=[VenueConfig(name="okx", role="backup")],
            health_threshold=0.5,
        )
        decision = await manager.evaluate()
        assert decision.state is FailoverState.ERROR
        assert manager.state is FailoverState.ERROR

    async def test_unhealthy_primary_fails_over_to_backup(self) -> None:
        manager = self._manager()
        manager.update_health("binance", 0.1)
        decision = await manager.evaluate()
        assert decision.state is FailoverState.FAILED_OVER
        assert decision.active_venue == "binance"
        assert decision.new_venue == "okx"

    async def test_complete_switch_updates_active_venue(self) -> None:
        manager = self._manager()
        manager.update_health("binance", 0.1)
        decision = await manager.evaluate()
        assert decision.new_venue == "okx"
        manager.complete_switch("okx")
        assert manager.active_venue == "okx"
        assert manager.state is FailoverState.FAILED_OVER
        # bereits auf Backup -> kein weiterer Switch
        again = await manager.evaluate()
        assert again.new_venue is None
        assert again.state is FailoverState.FAILED_OVER

    async def test_switch_back_to_recovered_primary(self) -> None:
        manager = self._manager()
        manager.update_health("binance", 0.1)
        await manager.evaluate()
        manager.complete_switch("okx")
        assert manager.state is FailoverState.FAILED_OVER

        manager.update_health("binance", 0.9)
        decision = await manager.evaluate()
        assert decision.state is FailoverState.NORMAL
        assert decision.new_venue == "binance"
        assert manager.state is FailoverState.SWITTING_BACK

        manager.complete_switch("binance")
        assert manager.active_venue == "binance"
        assert manager.state is FailoverState.NORMAL

    async def test_no_backup_available_is_degraded(self) -> None:
        manager = FailoverManager(
            venues=[VenueConfig(name="binance", role="primary")],
            health_threshold=0.5,
        )
        manager.update_health("binance", 0.1)
        decision = await manager.evaluate()
        assert decision.state is FailoverState.DEGRADED
        assert decision.new_venue is None

    async def test_all_unhealthy_escalates_to_error(self) -> None:
        manager = FailoverManager(
            venues=[
                VenueConfig(name="binance", role="primary", priority=1),
                VenueConfig(name="okx", role="backup", priority=2),
            ],
            health_threshold=0.5,
            max_consistency_failures=2,
        )
        manager.update_health("binance", 0.1)
        manager.update_health("okx", 0.1)
        first = await manager.evaluate()
        assert first.state is FailoverState.DEGRADED
        assert manager.state is FailoverState.DEGRADED
        second = await manager.evaluate()
        assert second.state is FailoverState.ERROR
        assert manager.state is FailoverState.ERROR
        assert manager.consistency_failures == 2

    def test_update_health_clamps_and_rejects_unknown(self) -> None:
        manager = self._manager()
        manager.update_health("binance", 5.0)
        assert manager.health_scores["binance"] == 1.0
        with pytest.raises(ValueError, match="Unknown venue"):
            manager.update_health("nope", 0.5)


class TestGapRecovery:
    """GapDetector / GapReport / GapRecoveryEngine."""

    def test_gap_report_severity_levels(self) -> None:
        def report(missing: int) -> GapReport:
            return GapReport(
                instrument="BTC", venue="binance", gap_type=GapType.SEQUENCE,
                first_gap_start=1, last_gap_end=missing,
                gap_count=1 if missing else 0, total_missing=missing,
            )

        assert report(0).severity == "none"
        assert report(2).severity == "low"
        assert report(8).severity == "medium"
        assert report(20).severity == "high"
        assert report(0).has_gaps is False
        assert report(8).needs_recovery is True
        assert report(2).needs_recovery is False  # low braucht keine Recovery

    def test_detect_no_gap(self) -> None:
        detector = GapDetector(gap_type=GapType.SEQUENCE)
        assert detector.detect("BTC", "binance", 100, 101) == []
        assert detector.detect("BTC", "binance", 105, 100) == []  # rückwärts

    def test_detect_single_gap_range(self) -> None:
        detector = GapDetector(gap_type=GapType.SEQUENCE)
        reports = detector.detect("BTC", "binance", 100, 110)
        assert len(reports) == 1
        report = reports[0]
        assert report.first_gap_start == 101
        assert report.last_gap_end == 109
        assert report.total_missing == 9
        assert report.gap_type is GapType.SEQUENCE
        assert report.has_gaps is True

    def test_detect_continuous_multiple_gaps(self) -> None:
        detector = GapDetector(gap_type=GapType.TRADE)
        reports = detector.detect_continuous("BTC", "binance", 10, 10, [1, 2, 4, 5, 9])
        ranges = [(r.first_gap_start, r.last_gap_end, r.total_missing) for r in reports]
        assert ranges == [(3, 3, 1), (6, 8, 3)]

    async def test_engine_detect_gaps_stores_report(self) -> None:
        engine = GapRecoveryEngine()
        reports = await engine.detect_gaps("BTC", "binance", 100, 120)
        assert len(reports) == 1
        assert engine.last_report is not None
        assert engine.last_report.total_missing == 19
        assert engine.state is RecoveryState.IDLE  # nach Detection zurück Idle

    async def test_engine_replay_historical(self) -> None:
        engine = GapRecoveryEngine(replay_limit=100)
        seen: list[dict[str, Any]] = []
        events = await engine.replay_historical(
            "BTC", "binance", 10, 12, callback=lambda e: seen.append(e)
        )
        assert [e["sequence"] for e in events] == [10, 11, 12]
        assert [e["sequence"] for e in seen] == [10, 11, 12]
        assert engine.state is RecoveryState.COMPLETED

    async def test_engine_replay_respects_limit(self) -> None:
        engine = GapRecoveryEngine(replay_limit=2)
        events = await engine.replay_historical("BTC", "binance", 1, 5)
        assert [e["sequence"] for e in events] == [1, 2]
        assert engine.state is RecoveryState.COMPLETED

    async def test_engine_reset_orderbook_by_threshold_and_force(self) -> None:
        engine = GapRecoveryEngine(max_gap_threshold=10)
        # ohne Report -> kein Reset
        assert await engine.reset_orderbook("BTC", "binance") is False
        # Lücke 8 < 10 -> kein Reset
        await engine.detect_gaps("BTC", "binance", 100, 109)
        assert await engine.reset_orderbook("BTC", "binance") is False
        # force=True -> Reset
        assert await engine.reset_orderbook("BTC", "binance", force=True) is True
        # Lücke 19 > 10 -> Reset ohne force
        await engine.detect_gaps("BTC", "binance", 100, 120)
        assert await engine.reset_orderbook("BTC", "binance") is True
