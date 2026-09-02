"""Tests für die Zyklus-Logik des Demo-Traders (Fehlerisolierung, Snapshot, Log)."""

from __future__ import annotations

import re
import time

import pytest
from apps.demo_trader.service import (
    INSERT_DEMO_TRADE,
    UPSERT_DEMO_ACCOUNT,
    DemoTraderConfig,
    make_run_id,
)

from .conftest import (
    BTC,
    ETH,
    FakeConnection,
    StubCandleSource,
    StubPipeline,
    make_ohlcv,
    make_result,
    make_trader,
)


class TestRunCycle:
    """Verhalten von run_cycle() mit gestubten Abhängigkeiten."""

    def test_cycle_persists_trades_snapshot_and_heartbeat(
        self, config: DemoTraderConfig, fake_conn: FakeConnection
    ) -> None:
        """Ein Zyklus: Trade-Zeile, Account-Upsert, Heartbeat-Datei."""
        provider = StubCandleSource(
            {BTC: make_ohlcv(100, start_price=100.0), ETH: make_ohlcv(100, start_price=50.0)}
        )
        pipeline = StubPipeline(
            {
                BTC: make_result(decision="LONG_BIAS", confidence=0.6),
                ETH: make_result(decision="NO_TRADE", confidence=0.0),
            }
        )
        trader = make_trader(config, provider, fake_conn, pipeline)

        executed = trader.run_cycle()

        assert executed == 1
        statements = [stmt for (stmt, _) in fake_conn.executed]
        assert statements.count(INSERT_DEMO_TRADE) == 1
        assert statements.count(UPSERT_DEMO_ACCOUNT) == 1
        assert fake_conn.commits == 2
        assert config.heartbeat_path.exists()
        written = int(config.heartbeat_path.read_text(encoding="utf-8").strip())
        assert written >= int(time.time()) - 5

    def test_run_id_format(self) -> None:
        """Run-ID folgt dem Muster demo-<UTC-Stempel>-<Instrument>."""
        from datetime import UTC, datetime

        moment = datetime(2026, 1, 1, 12, 30, 5, tzinfo=UTC)
        assert make_run_id(BTC, moment) == f"demo-20260101T123005Z-{BTC}"

    def test_pipeline_receives_active_ensemble(
        self, config: DemoTraderConfig, fake_conn: FakeConnection
    ) -> None:
        """Die Pipeline erhält OHLCV-Daten und ein frisches ACTIVE-Ensemble."""
        provider = StubCandleSource({BTC: make_ohlcv(200, start_price=100.0)})
        pipeline = StubPipeline({BTC: make_result()})
        trader = make_trader(config, provider, fake_conn, pipeline)

        trader._run_instrument(BTC)

        call = pipeline.calls[0]
        market_data = call["market_data"]
        assert set(market_data) == {"open", "high", "low", "close", "volume"}
        agents = call["agents"]
        assert len(agents) == 4
        assert {agent.agent_id for agent in agents} == {
            "trend",
            "mean_reversion",
            "volatility_regime",
            "volume_conviction",
        }
        for agent in agents:
            assert agent._agent.config.status.value == "active"

    def test_skips_instrument_without_candles(
        self, config: DemoTraderConfig, fake_conn: FakeConnection, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Keine Kerzen für das Venue → Skip, aber Snapshot und Heartbeat laufen."""
        provider = StubCandleSource({BTC: None, ETH: make_ohlcv(100, start_price=50.0)})
        pipeline = StubPipeline({ETH: make_result()})
        trader = make_trader(config, provider, fake_conn, pipeline)

        with caplog.at_level("WARNING"):
            executed = trader.run_cycle()

        assert executed == 0
        assert "übersprungen" in caplog.text
        statements = [stmt for (stmt, _) in fake_conn.executed]
        assert statements == [UPSERT_DEMO_ACCOUNT]
        assert config.heartbeat_path.exists()

    def test_logs_demo_cycle_line_per_instrument(
        self, config: DemoTraderConfig, fake_conn: FakeConnection, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Pro Instrument erscheint die Zeile 'Demo-Zyklus: ...' im vorgegebenen Format."""
        provider = StubCandleSource(
            {BTC: make_ohlcv(100, start_price=100.0), ETH: make_ohlcv(100, start_price=50.0)}
        )
        pipeline = StubPipeline(
            {
                BTC: make_result(decision="LONG_BIAS", confidence=0.62),
                ETH: make_result(decision="NO_TRADE", confidence=0.0),
            }
        )
        trader = make_trader(config, provider, fake_conn, pipeline)

        with caplog.at_level("INFO"):
            trader.run_cycle()

        assert re.search(
            r"Demo-Zyklus: BTC/USDT -> LONG_BIAS \(confidence=0\.6200, "
            r"Trade=BUY [\d.]+@[\d.]+\)",
            caplog.text,
        ) is not None
        assert re.search(
            r"Demo-Zyklus: ETH/USDT -> NO_TRADE \(confidence=0\.0000, Trade=—\)",
            caplog.text,
        ) is not None


class TestErrorIsolation:
    """Ein Fehler bei einem Instrument stoppt den Zyklus nicht."""

    def test_next_instrument_still_processed(
        self, config: DemoTraderConfig, fake_conn: FakeConnection
    ) -> None:
        """Pipeline-Fehler bei BTC → ETH wird trotzdem bearbeitet, Snapshot wird upgepusht."""
        provider = StubCandleSource(
            {BTC: make_ohlcv(100, start_price=100.0), ETH: make_ohlcv(100, start_price=50.0)}
        )
        # BTC fehlt im Stub → StubPipeline wirft für BTC eine RuntimeError
        pipeline = StubPipeline({ETH: make_result(decision="LONG_BIAS", confidence=0.6)})
        trader = make_trader(config, provider, fake_conn, pipeline)

        executed = trader.run_cycle()

        assert executed == 1
        statements = [stmt for (stmt, _) in fake_conn.executed]
        assert statements == [INSERT_DEMO_TRADE, UPSERT_DEMO_ACCOUNT]
        assert fake_conn.executed[0][1]["instrument"] == ETH
        assert config.heartbeat_path.exists()

    def test_provider_error_isolated(
        self, config: DemoTraderConfig, fake_conn: FakeConnection
    ) -> None:
        """ClickHouse-Fehler bei einem Instrument → das nächste läuft trotzdem."""
        provider = StubCandleSource(
            {BTC: make_ohlcv(100, start_price=100.0), ETH: make_ohlcv(100, start_price=50.0)}
        )
        provider.fail_for = (BTC,)
        pipeline = StubPipeline({ETH: make_result()})
        trader = make_trader(config, provider, fake_conn, pipeline)

        executed = trader.run_cycle()

        assert executed == 0
        # ETH wurde analysiert (kein Trade, aber kein Abbruch), Snapshot upgepusht
        assert [stmt for (stmt, _) in fake_conn.executed] == [UPSERT_DEMO_ACCOUNT]
        assert config.heartbeat_path.exists()

    def test_snapshot_failure_not_fatal(
        self, config: DemoTraderConfig, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Scheitert der Account-Upsert, überlebt der Zyklus (Heartbeat wird geschrieben)."""
        provider = StubCandleSource({BTC: make_ohlcv(100, start_price=100.0)})
        pipeline = StubPipeline({BTC: make_result()})
        failing_conn = FakeConnection()
        failing_conn.fail_commit = True
        trader = make_trader(config, provider, failing_conn, pipeline)

        with caplog.at_level("ERROR"):
            executed = trader.run_cycle()

        assert executed == 0
        assert failing_conn.commits == 0
        assert config.heartbeat_path.exists()
        assert "Account-Snapshot nicht persistierbar" in caplog.text

    def test_trade_persistence_failure_isolated_per_instrument(
        self, config: DemoTraderConfig, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Commit-Fehler beim Trade persistieren → Fehler wird isoliert, Zyklus läuft weiter."""
        provider = StubCandleSource(
            {BTC: make_ohlcv(100, start_price=100.0), ETH: make_ohlcv(100, start_price=50.0)}
        )

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

        conn = _FailingConnectionOnce()
        pipeline = StubPipeline(
            {
                BTC: make_result(decision="LONG_BIAS", confidence=0.6),
                ETH: make_result(decision="LONG_BIAS", confidence=0.6),
            }
        )
        trader = make_trader(config, provider, conn, pipeline)

        with caplog.at_level("ERROR"):
            executed = trader.run_cycle()

        # BTC-Trade scheitert am Commit → wird als Fehler geloggt, ETH-Trade gelingt.
        # Der in-Process-PaperZustand (Position BTC) bleibt erhalten; die
        # Trade-Zeile für BTC ist nur nicht persistiert.
        assert executed == 1
        assert "fehlgeschlagen" in caplog.text
        assert BTC in trader.account.positions
        assert ETH in trader.account.positions
        statements = [stmt for (stmt, _) in conn.executed]
        assert statements.count(INSERT_DEMO_TRADE) == 2
        assert statements.count(UPSERT_DEMO_ACCOUNT) == 1
