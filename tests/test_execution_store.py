"""Tests für ExecutionLogStore."""

from __future__ import annotations

from pathlib import Path

from trading_harness.services.execution_store import ExecutionLogStore


class TestExecutionLogStoreBasic:
    """Grundlegende ExecutionLogStore-Tests."""

    def test_add_log(self):
        """Log wird hinzugefügt."""
        store = ExecutionLogStore()
        entry = store.add(
            decision_id="dec-1",
            run_id="run-1",
            symbol="BTCUSDT",
            side="LONG",
            status="SUBMITTED",
        )
        assert entry.decision_id == "dec-1"
        assert entry.run_id == "run-1"
        assert entry.symbol == "BTCUSDT"
        assert entry.side == "LONG"
        assert entry.status == "SUBMITTED"

    def test_get_all(self):
        """Alle Logs abrufen."""
        store = ExecutionLogStore()
        store.add("dec-1", "run-1", "BTCUSDT", "LONG", "SUBMITTED")
        store.add("dec-2", "run-1", "ETHUSDT", "SHORT", "REJECTED")
        all_logs = store.get_all()
        assert len(all_logs) == 2

    def test_get_by_decision_id(self):
        """Logs nach decision_id filtern."""
        store = ExecutionLogStore()
        store.add("dec-1", "run-1", "BTCUSDT", "LONG", "SUBMITTED")
        store.add("dec-2", "run-1", "ETHUSDT", "SHORT", "REJECTED")
        logs = store.get_by_decision_id("dec-1")
        assert len(logs) == 1
        assert logs[0]["decision_id"] == "dec-1"

    def test_get_by_run(self):
        """Logs nach run_id filtern."""
        store = ExecutionLogStore()
        store.add("dec-1", "run-1", "BTCUSDT", "LONG", "SUBMITTED")
        store.add("dec-2", "run-2", "ETHUSDT", "SHORT", "REJECTED")
        logs = store.get_by_run("run-1")
        assert len(logs) == 1
        assert logs[0]["run_id"] == "run-1"

    def test_count(self):
        """Anzahl der Logs."""
        store = ExecutionLogStore()
        assert store.count == 0
        store.add("dec-1", "run-1", "BTCUSDT", "LONG", "SUBMITTED")
        assert store.count == 1


class TestExecutionLogStorePersist:
    """Persistenz-Tests."""

    def test_persist_and_load(self, tmp_path: Path):
        """Logs werden persistiert und geladen."""
        db_path = str(tmp_path / "execution_logs.json")
        store1 = ExecutionLogStore(db_path=db_path)
        store1.add("dec-1", "run-1", "BTCUSDT", "LONG", "SUBMITTED")
        del store1  # Speicher freigeben

        # Neuer Store sollte geladene Logs haben
        store2 = ExecutionLogStore(db_path=db_path)
        assert store2.count == 1
        assert store2.get_all()[0]["decision_id"] == "dec-1"

    def test_no_credentials_in_logs(self, tmp_path: Path):
        """Credentials werden nie in Logs gespeichert."""
        db_path = str(tmp_path / "execution_logs_no_creds.json")
        store = ExecutionLogStore(db_path=db_path)
        store.add(
            decision_id="dec-1",
            run_id="run-1",
            symbol="BTCUSDT",
            side="LONG",
            status="SUBMITTED",
            error="connection_error",
        )
        # Nach Persistenz laden und prüfen
        all_logs = store.get_all()
        log_str = str(all_logs)
        assert "key" not in log_str.lower() or "api_key" not in log_str.lower()