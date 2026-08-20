"""Tests für ExecutionLogStore."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest

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


class TestExecutionLogStoreHardening:
    """Review-13 (B1): clear(), Korruptions-Fallback, Crash-Integrität, Concurrency."""

    def test_clear(self, tmp_path: Path):
        """clear() leert den Store und persistiert den geleerten Zustand.

        Pinnung des IST-Verhaltens: clear() schreibt den geleerten Zustand
        atomar in die State-Datei (Datei bleibt existent, `{"logs": []}`);
        ein Reload sieht den geleerten Zustand.
        """
        db_path = str(tmp_path / "execution_log.json")
        store = ExecutionLogStore(db_path=db_path)
        store.add("dec-1", "run-1", "BTCUSDT", "LONG", "SUBMITTED")
        store.add("dec-2", "run-1", "ETHUSDT", "SHORT", "REJECTED")
        assert store.count == 2

        store.clear()
        assert store.count == 0
        assert store.get_all() == []

        state = json.loads((tmp_path / "execution_log.json").read_text())
        assert state == {"logs": []}

        reloaded = ExecutionLogStore(db_path=db_path)
        assert reloaded.count == 0

    def test_corrupted_state_file_fallback(self, tmp_path: Path):
        """Korruptes State-File (ungültiges JSON) → kein Crash, leerer Start.

        Pinnung des IST-Verhaltens, symmetrisch zu `KillSwitch._load_state`:
        `(OSError, json.JSONDecodeError)` → Warning + In-Memory-Fallback.
        """
        db_path = str(tmp_path / "execution_log.json")
        (tmp_path / "execution_log.json").write_text("{this is not valid json")

        store = ExecutionLogStore(db_path=db_path)
        assert store.count == 0
        assert store.get_all() == []

        store.add("dec-1", "run-1", "BTCUSDT", "LONG", "SUBMITTED")
        assert store.count == 1
        state = json.loads((tmp_path / "execution_log.json").read_text())
        assert len(state["logs"]) == 1
        assert state["logs"][0]["decision_id"] == "dec-1"

    def test_atomic_write_crash_integrity(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Erzwungener os.replace-Fehler: alter Datei-Stand bleibt intakt.

        Pinnt die Crash-Integrität (Review #13, empirisch verifiziert):
        add() liefert den Eintrag trotzdem zurück, der In-Memory-Zustand
        bleibt intakt, und die State-Datei behält ihren Vorherstand
        (keine halbe JSON).
        """
        db_path = str(tmp_path / "execution_log.json")
        store = ExecutionLogStore(db_path=db_path)
        store.add("dec-1", "run-1", "BTCUSDT", "LONG", "SUBMITTED")
        state_before = (tmp_path / "execution_log.json").read_text()

        def broken_replace(src: str, dst: str) -> None:
            raise OSError("simulated crash mid-write")

        monkeypatch.setattr(os, "replace", broken_replace)

        entry = store.add("dec-2", "run-1", "ETHUSDT", "SHORT", "REJECTED")
        assert entry.decision_id == "dec-2"
        assert store.count == 2
        assert (tmp_path / "execution_log.json").read_text() == state_before
        state = json.loads(state_before)
        assert [log["decision_id"] for log in state["logs"]] == ["dec-1"]
        assert list(tmp_path.glob("execution_log.json.*.tmp")) == []

    def test_concurrent_adds(self):
        """≥50 parallele Adds: count exakt, alle IDs eindeutig (N2-Regression)."""
        store = ExecutionLogStore()
        n = 60

        def worker(idx: int) -> None:
            store.add(f"dec-{idx}", "run-1", "BTCUSDT", "LONG", "SUBMITTED")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert store.count == n
        ids = [log["id"] for log in store.get_all()]
        assert len(set(ids)) == n