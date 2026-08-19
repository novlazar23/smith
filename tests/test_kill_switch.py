"""Tests für KillSwitch."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from trading_harness.services.kill_switch import KillSwitch


class TestKillSwitchBasic:
    """Grundlegende KillSwitch-Tests."""

    def test_initial_state_disabled(self):
        """Standardmäßig deaktiviert."""
        ks = KillSwitch(enabled=False)
        assert ks.is_active() is False

    def test_initial_state_enabled(self):
        """Explizit aktivierbar."""
        ks = KillSwitch(enabled=True)
        assert ks.is_active() is True

    def test_activate(self):
        """Activate setzt auf True."""
        ks = KillSwitch(enabled=False)
        ks.activate()
        assert ks.is_active() is True

    def test_deactivate(self):
        """Deactivate setzt auf False."""
        ks = KillSwitch(enabled=True)
        ks.deactivate()
        assert ks.is_active() is False

    def test_toggle_multiple_times(self):
        """Mehrfaches Umschalten."""
        ks = KillSwitch(enabled=False)
        ks.activate()
        assert ks.is_active() is True
        ks.deactivate()
        assert ks.is_active() is False
        ks.activate()
        assert ks.is_active() is True


class TestKillSwitchPersistence:
    """Persistenz-Tests."""

    def test_persist_and_load(self, tmp_path: Path):
        """Zustand wird persistiert und bei Neustart geladen."""
        db_path = str(tmp_path / "kill_switch.json")

        # Erster Switch — aktiv
        ks1 = KillSwitch(enabled=True, db_path=db_path)
        assert ks1.is_active() is True
        ks1.deactivate()

        # Neuer Switch — sollte geladenen Zustand haben
        ks2 = KillSwitch(enabled=False, db_path=db_path)
        assert ks2.is_active() is False  # sollte von Persistenz geladen sein

    def test_persist_toggle_count(self, tmp_path: Path):
        """Toggle-Count wird persistiert."""
        db_path = str(tmp_path / "kill_switch_count.json")
        ks = KillSwitch(enabled=False, db_path=db_path)

        ks.activate()
        ks.deactivate()
        ks.activate()

        config = ks.config
        assert config.toggle_count == 3


class TestKillSwitchThreadSafety:
    """Thread-Safety-Tests."""

    def test_concurrent_activate_deactivate(self):
        """Paralleles Aktivieren/Deaktivieren ist thread-sicher."""
        ks = KillSwitch(enabled=False)
        errors = []

        def toggle(n: int) -> None:
            try:
                for _ in range(100):
                    if n % 2 == 0:
                        ks.activate()
                    else:
                        ks.deactivate()
            except Exception as e:  # noqa: BLE001 — concurrent stress test: catch any thread error
                errors.append(e)

        threads = [threading.Thread(target=toggle, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors  # keine Exceptions
        # Zustand sollte konsistent sein
        assert ks.is_active() in (True, False)

    def test_is_active_performance(self):
        """is_active() muss <100ms sein."""
        ks = KillSwitch(enabled=False)
        start = time.monotonic()
        for _ in range(1000):
            _ = ks.is_active()
        elapsed = time.monotonic() - start
        # 1000 Aufrufe in <100ms = <0.1ms pro Aufruf
        assert elapsed < 0.1


class TestKillSwitchConfig:
    """KillSwitchConfig-Tests."""

    def test_config_defaults(self):
        """Standard-Werte."""
        ks = KillSwitch()
        config = ks.config
        assert config.enabled is False
        assert config.toggle_count == 0
        assert isinstance(config.last_toggled_at, float)

    def test_config_updated_on_toggle(self):
        """Config wird beim Toggle aktualisiert."""
        ks = KillSwitch()
        old_count = ks.config.toggle_count
        ks.activate()
        assert ks.config.toggle_count == old_count + 1


class TestKillSwitchAutoTrigger:
    """R5.6: Kill Switch automatisch auslösbar beim ersten Anomalie-Ereignis."""

    def test_no_trigger_below_threshold(self):
        """Unterhalb des Default-Schwellwerts (3) bleibt der Switch inaktiv."""
        ks = KillSwitch()
        assert ks.record_anomaly("EXCHANGE_ERROR: 5xx") is False
        assert ks.record_anomaly("EXCHANGE_ERROR: 5xx") is False
        assert ks.is_active() is False
        assert ks.config.anomaly_streak == 2

    def test_trigger_at_threshold(self):
        """Drittes Anomalie-Ereignis aktiviert den Kill Switch automatisch."""
        ks = KillSwitch()
        ks.record_anomaly("EXCHANGE_ERROR: 5xx")
        ks.record_anomaly("EXCHANGE_ERROR: 5xx")
        triggered = ks.record_anomaly("EXCHANGE_ERROR: auth-failure")
        assert triggered is True
        assert ks.is_active() is True
        config = ks.config
        assert config.auto_triggered is True
        assert "auth-failure" in config.trigger_reason
        assert config.anomaly_streak == 0

    def test_anomaly_after_active_is_noop(self):
        """Anomalie bei bereits aktivem Switch ändert nichts (kein doppeltes Toggle)."""
        ks = KillSwitch()
        for _ in range(3):
            ks.record_anomaly("EXCHANGE_ERROR: 5xx")
        toggles = ks.config.toggle_count
        assert ks.record_anomaly("EXCHANGE_ERROR: 5xx") is False
        assert ks.config.toggle_count == toggles

    def test_success_resets_streak(self):
        """Erfolgreiche Ausführung setzt den Anomalie-Streak zurück."""
        ks = KillSwitch()
        ks.record_anomaly("EXCHANGE_ERROR: 5xx")
        ks.record_anomaly("EXCHANGE_ERROR: 5xx")
        ks.record_success()
        assert ks.config.anomaly_streak == 0
        ks.record_anomaly("EXCHANGE_ERROR: 5xx")
        ks.record_anomaly("EXCHANGE_ERROR: 5xx")
        assert ks.is_active() is False
        assert ks.record_anomaly("EXCHANGE_ERROR: 5xx") is True

    def test_configurable_threshold(self):
        """Schwellwert ist konfigurierbar (threshold=1 -> sofortiger Trigger)."""
        ks = KillSwitch()
        ks._persisted_config.auto_trigger_threshold = 1
        assert ks.record_anomaly("EXCHANGE_ERROR: dns") is True
        assert ks.is_active() is True

    def test_auto_trigger_disabled(self):
        """Mit auto_trigger_enabled=False wird nie automatisch aktiviert."""
        ks = KillSwitch()
        ks._persisted_config.auto_trigger_enabled = False
        for _ in range(5):
            assert ks.record_anomaly("EXCHANGE_ERROR: 5xx") is False
        assert ks.is_active() is False
        assert ks.config.anomaly_streak == 0

    def test_manual_activate_marks_manual(self):
        """Manuelle Aktivierung überschreibt den Auto-Trigger-Zustand."""
        ks = KillSwitch()
        ks.activate()
        config = ks.config
        assert config.auto_triggered is False
        assert config.trigger_reason == "manual"

    def test_deactivate_resets_streak(self):
        """Manuelle Deaktivierung (Neustart durch Operator) setzt den Streak zurück."""
        ks = KillSwitch()
        for _ in range(3):
            ks.record_anomaly("EXCHANGE_ERROR: 5xx")
        assert ks.is_active() is True
        ks.deactivate()
        ks.record_anomaly("EXCHANGE_ERROR: 5xx")
        ks.record_anomaly("EXCHANGE_ERROR: 5xx")
        assert ks.is_active() is False
        assert ks.record_anomaly("EXCHANGE_ERROR: 5xx") is True

    def test_persistence_of_auto_trigger(self, tmp_path: Path):
        """Auto-Trigger-Zustand wird persistiert und bei Neustart geladen."""
        db_path = str(tmp_path / "kill_switch_auto.json")
        ks1 = KillSwitch(db_path=db_path)
        for _ in range(3):
            ks1.record_anomaly("EXCHANGE_ERROR: 5xx")
        assert ks1.is_active() is True

        ks2 = KillSwitch(enabled=False, db_path=db_path)
        assert ks2.is_active() is True
        assert ks2.config.auto_triggered is True
        assert "5xx" in (ks2.config.trigger_reason or "")

    def test_backwards_compatible_load(self, tmp_path: Path):
        """Alte JSON-Datei ohne neue Felder lädt mit Default-Werten."""
        import json

        db_path = str(tmp_path / "kill_switch_old.json")
        with open(db_path, "w") as f:
            json.dump(
                {"enabled": True, "last_toggled_at": 123.0, "toggle_count": 2}, f
            )
        ks = KillSwitch(enabled=False, db_path=db_path)
        assert ks.is_active() is True
        assert ks.config.auto_trigger_enabled is True
        assert ks.config.auto_trigger_threshold == 3
        assert ks.config.anomaly_streak == 0

    def test_concurrent_anomalies_trigger_exactly_once(self):
        """Parallele Anomalien: genau ein Auto-Trigger, thread-sicher."""
        ks = KillSwitch()
        errors = []

        def report(n: int) -> None:
            try:
                for i in range(100):
                    ks.record_anomaly(f"EXCHANGE_ERROR: thread-{n}-{i}")
            except Exception as e:  # noqa: BLE001 — concurrent stress test: catch any thread error
                errors.append(e)

        threads = [threading.Thread(target=report, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert ks.is_active() is True
        # Genau ein Auto-Trigger (1 Toggle), keine doppelte Aktivierung
        assert ks.config.toggle_count == 1


class TestKillSwitchPersistenceWiring:
    """Persistenz: manueller Kill-Switch-State überlebt einen Neustart (WI-P5-10)."""

    def test_manual_activation_survives_restart(self, tmp_path: Path):
        state_file = tmp_path / "kill_switch.json"
        ks = KillSwitch(db_path=str(state_file))
        ks.activate()

        reloaded = KillSwitch(db_path=str(state_file))
        assert reloaded.is_active() is True
        assert reloaded.config.auto_triggered is False
        assert reloaded.config.trigger_reason == "manual"

    def test_deactivation_survives_restart(self, tmp_path: Path):
        state_file = tmp_path / "kill_switch.json"
        ks = KillSwitch(db_path=str(state_file))
        ks.activate()
        ks.deactivate()

        reloaded = KillSwitch(db_path=str(state_file))
        assert reloaded.is_active() is False


class TestKillSwitchAtomicPersistence:
    """Atomarer State-Write: ein Crash mid-write lässt den vorherigen Stand intakt (WI-P5-10)."""

    def test_crash_before_replace_keeps_previous_state(
        self, tmp_path: Path, monkeypatch
    ):
        state_file = tmp_path / "kill_switch.json"
        ks = KillSwitch(db_path=str(state_file))
        ks.activate()
        assert state_file.exists()

        def boom(*args, **kwargs):
            raise OSError("simulated crash before atomic replace")

        monkeypatch.setattr("trading_harness.services.kill_switch.os.replace", boom)
        ks.deactivate()  # _save_state scheitert — alter File-Stand bleibt

        reloaded = KillSwitch(db_path=str(state_file))
        assert reloaded.is_active() is True

    def test_no_tmp_remains_after_successful_save(self, tmp_path: Path):
        state_file = tmp_path / "kill_switch.json"
        ks = KillSwitch(db_path=str(state_file))
        ks.activate()
        ks.deactivate()
        assert not (tmp_path / "kill_switch.json.tmp").exists()