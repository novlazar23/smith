"""Tests für KillSwitch."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import pytest

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


class TestKillSwitchMultiWriterIsolation:
    """WI-P5-12: Multi-Writer-Isolation (Review-MINOR-2 von WI-P5-10).

    Zwei `KillSwitch`-Instanzen, die sich einen State-Pfad teilen, dürfen
    nicht auf derselben Tmp-Datei kollidieren — sonst drohen Lost Updates
    und State-Divergenz zwischen den In-Memory-Zuständen.
    """

    @staticmethod
    def _snapshot_of(ks: KillSwitch) -> dict[str, object]:
        """Vollständiges In-Memory-Snapshot eines Writers (JSON-Format)."""
        cfg = ks.config
        return {
            "enabled": ks.is_active(),
            "last_toggled_at": cfg.last_toggled_at,
            "toggle_count": cfg.toggle_count,
            "auto_trigger_enabled": cfg.auto_trigger_enabled,
            "auto_trigger_threshold": cfg.auto_trigger_threshold,
            "anomaly_streak": cfg.anomaly_streak,
            "auto_triggered": cfg.auto_triggered,
            "trigger_reason": cfg.trigger_reason,
        }

    def test_multi_writer_collision_no_lost_update(self, tmp_path: Path, monkeypatch):
        """Deterministischer Multi-Writer-Kollisionstest (erzwungenes Interleaving).

        Zwei Writer teilen EINEN State-Pfad. Das Interleaving ist per
        Events erzwungen (kein Timing-Lottery):

        1. Writer A (zuerst gestartet) hat sein vollständiges Snapshot
           doc_a (per `activate()` → enabled=True) in die Tmp-Datei
           geschrieben (alt: geteilte `kill_switch.json.tmp`; neu:
           eigene mkstemp-Datei) und blockiert in `os.fsync`
           (erster Aufruf), bis er freigegeben wird.
        2. Erst danach startet Writer B: Er trunciert die Tmp-Datei
           (alter Code) und schreibt sein Snapshot doc_b (per
           `deactivate()` → enabled=False). B's `os.replace`
           (erster Aufruf) gibt A frei und wartet auf A's Replace.
        3. Writer A's `os.replace` (zweiter Aufruf) läuft sofort: Im
           alten Code konsumiert er die geteilte Tmp-Datei, die zu
           diesem Zeitpunkt doc_b enthält.

        Invariante: nach beiden Writes enthält die Datei das
        VOLLSTÄNDIGE Snapshot des Writers, dessen `os.replace` zuletzt
        erfolgreich ausgeführt hat. Alt: Die Datei enthält B's doc_b,
        obwohl A der zuletzt erfolgreiche Replacer war → A's doc_a ist
        verloren (Lost Update / State-Divergenz) → Test rot. Neu: Jeder
        Writer ersetzt seine eigene mkstemp-Datei; B's Replace ist
        zuletzt → die Datei enthält doc_b → Test grün.
        """
        state_file = tmp_path / "kill_switch.json"
        writer_a = KillSwitch(db_path=str(state_file))
        writer_a.deactivate()  # Seed: toggle_count = 1
        assert writer_a.config.toggle_count == 1
        writer_b = KillSwitch(db_path=str(state_file))

        first_fsync_reached = threading.Event()
        release_fsync = threading.Event()
        a_replace_done = threading.Event()
        replace_log: list[tuple[int, bool]] = []  # (Thread-Id, ok), Abschlussreihenfolge
        fsync_calls = 0
        fsync_lock = threading.Lock()
        replace_calls = 0
        replace_lock = threading.Lock()
        log_lock = threading.Lock()
        real_fsync = os.fsync
        real_replace = os.replace

        def fsync_wrapper(fd: int) -> None:
            nonlocal fsync_calls
            with fsync_lock:
                fsync_calls += 1
                is_first = fsync_calls == 1
            if is_first:
                # Erster fsync (Writer A): doc_a liegt vollständig in
                # der Datei — hier blockieren und Writer B komplett
                # durchlaufen lassen.
                first_fsync_reached.set()
                if not release_fsync.wait(timeout=10):
                    raise AssertionError("Deadlock: erster os.replace-Aufruf kam nicht")
            real_fsync(fd)

        def replace_wrapper(src: str, dst: str) -> None:
            nonlocal replace_calls
            with replace_lock:
                replace_calls += 1
                is_first = replace_calls == 1
            if is_first:
                # Erster Replace (Writer B): Writer A freigeben und auf
                # A's Replace-Abschluss warten, bevor unser Replace
                # läuft — erzwingt die Replace-Reihenfolge A → B.
                release_fsync.set()
                if not a_replace_done.wait(timeout=10):
                    raise AssertionError("Deadlock: zweiter os.replace-Aufruf kam nicht")
            exc: OSError | None = None
            try:
                real_replace(src, dst)
            except OSError as e:
                exc = e
            finally:
                with log_lock:
                    replace_log.append((threading.get_ident(), exc is None))
            if not is_first:
                a_replace_done.set()
            if exc is not None:
                raise exc

        monkeypatch.setattr("trading_harness.services.kill_switch.os.fsync", fsync_wrapper)
        monkeypatch.setattr("trading_harness.services.kill_switch.os.replace", replace_wrapper)

        errors: list[BaseException] = []
        thread_ids: dict[str, int] = {}

        def run_a() -> None:
            thread_ids["a"] = threading.get_ident()
            try:
                writer_a.activate()  # enabled=True → unterscheidbares Snapshot
            except BaseException as exc:  # noqa: BLE001 — Concurrent-Test: Thread-Fehler erfassen
                errors.append(exc)

        def run_b() -> None:
            thread_ids["b"] = threading.get_ident()
            try:
                writer_b.deactivate()  # enabled=False → unterscheidbares Snapshot
            except BaseException as exc:  # noqa: BLE001 — Concurrent-Test: Thread-Fehler erfassen
                errors.append(exc)

        thread_a = threading.Thread(target=run_a)
        thread_a.start()
        if not first_fsync_reached.wait(timeout=10):
            raise AssertionError("Writer A ist nicht in der fsync-Phase angekommen")
        thread_b = threading.Thread(target=run_b)
        thread_b.start()
        thread_a.join(timeout=15)
        thread_b.join(timeout=15)
        assert not thread_a.is_alive() and not thread_b.is_alive()
        assert not errors, errors

        data = json.loads(state_file.read_text())
        assert set(data) == {
            "enabled",
            "last_toggled_at",
            "toggle_count",
            "auto_trigger_enabled",
            "auto_trigger_threshold",
            "anomaly_streak",
            "auto_triggered",
            "trigger_reason",
        }
        successful = [tid for tid, ok in replace_log if ok]
        assert successful, "kein Writer hat erfolgreich ersetzt"
        writers_by_thread = {thread_ids["a"]: writer_a, thread_ids["b"]: writer_b}
        last_writer = writers_by_thread[successful[-1]]
        # Kein Lost Update / kein gemischtes Dokument: die Datei muss exakt
        # das vollständige Snapshot des zuletzt erfolgreich persistierenden
        # Writers enthalten.
        assert data == self._snapshot_of(last_writer)

    def test_concurrent_writers_stress(self, tmp_path: Path):
        """Stress: 2 Instanzen, 4 Threads × 25 Toggles auf einem State-Pfad.

        Invarianten nach Abschluss:
        - Datei ist valides JSON mit dem vollständigen Key-Set,
        - toggle_count == 50 (jeder Writer hat 2 × 25 Toggles ausgeführt —
          kein Partial-/Lost-Update),
        - keine *.tmp*-Rückstände im Verzeichnis,
        - frische Instanz lädt denselben Zustand (Reload-Konsistenz).
        """
        state_file = tmp_path / "kill_switch.json"
        writer_a = KillSwitch(db_path=str(state_file))
        writer_b = KillSwitch(db_path=str(state_file))
        writers = (writer_a, writer_b)
        errors: list[BaseException] = []
        barrier = threading.Barrier(4)

        def worker(n: int) -> None:
            try:
                barrier.wait(timeout=10)
                writer = writers[n % 2]
                for _ in range(25):
                    writer.deactivate()
            except BaseException as exc:  # noqa: BLE001 — Concurrent-Test: Thread-Fehler erfassen
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        assert not errors, errors
        assert all(not t.is_alive() for t in threads)

        data = json.loads(state_file.read_text())
        assert set(data) == {
            "enabled",
            "last_toggled_at",
            "toggle_count",
            "auto_trigger_enabled",
            "auto_trigger_threshold",
            "anomaly_streak",
            "auto_triggered",
            "trigger_reason",
        }
        assert data["toggle_count"] == 50
        assert not list(state_file.parent.glob("*.tmp*"))
        reloaded = KillSwitch(db_path=str(state_file))
        assert reloaded.is_active() is data["enabled"]
        assert reloaded.config.toggle_count == data["toggle_count"]

    def test_state_file_mode_0644_new_and_overwrite(self, tmp_path: Path):
        """State-Datei behält 0644 über Neuanlage UND Überschreiben.

        `tempfile.mkstemp` legt Tmp-Dateien per Default mit 0600 an — ohne
        explizites `os.chmod` würde der Modus der State-Datei wechseln.
        """
        state_file = tmp_path / "kill_switch.json"
        ks = KillSwitch(db_path=str(state_file))
        ks.activate()  # Neuanlage
        assert state_file.stat().st_mode & 0o777 == 0o644
        ks.deactivate()  # Überschreiben
        assert state_file.stat().st_mode & 0o777 == 0o644


class TestIsolationFixtureTeardown:
    """WI-P5-14: conftest-Teardown räumt den In-Memory-Log-State auf (NIT-A).

    Regressionstest für den Review-NIT aus dem WI-P5-11-Review (Review-ID 3):
    "marker-Opt-out überspringt auch den Teardown (zukunftssicherungs-relevant)"
    — der Marker ``real_execution_log_state`` schaltet in der conftest-Fixture
    die tmp-Umbindung UND (vor WI-P5-14) auch die Teardown-Aufräumung ab.
    Ein Marker-Opt-out-Test, der direkt hinter einem Test mit
    Log-Write läuft, erbt daher den In-Memory-Log-State des Vorgängers.
    Zukunftsrelevant: der heutige einzige Marker-Test ist rein lesend; ein
    schreibender Marker-Test hätte den Defekt ausgelöst.

    Wichtig: ``test_opt_out_test_sees_clean_log_state`` hängt bewusst von
    der Datei-Reihenfolge ab (pytest führt Tests in Datei-Reihenfolge aus)
    und muss direkt nach ``test_writes_log_entry_for_next_test`` laufen.

    Sicherheit: Die autouse-Fixture bindet den Log-Store pro Test auf einen
    tmp-Pfad um, daher landet der Write von Test 1 nie in der echten
    ``data/execution_log.json``; das WI-P5-14-Teardown-``clear()`` wirkt
    vor der Monkeypatch-Rücksetzung und berührt die echte State-Datei
    ebenfalls nicht.
    """

    def test_writes_log_entry_for_next_test(self):
        """Test 1 (ohne Marker): Log-Write landet im tmp-gebundenen Store."""
        from trading_harness.api import routes

        routes.execution_log_store.add(
            decision_id="nit-a-1",
            run_id="nit-a-run",
            symbol="BTCUSDT",
            side="buy",
            status="REJECTED",
            error="LIVE_EXECUTION_DISABLED",
        )
        assert routes.execution_log_store.count == 1

    @pytest.mark.real_execution_log_state
    def test_opt_out_test_sees_clean_log_state(self):
        """Test 2 (Marker-Opt-out) direkt nach Test 1 sieht einen leeren Store."""
        from trading_harness.api import routes

        # Ohne das WI-P5-14-Teardown-``clear()`` erbt dieser Test den
        # In-Memory-Eintrag von Test 1 (count == 1) → Test rot.
        assert routes.execution_log_store.count == 0


class TestCorruptedStateFileFallback:
    """WI-P5-14: Pin-Test für den Fail-Open-Fallback bei korrumpiertem State-File.

    Pinnt den bestehenden Fallback (NIT aus dem WI-P5-10-Review, Review-ID 2,
    NIT 6 "kein Pin-Test für Fail-Open-Fallback bei extern korrumpiertem
    State-File"): ``KillSwitch._load_state`` fängt
    ``(OSError, json.JSONDecodeError)`` und verwendet den Startzustand
    ("Fallback: Startzustand verwenden"). Dieses Fail-Open-Verhalten wird
    hier nur GEpinnt, nicht geändert — Fail-Closed wäre eine
    Sicherheitsgrenzen-Änderung und erfordert explizite Freigabe. Der Test
    muss daher auch vor jeder Quelländerung grün sein.
    """

    def test_corrupted_state_file_falls_back_to_init_state(self, tmp_path: Path):
        """Extern korrumpiertes State-File → Fallback auf Startzustand (gepinnt)."""
        state_file = tmp_path / "kill_switch.json"
        state_file.write_text("{not valid json!!")

        # (1) Korrumpiertes File + Startzustand enabled=False → keine
        #     Exception beim Laden, Switch bleibt im Startzustand inaktiv.
        ks = KillSwitch(enabled=False, db_path=str(state_file))
        assert ks.is_active() is False

        # (2) Erneut korrumpiertes File + Startzustand enabled=True →
        #     Startzustand bleibt aktiv (pinnt die Fail-Open-Semantik:
        #     Fallback auf den Konstruktor-Initialzustand, nicht auf
        #     einen "sicheren" Wert).
        state_file.write_text("{not valid json!!")
        ks_enabled = KillSwitch(enabled=True, db_path=str(state_file))
        assert ks_enabled.is_active() is True

        # (3) Folge-Write repariert die korrumpierte Datei: nach
        #     ``deactivate()`` muss das File wieder gültiges, parsebares
        #     JSON mit ``enabled == False`` enthalten.
        ks_enabled.deactivate()
        data = json.loads(state_file.read_text())
        assert data["enabled"] is False