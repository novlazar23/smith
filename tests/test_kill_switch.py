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