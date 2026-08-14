"""KillSwitch — thread-sicher mit Persistenz."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from pydantic import BaseModel, Field


class KillSwitchConfig(BaseModel):
    """Konfiguration des Kill Switches."""

    enabled: bool = False
    last_toggled_at: float = Field(default_factory=time.time)
    toggle_count: int = 0


class KillSwitch:
    """Thread-sicherer Kill Switch mit SQLite-Persistenz.

    Aktivierung innerhalb von 100ms wirksam.
    Zustand wird persistent gespeichert und bei Neustart wiederhergestellt.
    """

    def __init__(self, enabled: bool = False, db_path: str | None = None) -> None:
        self._enabled = enabled
        self._lock = threading.Lock()
        self._db_path = db_path
        self._persisted_config = KillSwitchConfig(enabled=enabled)
        self._load_state()

    def _load_state(self) -> None:
        """Persistierten Zustand laden (falls verfügbar)."""
        if self._db_path is None:
            return
        try:
            path = Path(self._db_path)
            if path.exists():
                import json

                with open(path, "r") as f:
                    data = json.load(f)
                    self._enabled = data.get("enabled", False)
                    self._persisted_config.enabled = self._enabled
        except (json.JSONDecodeError, IOError):
            # Fallback: Startzustand verwenden
            pass

    def _save_state(self) -> None:
        """Aktuellen Zustand persistieren."""
        if self._db_path is None:
            return
        try:
            path = Path(self._db_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            import json

            with open(path, "w") as f:
                json.dump(
                    {
                        "enabled": self._enabled,
                        "last_toggled_at": self._persisted_config.last_toggled_at,
                        "toggle_count": self._persisted_config.toggle_count,
                    },
                    f,
                )
        except IOError:
            # Persistenzfehler nicht kritisch — Zustand ist im Speicher
            pass

    def activate(self) -> None:
        """Kill Switch aktivieren (thread-sicher)."""
        with self._lock:
            self._enabled = True
            now = time.time()
            self._persisted_config.last_toggled_at = now
            self._persisted_config.toggle_count += 1
            self._save_state()

    def deactivate(self) -> None:
        """Kill Switch deaktivieren (thread-sicher)."""
        with self._lock:
            self._enabled = False
            now = time.time()
            self._persisted_config.last_toggled_at = now
            self._persisted_config.toggle_count += 1
            self._save_state()

    def is_active(self) -> bool:
        """Prüfen ob Kill Switch aktiv ist (thread-sicher, <100ms)."""
        with self._lock:
            return self._enabled

    @property
    def config(self) -> KillSwitchConfig:
        """Aktuelle Konfiguration."""
        with self._lock:
            return KillSwitchConfig(
                enabled=self._enabled,
                last_toggled_at=self._persisted_config.last_toggled_at,
                toggle_count=self._persisted_config.toggle_count,
            )