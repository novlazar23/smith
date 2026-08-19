"""KillSwitch — thread-sicher mit Persistenz."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from pydantic import BaseModel, Field


class KillSwitchConfig(BaseModel):
    """Konfiguration des Kill Switches."""

    enabled: bool = False
    last_toggled_at: float = Field(default_factory=time.time)
    toggle_count: int = 0
    # R5.6: automatische Auslösung bei Anomalie-Ereignissen
    auto_trigger_enabled: bool = True
    auto_trigger_threshold: int = Field(default=3, ge=1)
    anomaly_streak: int = 0
    auto_triggered: bool = False
    trigger_reason: str | None = None


class KillSwitch:
    """Thread-sicherer Kill Switch mit atomarer JSON-Persistenz.

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
                with open(path, "r") as f:
                    data = json.load(f)
                    self._enabled = data.get("enabled", False)
                    cfg = self._persisted_config
                    cfg.enabled = self._enabled
                    cfg.last_toggled_at = data.get("last_toggled_at", cfg.last_toggled_at)
                    cfg.toggle_count = data.get("toggle_count", cfg.toggle_count)
                    cfg.auto_trigger_enabled = data.get(
                        "auto_trigger_enabled", cfg.auto_trigger_enabled
                    )
                    cfg.auto_trigger_threshold = data.get(
                        "auto_trigger_threshold", cfg.auto_trigger_threshold
                    )
                    cfg.anomaly_streak = data.get("anomaly_streak", cfg.anomaly_streak)
                    cfg.auto_triggered = data.get("auto_triggered", cfg.auto_triggered)
                    cfg.trigger_reason = data.get("trigger_reason", cfg.trigger_reason)
        except (OSError, json.JSONDecodeError):
            # Fallback: Startzustand verwenden
            pass

    def _save_state(self) -> None:
        """Aktuellen Zustand persistieren (atomar: tmp-Datei + os.replace)."""
        if self._db_path is None:
            return
        try:
            path = Path(self._db_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_name(path.name + ".tmp")
            cfg = self._persisted_config
            with open(tmp_path, "w") as f:
                json.dump(
                    {
                        "enabled": self._enabled,
                        "last_toggled_at": cfg.last_toggled_at,
                        "toggle_count": cfg.toggle_count,
                        "auto_trigger_enabled": cfg.auto_trigger_enabled,
                        "auto_trigger_threshold": cfg.auto_trigger_threshold,
                        "anomaly_streak": cfg.anomaly_streak,
                        "auto_triggered": cfg.auto_triggered,
                        "trigger_reason": cfg.trigger_reason,
                    },
                    f,
                )
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        except OSError:
            # Persistenzfehler nicht kritisch — Zustand bleibt im Speicher.
            # Atomic-Write (tmp + os.replace) garantiert: nach einem Crash
            # ist der vorherige File-Stand intakt (keine halbe JSON).
            pass

    def activate(self) -> None:
        """Kill Switch aktivieren (thread-sicher, manuell)."""
        with self._lock:
            self._enabled = True
            now = time.time()
            cfg = self._persisted_config
            cfg.last_toggled_at = now
            cfg.toggle_count += 1
            cfg.auto_triggered = False
            cfg.trigger_reason = "manual"
            self._save_state()

    def deactivate(self) -> None:
        """Kill Switch deaktivieren (thread-sicher).

        Operator-Neustart: der Anomalie-Streak wird zurückgesetzt, damit
        der Count nach dem manuellen Resume von vorne beginnt.
        """
        with self._lock:
            self._enabled = False
            now = time.time()
            cfg = self._persisted_config
            cfg.last_toggled_at = now
            cfg.toggle_count += 1
            cfg.anomaly_streak = 0
            self._save_state()

    def is_active(self) -> bool:
        """Prüfen ob Kill Switch aktiv ist (thread-sicher, <100ms)."""
        with self._lock:
            return self._enabled

    @property
    def db_path(self) -> str | None:
        """Aktueller Persistenz-Pfad (None = nur In-Memory)."""
        return self._db_path

    def record_anomaly(self, reason: str) -> bool:
        """R5.6: Anomalie-Ereignis erfassen.

        Bei `auto_trigger_threshold` aufeinanderfolgenden Anomalien (ohne
        erfolgreiche Ausführung dazwischen) wird der Kill Switch
        automatisch aktiviert. Liefert True, wenn der Trigger ausgelöst
        wurde.
        """
        with self._lock:
            if self._enabled or not self._persisted_config.auto_trigger_enabled:
                return False
            cfg = self._persisted_config
            cfg.anomaly_streak += 1
            if cfg.anomaly_streak >= cfg.auto_trigger_threshold:
                cfg.anomaly_streak = 0
                cfg.auto_triggered = True
                cfg.trigger_reason = (
                    f"{reason} (auto, {cfg.auto_trigger_threshold} consecutive anomalies)"
                )
                self._enabled = True
                cfg.last_toggled_at = time.time()
                cfg.toggle_count += 1
                self._save_state()
                return True
            self._save_state()
            return False

    def record_success(self) -> None:
        """R5.6: Erfolgreiche Ausführung — setzt den Anomalie-Streak zurück."""
        with self._lock:
            if self._persisted_config.anomaly_streak:
                self._persisted_config.anomaly_streak = 0
                self._save_state()

    @property
    def config(self) -> KillSwitchConfig:
        """Aktuelle Konfiguration."""
        with self._lock:
            return KillSwitchConfig(
                enabled=self._enabled,
                last_toggled_at=self._persisted_config.last_toggled_at,
                toggle_count=self._persisted_config.toggle_count,
                auto_trigger_enabled=self._persisted_config.auto_trigger_enabled,
                auto_trigger_threshold=self._persisted_config.auto_trigger_threshold,
                anomaly_streak=self._persisted_config.anomaly_streak,
                auto_triggered=self._persisted_config.auto_triggered,
                trigger_reason=self._persisted_config.trigger_reason,
            )