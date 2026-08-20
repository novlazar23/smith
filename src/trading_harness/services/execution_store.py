"""ExecutionLog Store — persistierte Execution Logs mit In-Memory-Fallback."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ExecutionLogEntry(BaseModel):
    """Einzelner Execution Log Eintrag."""

    id: str
    decision_id: str
    run_id: str
    symbol: str
    side: str
    status: str
    order_id: str | None = None
    error: str | None = None
    timestamp: float = Field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "decision_id": self.decision_id,
            "run_id": self.run_id,
            "symbol": self.symbol,
            "side": self.side,
            "status": self.status,
            "order_id": self.order_id,
            "error": self.error,
            "timestamp": datetime.fromtimestamp(self.timestamp, tz=UTC).isoformat(),
        }


class ExecutionLogStore:
    """Persistente Execution Logs mit In-Memory-Fallback.

    Credentials werden nie in Logs geschrieben.
    """

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path
        self._logs: list[ExecutionLogEntry] = []
        self._lock = threading.Lock()
        self._load_state()

    @property
    def db_path(self) -> str | None:
        """Aktueller Persistenz-Pfad (None = nur In-Memory)."""
        return self._db_path

    def _load_state(self) -> None:
        """Persistierten Zustand laden."""
        if self._db_path is None:
            return
        try:
            path = Path(self._db_path)
            if path.exists():
                with open(path, "r") as f:
                    data = json.load(f)
                    for entry in data.get("logs", []):
                        # ISO-String nach Timestamp konvertieren
                        ts = entry.get("timestamp")
                        if isinstance(ts, str):
                            dt = datetime.fromisoformat(ts)
                            ts = dt.timestamp()
                        entry["timestamp"] = ts
                        self._logs.append(ExecutionLogEntry(**entry))
        except (OSError, json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to load execution logs: %s", e)

    def _save_state(self) -> None:
        """Zustand persistieren (atomar: unique mkstemp-Tmp + os.replace, ohne Credentials).

        Symmetrisch zu `KillSwitch._save_state` (WI-P5-12): Die Tmp-Datei wird
        per `tempfile.mkstemp` im Zielverzeichnis angelegt (gleiches
        Dateisystem → atomares `os.replace`). Jeder Write erhält einen
        eindeutigen Tmp-Namen, daher können mehrere Writer, die sich denselben
        State-Pfad teilen, keine gemeinsame Tmp-Datei mehr
        truncate/überschreiben (Lost Updates). Nach einem Crash bleibt der
        vorherige File-Stand intakt (keine halbe JSON). `mkstemp` legt mit
        Modus 0600 an — der Modus der State-Datei wird daher explizit
        übernommen (Neuanlage: 0644).
        """
        if self._db_path is None:
            return
        tmp_name: str | None = None
        tmp_fd: int | None = None
        replaced = False
        try:
            path = Path(self._db_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_fd, tmp_name = tempfile.mkstemp(
                dir=str(path.parent), prefix=path.name + ".", suffix=".tmp"
            )
            try:
                mode = path.stat().st_mode & 0o777
            except OSError:
                mode = 0o644
            os.chmod(tmp_name, mode)
            with os.fdopen(tmp_fd, "w") as f:
                tmp_fd = None  # fd wird jetzt vom File-Objekt geschlossen
                # Nie Credentials speichern
                clean_logs = [
                    {k: v for k, v in log.to_dict().items() if "key" not in k.lower()}
                    for log in self._logs
                ]
                json.dump({"logs": clean_logs}, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, path)
            replaced = True
        except OSError as e:
            # Persistenzfehler nicht kritisch — Zustand bleibt im Speicher.
            logger.warning("Failed to save execution logs: %s", e)
        finally:
            if tmp_fd is not None:
                try:
                    os.close(tmp_fd)
                except OSError:
                    pass
            if not replaced and tmp_name is not None:
                # Fehlgeschlagene Tmp-Datei aufräumen (best effort).
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass

    def add(
        self,
        decision_id: str,
        run_id: str,
        symbol: str,
        side: str,
        status: str,
        order_id: str | None = None,
        error: str | None = None,
    ) -> ExecutionLogEntry:
        """Execution Log hinzufügen.

        Die ID (Zeitstempel und Counter) wird vollständig innerhalb des
        Locks generiert, damit parallele Adds keine ID-Kollision erzeugen
        können (Review-13, N2).
        """
        with self._lock:
            entry = ExecutionLogEntry(
                id=f"exec-{int(time.time() * 1000)}-{len(self._logs)}",
                decision_id=decision_id,
                run_id=run_id,
                symbol=symbol,
                side=side,
                status=status,
                order_id=order_id,
                error=error,
            )
            self._logs.append(entry)
            self._save_state()
        return entry

    def get_by_decision_id(self, decision_id: str) -> list[dict[str, Any]]:
        """Logs für eine decision_id abrufen."""
        with self._lock:
            return [
                log.to_dict()
                for log in self._logs
                if log.decision_id == decision_id
            ]

    def get_all(self) -> list[dict[str, Any]]:
        """Alle Logs abrufen."""
        with self._lock:
            return [log.to_dict() for log in self._logs]

    def get_by_run(self, run_id: str) -> list[dict[str, Any]]:
        """Logs für einen Run abrufen."""
        with self._lock:
            return [log.to_dict() for log in self._logs if log.run_id == run_id]

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._logs)

    def clear(self) -> None:
        """Alle Logs löschen und den geleerten Zustand persistieren."""
        with self._lock:
            self._logs = []
            self._save_state()