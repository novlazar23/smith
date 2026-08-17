"""ExecutionLog Store — persistierte Execution Logs mit In-Memory-Fallback."""

from __future__ import annotations

import json
import logging
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

                            from datetime import datetime

                            dt = datetime.fromisoformat(ts)
                            ts = dt.timestamp()
                        entry["timestamp"] = ts
                        self._logs.append(ExecutionLogEntry(**entry))
        except (OSError, json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to load execution logs: %s", e)

    def _save_state(self) -> None:
        """Zustand persistieren (ohne Credentials)."""
        if self._db_path is None:
            return
        try:
            path = Path(self._db_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                # Nie Credentials speichern
                clean_logs = [
                    {k: v for k, v in log.to_dict().items() if "key" not in k.lower()}
                    for log in self._logs
                ]
                json.dump({"logs": clean_logs}, f)
        except OSError as e:
            logger.warning("Failed to save execution logs: %s", e)

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
        """Execution Log hinzufügen."""
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
        with self._lock:
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