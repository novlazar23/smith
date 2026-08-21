"""Persistent shadow-trading state store (ST.8/ST.15, F4/F5/F6, Z2).

Atomic checksum-verified JSON persistence (unique mkstemp temp in the
target directory + os.replace, same pattern as data/kill_switch.json),
quarantine on corruption (F5), no auto-start on restart (Z2), memory
bounded record/portfolio buffers (ST.15), consecutive-error autostop
(F6) and write-error survival (F4).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from trading_harness.models import (
    PortfolioState,
    ShadowLoopStatus,
    ShadowSessionState,
    ShadowTradingRecord,
)

logger = logging.getLogger(__name__)

MAX_RECORDS = 10_000
MAX_PORTFOLIO_HISTORY = 2_000
MAX_CONSECUTIVE_ERRORS = 10
_RECENT_PORTFOLIO_EXACT = 500  # newest history entries always kept verbatim (ST.15)


def compute_state_checksum(state: ShadowSessionState) -> str:
    """SHA-256 hex of the canonical JSON of `state` without the checksum field (F5)."""
    payload = {k: v for k, v in state.model_dump(mode="json").items() if k != "state_checksum"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class _RecordsSummary(BaseModel):
    """Lifetime record aggregates persisted next to the bounded buffer (ST.15)."""

    total_records: int = Field(ge=0)
    trimmed_count: int = Field(ge=0)
    by_status: dict[str, int]


class _PersistedDocument(BaseModel):
    """On-disk document: state + bounded buffers + lifetime aggregates."""

    state: ShadowSessionState
    records: list[ShadowTradingRecord]
    records_summary: _RecordsSummary
    portfolio_history: list[PortfolioState]


class ShadowTradingStateStore:
    """Thread-safe persistent store for the shadow-trading session (ST.8).

    Every public operation runs under one shared RLock (``store.lock``) so
    readers (API ``status()``) and the loop (writes) never observe torn
    state. The state file is rewritten atomically and is never left
    half-written; a corrupt or checksum-mismatched file is quarantined,
    never rewritten in place (F5).
    """

    def __init__(self, state_path: Path | str) -> None:
        self.state_path = Path(state_path)
        self.lock = threading.RLock()
        self._state: ShadowSessionState
        self._records: list[ShadowTradingRecord] = []
        self._portfolio_history: list[PortfolioState] = []
        self._lifetime_records = 0
        self._by_status: dict[str, int] = {}
        self._load()

    # ------------------------------------------------------------------
    # Session state
    # ------------------------------------------------------------------

    def status(self) -> ShadowSessionState:
        """Defensive deep copy of the current session state."""
        with self.lock:
            return self._state.model_copy(deep=True)

    def update_state(self, updates: dict[str, object]) -> None:
        """Apply a partial field update and refresh the checksum (ST.8).

        Raises pydantic.ValidationError on schema-invalid fields without
        touching the in-memory state or the state file. While the loop is
        RUNNING the update is persisted immediately (crash-durable audit
        trail); a quiescent (STOPPED) session is not rewritten — the data
        stays in memory and becomes durable via the next save() or RUNNING
        update. A lost STOP transition is unobservable after a restart
        because Z2 coerces any loaded RUNNING state to STOPPED +
        restart_required anyway.
        """
        with self.lock:
            merged = {**self._state.model_dump(mode="json"), **updates}
            state = ShadowSessionState.model_validate(merged)
            state.state_checksum = compute_state_checksum(state)
            self._state = state
            if state.status is ShadowLoopStatus.RUNNING:
                self._save()

    def save(self) -> None:
        """Persist the current state atomically; FS errors are logged, never raised (F4)."""
        with self.lock:
            self._save()

    def record_iteration_error(self, last_error: str) -> bool:
        """Count a consecutive iteration error; autostop on the 10th (F6).

        Returns True exactly when this call triggered the automatic stop.
        """
        with self.lock:
            state = self._state
            state.error_count += 1
            state.last_error = last_error
            tripped = state.error_count == MAX_CONSECUTIVE_ERRORS
            if tripped:
                state.status = ShadowLoopStatus.STOPPED
                state.restart_required = True
            state.state_checksum = compute_state_checksum(state)
            self._save()
            return tripped

    def record_iteration_success(self) -> None:
        """Break the consecutive-error streak (no-op, no write, when already clean)."""
        with self.lock:
            if self._state.error_count == 0:
                return
            self._state.error_count = 0
            self._state.last_error = None
            self._state.state_checksum = compute_state_checksum(self._state)
            self._save()

    # ------------------------------------------------------------------
    # Records (bounded, ST.15)
    # ------------------------------------------------------------------

    def records(self) -> list[ShadowTradingRecord]:
        """Bounded record buffer (newest last, at most MAX_RECORDS entries)."""
        with self.lock:
            return list(self._records)

    def add_record(self, record: ShadowTradingRecord) -> None:
        """Append one decision record, trim to MAX_RECORDS, persist (ST.15)."""
        with self.lock:
            self._append_records([record])

    def add_records(self, records: list[ShadowTradingRecord]) -> None:
        """Append decision records, trim to MAX_RECORDS, persist (ST.15)."""
        with self.lock:
            self._append_records(list(records))

    def records_summary(self) -> dict[str, object]:
        """Lifetime aggregates: totals plus by_status over all ingested records."""
        with self.lock:
            return {
                "total_records": self._lifetime_records,
                "trimmed_count": self._lifetime_records - len(self._records),
                "by_status": dict(self._by_status),
            }

    def _append_records(self, new: list[ShadowTradingRecord]) -> None:
        if not new:
            return
        self._records.extend(new)
        for record in new:
            self._lifetime_records += 1
            status = record.status.value
            self._by_status[status] = self._by_status.get(status, 0) + 1
        if len(self._records) > MAX_RECORDS:
            dropped = len(self._records) - MAX_RECORDS
            del self._records[:dropped]
            logger.warning(
                "SHADOW_STATE_MEMORY_LIMIT records: kept=%d dropped=%d total=%d max=%d",
                len(self._records),
                dropped,
                self._lifetime_records,
                MAX_RECORDS,
            )
        self._save()

    # ------------------------------------------------------------------
    # Portfolio history (bounded, ST.15)
    # ------------------------------------------------------------------

    def portfolio_history(self) -> list[PortfolioState]:
        """Bounded portfolio history (at most MAX_PORTFOLIO_HISTORY entries)."""
        with self.lock:
            return list(self._portfolio_history)

    def add_portfolio_state(self, state: PortfolioState) -> None:
        """Append one portfolio state, trim to MAX_PORTFOLIO_HISTORY, persist (ST.15)."""
        with self.lock:
            self._portfolio_history.append(state)
            self._trim_portfolio_history()
            self._save()

    def add_portfolio_states(self, states: list[PortfolioState]) -> None:
        """Append portfolio states, trim to MAX_PORTFOLIO_HISTORY, persist (ST.15)."""
        with self.lock:
            self._portfolio_history.extend(states)
            self._trim_portfolio_history()
            self._save()

    def _trim_portfolio_history(self) -> None:
        """Keep the newest 500 entries verbatim; downsample the rest equidistantly."""
        if len(self._portfolio_history) <= MAX_PORTFOLIO_HISTORY:
            return
        tail_start = len(self._portfolio_history) - _RECENT_PORTFOLIO_EXACT
        older = self._portfolio_history[:tail_start]
        tail = self._portfolio_history[tail_start:]
        keep = MAX_PORTFOLIO_HISTORY - _RECENT_PORTFOLIO_EXACT
        span = len(older) - 1
        sampled = [older[(2 * j * span + keep - 1) // (2 * (keep - 1))] for j in range(keep)]
        self._portfolio_history = [*sampled, *tail]
        logger.warning(
            "SHADOW_STATE_MEMORY_LIMIT portfolio_history: kept=%d max=%d",
            len(self._portfolio_history),
            MAX_PORTFOLIO_HISTORY,
        )

    # ------------------------------------------------------------------
    # Load / persist internals
    # ------------------------------------------------------------------

    def _load(self) -> None:
        path = self.state_path
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            self._state = self._fresh_state(restart_required=False)
            return
        except OSError as exc:
            self._quarantine(path, f"unreadable state file: {exc!r}")
            self._state = self._fresh_state(restart_required=True)
            return
        try:
            parsed = _PersistedDocument.model_validate(json.loads(raw))
            if compute_state_checksum(parsed.state) != parsed.state.state_checksum:
                raise ValueError("state_checksum mismatch")
        except ValueError as exc:
            self._quarantine(path, f"corrupt state document: {exc!r}")
            self._state = self._fresh_state(restart_required=True)
            return
        self._records = list(parsed.records)
        self._lifetime_records = parsed.records_summary.total_records
        self._by_status = dict(parsed.records_summary.by_status)
        self._portfolio_history = list(parsed.portfolio_history)
        self._state = parsed.state
        self._apply_no_autostart()
        self._state.state_checksum = compute_state_checksum(self._state)

    def _fresh_state(self, restart_required: bool) -> ShadowSessionState:
        state = ShadowSessionState(restart_required=restart_required)
        state.state_checksum = compute_state_checksum(state)
        return state

    def _apply_no_autostart(self) -> None:
        """Z2: a persisted RUNNING/STOPPING loop is never auto-resumed on restart."""
        if self._state.status in (ShadowLoopStatus.RUNNING, ShadowLoopStatus.STOPPING):
            self._state.status = ShadowLoopStatus.STOPPED
            self._state.restart_required = True

    def _quarantine(self, path: Path, reason: str) -> None:
        """Move the corrupt file aside as <path>.corrupt-<UTC timestamp> (F5)."""
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        target = path.with_name(f"{path.name}.corrupt-{stamp}")
        try:
            path.rename(target)
        except OSError as exc:
            logger.warning("SHADOW_STATE_QUARANTINED rename failed: %s (%r)", reason, exc)
        else:
            logger.warning(
                "SHADOW_STATE_QUARANTINED path=%s quarantined_as=%s reason=%s",
                path,
                target.name,
                reason,
            )

    def _save(self) -> None:
        """Atomic write: unique mkstemp temp in the target dir + os.replace.

        FS errors are logged (SHADOW_STATE_WRITE_FAILED) and swallowed (F4):
        the in-memory state stays intact and is retried on the next write.
        """
        doc = {
            "state": self._state.model_dump(mode="json"),
            "records": [r.model_dump(mode="json") for r in self._records],
            "records_summary": {
                "total_records": self._lifetime_records,
                "trimmed_count": self._lifetime_records - len(self._records),
                "by_status": dict(self._by_status),
            },
            "portfolio_history": [p.model_dump(mode="json") for p in self._portfolio_history],
        }
        path = self.state_path
        tmp_fd: int | None = None
        tmp_name: str | None = None
        replaced = False
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_fd, tmp_name = tempfile.mkstemp(
                dir=str(path.parent), prefix=path.name + ".", suffix=".tmp"
            )
            os.chmod(tmp_name, 0o600)
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                tmp_fd = None  # fd is now owned by the file object
                json.dump(doc, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, path)
            replaced = True
        except OSError as exc:
            logger.warning("SHADOW_STATE_WRITE_FAILED path=%s error=%r", path, exc)
        finally:
            if tmp_fd is not None:
                try:
                    os.close(tmp_fd)
                except OSError:
                    pass
            if not replaced and tmp_name is not None:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
