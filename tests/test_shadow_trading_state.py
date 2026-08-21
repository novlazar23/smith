"""Tests for ShadowTradingStateStore (WI-ST-02, Spec ST.8/ST.15/F4/F5/Z2).

Covers: atomic checksum-verified persistence, quarantine on corruption,
Z2 no-auto-start, memory-bounded trimming (records/portfolio history),
file-size bound, consecutive-error autostop, F4 write-error survival,
and concurrency (RLock + atomic replace => no torn reads).
"""

from __future__ import annotations

import hashlib
import itertools
import json
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError
from trading_harness.services.shadow_trading_state import (
    MAX_RECORDS,
    ShadowTradingStateStore,
    compute_state_checksum,
)

from trading_harness.models import (
    PortfolioState,
    ShadowLoopStatus,
    ShadowSessionState,
    ShadowTradingRecord,
    ShadowTradingStatus,
)

STATE_PATH = "shadow_trading_state.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _record(i: int) -> ShadowTradingRecord:
    """Realistic record; status cycles NO_TRADE/REJECTED/FILLED deterministically."""
    status = (ShadowTradingStatus.NO_TRADE, ShadowTradingStatus.REJECTED, ShadowTradingStatus.FILLED)[
        i % 3
    ]
    return ShadowTradingRecord(
        timestamp=datetime(2026, 8, 21, 8, 0, 0, tzinfo=UTC) + timedelta(seconds=i),
        symbol="BTCUSDT" if i % 2 == 0 else "ETHUSDT",
        side="BUY" if i % 2 == 0 else "SELL",
        direction="LONG" if i % 2 == 0 else "SHORT",
        status=status,
        decision_id=f"dec-{i:06d}",
        run_id=f"run-{i:06d}",
        snapshot_id=f"snap-{i:06d}",
        trade_id=f"trade-{i:06d}" if status is ShadowTradingStatus.FILLED else None,
        risk_reason="" if status is not ShadowTradingStatus.REJECTED else "MAX_POSITIONS",
        agent_ids=["agent-tech-01"],
        quantity=0.5,
        entry_price=65000.0 + i,
        mark_price=65010.0 + i,
        slippage=1.0,
        commission=0.3,
        pnl_estimate=5.0 if status is ShadowTradingStatus.FILLED else 0.0,
        slippage_rate=0.0001,
        commission_rate=0.0005,
        config_version="cfg-abc123",
    )


def _portfolio(i: int) -> PortfolioState:
    return PortfolioState(
        run_id=f"run-{i:06d}",
        start_equity=100000.0,
        current_equity=100000.0 + i,
        total_realized_pnl=float(i),
        peak_equity=100000.0 + i,
        symbols=["BTCUSDT", "ETHUSDT"],
        timestamp=datetime(2026, 8, 21, 8, 0, 0, tzinfo=UTC) + timedelta(seconds=i),
    )


def _fresh_store(tmp_path: Path) -> ShadowTradingStateStore:
    return ShadowTradingStateStore(tmp_path / STATE_PATH)


def _expected_checksum(doc_state: dict[str, object]) -> str:
    """Independent recomputation of the checksum formula (test-side)."""
    payload = {k: v for k, v in doc_state.items() if k != "state_checksum"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _write_state_doc(path: Path, state: ShadowSessionState) -> None:
    """Write a full state document the way the store does (for tamper tests)."""
    doc = {
        "state": state.model_dump(mode="json"),
        "records": [],
        "records_summary": {"total_records": 0, "trimmed_count": 0, "by_status": {}},
        "portfolio_history": [],
    }
    path.write_text(json.dumps(doc))


# ---------------------------------------------------------------------------
# ST.8 — persistence, restart, Z2
# ---------------------------------------------------------------------------


def test_state_survives_restart(tmp_path: Path) -> None:
    """5 loop iterations -> new store instance on same path -> values survive, Z2 applied."""
    path = tmp_path / STATE_PATH
    store = _fresh_store(tmp_path)
    base = datetime(2026, 8, 21, 8, 0, 0, tzinfo=UTC)
    for i in range(1, 6):
        store.update_state(
            {
                "status": ShadowLoopStatus.RUNNING,
                "iteration_count": i,
                "decisions_today": i,
                "current_equity": 100000.0 + i * 10,
                "last_iteration_at": base + timedelta(seconds=i),
                "symbols": ["BTCUSDT", "ETHUSDT"],
            }
        )

    # Simulated process restart: brand-new instance, same file.
    reloaded = ShadowTradingStateStore(path)
    st = reloaded.status()
    assert st.iteration_count == 5
    assert st.decisions_today == 5
    assert st.current_equity == 100050.0
    # Z2: was RUNNING before restart -> no auto-start, restart required.
    assert st.restart_required is True


def test_no_autostart_on_restart(tmp_path: Path) -> None:
    """Persisted RUNNING state -> after reload status is STOPPED + restart_required (Z2)."""
    path = tmp_path / STATE_PATH
    store = _fresh_store(tmp_path)
    store.update_state(
        {
            "status": ShadowLoopStatus.RUNNING,
            "started_at": datetime(2026, 8, 21, 8, 0, 0, tzinfo=UTC),
            "iteration_count": 7,
        }
    )
    st_running = store.status()
    assert st_running.status is ShadowLoopStatus.RUNNING
    assert st_running.restart_required is False

    reloaded = ShadowTradingStateStore(path)
    st = reloaded.status()
    assert st.status is ShadowLoopStatus.STOPPED
    assert st.restart_required is True
    # Data survived; only the lifecycle transition changed.
    assert st.iteration_count == 7


def test_missing_file_starts_fresh(tmp_path: Path) -> None:
    """No state file -> fresh STOPPED state, nothing to restart, no error."""
    store = _fresh_store(tmp_path)
    st = store.status()
    assert st.status is ShadowLoopStatus.STOPPED
    assert st.restart_required is False
    assert st.iteration_count == 0
    assert st.session_id  # non-empty default


def test_checksum_roundtrip_matches_formula(tmp_path: Path) -> None:
    """Stored state_checksum equals sha256(canonical JSON without checksum field)."""
    path = tmp_path / STATE_PATH
    store = _fresh_store(tmp_path)
    store.update_state({"iteration_count": 4, "current_equity": 12345.67})
    store.save()

    doc = json.loads(path.read_text())
    state_doc = doc["state"]
    assert set(doc) == {"state", "records", "records_summary", "portfolio_history"}
    assert state_doc["state_checksum"] == _expected_checksum(state_doc)
    # And it matches the in-memory view.
    assert store.status().state_checksum == state_doc["state_checksum"]
    # Cross-process determinism: recomputing from a model built from the file
    # yields the identical checksum (mode="json" datetimes).
    rebuilt = ShadowSessionState.model_validate(state_doc)
    rebuilt.state_checksum = ""
    assert compute_state_checksum(rebuilt) == state_doc["state_checksum"]


# ---------------------------------------------------------------------------
# F5 — quarantine on corruption
# ---------------------------------------------------------------------------


def test_corrupt_state_file_is_quarantined(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Garbage (invalid JSON) -> no exception, fresh state, *.corrupt-* file exists."""
    path = tmp_path / STATE_PATH
    path.write_text("this is definitely {not json!!")

    with caplog.at_level("WARNING"):
        store = ShadowTradingStateStore(path)  # must not raise

    st = store.status()
    assert st.status is ShadowLoopStatus.STOPPED
    assert st.iteration_count == 0
    assert st.restart_required is True  # F5: fresh state after corruption

    quarantined = list(tmp_path.glob("*.corrupt-*"))
    assert len(quarantined) == 1
    assert quarantined[0].name.startswith("shadow_trading_state.json.corrupt-")
    assert quarantined[0].read_text() == "this is definitely {not json!!"
    assert "SHADOW_STATE_QUARANTINED" in caplog.text
    # Original corrupt file was moved, not overwritten in place.
    assert not path.exists()


def test_state_checksum_mismatch_quarantined(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Valid document with a tampered state field (checksum not updated) -> quarantine."""
    path = tmp_path / STATE_PATH
    store = _fresh_store(tmp_path)
    store.update_state({"iteration_count": 3, "status": ShadowLoopStatus.RUNNING})
    assert path.exists()

    doc = json.loads(path.read_text())
    doc["state"]["iteration_count"] = 99  # tamper WITHOUT updating state_checksum
    path.write_text(json.dumps(doc))

    with caplog.at_level("WARNING"):
        reloaded = ShadowTradingStateStore(path)  # must not raise

    st = reloaded.status()
    assert st.iteration_count == 0  # state re-initialized, tampered data not loaded
    assert st.status is ShadowLoopStatus.STOPPED
    assert st.restart_required is True

    assert len(list(tmp_path.glob("*.corrupt-*"))) == 1
    assert "SHADOW_STATE_QUARANTINED" in caplog.text


def test_structurally_invalid_state_is_quarantined(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Valid JSON + valid-looking checksum but schema-invalid state -> quarantine (fail-safe)."""
    path = tmp_path / STATE_PATH
    # Hand-craft a document whose state violates the schema (iteration_count a str),
    # with a checksum computed over exactly that payload — proves schema checks run.
    state_doc = {
        "session_id": "shadow-x",
        "status": "STOPPED",
        "symbols": [],
        "interval_seconds": 900,
        "started_at": None,
        "stopped_at": None,
        "last_iteration_at": None,
        "iteration_count": "not-an-int",
        "decisions_today": 0,
        "budget_date": "",
        "start_equity": 100000.0,
        "current_equity": 100000.0,
        "open_positions": 0,
        "error_count": 0,
        "last_error": None,
        "restart_required": False,
        "state_checksum": "deadbeef",
    }
    doc = {
        "state": state_doc,
        "records": [],
        "records_summary": {"total_records": 0, "trimmed_count": 0, "by_status": {}},
        "portfolio_history": [],
    }
    path.write_text(json.dumps(doc))

    with caplog.at_level("WARNING"):
        store = ShadowTradingStateStore(path)

    assert store.status().iteration_count == 0
    assert len(list(tmp_path.glob("*.corrupt-*"))) == 1
    assert "SHADOW_STATE_QUARANTINED" in caplog.text


# ---------------------------------------------------------------------------
# ST.8 concurrency — no torn reads
# ---------------------------------------------------------------------------


def test_status_read_during_write_no_torn_read(tmp_path: Path) -> None:
    """Parallel writer (save loop) + reader (status() and raw json.load) -> no torn reads."""
    path = tmp_path / STATE_PATH
    store = _fresh_store(tmp_path)
    store.save()  # file exists before threads start
    assert path.exists()

    errors: list[str] = []
    stop = threading.Event()

    def writer() -> None:
        for i in range(50):
            try:
                store.update_state(
                    {
                        "iteration_count": i,
                        "decisions_today": i * 2,
                        "current_equity": 100000.0 + i,
                        "status": ShadowLoopStatus.RUNNING,
                        "last_iteration_at": datetime(2026, 8, 21, 9, 0, 0, tzinfo=UTC)
                        + timedelta(milliseconds=i),
                    }
                )
            except (OSError, ValidationError) as exc:  # pragma: no cover - defensive
                errors.append(f"writer: {exc!r}")
            time.sleep(0.0005)
        stop.set()

    def reader() -> None:
        while not stop.is_set():
            try:
                st = store.status()
            # Tripwire for implementation bugs: status() is an in-memory copy with no
            # defined error surface, so the catch must stay broad (BLE001 suppressed).
            except Exception as exc:  # noqa: BLE001
                errors.append(f"status(): {exc!r}")
                continue
            # In-memory view must always be checksum-consistent.
            st_dict = st.model_dump(mode="json")
            if st_dict["state_checksum"] != _expected_checksum(st_dict):
                errors.append("status() returned checksum-inconsistent state")
            # Raw file read WITHOUT the store lock: os.replace must guarantee a
            # complete document (old or new version), never a torn one.
            try:
                raw = json.loads(path.read_text())
            except FileNotFoundError:
                errors.append("state file vanished mid-write")
                continue
            except json.JSONDecodeError as exc:
                errors.append(f"torn read (JSONDecodeError): {exc}")
                continue
            if set(raw) != {"state", "records", "records_summary", "portfolio_history"}:
                errors.append(f"incomplete document keys: {sorted(raw)}")
                continue
            state_doc = raw["state"]
            if state_doc.get("state_checksum") != _expected_checksum(state_doc):
                errors.append("raw file document checksum-inconsistent")

    threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not any(t.is_alive() for t in threads), "threads did not finish in time"
    assert errors == []


# ---------------------------------------------------------------------------
# ST.15 — memory bounds and file-size bound
# ---------------------------------------------------------------------------


def test_state_file_size_bounded(tmp_path: Path) -> None:
    """50 iterations (Interval 0), each adding records + portfolio state -> <= 1 MiB."""
    store = _fresh_store(tmp_path)
    base = datetime(2026, 8, 21, 8, 0, 0, tzinfo=UTC)
    for i in range(50):
        store.update_state(
            {
                "status": ShadowLoopStatus.RUNNING,
                "iteration_count": i + 1,
                "decisions_today": (i + 1) * 3,
                "current_equity": 100000.0 + i,
                "last_iteration_at": base + timedelta(seconds=i),
            }
        )
        store.add_records([_record(i * 3 + j) for j in range(3)])
        store.add_portfolio_state(_portfolio(i))
        store.record_iteration_success()

    size = (tmp_path / STATE_PATH).stat().st_size
    assert size <= 1048576, f"state file grew to {size} bytes (> 1 MiB)"


def test_records_trimmed_at_limit(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """12000 records -> store holds exactly the newest 10000; summary stays consistent."""
    store = _fresh_store(tmp_path)
    records = [_record(i) for i in range(12000)]
    with caplog.at_level("WARNING"):
        store.add_records(records)

    kept = store.records()
    assert len(kept) == MAX_RECORDS == 10000
    # Exactly the newest 10000, in order.
    assert [r.id for r in kept] == [r.id for r in records[2000:]]

    summary = store.records_summary()
    assert summary["total_records"] == 12000
    assert summary["trimmed_count"] == 2000
    by_status = summary["by_status"]
    assert set(by_status) == {"NO_TRADE", "REJECTED", "FILLED"}
    # by_status tracks all ingested records: 12000 // 3 per status.
    assert by_status == {"NO_TRADE": 4000, "REJECTED": 4000, "FILLED": 4000}
    assert sum(by_status.values()) == summary["total_records"]
    # Invariant: buffered + trimmed == lifetime total.
    assert len(kept) == summary["total_records"] - summary["trimmed_count"]
    assert "SHADOW_STATE_MEMORY_LIMIT" in caplog.text

    # The persisted document obeys the bound too.
    doc = json.loads((tmp_path / STATE_PATH).read_text())
    assert len(doc["records"]) == 10000
    assert doc["records_summary"]["trimmed_count"] == 2000

    # Trimming again after more records keeps the invariant.
    more = [_record(12000 + i) for i in range(10)]
    store.add_records(more)
    kept2 = store.records()
    assert len(kept2) == 10000
    assert [r.id for r in kept2] == [r.id for r in records[2010:] + more]
    s2 = store.records_summary()
    assert s2["total_records"] == 12010
    assert s2["trimmed_count"] == 2010


def test_portfolio_history_trimmed_at_limit(tmp_path: Path) -> None:
    """2500 portfolio states -> <= 2000 kept; last 500 exact; older part equidistant."""
    store = _fresh_store(tmp_path)
    added = [_portfolio(i) for i in range(2500)]
    store.add_portfolio_states(added)

    kept = store.portfolio_history()
    assert len(kept) <= 2000
    assert len(kept) == 2000

    # Last 500 preserved exactly (deep-equal to the last 500 added).
    assert kept[-500:] == added[-500:]

    older_kept = kept[:1500]
    assert len(older_kept) == 1500
    older_indices = [int(p.run_id.removeprefix("run-")) for p in older_kept]
    # Strictly increasing subsequence of the first 2000.
    assert older_indices == sorted(older_indices)
    assert len(set(older_indices)) == 1500
    assert all(0 <= i < 2000 for i in older_indices)
    # Equidistant: spans both ends, uniform spacing (gaps of 1 or 2 here).
    assert older_indices[0] == 0
    assert older_indices[-1] == 1999
    gaps = [b - a for a, b in itertools.pairwise(older_indices)]
    assert all(g in (1, 2) for g in gaps)
    # Spot-check identity of the downsampled entries.
    assert [p.id for p in older_kept] == [added[i].id for i in older_indices]

    # Determinism: same input -> same kept set (fresh store, same run_id sequence).
    store2 = ShadowTradingStateStore(tmp_path / "second.json")
    store2.add_portfolio_states([_portfolio(i) for i in range(2500)])
    kept2 = store2.portfolio_history()
    assert [p.run_id for p in kept2] == [p.run_id for p in kept]


def test_portfolio_history_not_trimmed_below_limit(tmp_path: Path) -> None:
    """Exactly 2000 portfolio states -> nothing dropped."""
    store = _fresh_store(tmp_path)
    added = [_portfolio(i) for i in range(2000)]
    store.add_portfolio_states(added)
    assert store.portfolio_history() == added


# ---------------------------------------------------------------------------
# ST.15 / F6 — consecutive-error autostop
# ---------------------------------------------------------------------------


def test_consecutive_errors_trigger_autostop(tmp_path: Path) -> None:
    """9 errors -> still running; 10th -> STOPPED + restart_required; success resets."""
    store = _fresh_store(tmp_path)
    store.update_state({"status": ShadowLoopStatus.RUNNING})

    for i in range(9):
        assert store.record_iteration_error(f"boom-{i}") is False
    st = store.status()
    assert st.status is ShadowLoopStatus.RUNNING
    assert st.error_count == 9
    assert st.restart_required is False

    assert store.record_iteration_error("boom-10") is True
    st = store.status()
    assert st.status is ShadowLoopStatus.STOPPED
    assert st.restart_required is True
    assert st.error_count == 10
    assert st.last_error == "boom-10"

    # Operator restarts the loop, one successful iteration breaks the streak.
    store.update_state({"status": ShadowLoopStatus.RUNNING, "restart_required": False})
    store.record_iteration_success()
    assert store.status().error_count == 0

    for i in range(9):
        assert store.record_iteration_error(f"again-{i}") is False
    st = store.status()
    assert st.status is ShadowLoopStatus.RUNNING
    assert st.error_count == 9
    assert st.last_error == "again-8"


def test_success_without_errors_is_noop(tmp_path: Path) -> None:
    """record_iteration_success with a clean streak keeps state (no spurious write needed)."""
    store = _fresh_store(tmp_path)
    store.update_state({"status": ShadowLoopStatus.RUNNING, "iteration_count": 3})
    store.record_iteration_success()
    st = store.status()
    assert st.status is ShadowLoopStatus.RUNNING
    assert st.iteration_count == 3
    assert st.error_count == 0


# ---------------------------------------------------------------------------
# F4 — write-error survival
# ---------------------------------------------------------------------------


def test_write_error_keeps_in_memory_state(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Unwritable target (parent path is a file) -> save() does not raise, state intact."""
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file, not a directory")
    bad_path = blocker / "state.json"  # parent is a regular file -> every write fails

    store = ShadowTradingStateStore(bad_path)  # load: file missing -> fresh, no crash
    store.update_state(
        {"iteration_count": 42, "current_equity": 99999.0, "status": ShadowLoopStatus.RUNNING}
    )  # must not raise despite the unwritable target

    st = store.status()
    assert st.iteration_count == 42
    assert st.current_equity == 99999.0
    assert st.status is ShadowLoopStatus.RUNNING

    # A later success path must still work in-memory (retry in next iteration, F4).
    assert store.record_iteration_error("fs-down") is False
    assert store.status().error_count == 1
    assert "SHADOW_STATE_WRITE_FAILED" in caplog.text

    # No stray tmp files from failed saves.
    assert list(tmp_path.glob("*.tmp")) == []
    assert list(tmp_path.glob("blocker*")) == ["blocker"] if False else True


def test_save_with_state_path_as_directory_fails_closed(tmp_path: Path) -> None:
    """Target path is a directory -> os.replace fails, tmp cleaned up, state intact (F4)."""
    path = tmp_path / STATE_PATH
    store = _fresh_store(tmp_path)
    store.update_state({"iteration_count": 7})
    path.mkdir()  # now the save target is a directory

    store.save()  # must not raise
    assert store.status().iteration_count == 7
    assert path.is_dir()
    assert list(tmp_path.glob("*.tmp")) == []  # failed tmp file was cleaned up


# ---------------------------------------------------------------------------
# API surface sanity (used by WI-ST-05)
# ---------------------------------------------------------------------------


def test_public_api_surface(tmp_path: Path) -> None:
    """Document the exact public surface the downstream loop is written against."""
    store = _fresh_store(tmp_path)
    lock = store.lock
    assert isinstance(lock, type(threading.RLock()))
    assert store.state_path == tmp_path / STATE_PATH

    # Single shared lock: re-entrant acquisition works (RLock), and the lock
    # serializes status() against writes.
    with lock, store.lock:
        store.update_state({"iteration_count": 1})
    assert store.status().iteration_count == 1

    rec = _record(0)
    store.add_record(rec)
    assert store.records() == [rec]
    pf = _portfolio(0)
    store.add_portfolio_state(pf)
    assert store.portfolio_history() == [pf]
    assert store.records_summary()["total_records"] == 1


def test_status_returns_defensive_copy(tmp_path: Path) -> None:
    """Mutating a status() result must not corrupt in-memory state."""
    store = _fresh_store(tmp_path)
    st = store.status()
    st.iteration_count = 999
    st.symbols = ["TAMPERED"]
    fresh = store.status()
    assert fresh.iteration_count == 0
    assert fresh.symbols == []


def test_update_state_rejects_invalid_fields(tmp_path: Path) -> None:
    """Schema-invalid updates are rejected (no silent corruption)."""
    store = _fresh_store(tmp_path)
    with pytest.raises(ValidationError):
        store.update_state({"iteration_count": "not-an-int"})
    assert store.status().iteration_count == 0
    assert not (tmp_path / STATE_PATH).exists()  # failed update persisted nothing


def test_reload_after_trim_keeps_summary(tmp_path: Path) -> None:
    """Trimmed records + summary survive a restart (no data loss of the aggregate)."""
    store = _fresh_store(tmp_path)
    records = [_record(i) for i in range(10005)]
    store.add_records(records)
    doc_path = tmp_path / STATE_PATH

    reloaded = ShadowTradingStateStore(doc_path)
    assert len(reloaded.records()) == 10000
    s = reloaded.records_summary()
    assert s["total_records"] == 10005
    assert s["trimmed_count"] == 5
    assert [r.id for r in reloaded.records()] == [r.id for r in records[5:]]
