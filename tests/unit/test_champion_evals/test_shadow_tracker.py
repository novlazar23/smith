"""Tests für den Shadow-Survival-Tracker (Promotions-Kriterien 1+2)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from apps.champion_evals.shadow_tracker import (
    MAX_SCORE_HISTORY,
    PROMOTION_MIN_PASSES,
    SHADOW_FILENAME,
    load_shadow_state,
    promotion_status,
    update_shadow_state,
    write_shadow_state,
)


def _entry(code: str = "Code A", score: float = 0.40, admitted_at: str = "2026-09-01T00:00:00+00:00") -> dict[str, object]:
    return {"code": code, "claim": "c", "score": score, "admitted_at": admitted_at}


def test_new_agent_starts_at_zero_passes() -> None:
    artifact = {"agent_x": _entry(score=0.38)}
    state = update_shadow_state({}, artifact, {}, "2026-09-01T04:00:00+00:00")
    assert state["agent_x"]["consecutive_passes"] == 0
    assert state["agent_x"]["score_history"] == [0.38]
    assert state["agent_x"]["admitted_at"] == "2026-09-01T00:00:00+00:00"
    assert state["agent_x"]["last_checked"] == "2026-09-01T04:00:00+00:00"


def test_recheck_pass_increments() -> None:
    artifact = {"agent_x": _entry(score=0.39)}
    previous = {"agent_x": _entry()}
    state = {"agent_x": {"consecutive_passes": 3, "score_history": [0.38], "admitted_at": "d0"}}
    out = update_shadow_state(state, artifact, previous, "d4")
    assert out["agent_x"]["consecutive_passes"] == 4
    assert out["agent_x"]["score_history"] == [0.38, 0.39]


def test_ready_exactly_after_min_passes() -> None:
    artifact = {"agent_x": _entry()}
    previous = {"agent_x": _entry()}
    state: dict[str, dict[str, object]] = {}
    for day in range(PROMOTION_MIN_PASSES + 1):
        state = update_shadow_state(state, artifact, previous, f"day-{day}")
    assert state["agent_x"]["consecutive_passes"] == PROMOTION_MIN_PASSES
    assert promotion_status(state)["agent_x"]["ready"] is True
    short = {"agent_x": {"consecutive_passes": PROMOTION_MIN_PASSES - 1, "score_history": [0.4]}}
    assert promotion_status(short)["agent_x"]["ready"] is False


def test_removed_agent_dropped() -> None:
    state = {"agent_x": {"consecutive_passes": 5, "score_history": [0.4]}}
    out = update_shadow_state(state, {}, {}, "d6")
    assert out == {}


def test_code_change_resets_streak() -> None:
    artifact = {"agent_x": _entry(code="Code B", score=0.41)}
    previous = {"agent_x": _entry(code="Code A")}
    state = {"agent_x": {"consecutive_passes": 7, "score_history": [0.4]}}
    out = update_shadow_state(state, artifact, previous, "d8")
    assert out["agent_x"]["consecutive_passes"] == 0
    assert out["agent_x"]["score_history"] == [0.4, 0.41]


def test_history_capped_at_max() -> None:
    artifact = {"agent_x": _entry()}
    previous = {"agent_x": _entry()}
    state: dict[str, Any] = {"agent_x": {"consecutive_passes": 0, "score_history": [0.1] * (MAX_SCORE_HISTORY - 2)}}
    for day in range(5):
        state = update_shadow_state(state, artifact, previous, f"d{day}")
    history: list[float] = state["agent_x"]["score_history"]
    assert len(history) == MAX_SCORE_HISTORY
    assert history[-1] == 0.4


def test_load_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_shadow_state(tmp_path / SHADOW_FILENAME) == {}


def test_load_corrupt_file_returns_empty(tmp_path: Path) -> None:
    path = tmp_path / SHADOW_FILENAME
    path.write_text("{kaputt", encoding="utf-8")
    assert load_shadow_state(path) == {}


def test_load_non_dict_returns_empty(tmp_path: Path) -> None:
    path = tmp_path / SHADOW_FILENAME
    path.write_text("[1, 2, 3]", encoding="utf-8")
    assert load_shadow_state(path) == {}


def test_load_skips_non_dict_entries(tmp_path: Path) -> None:
    path = tmp_path / SHADOW_FILENAME
    path.write_text(json.dumps({"good": {"a": 1}, "bad": 42}), encoding="utf-8")
    assert load_shadow_state(path) == {"good": {"a": 1}}


def test_write_and_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / SHADOW_FILENAME
    state = {"agent_x": {"consecutive_passes": 2, "score_history": [0.4, 0.41]}}
    write_shadow_state(path, state)
    assert load_shadow_state(path) == state


def test_promotion_status_spread() -> None:
    state = {"agent_x": {"consecutive_passes": 14, "score_history": [0.40, 0.37, 0.42]}}
    status = promotion_status(state)["agent_x"]
    assert status == {
        "passes": 14,
        "ready": True,
        "min_score": 0.37,
        "max_score": 0.42,
        "spread": 0.42 - 0.37,
    }
    assert promotion_status({}) == {}
