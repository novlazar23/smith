"""Fail-soft-Reader der Evolved Agents im Dashboard-Endpunkt."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from apps.api.routers.dashboard import (
    _fetch_evolved_agents,
    _fetch_evolved_archive,
    _fetch_evolved_last_run,
    _fetch_evolved_shadow,
)


def _write_artifact(tmp_path: Path, content: str) -> Path:
    file = tmp_path / "evolved_agents.json"
    file.write_text(content, encoding="utf-8")
    return file


def _write_jsonl(tmp_path: Path, lines: list[str]) -> Path:
    file = tmp_path / "evolved_agents_archive.jsonl"
    file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return file


def test_parses_valid_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifact = {
        "momentum_5m": {
            "version": 2,
            "code": "def predict(o, h, l, c, v, t):\n    return 0.4, 0.3, 0.3",
            "claim": "Momentum auf 5m-Fenster",
            "score": 0.37,
            "admitted_at": "2026-09-14T08:00:00+00:00",
        }
    }
    monkeypatch.setenv("EVOLVED_AGENTS_PATH", str(_write_artifact(tmp_path, json.dumps(artifact))))
    agents = _fetch_evolved_agents()
    assert agents == [
        {
            "name": "momentum_5m",
            "claim": "Momentum auf 5m-Fenster",
            "version": 2,
            "score": 0.37,
            "admitted_at": "2026-09-14T08:00:00+00:00",
            "status": "SHADOW",
        }
    ]


def test_missing_file_returns_empty_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVOLVED_AGENTS_PATH", str(tmp_path / "fehlt.json"))
    assert _fetch_evolved_agents() == []


def test_corrupt_json_raises_for_run_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Defekte Datei wirft — ``_run_source`` setzt das wie bei allen
    # anderen Quellen in die leere Liste um (fail-soft).
    monkeypatch.setenv("EVOLVED_AGENTS_PATH", str(_write_artifact(tmp_path, "{kaputt")))
    with pytest.raises(json.JSONDecodeError):
        _fetch_evolved_agents()


def test_last_run_parses_valid_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVOLVED_AGENTS_PATH", str(tmp_path / "evolved_agents.json"))
    last_run = {
        "run_at": "2026-09-16T07:39:12+00:00",
        "candidates": [
            {"name": "wick_x", "kind": "kandidat", "admitted": True, "score": 0.6317, "reasons": []},
            {
                "name": "bad_x",
                "kind": "kandidat",
                "admitted": False,
                "score": 0.295,
                "reasons": ["OOS-Score 0.2950 < Zufalls-Basis 0.3333 + 0.02"],
            },
        ],
    }
    (tmp_path / "evolved_agents_last_run.json").write_text(json.dumps(last_run), encoding="utf-8")
    assert _fetch_evolved_last_run() == {
        "run_at": "2026-09-16T07:39:12+00:00",
        "verdicts": [
            {"name": "wick_x", "kind": "kandidat", "admitted": True, "score": 0.6317, "reasons": []},
            {
                "name": "bad_x",
                "kind": "kandidat",
                "admitted": False,
                "score": 0.295,
                "reasons": ["OOS-Score 0.2950 < Zufalls-Basis 0.3333 + 0.02"],
            },
        ],
        "shadow": {},
    }


def test_last_run_missing_file_returns_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVOLVED_AGENTS_PATH", str(tmp_path / "evolved_agents.json"))
    assert _fetch_evolved_last_run() == {}


def test_last_run_passes_shadow_field_through(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVOLVED_AGENTS_PATH", str(tmp_path / "evolved_agents.json"))
    shadow = {
        "wick_x": {
            "passes": 15,
            "ready": True,
            "min_score": 0.36,
            "max_score": 0.39,
            "spread": 0.03,
        }
    }
    last_run = {
        "run_at": "2026-09-29T07:39:12+00:00",
        "candidates": [{"name": "wick_x", "kind": "bestand", "admitted": True, "score": 0.38, "reasons": []}],
        "shadow": shadow,
    }
    (tmp_path / "evolved_agents_last_run.json").write_text(json.dumps(last_run), encoding="utf-8")
    result = _fetch_evolved_last_run()
    assert result["shadow"] == shadow
    assert result["run_at"] == "2026-09-29T07:39:12+00:00"
    assert result["verdicts"][0]["name"] == "wick_x"


def test_last_run_shadow_defaults_to_empty_dict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Kein ``shadow``-Feld (vor dem ersten Shadow-Tracker-Lauf) → ``{}``.
    monkeypatch.setenv("EVOLVED_AGENTS_PATH", str(tmp_path / "evolved_agents.json"))
    last_run = {"run_at": "2026-09-16T07:39:12+00:00", "candidates": []}
    (tmp_path / "evolved_agents_last_run.json").write_text(json.dumps(last_run), encoding="utf-8")
    assert _fetch_evolved_last_run()["shadow"] == {}


def test_shadow_parses_valid_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVOLVED_AGENTS_PATH", str(tmp_path / "evolved_agents.json"))
    shadow = {
        "momentum_5m": {
            "admitted_at": "2026-09-14T08:00:00+00:00",
            "consecutive_passes": 15,
            "score_history": [0.37, 0.38, 0.36],
            "last_checked": "2026-09-29T07:39:12+00:00",
        }
    }
    (tmp_path / "evolved_agents_shadow.json").write_text(json.dumps(shadow), encoding="utf-8")
    assert _fetch_evolved_shadow() == shadow


def test_shadow_missing_file_returns_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVOLVED_AGENTS_PATH", str(tmp_path / "evolved_agents.json"))
    assert _fetch_evolved_shadow() == {}


def test_shadow_corrupt_json_raises_for_run_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Defekte Datei wirft — ``_run_source`` setzt das wie bei allen
    # anderen Quellen in ``{}`` um (fail-soft).
    monkeypatch.setenv("EVOLVED_AGENTS_PATH", str(tmp_path / "evolved_agents.json"))
    (tmp_path / "evolved_agents_shadow.json").write_text("{kaputt", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        _fetch_evolved_shadow()


def test_archive_returns_rejections_in_file_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Gemischte JSONL: nur die admitted-False-Zeilen, in
    # Datei-Reihenfolge, unverändert weitergereicht.
    monkeypatch.setenv("EVOLVED_AGENTS_PATH", str(tmp_path / "evolved_agents.json"))
    entries = [
        {"run_at": "2026-09-01T00:00:00+00:00", "name": "a1", "admitted": False, "score": 0.31},
        {"run_at": "2026-09-02T00:00:00+00:00", "name": "ok1", "admitted": True, "score": 0.36},
        {"run_at": "2026-09-03T00:00:00+00:00", "name": "a2", "admitted": False, "score": 0.30},
        {"run_at": "2026-09-04T00:00:00+00:00", "name": "ok2", "admitted": True, "score": 0.37},
        {"run_at": "2026-09-05T00:00:00+00:00", "name": "a3", "admitted": False, "score": 0.32},
    ]
    _write_jsonl(tmp_path, [json.dumps(entry) for entry in entries])
    result = _fetch_evolved_archive()
    assert [entry["name"] for entry in result] == ["a1", "a2", "a3"]
    assert result[0] == entries[0]


def test_archive_missing_file_returns_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVOLVED_AGENTS_PATH", str(tmp_path / "evolved_agents.json"))
    assert _fetch_evolved_archive() == []


def test_archive_skips_corrupt_lines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Korrupte Einzelseilen werden übersprungen (Writer-Seiten-Konvention),
    # lesbare Zeilen bleiben erhalten.
    monkeypatch.setenv("EVOLVED_AGENTS_PATH", str(tmp_path / "evolved_agents.json"))
    _write_jsonl(
        tmp_path,
        [
            json.dumps({"name": "a1", "admitted": False}),
            "{kaputt",
            json.dumps({"name": "ok1", "admitted": True}),
            "definitely not json",
            json.dumps({"name": "a2", "admitted": False}),
        ],
    )
    assert [entry["name"] for entry in _fetch_evolved_archive()] == ["a1", "a2"]


def test_archive_caps_at_fifteen_newest_rejections(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Append-only (älteste → neueste) → bei > 15 Ablehnungen bleiben
    # exakt die 15 jüngsten.
    monkeypatch.setenv("EVOLVED_AGENTS_PATH", str(tmp_path / "evolved_agents.json"))
    lines = [json.dumps({"name": f"agent_{i}", "admitted": False}) for i in range(20)]
    _write_jsonl(tmp_path, lines)
    result = _fetch_evolved_archive()
    assert len(result) == 15
    assert [entry["name"] for entry in result] == [f"agent_{i}" for i in range(5, 20)]
