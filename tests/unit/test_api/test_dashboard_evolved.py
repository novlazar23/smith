"""Fail-soft-Reader der Evolved Agents im Dashboard-Endpunkt."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from apps.api.routers.dashboard import _fetch_evolved_agents


def _write_artifact(tmp_path: Path, content: str) -> Path:
    file = tmp_path / "evolved_agents.json"
    file.write_text(content, encoding="utf-8")
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
