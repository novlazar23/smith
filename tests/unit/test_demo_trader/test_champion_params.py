"""Tests für die Champion-Parameter-Injection im ACTIVE-Ensemble des Demo-Traders."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from apps.demo_trader.service import build_active_ensemble
from apps.orchestrator_service.service import ContextualAgent
from packages.agents.trend_agent import TrendAgent, TrendParams
from packages.schemas.agent_report import AgentStatus


def _trend(agents: list[ContextualAgent]) -> TrendAgent:
    inner = next(agent._agent for agent in agents if agent.agent_id == "trend")  # type: ignore[attr-defined]
    assert isinstance(inner, TrendAgent)
    return inner


def test_injects_champion_params_from_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "champion_configs.json"
    path.write_text(
        json.dumps({"trend": {"version": 1, "params": {"ema_fast": 7, "p_cap": 0.9}, "score": 0.5}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEMO_CHAMPION_CONFIGS", str(path))

    agents = build_active_ensemble("BTC/USDT", "15m")

    trend = _trend(agents)
    assert trend._params.ema_fast == 7
    assert trend._params.p_cap == 0.9
    for agent in agents:
        assert agent._agent.config.status is AgentStatus.ACTIVE  # type: ignore[attr-defined]


def test_missing_file_falls_back_to_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEMO_CHAMPION_CONFIGS", str(tmp_path / "fehlt.json"))

    agents = build_active_ensemble("BTC/USDT", "15m")

    assert _trend(agents)._params == TrendParams()


def test_corrupt_file_falls_back_to_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "champion_configs.json"
    path.write_text("kein JSON", encoding="utf-8")
    monkeypatch.setenv("DEMO_CHAMPION_CONFIGS", str(path))

    agents = build_active_ensemble("BTC/USDT", "15m")

    assert _trend(agents)._params == TrendParams()
