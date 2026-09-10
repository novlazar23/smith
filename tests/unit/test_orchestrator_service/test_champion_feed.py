"""Tests für den Champion-Challenger-Feed (Evaluations-Artefakt → Overrides)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from apps.orchestrator_service.champion_feed import load_status_overrides, load_version_pairs
from apps.orchestrator_service.service import OrchestratorServiceConfig, build_service
from packages.governance.champion_challenger import AgentVersionPair
from packages.schemas.agent_report import AgentStatus

from .conftest import FakeDB, StubProvider


def _write(tmp_path: Path, data: object) -> Path:
    path = tmp_path / "champion_evals.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _good_block(challenger_oos: float = 0.75) -> dict:
    return {
        "champion": {
            "version": "v1",
            "oos_score": 0.70,
            "calibration_score": 0.80,
            "stability_score": 0.95,
            "marginal_contribution": 0.01,
            "shadow_days": 10,
            "samples": 100,
        },
        "challenger": {
            "version": "v2",
            "oos_score": challenger_oos,
            "calibration_score": 0.82,
            "stability_score": 0.94,
            "marginal_contribution": 0.02,
            "shadow_days": 10,
            "samples": 100,
        },
    }


class TestLoadVersionPairs:
    def test_parses_artifact(self, tmp_path: Path) -> None:
        path = _write(tmp_path, {"trend": _good_block()})
        pairs = load_version_pairs(path)
        assert set(pairs) == {"trend"}
        pair = pairs["trend"]
        assert isinstance(pair, AgentVersionPair)
        assert pair.champion.version == "v1"
        assert pair.challenger.version == "v2"
        assert pair.challenger.oos_score == 0.75
        assert pair.challenger.marginal_contribution == 0.02
        assert pair.new_risks == ()
        assert pair.shadow_success is True

    def test_defaults_for_missing_fields(self, tmp_path: Path) -> None:
        path = _write(tmp_path, {"trend": {"champion": {}, "challenger": {"oos_score": 0.5}}})
        pair = load_version_pairs(path)["trend"]
        assert pair.champion.version == "unknown"
        assert pair.champion.oos_score == 0.0
        assert pair.champion.stability_score == 1.0
        assert pair.challenger.oos_score == 0.5

    def test_rejects_missing_role(self, tmp_path: Path) -> None:
        path = _write(tmp_path, {"trend": {"champion": {}}})
        with pytest.raises(ValueError, match="champion"):
            load_version_pairs(path)

    def test_rejects_non_object_top_level(self, tmp_path: Path) -> None:
        path = _write(tmp_path, ["not", "a", "dict"])
        with pytest.raises(ValueError, match="JSON-Objekt"):
            load_version_pairs(path)


class TestLoadStatusOverrides:
    def test_promotes_good_challenger(self, tmp_path: Path) -> None:
        path = _write(tmp_path, {"trend": _good_block()})
        assert load_status_overrides(path) == {"trend": AgentStatus.ACTIVE}

    def test_keeps_shadow_when_oos_flat(self, tmp_path: Path) -> None:
        path = _write(tmp_path, {"trend": _good_block(challenger_oos=0.71)})
        assert load_status_overrides(path) == {"trend": AgentStatus.SHADOW}

    def test_blocks_promotion_on_new_risks(self, tmp_path: Path) -> None:
        block = _good_block()
        block["new_risks"] = ["drift"]
        path = _write(tmp_path, {"trend": block})
        assert load_status_overrides(path) == {"trend": AgentStatus.SHADOW}

    def test_empty_artifact_yields_no_overrides(self, tmp_path: Path) -> None:
        path = _write(tmp_path, {})
        assert load_status_overrides(path) == {}


class TestBuildServiceWiring:
    def test_build_service_loads_overrides_from_path(
        self, tmp_path: Path, fake_db: FakeDB, stub_provider: StubProvider
    ) -> None:
        path = _write(tmp_path, {"trend": _good_block()})
        config = OrchestratorServiceConfig(instruments=("BTC/USDT",), status_overrides_path=path)
        service = build_service(config=config, provider=stub_provider, db=fake_db)
        assert service._status_overrides == {"trend": AgentStatus.ACTIVE}

    def test_build_service_without_path_has_no_overrides(
        self, fake_db: FakeDB, stub_provider: StubProvider
    ) -> None:
        config = OrchestratorServiceConfig(instruments=("BTC/USDT",))
        service = build_service(config=config, provider=stub_provider, db=fake_db)
        assert service._status_overrides is None
