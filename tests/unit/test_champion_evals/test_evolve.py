"""Tests für den Evolutions-Kern: Mutation, Varianten, Selektion, Config-Artefakt."""

from __future__ import annotations

import json
from pathlib import Path
from random import Random

import pytest
from apps.champion_evals.agent_params import AGENT_TYPES, build_agent, default_params, param_space
from apps.champion_evals.evolve import (
    build_configs_artifact,
    generate_variants,
    load_champion_configs,
    load_champion_params,
    mutate,
    select,
    write_json_atomic,
)
from apps.champion_evals.score import AgentMetrics
from packages.agents.trend_agent import TrendAgent


def _m(
    agent_id: str,
    cal_brier: float,
    oos_brier: float,
    cal_stab: float = 0.5,
    oos_stab: float = 0.5,
) -> AgentMetrics:
    return AgentMetrics(
        agent_id=agent_id,
        cal_samples=10,
        oos_samples=10,
        cal_brier=cal_brier,
        oos_brier=oos_brier,
        cal_stability=cal_stab,
        oos_stability=oos_stab,
        oos_marginal=0.0,
    )


class TestBuildAgent:
    def test_families_and_defaults(self) -> None:
        assert AGENT_TYPES == ("trend", "mean_reversion", "volatility_regime", "volume_conviction")
        for agent_id in AGENT_TYPES:
            assert build_agent(agent_id).agent_id == agent_id

    def test_builds_with_dict_params(self) -> None:
        agent = build_agent("trend", {"ema_fast": 20})
        assert isinstance(agent, TrendAgent)
        assert agent._params.ema_fast == 20

    def test_unknown_agent_raises(self) -> None:
        with pytest.raises(KeyError):
            build_agent("kein_agent")


class TestMutate:
    def test_stays_in_bounds_and_keeps_types(self) -> None:
        space = param_space("trend")
        params = default_params("trend")
        rng = Random(42)
        for _ in range(200):
            mutated = mutate(params, space, rng, strength=0.5)
            for name, (kind, lo, hi, _step) in space.items():
                value = getattr(mutated, name)
                assert lo <= value <= hi
                if kind == "int":
                    assert isinstance(value, int)

    def test_deterministic_for_same_seed(self) -> None:
        space = param_space("trend")
        params = default_params("trend")
        assert mutate(params, space, Random(7), strength=0.3) == mutate(params, space, Random(7), strength=0.3)

    def test_changes_values_for_some_seed(self) -> None:
        space = param_space("trend")
        base = default_params("trend")
        assert any(mutate(base, space, Random(s), strength=0.5).to_dict() != base.to_dict() for s in range(10))


class TestGenerateVariants:
    def test_deterministic_ids_and_distinctness(self) -> None:
        champion = default_params("trend").to_dict()
        v1 = generate_variants("trend", champion, 8, seed=42)
        v2 = generate_variants("trend", champion, 8, seed=42)
        assert [vid for vid, _ in v1] == [f"trend:v{i}" for i in range(8)]
        assert [(vid, p.to_dict()) for vid, p in v1] == [(vid, p.to_dict()) for vid, p in v2]
        dicts = [p.to_dict() for _, p in v1]
        assert len({tuple(sorted(d.items())) for d in dicts}) == 8
        assert champion not in dicts

    def test_generates_for_all_families(self) -> None:
        for agent_id in AGENT_TYPES:
            variants = generate_variants(agent_id, default_params(agent_id).to_dict(), 4, seed=1)
            assert len(variants) == 4


class TestSelect:
    def test_promotes_better_stable_variant(self) -> None:
        metrics = {
            "trend": _m("trend", 0.40, 0.40),
            "trend:v0": _m("trend:v0", 0.35, 0.30, cal_stab=0.6, oos_stab=0.6),
        }
        result = select("trend", "trend", metrics)
        assert result.promoted is True
        assert result.selected_id == "trend:v0"
        assert result.champion_score == pytest.approx(0.60)
        assert result.best_score == pytest.approx(0.70)

    def test_keeps_champion_when_worse(self) -> None:
        metrics = {
            "trend": _m("trend", 0.30, 0.30),
            "trend:v0": _m("trend:v0", 0.40, 0.40),
        }
        result = select("trend", "trend", metrics)
        assert result.promoted is False
        assert result.selected_id == "trend"

    def test_requires_promotion_margin(self) -> None:
        metrics = {
            "trend": _m("trend", 0.30, 0.40),
            "trend:v0": _m("trend:v0", 0.30, 0.39),
        }
        assert select("trend", "trend", metrics, promotion_margin=0.02).promoted is False
        assert select("trend", "trend", metrics, promotion_margin=0.005).promoted is True

    def test_stability_guard_blocks_overfit(self) -> None:
        metrics = {
            "trend": _m("trend", 0.30, 0.40),
            "trend:v0": _m("trend:v0", 0.20, 0.25, cal_stab=0.9, oos_stab=0.3),
        }
        result = select("trend", "trend", metrics)
        assert result.promoted is False
        assert result.selected_id == "trend"

    def test_picks_best_variant_among_many(self) -> None:
        metrics = {
            "trend": _m("trend", 0.30, 0.40),
            "trend:v0": _m("trend:v0", 0.30, 0.39),
            "trend:v1": _m("trend:v1", 0.30, 0.20, cal_stab=0.5, oos_stab=0.5),
            "trend:v2": _m("trend:v2", 0.30, 0.35),
        }
        result = select("trend", "trend", metrics)
        assert result.promoted is True
        assert result.selected_id == "trend:v1"


class TestConfigsArtifact:
    def test_first_run_version_one(self) -> None:
        artifact = build_configs_artifact({"trend": ({"ema_fast": 12, "p_cap": 0.85}, 0.62)}, None)
        assert artifact["trend"] == {"version": 1, "params": {"ema_fast": 12, "p_cap": 0.85}, "score": 0.62}

    def test_unchanged_params_keep_block_and_version(self) -> None:
        previous = {"trend": {"version": 3, "params": {"ema_fast": 12}, "score": 0.6}}
        assert build_configs_artifact({"trend": ({"ema_fast": 12}, 0.9)}, previous) == previous

    def test_changed_params_bump_version(self) -> None:
        previous = {"trend": {"version": 3, "params": {"ema_fast": 12}, "score": 0.6}}
        artifact = build_configs_artifact({"trend": ({"ema_fast": 21}, 0.7)}, previous)
        assert artifact["trend"] == {"version": 4, "params": {"ema_fast": 21}, "score": 0.7}


class TestConfigsIO:
    def test_load_missing_returns_none(self, tmp_path: Path) -> None:
        assert load_champion_configs(tmp_path / "champion_configs.json") is None

    def test_write_and_load_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "champion_configs.json"
        payload = {"trend": {"version": 1, "params": {"ema_fast": 12}, "score": 0.62}}
        write_json_atomic(path, payload)
        assert load_champion_configs(path) == payload


class TestLoadChampionParams:
    """Fail-soft-Loader: liefert {agent_id: params}, wirft nie aus."""

    @staticmethod
    def _write(tmp_path: Path, data: object) -> Path:
        path = tmp_path / "champion_configs.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_valid_file_yields_params_per_agent(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path,
            {
                "trend": {"version": 1, "params": {"ema_fast": 7, "p_cap": 0.9}, "score": 0.55},
                "mean_reversion": {"version": 2, "params": {"sma_period": 40.0}, "score": 0.79},
            },
        )
        assert load_champion_params(path) == {
            "trend": {"ema_fast": 7, "p_cap": 0.9},
            "mean_reversion": {"sma_period": 40.0},
        }

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert load_champion_params(tmp_path / "fehlt.json") == {}

    def test_corrupt_file_returns_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "champion_configs.json"
        path.write_text("kein JSON", encoding="utf-8")
        assert load_champion_params(path) == {}

    def test_non_object_top_level_returns_empty(self, tmp_path: Path) -> None:
        assert load_champion_params(self._write(tmp_path, ["a", "list"])) == {}

    def test_agent_without_params_mapping_is_skipped(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path,
            {
                "trend": {"version": 1, "score": 0.55},
                "mean_reversion": {"version": 1, "params": {"sma_period": 40}, "score": 0.79},
            },
        )
        assert load_champion_params(path) == {"mean_reversion": {"sma_period": 40}}

    def test_non_object_agent_block_is_skipped(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, {"trend": "kaputt", "volume_conviction": {"params": {"window": 30}}})
        assert load_champion_params(path) == {"volume_conviction": {"window": 30}}

    def test_non_numeric_param_value_skips_agent(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path,
            {
                "trend": {"params": {"ema_fast": "zwölf"}},
                "mean_reversion": {"params": {"sma_period": 40}},
            },
        )
        assert load_champion_params(path) == {"mean_reversion": {"sma_period": 40}}
