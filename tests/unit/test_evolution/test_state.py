"""Tests für den dateibasierten Evolutions-State (JSONL, Meta, Grab, Registry)."""

from __future__ import annotations

from pathlib import Path

import pytest
from apps.evolution import state as state_mod
from apps.evolution.models import Variant, variant_key
from apps.evolution.state import DEFAULT_BASELINE, EvolutionStore, default_state_dir
from tests.unit.test_evolution.conftest import make_hypothesis


def test_default_state_dir_uses_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVOLUTION_STATE_DIR", "/tmp/evo-state")
    assert default_state_dir() == Path("/tmp/evo-state")


def test_default_state_dir_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EVOLUTION_STATE_DIR", raising=False)
    expected = (
        Path("/app/backtest_reports/evolution")
        if Path("/app/backtest_reports").is_dir()
        else Path("evolution")
    )
    assert default_state_dir() == expected


def test_store_creates_root_dir(tmp_path: Path) -> None:
    store = EvolutionStore(tmp_path / "nested" / "root")
    assert store.root.is_dir()


def test_meta_defaults_and_persistence(tmp_path: Path) -> None:
    store = EvolutionStore(tmp_path / "evo")
    meta = store.meta()
    assert meta["baseline"] == DEFAULT_BASELINE
    assert meta["families"] == {}
    assert meta["last_cycle"] is None

    store.set_meta(last_cycle={"n_tested": 2})
    reloaded = EvolutionStore(tmp_path / "evo")
    assert reloaded.meta()["last_cycle"] == {"n_tested": 2}
    assert reloaded.meta()["baseline"] == DEFAULT_BASELINE


def test_meta_recovers_from_corrupt_state_json(tmp_path: Path) -> None:
    store = EvolutionStore(tmp_path / "evo")
    (store.root / state_mod.META_FILE).write_text("kein json", encoding="utf-8")
    assert store.meta()["baseline"] == DEFAULT_BASELINE


def test_registry_recovers_from_corrupt_json(tmp_path: Path) -> None:
    store = EvolutionStore(tmp_path / "evo")
    (store.root / state_mod.REGISTRY_FILE).write_text("{kaputt", encoding="utf-8")
    assert store.registry() == {"promoted": []}


def test_baseline_variant(tmp_path: Path) -> None:
    store = EvolutionStore(tmp_path / "evo")
    base = store.baseline()
    assert base.strategy == "rsi_mean_reversion"
    assert base.params == {"period": 30.0, "buy_below": 20.0, "sell_above": 80.0}


def test_bump_family_and_family_count(tmp_path: Path) -> None:
    store = EvolutionStore(tmp_path / "evo")
    assert store.family_count("rsi_mean_reversion") == 0
    assert store.bump_family("rsi_mean_reversion") == 1
    assert store.bump_family("rsi_mean_reversion") == 2
    assert store.family_count("rsi_mean_reversion") == 2


def test_family_count_falls_back_to_hypothesis_scan(tmp_path: Path) -> None:
    store = EvolutionStore(tmp_path / "evo")
    store.add_hypothesis(make_hypothesis(params={"buy_below": 21.0}))
    store.add_hypothesis(make_hypothesis(hid="rsi_mean_reversion-20240101-2", params={"buy_below": 22.0}))
    store.set_meta(families={})
    assert store.family_count("rsi_mean_reversion") == 2


def test_add_and_update_hypothesis_are_append_only(store: EvolutionStore) -> None:
    hyp = make_hypothesis()
    store.add_hypothesis(hyp)
    file = store.root / state_mod.HYPOTHESES_FILE
    assert len(file.read_text(encoding="utf-8").splitlines()) == 1

    store.update_hypothesis(make_hypothesis(status="rejected"))
    assert len(file.read_text(encoding="utf-8").splitlines()) == 2
    assert [h.status for h in store.all_hypotheses()] == ["rejected"]


def test_all_hypotheses_skips_corrupt_lines(store: EvolutionStore) -> None:
    hyp = make_hypothesis()
    store.add_hypothesis(hyp)
    file = store.root / state_mod.HYPOTHESES_FILE
    file.write_text("kaputte zeile\n" + file.read_text(encoding="utf-8"), encoding="utf-8")
    assert [h.id for h in store.all_hypotheses()] == [hyp.id]


def test_all_hypotheses_sorted_by_id(store: EvolutionStore) -> None:
    store.add_hypothesis(make_hypothesis(hid="rsi_mean_reversion-20240102-1"))
    store.add_hypothesis(make_hypothesis(hid="rsi_mean_reversion-20240101-1"))
    assert [h.id for h in store.all_hypotheses()] == [
        "rsi_mean_reversion-20240101-1",
        "rsi_mean_reversion-20240102-1",
    ]


def test_find_duplicate_matches_strategy_and_params(store: EvolutionStore) -> None:
    variant = Variant(strategy="rsi_mean_reversion", params={"buy_below": 25.0, "period": 30.0})
    store.add_hypothesis(make_hypothesis(params={"period": 30.0, "buy_below": 25.0}))
    assert store.find_duplicate(variant) is not None  # Parameter-Reihenfolge egal
    assert store.find_duplicate(Variant(strategy="rsi_mean_reversion", params={"period": 30.0})) is None
    assert store.find_duplicate(Variant(strategy="ema_cross", params={"period": 30.0, "buy_below": 25.0})) is None


def test_graveyard_roundtrip_and_in_graveyard(store: EvolutionStore) -> None:
    variant = Variant(strategy="rsi_mean_reversion", params={"buy_below": 25.0})
    key = variant_key(variant)
    assert store.in_graveyard(variant) is None
    store.add_graveyard({"id": "g1", "variant_key": key, "reasons": ["OOS-Marge"]})
    assert store.in_graveyard(variant) is not None
    assert store.in_graveyard(variant)["id"] == "g1"
    assert store.in_graveyard(Variant(strategy="rsi_mean_reversion")) is None
    assert len(store.graveyard()) == 1


def test_promote_dedupes_by_id(store: EvolutionStore) -> None:
    entry = {
        "id": "p1",
        "family": "rsi_mean_reversion",
        "variant": {"strategy": "rsi_mean_reversion", "params": {}},
    }
    store.promote(entry)
    store.promote(entry)
    assert len(store.registry()["promoted"]) == 1
    assert store.promoted_strategy_names() == {"rsi_mean_reversion"}
