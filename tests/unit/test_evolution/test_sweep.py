"""Tests für Grid-Erzeugung, Validierung und Preregistrierung (Sweep)."""

from __future__ import annotations

from apps.evolution.models import Variant, variant_key
from apps.evolution.state import EvolutionStore
from apps.evolution.sweep import grid_combinations, propose_grid, sweep_report, validate_grid
from tests.unit.test_evolution.conftest import make_hypothesis


def test_grid_combinations_empty_grid_yields_single_empty_combo() -> None:
    assert grid_combinations({}) == [{}]


def test_grid_combinations_single_and_multiple_axes() -> None:
    assert grid_combinations({"a": [1.0, 2.0]}) == [{"a": 1.0}, {"a": 2.0}]
    combos = grid_combinations({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    assert combos == [
        {"a": 1.0, "b": 3.0},
        {"a": 1.0, "b": 4.0},
        {"a": 2.0, "b": 3.0},
        {"a": 2.0, "b": 4.0},
    ]


def test_validate_grid_unknown_strategy() -> None:
    assert validate_grid({"a": [1.0]}, "nope_strategy") == ["unbekannte Strategie 'nope_strategy'"]


def test_validate_grid_unknown_and_out_of_bounds_params() -> None:
    problems = validate_grid({"foo": [1.0], "buy_below": [50.0]}, "rsi_mean_reversion")
    assert any("foo" in p and "Manifest" in p for p in problems)
    assert any("buy_below=50.0" in p for p in problems)


def test_validate_grid_accepts_in_bounds_params() -> None:
    assert validate_grid({"period": [30.0], "buy_below": [25.0]}, "rsi_mean_reversion") == []


def test_propose_grid_skips_empty_params(store: EvolutionStore) -> None:
    new, skipped = propose_grid(store, "rsi_mean_reversion", {})
    assert new == []
    assert any("Baseline-Konfiguration" in s for s in skipped)


def test_propose_grid_preregisters_and_dedups(store: EvolutionStore) -> None:
    new, skipped = propose_grid(store, "rsi_mean_reversion", {"buy_below": [25.0, 30.0]})
    assert skipped == []
    assert [h.variant.params for h in new] == [{"buy_below": 25.0}, {"buy_below": 30.0}]
    assert all(h.status == "proposed" and h.source == "sweep" for h in new)
    assert all(h.kind == "config" for h in new)

    again, skipped_again = propose_grid(store, "rsi_mean_reversion", {"buy_below": [25.0, 30.0]})
    assert again == []
    assert len(skipped_again) == 2
    assert all("bereits preregistriert" in s for s in skipped_again)


def test_propose_grid_skips_graveyarded_variants(store: EvolutionStore) -> None:
    variant = Variant(strategy="rsi_mean_reversion", params={"buy_below": 27.0})
    store.add_graveyard({"id": "g1", "variant_key": variant_key(variant), "reasons": []})
    new, skipped = propose_grid(store, "rsi_mean_reversion", {"buy_below": [27.0]})
    assert new == []
    assert any("liegt im Grab" in s for s in skipped)


def test_propose_grid_rejects_invalid_grid(store: EvolutionStore) -> None:
    new, skipped = propose_grid(store, "nope_strategy", {"a": [1.0]})
    assert new == []
    assert skipped == ["unbekannte Strategie 'nope_strategy'"]


def test_propose_grid_respects_max_variants(store: EvolutionStore) -> None:
    new, skipped = propose_grid(
        store,
        "rsi_mean_reversion",
        {"period": [5.0, 10.0, 15.0, 20.0]},
        max_variants=2,
    )
    assert len(new) == 2
    assert skipped == []


def test_sweep_report_counts(store: EvolutionStore) -> None:
    store.add_hypothesis(make_hypothesis(params={"buy_below": 21.0}))
    store.add_hypothesis(make_hypothesis(hid="rsi_mean_reversion-20240101-2", params={"buy_below": 22.0}, status="rejected"))
    store.add_graveyard({"id": "g1", "variant_key": "k", "reasons": []})
    store.promote({"id": "p1", "variant": {"strategy": "rsi_mean_reversion", "params": {}}})
    report = sweep_report(store)
    assert report["n_hypotheses"] == 2
    assert report["by_status"] == {"proposed": 1, "rejected": 1}
    assert report["families"] == ["rsi_mean_reversion"]
    assert report["n_graveyard"] == 1
    assert report["n_promoted"] == 1
    assert report["last_cycle"] is None
