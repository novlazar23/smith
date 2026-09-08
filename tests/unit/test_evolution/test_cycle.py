"""Tests für die Zyklus-Orchestrierung (Budget, Preregistrierung, Verdict, State)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import pytest
from apps.evolution import cycle as cycle_mod
from apps.evolution import evaluate as evaluate_mod
from apps.evolution.cycle import CycleBudget, run_cycle
from apps.evolution.models import Proposal, TestPlan, Variant, variant_key
from apps.evolution.state import EvolutionStore
from packages.backtesting.core import Candle
from tests.unit.test_evolution.conftest import make_feed_factory, make_hypothesis

BASELINE_PARAMS = {"period": 30.0, "buy_below": 20.0, "sell_above": 80.0}
CANDIDATE_PARAMS = {"period": 30.0, "buy_below": 25.0, "sell_above": 75.0}


def _candidate_proposal(test_plan: TestPlan) -> Proposal:
    return Proposal(
        family="rsi_mean_reversion",
        kind="config",
        claim="Tieferer Oversold-Entry (b25) soll die OOS-Marge ueber der Baseline erhoeen.",
        variant=Variant(strategy="rsi_mean_reversion", params=dict(CANDIDATE_PARAMS)),
        test_plan=test_plan,
    )


def _stub_run_window(*, weak: bool) -> Callable[..., dict[str, Any]]:
    """Ersetzt die Engine: Baseline schwach, Kandidat stark bzw. knapp."""

    def fake(
        candles: Sequence[Candle],
        strategy_name: str,
        params: dict[str, float],
        *,
        initial_capital: float = 100_000.0,
        trade_notional: float = 10_000.0,
        timeframe: str = "5m",
    ) -> dict[str, Any]:
        if params == BASELINE_PARAMS:
            return {
                "return_pct": 0.0,
                "max_dd_pct": 1.0,
                "legs": 0,
                "win_rate": None,
                "n_candles": len(candles),
                "sharpe_ratio": None,
                "profit_factor": None,
            }
        return {
            "return_pct": 0.5 if weak else 5.0,
            "max_dd_pct": 2.0,
            "legs": 30,
            "win_rate": 0.8,
            "n_candles": len(candles),
            "sharpe_ratio": None,
            "profit_factor": None,
        }

    return fake


# ─── Budget ──────────────────────────────────────────────────────────────────


def test_cycle_budget_tracks_runs() -> None:
    budget = CycleBudget(max_runs=3)
    assert budget.runs_left == 3
    assert budget.exhausted() is False
    budget.spend_runs(2)
    assert budget.runs_left == 1
    budget.spend_runs(1)
    assert budget.runs_left == 0
    assert budget.exhausted() is True


def test_cycle_budget_exhausts_on_zero_minutes() -> None:
    assert CycleBudget(max_minutes=0.0).exhausted() is True


# ─── Zyklus-Verhalten ────────────────────────────────────────────────────────


def test_run_cycle_promotes_strong_candidate(
    store: EvolutionStore,
    small_plan: TestPlan,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(evaluate_mod, "run_window", _stub_run_window(weak=False))
    monkeypatch.setattr(cycle_mod, "fetch_live_paper", lambda: None)
    summary = run_cycle(store, make_feed_factory(), proposals=[_candidate_proposal(small_plan)])
    assert summary["new_proposals"] == 1
    assert summary["promoted"] == 1
    assert summary["rejected"] == 0
    assert summary["error"] == 0
    assert summary["tested"] == 1
    hypothesis = store.all_hypotheses()[0]
    assert hypothesis.status == "passed"
    assert hypothesis.verdict is not None
    assert hypothesis.verdict["decision"] == "promoted"
    assert store.registry()["promoted"][0]["id"] == hypothesis.id
    assert store.promoted_strategy_names() == {"rsi_mean_reversion"}
    assert (store.root / "digests").is_dir()
    assert summary["last_cycle"]["n_promoted"] == 1


def test_run_cycle_rejects_weak_candidate_and_graves_it(
    store: EvolutionStore,
    small_plan: TestPlan,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(evaluate_mod, "run_window", _stub_run_window(weak=True))
    monkeypatch.setattr(cycle_mod, "fetch_live_paper", lambda: None)
    summary = run_cycle(store, make_feed_factory(), proposals=[_candidate_proposal(small_plan)])
    assert summary["rejected"] == 1
    assert summary["promoted"] == 0
    hypothesis = store.all_hypotheses()[0]
    assert hypothesis.status == "rejected"
    assert hypothesis.verdict is not None
    assert any(r.startswith("OOS-Marge") for r in hypothesis.verdict["reasons"])
    grave = store.graveyard()[0]
    assert grave["id"] == hypothesis.id
    assert store.registry()["promoted"] == []


def test_run_cycle_keeps_hypothesis_pending_when_budget_exhausted(
    store: EvolutionStore,
    small_plan: TestPlan,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(evaluate_mod, "run_window", _stub_run_window(weak=False))
    monkeypatch.setattr(cycle_mod, "fetch_live_paper", lambda: None)
    summary = run_cycle(
        store,
        make_feed_factory(),
        proposals=[_candidate_proposal(small_plan)],
        budget=CycleBudget(max_runs=0),
    )
    assert summary["pending"] == 1
    assert summary["tested"] == 0
    assert store.all_hypotheses()[0].status == "pending"


def test_run_cycle_skips_duplicate_variant(
    store: EvolutionStore,
    small_plan: TestPlan,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cycle_mod, "fetch_live_paper", lambda: None)
    store.add_hypothesis(make_hypothesis(status="rejected", params=dict(CANDIDATE_PARAMS)))
    summary = run_cycle(store, make_feed_factory(), proposals=[_candidate_proposal(small_plan)])
    assert summary["new_proposals"] == 0
    assert summary["tested"] == 0
    assert len(store.all_hypotheses()) == 1  # nur die bereits rejected Vor-Registrierung


def test_run_cycle_skips_graveyarded_variant(
    store: EvolutionStore,
    small_plan: TestPlan,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cycle_mod, "fetch_live_paper", lambda: None)
    variant = Variant(strategy="rsi_mean_reversion", params=dict(CANDIDATE_PARAMS))
    store.add_graveyard({"id": "g1", "variant_key": variant_key(variant), "reasons": []})
    summary = run_cycle(store, make_feed_factory(), proposals=[_candidate_proposal(small_plan)])
    assert summary["new_proposals"] == 0
    assert summary["tested"] == 0
    assert store.all_hypotheses() == []


def test_run_cycle_graves_mechanism_on_jail_failure(
    store: EvolutionStore,
    small_plan: TestPlan,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cycle_mod, "fetch_live_paper", lambda: None)
    proposal = Proposal(
        family="test_strategy",
        kind="mechanism",
        claim="Neuer Mechanismus, der im Jail scheitern muss (Import os).",
        variant=Variant(
            strategy="test_strategy",
            code="import os\n",
            code_file="packages/strategies/test_strategy.py",
        ),
        test_plan=small_plan,
    )
    summary = run_cycle(store, make_feed_factory(), proposals=[proposal])
    assert summary["error"] == 1
    assert summary["tested"] == 1
    hypothesis = store.all_hypotheses()[0]
    assert hypothesis.status == "error"
    assert hypothesis.verdict is not None
    assert hypothesis.verdict["reasons"][0].startswith("Jail")
    assert len(store.graveyard()) == 1
    assert not (tmp_path / "packages" / "strategies" / "test_strategy.py").exists()


def test_run_cycle_end_to_end_with_real_engine(
    store: EvolutionStore,
    small_plan: TestPlan,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cycle_mod, "fetch_live_paper", lambda: None)
    proposal = Proposal(
        family="rsi_mean_reversion",
        kind="config",
        claim="Tieferer Oversold-Entry (b25) auf synthetischen 5m-Kerzen.",
        variant=Variant(strategy="rsi_mean_reversion", params={"buy_below": 25.0}),
        test_plan=small_plan,
    )
    summary = run_cycle(store, make_feed_factory(), proposals=[proposal])
    assert summary["tested"] == 1
    hypothesis = store.all_hypotheses()[0]
    assert hypothesis.status in ("passed", "rejected")
    assert hypothesis.results is not None
    assert hypothesis.results["portfolio"]["oos"]["n"] == 1
    assert (store.root / "digests").is_dir()
