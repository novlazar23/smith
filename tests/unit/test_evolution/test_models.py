"""Tests für die Preregistrierungs-Modelle (Schema, IDs, Deduplication-Key)."""

from __future__ import annotations

from datetime import UTC, datetime, timezone

import pytest
from apps.evolution import models
from apps.evolution.models import (
    NAME_PATTERN,
    DecisionRule,
    Hypothesis,
    Proposal,
    TestPlan,
    Variant,
    hypothesis_from_proposal,
    make_hypothesis_id,
    utcnow_iso,
    variant_key,
)
from tests.unit.test_evolution.conftest import make_hypothesis


class _FakeDatetime(datetime):
    """Feste UTC-Zeit für deterministische ID- und Zeitstempel-Tests."""

    @classmethod
    def now(cls, tz: timezone | None = None) -> datetime:
        return datetime(2026, 9, 7, 12, 30, 45, tzinfo=UTC)


def test_name_pattern_accepts_valid_names() -> None:
    for name in ("abc", "rsi_mean_reversion", "a1_b", "a" * 40):
        assert NAME_PATTERN.match(name) is not None


def test_name_pattern_rejects_invalid_names() -> None:
    for name in ("ab", "Abc", "1abc", "ab-c", "a" * 41, ""):
        assert NAME_PATTERN.match(name) is None


def test_utcnow_iso_is_deterministic_with_patched_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(models, "datetime", _FakeDatetime)
    assert utcnow_iso() == "2026-09-07T12:30:45+00:00"


def test_make_hypothesis_id_next_free_index(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(models, "datetime", _FakeDatetime)
    assert make_hypothesis_id("testfam", set()) == "testfam-20260907-1"
    assert make_hypothesis_id("testfam", {"testfam-20260907-1"}) == "testfam-20260907-2"
    assert make_hypothesis_id("testfam", {"testfam-20260907-1", "testfam-20260907-3"}) == "testfam-20260907-4"


def test_make_hypothesis_id_ignores_other_dates_and_bad_suffixes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(models, "datetime", _FakeDatetime)
    assert make_hypothesis_id("testfam", {"testfam-20250101-5", "testfam-20260907-x"}) == "testfam-20260907-1"


def test_test_plan_defaults_are_valid() -> None:
    plan = TestPlan()
    assert plan.instruments == ("BTC/USDT", "ETH/USDT")
    assert plan.timeframe == "5m"
    assert plan.oos_end is None


def test_test_plan_rejects_empty_instruments() -> None:
    with pytest.raises(ValueError, match="instruments"):
        TestPlan(instruments=())


def test_test_plan_rejects_invalid_calibration_window() -> None:
    with pytest.raises(ValueError, match="Kalibrierungsfenster"):
        TestPlan(calibration_start="2021-05-01", calibration_end="2021-05-01")
    with pytest.raises(ValueError, match="Kalibrierungsfenster"):
        TestPlan(calibration_start="2022-01-01", calibration_end="2021-01-01")


def test_test_plan_rejects_overlapping_oos_window() -> None:
    with pytest.raises(ValueError, match="überlappt"):
        TestPlan(calibration_start="2021-05-01", calibration_end="2022-12-31", oos_start="2022-12-31")
    with pytest.raises(ValueError, match="überlappt"):
        TestPlan(calibration_start="2021-05-01", calibration_end="2022-12-31", oos_start="2022-06-01")


def test_test_plan_accepts_disjoint_windows() -> None:
    plan = TestPlan(calibration_start="2021-01-01", calibration_end="2021-06-01", oos_start="2021-06-02")
    assert plan.oos_start == "2021-06-02"


def test_test_plan_min_candles_bound() -> None:
    with pytest.raises(ValueError, match="min_candles"):
        TestPlan(min_candles=9)


def test_decision_rule_defaults() -> None:
    rule = DecisionRule()
    assert rule.oos_margin_min_pct == 1.0
    assert rule.min_legs == 10
    assert rule.max_dd_pct == 8.0
    assert rule.majority_positive is True


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"oos_margin_min_pct": -0.1}, "oos_margin_min_pct"),
        ({"min_legs": 0}, "min_legs"),
        ({"max_dd_pct": 0.0}, "max_dd_pct"),
    ],
)
def test_decision_rule_rejects_invalid_bounds(kwargs: dict[str, float | int], field: str) -> None:
    with pytest.raises(ValueError, match=field):
        DecisionRule(**kwargs)


def test_variant_accepts_config_and_mechanism() -> None:
    assert Variant(strategy="rsi_mean_reversion").params == {}
    mech = Variant(strategy="new_mech", code="x", code_file="packages/strategies/new_mech.py")
    assert mech.code_file == "packages/strategies/new_mech.py"


def test_variant_rejects_bad_strategy_name() -> None:
    with pytest.raises(ValueError, match="Strategie-Name"):
        Variant(strategy="Bad_Name")


def test_variant_requires_code_and_code_file_together() -> None:
    with pytest.raises(ValueError, match="zusammen"):
        Variant(strategy="new_mech", code="x")
    with pytest.raises(ValueError, match="zusammen"):
        Variant(strategy="new_mech", code_file="packages/strategies/new_mech.py")


def test_hypothesis_defaults() -> None:
    hyp = make_hypothesis()
    assert hyp.status == "proposed"
    assert hyp.source == "manual"
    assert hyp.results is None
    assert hyp.verdict is None
    assert isinstance(hyp.test_plan, TestPlan)
    assert isinstance(hyp.decision_rule, DecisionRule)


def test_hypothesis_rejects_short_claim() -> None:
    with pytest.raises(ValueError, match="claim"):
        make_hypothesis(claim="zu kurz")


def test_hypothesis_rejects_bad_family() -> None:
    with pytest.raises(ValueError, match="Family-Name"):
        Hypothesis(
            id="abc-20240101-1",
            created_at="2024-01-01T00:00:00+00:00",
            family="Bad_Family",
            kind="config",
            claim="Ein belastbarer Claim.",
            variant=Variant(strategy="abc"),
        )


def test_hypothesis_mechanism_requires_code() -> None:
    with pytest.raises(ValueError, match=r"variant\.code"):
        make_hypothesis(kind="mechanism")


def test_hypothesis_rejects_unknown_status() -> None:
    with pytest.raises(ValueError, match="status"):
        make_hypothesis(status="bogus")


def test_proposal_requires_family_to_match_strategy() -> None:
    with pytest.raises(ValueError, match=r"variant\.strategy"):
        Proposal(
            family="other_fam",
            kind="config",
            claim="Claim mit genug Laenge.",
            variant=Variant(strategy="abc"),
        )


def test_proposal_mechanism_requires_code() -> None:
    with pytest.raises(ValueError, match=r"variant\.code"):
        Proposal(
            family="new_mech",
            kind="mechanism",
            claim="Claim mit genug Laenge.",
            variant=Variant(strategy="new_mech"),
        )


def test_hypothesis_from_proposal_sets_id_timestamp_and_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(models, "datetime", _FakeDatetime)
    proposal = Proposal(
        family="rsi_mean_reversion",
        kind="config",
        claim="  Claim mit Whitespace.  ",
        variant=Variant(strategy="rsi_mean_reversion", params={"buy_below": 25.0}),
    )
    hyp = hypothesis_from_proposal(proposal, set(), source="sweep")
    assert hyp.id == "rsi_mean_reversion-20260907-1"
    assert hyp.created_at == "2026-09-07T12:30:45+00:00"
    assert hyp.source == "sweep"
    assert hyp.claim == "Claim mit Whitespace."
    assert hyp.status == "proposed"


def test_hypothesis_from_proposal_uses_defaults_and_next_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(models, "datetime", _FakeDatetime)
    proposal = Proposal(
        family="rsi_mean_reversion",
        kind="config",
        claim="Claim mit genug Laenge.",
        variant=Variant(strategy="rsi_mean_reversion"),
    )
    existing = {"rsi_mean_reversion-20260907-1", "rsi_mean_reversion-20260907-2"}
    hyp = hypothesis_from_proposal(proposal, existing, source="llm:personas")
    assert hyp.id == "rsi_mean_reversion-20260907-3"
    assert hyp.test_plan == TestPlan()
    assert hyp.decision_rule == DecisionRule()


def test_variant_key_sorts_params_and_uses_general_format() -> None:
    variant = Variant(strategy="rsi_mean_reversion", params={"buy_below": 25.0, "period": 30.0})
    assert variant_key(variant) == "rsi_mean_reversion::buy_below=25,period=30"


def test_variant_key_ignores_code() -> None:
    base = Variant(strategy="rsi_mean_reversion", params={"period": 30.0})
    with_code = Variant(
        strategy="rsi_mean_reversion",
        params={"period": 30.0},
        code="x",
        code_file="packages/strategies/rsi_mean_reversion.py",
    )
    assert variant_key(base) == variant_key(with_code)


def test_variant_key_distinguishes_params() -> None:
    a = Variant(strategy="abc", params={"period": 14.0})
    b = Variant(strategy="abc", params={"period": 15.0})
    assert variant_key(a) != variant_key(b)
