"""Tests für die CLI-Wiring (Strategie-Modi, Parameter-Parsee, Validierung)."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest
from apps.backtest import __main__ as cli
from apps.backtest.agent_strategy import AgentEnsembleStrategy
from apps.backtest.prompt_strategy import PromptStrategy
from packages.strategies import RuleStrategy


def _args(*argv: str) -> argparse.Namespace:
    return cli.build_parser().parse_args(list(argv))


# ── parse_params ──────────────────────────────────────────────────────────


def test_parse_params_valid() -> None:
    assert cli.parse_params("fast=8,slow=30") == {"fast": 8.0, "slow": 30.0}


def test_parse_params_ignores_empty_segments() -> None:
    assert cli.parse_params("fast=8,, slow = 30,") == {"fast": 8.0, "slow": 30.0}


def test_parse_params_rejects_missing_equals() -> None:
    with pytest.raises(ValueError, match="im Format k=v erwartet"):
        cli.parse_params("fast8")


def test_parse_params_rejects_non_numeric() -> None:
    with pytest.raises(ValueError, match="ist keine Zahl"):
        cli.parse_params("fast=abc")


# ── make_strategy-Modi-Routing ────────────────────────────────────────────


def test_make_strategy_ensemble_by_default() -> None:
    args = _args("--gate", "0.4")
    strategy = cli.make_strategy(args)
    assert isinstance(strategy, AgentEnsembleStrategy)
    assert strategy.min_confidence == 0.4


def test_make_strategy_library_from_name() -> None:
    args = _args("--strategy", "ema_cross")
    strategy = cli.make_strategy(args)
    assert isinstance(strategy, RuleStrategy)
    assert strategy.params == {"fast": 12.0, "slow": 26.0}


def test_make_strategy_library_with_params() -> None:
    args = _args("--strategy", "ema_cross", "--params", "fast=8,slow=30")
    strategy = cli.make_strategy(args)
    assert isinstance(strategy, RuleStrategy)
    assert strategy.params == {"fast": 8.0, "slow": 30.0}


def test_make_strategy_library_explicit_name_overrides_args() -> None:
    args = _args("--strategy", "ema_cross")
    strategy = cli.make_strategy(args, strategy_name="supertrend")
    assert isinstance(strategy, RuleStrategy)
    assert strategy.params["period"] == 10.0


def test_make_strategy_unknown_name_raises() -> None:
    args = _args("--strategy", "does_not_exist")
    with pytest.raises(Exception, match="Unbekannte Strategie"):
        cli.make_strategy(args)


def test_make_strategy_prompt_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_API_KEY", "test-key")
    monkeypatch.setenv("LITELLM_BASE_URL", "http://localhost/v1")
    args = _args("--prompt-strategy", "--llm-model", "local-fast", "--llm-every", "7")
    strategy = cli.make_strategy(args)
    assert isinstance(strategy, PromptStrategy)
    assert strategy.llm_every == 7
    assert strategy.model_name == "local-fast"


# ── validate_strategy_args ────────────────────────────────────────────────


def test_validate_rejects_prompt_plus_strategy() -> None:
    args = _args("--prompt-strategy", "--strategy", "ema_cross")
    with pytest.raises(SystemExit):
        cli.validate_strategy_args(cli.build_parser(), args)


def test_validate_rejects_prompt_plus_sweep_library() -> None:
    args = _args("--prompt-strategy", "--sweep-library")
    with pytest.raises(SystemExit):
        cli.validate_strategy_args(cli.build_parser(), args)


def test_validate_rejects_strategy_plus_sweep_library() -> None:
    args = _args("--strategy", "ema_cross", "--sweep-library")
    with pytest.raises(SystemExit):
        cli.validate_strategy_args(cli.build_parser(), args)


def test_validate_rejects_sweep_gates_plus_strategy() -> None:
    args = _args("--strategy", "ema_cross", "--sweep-gates", "0.3,0.5")
    with pytest.raises(SystemExit):
        cli.validate_strategy_args(cli.build_parser(), args)


def test_validate_rejects_params_without_strategy() -> None:
    args = _args("--params", "fast=8")
    with pytest.raises(SystemExit):
        cli.validate_strategy_args(cli.build_parser(), args)


def test_validate_accepts_library_sweep() -> None:
    args = _args("--sweep-library")
    cli.validate_strategy_args(cli.build_parser(), args)  # darf nicht werfen


def test_validate_accepts_prompt_alone() -> None:
    args = _args("--prompt-strategy")
    cli.validate_strategy_args(cli.build_parser(), args)  # darf nicht werfen


# ── load_feed / run_scenario-Refactoring: Feed-Reuse-Vertrag ──────────────


def test_load_feed_returns_none_on_missing_candles(monkeypatch: pytest.MonkeyPatch) -> None:
    args = _args("--min-candles", "10")
    engine = SimpleNamespace()
    monkeypatch.setattr(
        cli,
        "ClickHouseDataFeed",
        lambda *a, **k: SimpleNamespace(get_candles=list),
    )
    assert cli.load_feed(args, engine, "s", None, None) is None
