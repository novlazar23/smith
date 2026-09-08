"""Tests für den Mechanismus-Pfad (Jail, Smoke-Test, Registry, Clone-Check)."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
from apps.evolution.mechanism import (
    apply_registry,
    clone_similarity,
    find_clone,
    revert_registry,
    run_clone_check,
    sample_clone_candles,
    signal_bars,
    smoke_test,
    validate_mechanism_code,
    write_strategy_file,
)
from packages.backtesting.core import Candle
from tests.unit.test_evolution.conftest import BASE_TIME, VALID_CODE, make_candles

TWO_CLASSES = VALID_CODE + '''

class Zweite(RuleStrategy):
    strategy_name = "zweite"
    description = "Zweite Klasse."
    param_specs: ClassVar[dict[str, tuple[float, float, float]]] = {}
    min_bars = 60

    def _evaluate(self, candle: Candle) -> StrategySignal | None:
        return None
'''

WRONG_BASE = '''"""Falsche Base."""
from ._common import RuleStrategy


class FalscheBase(object):
    strategy_name = "test_strategy"
    description = "Falsche Base."
    param_specs = {}

    def _evaluate(self):
        return None
'''

MISSING_CLASSVARS = '''"""Fehlende ClassVars."""
from ._common import RuleStrategy


class Unvollstaendig(RuleStrategy):
    strategy_name = "test_strategy"

    def _evaluate(self, candle):
        return None
'''

MISSING_EVALUATE = '''"""Fehlender _evaluate."""
from ._common import RuleStrategy


class OhneEvaluate(RuleStrategy):
    strategy_name = "test_strategy"
    description = "Ohne Evaluate."
    param_specs = {}

    def warmup(self):
        return None
'''

RUNTIME_FAIL = '''"""Runtime-Fehler."""
from typing import ClassVar

from packages.backtesting.core import Candle
from packages.backtesting.strategies import StrategySignal

from ._common import RuleStrategy


class RuntimeFailStrategy(RuleStrategy):
    strategy_name = "runtime_fail"
    description = "Faelt bewusst im Smoke-Test."
    param_specs: ClassVar[dict[str, tuple[float, float, float]]] = {}
    min_bars = 2

    def _evaluate(self, candle: Candle) -> StrategySignal | None:
        raise ZeroDivisionError("kaputt")
'''

BAD_IMPORT = '''"""Bogus-Import."""
from .nonexistent import X
from typing import ClassVar

from packages.backtesting.core import Candle
from packages.backtesting.strategies import StrategySignal

from ._common import RuleStrategy


class BadImportStrategy(RuleStrategy):
    strategy_name = "bad_import"
    description = "Falscher relativer Import."
    param_specs: ClassVar[dict[str, tuple[float, float, float]]] = {}
    min_bars = 2

    def _evaluate(self, candle: Candle) -> StrategySignal | None:
        return None
'''

MINI_REGISTRY = '''"""Mini-Registry."""

from ._common import RuleStrategy
from .ema_cross import EmaCrossStrategy

_STRATEGY_CLASSES: tuple[type[RuleStrategy], ...] = (
    EmaCrossStrategy,
)
'''


# ─── Jail ────────────────────────────────────────────────────────────────────


def test_validate_accepts_valid_code() -> None:
    assert validate_mechanism_code(VALID_CODE, "test_strategy") is None


def test_validate_accepts_numpy_import() -> None:
    assert validate_mechanism_code("import numpy\n" + VALID_CODE, "test_strategy") is None


def test_validate_rejects_empty_code() -> None:
    assert validate_mechanism_code("", "test_strategy") == "leerer Code"
    assert validate_mechanism_code("   \n", "test_strategy") == "leerer Code"


def test_validate_rejects_invalid_strategy_name() -> None:
    result = validate_mechanism_code(VALID_CODE, "Bad_Name")
    assert result is not None
    assert "Strategie-Name" in result


def test_validate_rejects_syntax_error() -> None:
    result = validate_mechanism_code("def x(:", "test_strategy")
    assert result is not None
    assert "Syntaxfehler" in result


def test_validate_rejects_banned_import() -> None:
    assert validate_mechanism_code("import os\n" + VALID_CODE, "test_strategy") == "Import 'os' nicht erlaubt"


def test_validate_rejects_future_import() -> None:
    code = "from __future__ import annotations\n" + VALID_CODE
    result = validate_mechanism_code(code, "test_strategy")
    assert result is not None
    assert "__future__" in result


def test_validate_rejects_too_deep_relative_import() -> None:
    code = "from ..outside import X\n" + VALID_CODE
    result = validate_mechanism_code(code, "test_strategy")
    assert result is not None
    assert "zu tief" in result


def test_validate_rejects_multiple_classes() -> None:
    result = validate_mechanism_code(TWO_CLASSES, "test_strategy")
    assert result is not None
    assert "gefunden 2" in result


def test_validate_rejects_missing_class() -> None:
    result = validate_mechanism_code('"""Nur Docstring."""\n', "test_strategy")
    assert result is not None
    assert "gefunden 0" in result


def test_validate_rejects_wrong_base_class() -> None:
    result = validate_mechanism_code(WRONG_BASE, "test_strategy")
    assert result is not None
    assert "exakt eine Base" in result


def test_validate_rejects_module_level_statement() -> None:
    result = validate_mechanism_code("X = 1\n" + VALID_CODE, "test_strategy")
    assert result is not None
    assert "nicht erlaubtes Modul-Element" in result


def test_validate_rejects_missing_classvars() -> None:
    result = validate_mechanism_code(MISSING_CLASSVARS, "test_strategy")
    assert result is not None
    assert "fehlende ClassVars: description, param_specs" in result


def test_validate_rejects_missing_evaluate() -> None:
    result = validate_mechanism_code(MISSING_EVALUATE, "test_strategy")
    assert result is not None
    assert "Methode _evaluate fehlt" in result


def test_validate_rejects_strategy_name_mismatch() -> None:
    code = VALID_CODE.replace('strategy_name = "test_strategy"', 'strategy_name = "anderer_name"')
    result = validate_mechanism_code(code, "test_strategy")
    assert result is not None
    assert "!= erwarteter Name" in result


def test_validate_rejects_out_of_range_min_bars() -> None:
    code = VALID_CODE.replace("min_bars = 60", "min_bars = 500")
    result = validate_mechanism_code(code, "test_strategy")
    assert result is not None
    assert "außerhalb [1, 300]" in result


def test_validate_rejects_forbidden_call_in_evaluate() -> None:
    code = VALID_CODE.replace("return None", 'return open("/etc/passwd")', 1)
    result = validate_mechanism_code(code, "test_strategy")
    assert result is not None
    assert "verbotener Aufruf in _evaluate" in result


def test_validate_rejects_forbidden_call_in_class_body() -> None:
    code = VALID_CODE.replace(
        'description = "Synthetischer Test-Mechanismus fuer die Evolution-Unit-Tests."',
        'description = eval("x")',
    )
    result = validate_mechanism_code(code, "test_strategy")
    assert result is not None
    assert "verboten" in result


# ─── Smoke-Test ──────────────────────────────────────────────────────────────


def test_smoke_test_passes_for_valid_code() -> None:
    assert smoke_test(VALID_CODE, "test_strategy") is None


def test_smoke_test_reports_runtime_error() -> None:
    result = smoke_test(RUNTIME_FAIL, "runtime_fail")
    assert result is not None
    assert result.startswith("Smoke-Test")
    assert "ZeroDivisionError" in result


def test_smoke_test_reports_non_executable_code() -> None:
    result = smoke_test(BAD_IMPORT, "bad_import")
    assert result is not None
    assert result.startswith("Code nicht ausführbar")
    assert "ModuleNotFoundError" in result


# ─── Signale und Clone-Check ─────────────────────────────────────────────────


def _oversold_rebound_candles() -> list[Candle]:
    """70 Abwärts-Kerzen (-2 %), dann 10 Aufwärts-Kerzen (+1,5 %)."""
    candles: list[Candle] = []
    price = 100.0
    for i in range(80):
        step = -0.02 if i < 70 else 0.015
        close = max(1.0, price * (1.0 + step))
        candles.append(
            Candle(
                timestamp=BASE_TIME + timedelta(minutes=5 * i),
                symbol="BTC/USDT",
                open=price,
                high=max(price, close),
                low=min(price, close),
                close=close,
                volume=1000.0,
            )
        )
        price = close
    return candles


def test_signal_bars_detects_buy_after_oversold_rebound() -> None:
    bars = signal_bars("rsi_mean_reversion", {}, _oversold_rebound_candles())
    assert bars, "erwartet mindestens ein BUY-Signal nach Oversold-Rebound"
    assert all(59 <= i < 80 for i in bars)


def test_clone_similarity_variants() -> None:
    assert clone_similarity({1, 2, 3}, {1, 2, 3}) == 1.0
    assert clone_similarity({1, 2}, {3, 4}) == 0.0
    assert clone_similarity(set(), {1}) == 0.0
    assert clone_similarity({1, 2}, set()) == 0.0
    assert clone_similarity({1, 2, 3, 4, 5}, {1, 2}) == 1.0  # containment
    assert clone_similarity({1, 2, 3}, {2, 3, 4}) == pytest.approx(2.0 / 3.0)


def test_find_clone_returns_first_alpha_match() -> None:
    zoo = {"bbb": {2, 3, 4}, "aaa": {1, 2, 3}}
    assert find_clone({1, 2, 3}, zoo) == "aaa"


def test_find_clone_none_below_threshold() -> None:
    zoo = {"aaa": {1, 2, 3, 6, 7}}
    assert find_clone({1, 2, 3, 4, 5}, zoo) is None


def test_run_clone_check_skips_short_series() -> None:
    assert run_clone_check("rsi_mean_reversion", {}, make_candles(300)) is None


def test_run_clone_check_runs_on_long_series() -> None:
    result = run_clone_check("rsi_mean_reversion", {}, make_candles(500))
    assert result is None or isinstance(result, str)


def test_sample_clone_candles_empty() -> None:
    assert sample_clone_candles([]) == []


def test_sample_clone_candles_keeps_last_days() -> None:
    candles = make_candles(4000)
    sample = sample_clone_candles(candles, days=3)
    assert sample
    assert len(sample) < len(candles)
    assert sample[-1] is candles[-1]


# ─── Datei und Registry ──────────────────────────────────────────────────────


def test_write_strategy_file_is_idempotent(tmp_path: Path) -> None:
    path = write_strategy_file(VALID_CODE, "test_strategy", tmp_path)
    assert path == tmp_path / "packages" / "strategies" / "test_strategy.py"
    assert path.read_text(encoding="utf-8") == VALID_CODE.rstrip() + "\n"
    again = write_strategy_file(VALID_CODE, "test_strategy", tmp_path)
    assert again == path
    assert path.read_text(encoding="utf-8") == VALID_CODE.rstrip() + "\n"


def _mini_repo(tmp_path: Path) -> Path:
    pkg = tmp_path / "packages" / "strategies"
    pkg.mkdir(parents=True)
    (pkg / "registry.py").write_text(MINI_REGISTRY, encoding="utf-8")
    return tmp_path


def test_apply_registry_inserts_import_and_tuple_entry(tmp_path: Path) -> None:
    repo = _mini_repo(tmp_path)
    class_name = apply_registry(VALID_CODE, "test_strategy", repo)
    assert class_name == "TestStrategyStrategy"
    text = (repo / "packages" / "strategies" / "registry.py").read_text(encoding="utf-8")
    assert "from .test_strategy import TestStrategyStrategy\n" in text
    assert "    TestStrategyStrategy,\n" in text


def test_apply_registry_is_idempotent(tmp_path: Path) -> None:
    repo = _mini_repo(tmp_path)
    apply_registry(VALID_CODE, "test_strategy", repo)
    first = (repo / "packages" / "strategies" / "registry.py").read_text(encoding="utf-8")
    apply_registry(VALID_CODE, "test_strategy", repo)
    second = (repo / "packages" / "strategies" / "registry.py").read_text(encoding="utf-8")
    assert first == second
    assert second.count("from .test_strategy import TestStrategyStrategy\n") == 1


def test_revert_registry_restores_original(tmp_path: Path) -> None:
    repo = _mini_repo(tmp_path)
    apply_registry(VALID_CODE, "test_strategy", repo)
    revert_registry(VALID_CODE, "test_strategy", repo)
    text = (repo / "packages" / "strategies" / "registry.py").read_text(encoding="utf-8")
    assert text == MINI_REGISTRY
