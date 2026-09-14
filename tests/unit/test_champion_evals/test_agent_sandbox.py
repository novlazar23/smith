"""Tests für die Agent-Sandbox: Jail, eingeschränkte Ausführung, Smoke-Test, Adapter, Artefakt-IO."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from apps.champion_evals.agent_sandbox import (
    DEFAULT_MAX_MS,
    build_evolved_agent,
    load_evolved_agents,
    load_predict_fn,
    normalize_raw_probs,
    smoke_test_predict,
    validate_agent_code,
)
from packages.schemas.agent_report import AgentStatus

GOOD_CODE = """import numpy as np

def predict(open, high, low, close, volume):
    m = float(close[-1] - close[-6])
    if m > 0:
        return (0.8, 0.1, 0.1)
    if m < 0:
        return (0.1, 0.8, 0.1)
    return (0.34, 0.33, 0.33)
"""


class TestValidateAgentCode:
    def test_valid_code_passes(self) -> None:
        assert validate_agent_code(GOOD_CODE, "momentum_test") is None

    @pytest.mark.parametrize(
        "code",
        [
            "import os\ndef predict(o,h,l,c,v): return (1,0,0)",
            "from subprocess import run\ndef predict(o,h,l,c,v): return (1,0,0)",
            "def predict(o,h,l,c,v):\n    open('/etc/passwd')\n    return (1,0,0)",
            "def predict(o,h,l,c,v):\n    __import__('os')\n    return (1,0,0)",
            "def predict(o,h,l,c,v):\n    eval('1')\n    return (1,0,0)",
            "def predict(o,h,l,c,v):\n    exec('x=1')\n    return (1,0,0)",
            "def predict(o,h,l,c,v):\n    np.fromfile('/etc/passwd')\n    return (1,0,0)",
            "def predict(o,h,l,c,v):\n    getattr(np, 'load')\n    return (1,0,0)",
            "x = 1\ndef predict(o,h,l,c,v): return (1,0,0)",
            "def predict(o,h,l,c,v): return (1,0,0)\ndef predict2(o,h,l,c,v): return (1,0,0)",
            "def predict(o,h,l,c): return (1,0,0)",
            "def predict(o,h,l,c,v,*a): return (1,0,0)",
        ],
    )
    def test_forbidden_code_rejected(self, code: str) -> None:
        assert validate_agent_code(code, "test_agent") is not None

    @pytest.mark.parametrize("name", ["", "a", "UPPER", "has space", "a" * 42])
    def test_invalid_name_rejected(self, name: str) -> None:
        assert validate_agent_code(GOOD_CODE, name) is not None

    def test_syntax_error_rejected(self) -> None:
        assert validate_agent_code("def predict(:", "test_agent") is not None

    def test_empty_code_rejected(self) -> None:
        assert validate_agent_code("   ", "test_agent") is not None


class TestLoadPredictFn:
    def test_returns_callable_predict(self) -> None:
        fn = load_predict_fn(GOOD_CODE, "momentum_test")
        a = np.ones(10)
        assert fn(a, a, a, a, a) == (0.34, 0.33, 0.33)

    def test_runtime_import_blocked(self) -> None:
        code = "import os\ndef predict(o,h,l,c,v):\n    return (1,0,0)"
        with pytest.raises(ImportError):
            load_predict_fn(code, "t")

    def test_unsafe_builtin_unreachable(self) -> None:
        code = "def predict(o,h,l,c,v):\n    return (input('x'), 0, 0)"
        fn = load_predict_fn(code, "t")
        a = np.ones(5)
        with pytest.raises(NameError):
            fn(a, a, a, a, a)

    def test_missing_predict_raises(self) -> None:
        with pytest.raises(ValueError):
            load_predict_fn("import math\n", "t")


class TestSmokeTest:
    def test_valid_function_passes(self) -> None:
        assert smoke_test_predict(load_predict_fn(GOOD_CODE, "momentum_test")) is None

    @pytest.mark.parametrize(
        "code",
        [
            "def predict(o,h,l,c,v):\n    return (1,0)",
            "def predict(o,h,l,c,v):\n    return (float('nan'),0,0)",
            "def predict(o,h,l,c,v):\n    return (-1,-1,-1)",
            "def predict(o,h,l,c,v):\n    raise RuntimeError('boom')",
        ],
    )
    def test_invalid_output_rejected(self, code: str) -> None:
        assert smoke_test_predict(load_predict_fn(code, "t")) is not None

    def test_slow_function_rejected(self) -> None:
        code = "import math\ndef predict(o,h,l,c,v):\n    for _ in range(2_000_000):\n        math.sqrt(2.0)\n    return (1,0,0)"
        fn = load_predict_fn(code, "t")
        assert smoke_test_predict(fn, max_ms=DEFAULT_MAX_MS) is not None

    def test_short_window_used(self) -> None:
        code = "def predict(open, high, low, close, volume):\n    if len(close) < 31: return (1,0,0)\n    return (1,0,0)"
        assert smoke_test_predict(load_predict_fn(code, "t")) is None


class TestNormalizeRawProbs:
    def test_sums_to_one(self) -> None:
        assert sum(normalize_raw_probs((2.0, 2.0, 2.0)).values()) == pytest.approx(1.0)

    def test_negative_clipped_to_zero(self) -> None:
        probs = normalize_raw_probs((-0.1, 0.4, 0.5))
        assert probs["up"] == 0.0
        assert sum(probs.values()) == pytest.approx(1.0)

    @pytest.mark.parametrize("raw", ["abc", (1, 0), (float("nan"), 0, 0), (-1, -1, -1)])
    def test_invalid_raises(self, raw: object) -> None:
        with pytest.raises(ValueError):
            normalize_raw_probs(raw)


class TestEvolvedAgent:
    def test_report_contract(self) -> None:
        agent = build_evolved_agent(
            "momentum_test", {"code": GOOD_CODE}, AgentStatus.SHADOW, instrument="BTC/USDT", horizon="15m"
        )
        assert agent is not None
        data = {key: np.ones(30) for key in ("open", "high", "low", "close", "volume")}
        data["close"] = np.linspace(100, 110, 30)
        report = agent.analyze(data)
        assert abs(sum(report.probabilities.values()) - 1.0) < 1e-6
        assert report.status == AgentStatus.SHADOW
        assert report.agent_id == "momentum_test"
        assert report.instrument == "BTC/USDT"
        assert report.horizon == "15m"
        assert len(report.evidence) >= 1

    def test_rejects_invalid_code(self) -> None:
        assert build_evolved_agent("t", {"code": "def predict(o,h,l,c,v):\n    eval('x')\n    return (1,0,0)"}, AgentStatus.SHADOW) is None

    def test_rejects_invalid_status(self) -> None:
        assert build_evolved_agent("t", {"code": GOOD_CODE}, "bogus") is None


class TestArtifactIO:
    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert load_evolved_agents(tmp_path / "keine.json") == {}

    def test_corrupt_file_returns_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "evolved_agents.json"
        path.write_text("{kein json", encoding="utf-8")
        assert load_evolved_agents(path) == {}

    def test_valid_entry_loaded(self, tmp_path: Path) -> None:
        path = tmp_path / "evolved_agents.json"
        path.write_text(json.dumps({"momentum_test": {"version": 2, "code": GOOD_CODE, "claim": "c"}}), encoding="utf-8")
        agents = load_evolved_agents(path)
        assert agents["momentum_test"]["version"] == 2
        assert agents["momentum_test"]["code"] == GOOD_CODE

    def test_bad_entries_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "evolved_agents.json"
        payload = {
            "UPPER_BAD": {"code": GOOD_CODE},
            "ok_agent": {"code": GOOD_CODE},
            "no_code": {"version": 1},
            "not_obj": 42,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        assert list(load_evolved_agents(path)) == ["ok_agent"]
