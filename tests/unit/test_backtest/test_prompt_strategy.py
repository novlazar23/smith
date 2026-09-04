"""Tests für apps.backtest.prompt_strategy (PromptStrategy, Phase 2).

Kein reales Netzwerk, keine echten LiteLLM-Aufrufe: der Client wird durch
`_FakeClient` injiziert, die Kerzen sind synthetische `Candle`-Objekte aus
dem Backtest-conftest (flache Serie → deterministisch identische
Snapshot-Fenster → Cache-Hits prüfbar).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from apps.backtest.prompt_strategy import PROMPT_VERSION, PromptStrategy
from packages.backtesting.strategies import SignalAction
from packages.llm.errors import LLMError
from tests.unit.test_backtest.conftest import BTC, drive, make_candles

_BUY_JSON = (
    '{"decision": "BUY", "confidence": 0.7, "up": 0.6, "down": 0.2, "range": 0.2, '
    '"hypothesis": "Trend continuation on rising volume."}'
)
_SELL_JSON = (
    '{"decision": "SELL", "confidence": 0.8, "up": 0.2, "down": 0.5, "range": 0.3, '
    '"hypothesis": "Trend exhaustion near range high."}'
)
_NONE_JSON = (
    '{"decision": "NONE", "confidence": 0.4, "up": 0.3, "down": 0.3, "range": 0.4, '
    '"hypothesis": "Range-bound market, no edge."}'
)
_SURROUNDED_JSON = f'Sure, here is my take: {_BUY_JSON} Hope that helps.'
_MALFORMED_JSON = "I cannot decide right now."
_BAD_SUM_JSON = (
    '{"decision": "BUY", "confidence": 0.7, "up": 0.8, "down": 0.5, "range": 0.2, '
    '"hypothesis": "probability sum violates the schema"}'
)
_WARMUP_BARS = 120  # min_candles-Default: erste Evaluation exakt bei Bar 120 (1-basiert)


class _FakeClient:
    """Duck-Typ-Ersatz für LLMClient: liefert vordefinierte Antworten, zählt Aufrufe."""

    def __init__(self, response: str = _BUY_JSON, model: str = "fake-model") -> None:
        self.response = response
        self.model = model
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]], *, temperature: float = 0.0) -> str:
        self.calls.append(messages)
        return self.response


class _ErrorClient(_FakeClient):
    """Client, der pro Aufruf einen LLMError wirft (z.B. Timeout)."""

    def __init__(self) -> None:
        super().__init__()
        self.error = LLMError("timeout", "timed out")

    def complete(self, messages: list[dict[str, str]], *, temperature: float = 0.0) -> str:
        self.calls.append(messages)
        raise self.error


def make_prompt_strategy(
    client: _FakeClient, cache_path: Path, **overrides: Any
) -> PromptStrategy:
    """PromptStrategy mit injiziertem Client und tmp-Datei-Cache (keine Netzwerk-Zugriffe)."""
    defaults: dict[str, Any] = {
        "llm_every": 15,
        "min_candles": 120,
        "initial_capital": 100_000.0,
        "trade_notional": 2000.0,
    }
    defaults.update(overrides)
    return PromptStrategy(BTC, client=client, cache_path=str(cache_path), **defaults)


class TestSignalMapping:
    """Entscheidung → StrategySignal (long-only-Semantik wie AgentEnsembleStrategy)."""

    def test_buy_response_emits_buy_signal(self, tmp_path: Path) -> None:
        client = _FakeClient(_BUY_JSON)
        strategy = make_prompt_strategy(client, tmp_path / "cache.jsonl")
        candles = make_candles(_WARMUP_BARS)

        signals = drive(strategy, candles)

        assert len(signals) == 1
        signal = signals[0]
        assert signal.action == SignalAction.BUY
        assert signal.symbol == BTC
        assert signal.confidence == pytest.approx(0.7)
        assert signal.position_size == pytest.approx(2000.0 / 100_000.0)
        assert signal.reason == "prompt-llm: BUY (decision confidence=0.700)"
        assert signal.timestamp == candles[119].timestamp
        assert signal.metadata["decision"] == "BUY"
        assert signal.metadata["confidence"] == pytest.approx(0.7)
        assert signal.metadata["hypothesis"] == "Trend continuation on rising volume."
        assert signal.metadata["up"] == pytest.approx(0.6)
        assert signal.metadata["down"] == pytest.approx(0.2)
        assert signal.metadata["range"] == pytest.approx(0.2)
        assert signal.metadata["prompt_version"] == PROMPT_VERSION
        assert signal.metadata["instrument"] == BTC

    def test_sell_response_emits_sell_signal(self, tmp_path: Path) -> None:
        client = _FakeClient(_SELL_JSON)
        strategy = make_prompt_strategy(client, tmp_path / "cache.jsonl")

        signals = drive(strategy, make_candles(_WARMUP_BARS))

        assert len(signals) == 1
        signal = signals[0]
        assert signal.action == SignalAction.SELL
        assert signal.position_size == 0.0
        assert signal.confidence == pytest.approx(0.8)
        assert strategy.n_sell_signals == 1
        assert strategy.n_buy_signals == 0

    def test_none_response_abstains(self, tmp_path: Path) -> None:
        client = _FakeClient(_NONE_JSON)
        strategy = make_prompt_strategy(client, tmp_path / "cache.jsonl")

        signals = drive(strategy, make_candles(_WARMUP_BARS))

        assert signals == []
        assert strategy.llm_calls == 1
        assert strategy.llm_failures == 0
        assert strategy.n_buy_signals == 0
        assert strategy.n_sell_signals == 0

    def test_json_with_surrounding_text_is_accepted(self, tmp_path: Path) -> None:
        client = _FakeClient(_SURROUNDED_JSON)
        strategy = make_prompt_strategy(client, tmp_path / "cache.jsonl")

        signals = drive(strategy, make_candles(_WARMUP_BARS))

        assert len(signals) == 1
        assert signals[0].action == SignalAction.BUY
        assert strategy.llm_failures == 0

    def test_prompt_messages_are_system_plus_summary(self, tmp_path: Path) -> None:
        client = _FakeClient(_BUY_JSON)
        strategy = make_prompt_strategy(client, tmp_path / "cache.jsonl")

        drive(strategy, make_candles(_WARMUP_BARS))

        messages = client.calls[0]
        assert [message["role"] for message in messages] == ["system", "user"]
        assert messages[0]["content"].startswith("You are a disciplined crypto swing trader")
        assert messages[1]["content"].startswith("BTC/USDT-style OHLCV snapshot")


class TestCache:
    """Response-Cache: flaches Fenster → zweites Fenster ist ein Cache-Hit."""

    def test_flat_window_second_evaluation_is_cache_hit(self, tmp_path: Path) -> None:
        client = _FakeClient(_BUY_JSON)
        strategy = make_prompt_strategy(client, tmp_path / "cache.jsonl")

        # Bewertungen bei Bar 120 und 135 — flache Kerzen → identische Snapshots
        signals = drive(strategy, make_candles(135))

        assert len(signals) == 2
        assert len(client.calls) == 1
        assert strategy.llm_calls == 1
        assert strategy.llm_cache_hits == 1
        assert strategy.llm_failures == 0

    def test_cache_file_is_reused_across_strategy_instances(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "cache.jsonl"
        first = make_prompt_strategy(_FakeClient(_BUY_JSON), cache_path)
        drive(first, make_candles(_WARMUP_BARS))
        assert (cache_path).is_file()

        second_client = _FakeClient(_BUY_JSON)
        second = make_prompt_strategy(second_client, cache_path)
        drive(second, make_candles(_WARMUP_BARS))

        assert len(second_client.calls) == 0
        assert second.llm_cache_hits == 1


class TestFailures:
    """Parse-/Validierungs-/Client-Fehler → Abstinenz statt Abbruch."""

    def test_malformed_json_abstains_and_counts_failure(self, tmp_path: Path) -> None:
        client = _FakeClient(_MALFORMED_JSON)
        strategy = make_prompt_strategy(client, tmp_path / "cache.jsonl")

        signals = drive(strategy, make_candles(_WARMUP_BARS))

        assert signals == []
        assert strategy.llm_calls == 1
        assert strategy.llm_failures == 1

    def test_probability_sum_violation_abstains_and_counts_failure(
        self, tmp_path: Path
    ) -> None:
        client = _FakeClient(_BAD_SUM_JSON)
        strategy = make_prompt_strategy(client, tmp_path / "cache.jsonl")

        signals = drive(strategy, make_candles(_WARMUP_BARS))

        assert signals == []
        assert strategy.llm_failures == 1

    def test_client_error_abstains_and_is_not_cached(self, tmp_path: Path) -> None:
        client = _ErrorClient()
        strategy = make_prompt_strategy(client, tmp_path / "cache.jsonl")

        signals = drive(strategy, make_candles(135))  # Bewertungen bei Bar 120 und 135

        assert signals == []
        assert len(client.calls) == 2  # Fehlschlag wird nicht gecacht
        assert strategy.llm_calls == 2
        assert strategy.llm_failures == 2


class TestLifecycle:
    """Warmup, Rhythmus, Validierung, to_dict."""

    def test_no_evaluation_before_warmup(self, tmp_path: Path) -> None:
        client = _FakeClient(_BUY_JSON)
        strategy = make_prompt_strategy(client, tmp_path / "cache.jsonl")

        drive(strategy, make_candles(119))

        assert client.calls == []
        assert strategy.llm_calls == 0
        assert strategy.llm_failures == 0

    def test_constructor_validation(self) -> None:
        client = _FakeClient()
        with pytest.raises(ValueError):
            PromptStrategy(BTC, client=client, llm_every=0)
        with pytest.raises(ValueError):
            PromptStrategy(BTC, client=client, min_candles=10)

    def test_to_dict_shape(self, tmp_path: Path) -> None:
        client = _FakeClient(_BUY_JSON)
        strategy = make_prompt_strategy(client, tmp_path / "cache.jsonl")
        drive(strategy, make_candles(135))

        data = strategy.to_dict()

        assert data["name"] == "prompt-strategy"
        assert data["instrument"] == BTC
        assert data["prompt_version"] == PROMPT_VERSION
        assert data["model"] == "fake-model"
        assert data["llm_every"] == 15
        assert data["min_candles"] == 120
        assert data["llm_calls"] == 1
        assert data["llm_cache_hits"] == 1
        assert data["llm_failures"] == 0
        assert data["n_buy_signals"] == 2
        assert data["n_sell_signals"] == 0
