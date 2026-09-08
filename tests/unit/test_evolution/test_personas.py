"""Tests für die LLM-Phase „Diskutieren" (Messages, Parse, Validierung)."""

from __future__ import annotations

import json

import pytest
from apps.evolution.personas import PERSONA_TIMEOUT, build_messages, parse_proposals, propose
from packages.llm.errors import LLMError
from tests.unit.test_evolution.conftest import VALID_CODE

CLAIM = "Deep-Oversold-Entry verbessert die Win-Rate."


class FakeClient:
    """Duck-Typ-LLM-Client: rekordiert Aufrufe, liefert Antwort oder raise."""

    def __init__(self, answer: str = "[]", error: BaseException | None = None) -> None:
        self.answer = answer
        self.error = error
        self.timeout: float = 0.0
        self.messages: list[dict[str, str]] | None = None
        self.temperature: float | None = None

    def complete(self, messages: list[dict[str, str]], *, temperature: float) -> str:
        self.messages = messages
        self.temperature = temperature
        if self.error is not None:
            raise self.error
        return self.answer


def test_build_messages_contains_system_user_context() -> None:
    messages = build_messages("DIGEST-CONTENT", max_proposals=3)
    assert [m["role"] for m in messages] == ["system", "user"]
    user = messages[1]["content"]
    assert "Maximale Anzahl Vorschläge: 3" in user
    assert "rsi_mean_reversion" in user  # Zoo-Manifest
    assert "DIGEST-CONTENT" in user
    assert "JSON-Array" in user


def test_parse_proposals_plain_array() -> None:
    assert parse_proposals('[{"family": "abc"}]') == [{"family": "abc"}]


def test_parse_proposals_fenced_array() -> None:
    raw = '```json\n[{"family": "abc"}]\n```'
    assert parse_proposals(raw) == [{"family": "abc"}]


def test_parse_proposals_prose_wrapped_array() -> None:
    raw = 'Hier sind die Vorschläge: [{"family": "abc"}] — fertig.'
    assert parse_proposals(raw) == [{"family": "abc"}]


def test_parse_proposals_empty_array() -> None:
    assert parse_proposals("[]") == []


def test_parse_proposals_filters_non_dict_items() -> None:
    assert parse_proposals('[1, "x", {"family": "abc"}]') == [{"family": "abc"}]


def test_parse_proposals_raises_without_array() -> None:
    with pytest.raises(ValueError, match="kein JSON-Array"):
        parse_proposals("hier ist kein array")


def test_parse_proposals_raises_on_invalid_json() -> None:
    with pytest.raises(json.JSONDecodeError, match="Expecting"):
        parse_proposals('[{"foo": }]')


def test_propose_returns_validated_config_proposal() -> None:
    answer = json.dumps(
        [{"family": "rsi_mean_reversion", "kind": "config", "params": {"buy_below": 25.0}, "claim": CLAIM}]
    )
    client = FakeClient(answer=answer)
    proposals = propose(client, "digest")
    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal.family == "rsi_mean_reversion"
    assert proposal.kind == "config"
    assert proposal.variant.params == {"buy_below": 25.0}
    assert proposal.claim == CLAIM


def test_propose_sets_persona_timeout_and_temperature() -> None:
    client = FakeClient()
    propose(client, "digest")
    assert client.timeout == PERSONA_TIMEOUT
    assert client.temperature == 0.2
    assert client.messages is not None


def test_propose_returns_validated_mechanism_proposal() -> None:
    answer = json.dumps(
        [
            {
                "family": "new_mechanism",
                "kind": "mechanism",
                "code": VALID_CODE,
                "claim": "Aufwaerts-Kerzen-Momentum als neuer Mechanismus.",
            }
        ]
    )
    client = FakeClient(answer=answer)
    proposals = propose(client, "digest")
    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal.kind == "mechanism"
    assert proposal.variant.code == VALID_CODE
    assert proposal.variant.code_file == "packages/strategies/new_mechanism.py"


def test_propose_returns_empty_on_llm_error() -> None:
    client = FakeClient(error=LLMError("timeout", "zu langsam"))
    assert propose(client, "digest") == []


def test_propose_returns_empty_on_unexpected_error() -> None:
    client = FakeClient(error=RuntimeError("boom"))
    assert propose(client, "digest") == []


def test_propose_returns_empty_on_unparsable_answer() -> None:
    client = FakeClient(answer="hier ist kein json")
    assert propose(client, "digest") == []


def test_propose_discards_invalid_items() -> None:
    answer = json.dumps(
        [
            {"family": "unknown_strategy", "kind": "config", "params": {"x": 1.0}, "claim": "Unbekannte Strategie im Zoo."},
            {"family": "rsi_mean_reversion", "kind": "config", "params": {"buy_below": 25.0}, "claim": CLAIM},
        ]
    )
    client = FakeClient(answer=answer)
    proposals = propose(client, "digest")
    assert len(proposals) == 1
    assert proposals[0].variant.params == {"buy_below": 25.0}


def test_propose_respects_max_proposals() -> None:
    items = [
        {"family": "rsi_mean_reversion", "kind": "config", "params": {"buy_below": float(20 + i)}, "claim": CLAIM}
        for i in range(3)
    ]
    client = FakeClient(answer=json.dumps(items))
    assert len(propose(client, "digest", max_proposals=2)) == 2
