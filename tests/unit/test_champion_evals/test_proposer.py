"""Tests für den LLM-Proposer: Parsing, Validierung, Fail-Soft."""

from __future__ import annotations

import json

import pytest
from apps.champion_evals.proposer import build_messages, parse_agent_proposals, propose
from packages.llm.errors import LLMError

VALID_PROPOSAL = {
    "name": "volume_drift",
    "claim": "Anhaltendes Übermaß an Up-Volumen vorwärts-indiziert Fortsetzung der Bewegung.",
    "code": (
        "import numpy as np\n"
        "\n"
        "def predict(open, high, low, close, volume):\n"
        "    v = np.asarray(volume, dtype=float)\n"
        "    if len(v) < 10:\n"
        "        return (0.34, 0.33, 0.33)\n"
        "    recent = float(v[-3:].mean()) / max(float(v[-10:].mean()), 1e-9)\n"
        "    if recent > 1.2:\n"
        "        return (0.6, 0.2, 0.2)\n"
        "    return (0.34, 0.33, 0.33)\n"
    ),
}


class FakeClient:
    def __init__(self, answer: str | Exception) -> None:
        self.answer = answer
        self.timeout = 60.0
        self.messages: list[dict[str, str]] | None = None

    def complete(self, messages: list[dict[str, str]], *, temperature: float = 0.0) -> str:
        self.messages = messages
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def _answer(payload: object) -> str:
    return json.dumps(payload)


class TestParseAgentProposals:
    def test_plain_json(self) -> None:
        assert parse_agent_proposals(_answer([VALID_PROPOSAL])) == [VALID_PROPOSAL]

    def test_fenced_json(self) -> None:
        raw = "```json\n" + _answer([VALID_PROPOSAL]) + "\n```"
        assert parse_agent_proposals(raw) == [VALID_PROPOSAL]

    def test_prose_around_json(self) -> None:
        raw = "Hier sind die Vorschläge: " + _answer([VALID_PROPOSAL]) + " — Ende."
        assert parse_agent_proposals(raw) == [VALID_PROPOSAL]

    def test_no_array_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_agent_proposals("keine JSON-Antwort")

    def test_non_dict_items_dropped(self) -> None:
        assert parse_agent_proposals(_answer([VALID_PROPOSAL, 42, "x"])) == [VALID_PROPOSAL]


class TestPropose:
    def test_valid_proposals_returned(self) -> None:
        client = FakeClient(_answer([VALID_PROPOSAL]))
        proposals = propose(client, "Digest", ("trend",), max_candidates=3)
        assert [p["name"] for p in proposals] == ["volume_drift"]
        assert client.messages is not None
        assert len(client.messages) == 2
        assert client.timeout == 600.0

    def test_llm_error_is_fail_soft(self) -> None:
        client = FakeClient(LLMError("timeout", "Zeitüberschreitung"))
        assert propose(client, "Digest", (), max_candidates=3) == []

    def test_unparseable_answer_is_fail_soft(self) -> None:
        client = FakeClient("Das ist kein JSON.")
        assert propose(client, "Digest", (), max_candidates=3) == []

    def test_unexpected_exception_is_fail_soft(self) -> None:
        client = FakeClient(RuntimeError("boom"))
        assert propose(client, "Digest", (), max_candidates=3) == []

    def test_taken_name_rejected(self) -> None:
        client = FakeClient(_answer([VALID_PROPOSAL]))
        assert propose(client, "Digest", ("volume_drift",), max_candidates=3) == []

    def test_bad_name_rejected(self) -> None:
        proposal = dict(VALID_PROPOSAL, name="BAD NAME")
        client = FakeClient(_answer([proposal]))
        assert propose(client, "Digest", (), max_candidates=3) == []

    def test_short_claim_rejected(self) -> None:
        proposal = dict(VALID_PROPOSAL, claim="zu kurz")
        client = FakeClient(_answer([proposal]))
        assert propose(client, "Digest", (), max_candidates=3) == []

    def test_short_code_rejected(self) -> None:
        proposal = dict(VALID_PROPOSAL, code="def predict(o,h,l,c,v): return (1,0,0)")
        client = FakeClient(_answer([proposal]))
        assert propose(client, "Digest", (), max_candidates=3) == []

    def test_duplicate_names_deduped(self) -> None:
        client = FakeClient(_answer([VALID_PROPOSAL, dict(VALID_PROPOSAL)]))
        proposals = propose(client, "Digest", (), max_candidates=3)
        assert len(proposals) == 1

    def test_max_candidates_capped(self) -> None:
        items = [dict(VALID_PROPOSAL, name=f"agent_{i}") for i in range(5)]
        client = FakeClient(_answer(items))
        proposals = propose(client, "Digest", (), max_candidates=2)
        assert [p["name"] for p in proposals] == ["agent_0", "agent_1"]

    def test_max_candidates_in_prompt(self) -> None:
        messages = build_messages("Digest", ("trend",), 3)
        assert "3" in messages[1]["content"]
        assert "trend" in messages[1]["content"]
