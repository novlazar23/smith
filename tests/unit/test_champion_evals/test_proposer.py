"""Tests für den LLM-Proposer: Parsing, Validierung, Persona-Panel, Fail-Soft."""

from __future__ import annotations

import json
import threading

import pytest
from apps.champion_evals.proposer import PERSONAS, build_messages, parse_agent_proposals, propose
from packages.llm.errors import LLMError

VALID_PROPOSAL = {
    "name": "volume_drift",
    "claim": "Anhaltendes Übermaß an Up-Volumen vorwärts-indiziert Fortsetzung der Bewegung.",
    "code": (
        "import numpy as np\n"
        "\n"
        "def predict(open, high, low, close, volume, timestamps):\n"
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
    """LLM-Client-Doppel (thread-safe für die parallelen Panel-Aufrufe).

    ``answer`` ist entweder ein einzelner Wert (jeder Aufruf liefert ihn)
    oder eine Liste (Aufruf i liefert ``answer[i]``; ab Ende der Liste
    der letzte Eintrag). ``calls`` protokolliert alle empfangenen
    Message-Listen (einschließlich der Persona-System-Prompts).
    """

    def __init__(self, answer: str | Exception | list[str | Exception] | None = None) -> None:
        self.answer = answer
        self.timeout = 60.0
        self.messages: list[dict[str, str]] | None = None
        self.calls: list[list[dict[str, str]]] = []
        self._lock = threading.Lock()

    def complete(self, messages: list[dict[str, str]], *, temperature: float = 0.0) -> str:
        with self._lock:
            idx = len(self.calls)
            self.calls.append(messages)
            self.messages = messages
            if isinstance(self.answer, list):
                current: str | Exception = self.answer[min(idx, len(self.answer) - 1)]
            else:
                current = self.answer if self.answer is not None else ""
        if isinstance(current, Exception):
            raise current
        return current


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

    def test_bare_object_accepted(self) -> None:
        # Modell-Verhalten aus der Refine-Runde: einzelnes Objekt statt Array
        assert parse_agent_proposals(json.dumps(VALID_PROPOSAL)) == [VALID_PROPOSAL]

    def test_bare_object_with_prose(self) -> None:
        raw = "Hier ist die Überarbeitung: " + json.dumps(VALID_PROPOSAL, indent=2) + " — Ende."
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
        answers = [
            _answer([dict(VALID_PROPOSAL, name=f"agent_{i}")]) for i in range(len(PERSONAS))
        ]
        client = FakeClient(answers)
        proposals = propose(client, "Digest", (), max_candidates=2)
        assert [p["name"] for p in proposals] == ["agent_0", "agent_1"]

    def test_max_candidates_in_prompt(self) -> None:
        messages = build_messages("Digest", ("trend",), 3)
        assert "3" in messages[1]["content"]
        assert "trend" in messages[1]["content"]


class TestPersonaPanel:
    def test_one_llm_call_per_persona(self) -> None:
        client = FakeClient(["[]"] * len(PERSONAS))
        assert propose(client, "Digest", (), max_candidates=3) == []
        assert len(client.calls) == len(PERSONAS)

    def test_persona_failure_is_fail_soft(self) -> None:
        answers = [
            LLMError("timeout", "Gateway-Timeout"),
            _answer([VALID_PROPOSAL]),
            "[]",
            "[]",
        ]
        client = FakeClient(answers)
        proposals = propose(client, "Digest", (), max_candidates=3)
        assert [p["name"] for p in proposals] == ["volume_drift"]
        # 6 Persona-Aufrufe + 1 Skeptiker-Aufruf (Kritik leer → fail-soft)
        assert len(client.calls) == len(PERSONAS) + 1

    def test_persona_system_prompts_are_distinct(self) -> None:
        client = FakeClient(["[]"] * len(PERSONAS))
        propose(client, "Digest", (), max_candidates=3)
        systems = [call[0]["content"] for call in client.calls]
        assert len(set(systems)) == len(systems)

    def test_role_prompt_contains_prior(self) -> None:
        messages = build_messages("Digest", (), 2, role="trend")
        assert "trend" in messages[0]["content"]
        assert "Autokorrelation" in messages[0]["content"]

    def test_legacy_prompt_without_role(self) -> None:
        messages = build_messages("Digest", ("trend",), 3)
        assert "moderierst" in messages[0]["content"]

    def test_per_persona_budget_scales_with_max_candidates(self) -> None:
        client = FakeClient(["[]"] * len(PERSONAS))
        propose(client, "Digest", (), max_candidates=8)
        assert "Maximale Anzahl Vorschläge: 2" in client.calls[0][1]["content"]


REVISED_CODE = VALID_PROPOSAL["code"] + "    # revidiert: robustere Schwellen (Skeptiker-Kritik)\n"
SKEPTIC_CRITIQUE = _answer(
    [{"name": "volume_drift", "critique": "Schwelle 1.2 ist auf den letzten Spike gefittet; Overfitting-Risiko, robuste Schätzung nötig."}]
)


def _round1_answers() -> list[str]:
    """Sechs Persona-Antworten, von denen genau eine einen Vorschlag liefert."""
    return ["[]"] * (len(PERSONAS) - 1) + [_answer([VALID_PROPOSAL])]


class TestCritiqueRound:
    def test_critique_triggers_refine(self) -> None:
        answers = [*_round1_answers(), SKEPTIC_CRITIQUE, _answer([dict(VALID_PROPOSAL, code=REVISED_CODE)])]
        client = FakeClient(answers)
        proposals = propose(client, "Digest", (), max_candidates=3)
        assert [p["name"] for p in proposals] == ["volume_drift"]
        assert proposals[0]["code"] == REVISED_CODE
        assert len(client.calls) == len(PERSONAS) + 2

    def test_skeptic_failure_keeps_original(self) -> None:
        answers = [*_round1_answers(), "keine JSON-Antwort"]
        client = FakeClient(answers)
        proposals = propose(client, "Digest", (), max_candidates=3)
        assert proposals[0]["code"] == VALID_PROPOSAL["code"]
        assert len(client.calls) == len(PERSONAS) + 1

    def test_refine_failure_keeps_original(self) -> None:
        answers = [*_round1_answers(), SKEPTIC_CRITIQUE, LLMError("timeout", "Gateway-Timeout")]
        client = FakeClient(answers)
        proposals = propose(client, "Digest", (), max_candidates=3)
        assert proposals[0]["code"] == VALID_PROPOSAL["code"]
        assert len(client.calls) == len(PERSONAS) + 2

    def test_refine_keeps_original_name(self) -> None:
        renamed = dict(VALID_PROPOSAL, name="other_name", code=REVISED_CODE)
        answers = [*_round1_answers(), SKEPTIC_CRITIQUE, _answer([renamed])]
        client = FakeClient(answers)
        proposals = propose(client, "Digest", (), max_candidates=3)
        assert proposals[0]["name"] == "volume_drift"
        assert proposals[0]["code"] == REVISED_CODE

    def test_critique_for_foreign_name_skips_refine(self) -> None:
        foreign = _answer(
            [{"name": "anderer_agent", "critique": "Dieser Vorschlag ist redundant und sollte entfernt werden."}]
        )
        answers = [*_round1_answers(), foreign]
        client = FakeClient(answers)
        proposals = propose(client, "Digest", (), max_candidates=3)
        assert proposals[0]["code"] == VALID_PROPOSAL["code"]
        assert len(client.calls) == len(PERSONAS) + 1
