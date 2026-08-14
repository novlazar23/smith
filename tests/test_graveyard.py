from __future__ import annotations

from trading_harness.services.graveyard import Graveyard

# ---------------------------------------------------------------------------
# Basic operations
# ---------------------------------------------------------------------------


def test_add_entry():
    grave = Graveyard()
    record = grave.add("agent-1", "technical", final_score=0.15, reason="Poor performance")
    assert record.agent_id == "agent-1"
    assert record.category == "technical"
    assert record.final_score == 0.15
    assert record.reason == "Poor performance"


def test_add_multiple_entries():
    grave = Graveyard()
    grave.add("agent-1", "technical", final_score=0.10, reason="Underperformed")
    grave.add("agent-2", "macro", final_score=0.05, reason="Failed OOS")
    entries = grave.get_entries()
    assert len(entries) == 2


def test_get_entries():
    grave = Graveyard()
    grave.add("agent-1", "technical", final_score=0.20, reason="Retired")
    grave.add("agent-2", "technical", final_score=0.05, reason="Rejected")
    entries = grave.get_entries()
    assert len(entries) == 2
    assert all(r.category == "technical" for r in entries)


def test_get_by_agent():
    grave = Graveyard()
    grave.add("agent-1", "technical", final_score=0.15, reason="Retired")
    record = grave.get_by_agent("agent-1")
    assert record is not None
    assert record.final_score == 0.15
    assert grave.get_by_agent("nonexistent") is None


def test_get_by_category():
    grave = Graveyard()
    grave.add("agent-1", "technical", final_score=0.10, reason="Retired")
    grave.add("agent-2", "macro", final_score=0.05, reason="Failed")
    grave.add("agent-3", "technical", final_score=0.01, reason="Rejected")
    tech = grave.get_by_category("technical")
    assert len(tech) == 2
    assert all(r.category == "technical" for r in tech)
    macro = grave.get_by_category("macro")
    assert len(macro) == 1
    empty = grave.get_by_category("orderflow")
    assert len(empty) == 0


def test_empty_graveyard():
    grave = Graveyard()
    assert grave.get_entries() == []
    assert grave.get_by_agent("anything") is None
    assert grave.get_by_category("technical") == []