from __future__ import annotations

from trading_harness.services.hall_of_fame import HallOfFame

# ---------------------------------------------------------------------------
# Hall of Fame basic operations
# ---------------------------------------------------------------------------


def test_add_first_entry():
    hof = HallOfFame()
    record = hof.add("agent-1", "technical", score=0.85, observations=100)
    assert record.agent_id == "agent-1"
    assert record.category == "technical"
    assert record.score == 0.85
    assert record.observations == 100


def test_add_updates_existing_higher_score():
    hof = HallOfFame()
    hof.add("agent-1", "technical", score=0.70, observations=50)
    record = hof.add("agent-1", "technical", score=0.85, observations=100, reason="Improved")
    assert record.score == 0.85
    assert record.observations == 100
    assert record.reason == "Improved"
    assert len(hof.get_entries()) == 1


def test_add_does_not_update_lower_score():
    hof = HallOfFame()
    hof.add("agent-1", "technical", score=0.85, observations=100)
    record = hof.add("agent-1", "technical", score=0.70, observations=50)
    assert record.score == 0.85  # Original score preserved
    assert len(hof.get_entries()) == 1


def test_get_entries_sorted_by_score():
    hof = HallOfFame()
    hof.add("agent-c", "technical", score=0.50, observations=1)
    hof.add("agent-a", "macro", score=0.90, observations=1)
    hof.add("agent-b", "technical", score=0.75, observations=1)
    entries = hof.get_entries()
    assert len(entries) == 3
    assert entries[0].score == 0.90
    assert entries[1].score == 0.75
    assert entries[2].score == 0.50


def test_get_top_n():
    hof = HallOfFame()
    for i in range(5):
        hof.add(f"agent-{i}", "technical", score=float(i), observations=1)
    top2 = hof.get_top_n(2)
    assert len(top2) == 2
    assert top2[0].score == 4.0
    assert top2[1].score == 3.0


def test_get_by_agent():
    hof = HallOfFame()
    hof.add("agent-1", "technical", score=0.85, observations=1)
    record = hof.get_by_agent("agent-1")
    assert record is not None
    assert record.score == 0.85
    assert hof.get_by_agent("nonexistent") is None


def test_get_by_category():
    hof = HallOfFame()
    hof.add("agent-1", "technical", score=0.85, observations=1)
    hof.add("agent-2", "macro", score=0.70, observations=1)
    hof.add("agent-3", "technical", score=0.60, observations=1)
    tech = hof.get_by_category("technical")
    assert len(tech) == 2
    macro = hof.get_by_category("macro")
    assert len(macro) == 1
    empty = hof.get_by_category("orderflow")
    assert len(empty) == 0


def test_max_entries_limit():
    hof = HallOfFame(max_entries=3)
    for i in range(5):
        hof.add(f"agent-{i}", "technical", score=float(i), observations=1)
    entries = hof.get_entries()
    assert len(entries) == 3
    assert entries[0].score == 4.0
    assert entries[2].score == 2.0


def test_empty_hof():
    hof = HallOfFame()
    assert hof.get_entries() == []
    assert hof.get_top_n(5) == []
    assert hof.get_by_category("technical") == []