from __future__ import annotations

import json
from pathlib import Path


def test_opencode_project_setup_is_self_contained() -> None:
    """OpenCode must receive versioned rules, commands, and safe permissions."""
    repository_root = Path(__file__).resolve().parents[1]

    agents = (repository_root / "AGENTS.md").read_text(encoding="utf-8")
    assert "make check" in agents
    assert "Live-Execution" in agents
    assert "Codex" not in agents

    config = json.loads((repository_root / "opencode.json").read_text(encoding="utf-8"))
    assert "model" not in config
    assert config["permission"]["external_directory"] == "deny"
    assert config["permission"]["bash"]["git push*"] == "ask"
    assert config["permission"]["bash"]["git push --force*"] == "deny"

    commands = repository_root / ".opencode" / "commands"
    assert {"check.md", "handoff.md", "resume.md"} <= {
        path.name for path in commands.glob("*.md")
    }
