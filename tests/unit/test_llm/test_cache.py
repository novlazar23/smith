"""Tests für packages.llm.cache (JSONL-Roundtrip, Reload, Last-Write-Wins, Key)."""

from __future__ import annotations

from pathlib import Path

import pytest
from packages.llm.cache import LLMResponseCache


class TestKey:
    """Stabilität des Cache-Keys (SHA-256-Hex)."""

    def test_key_is_stable_sha256_hex(self) -> None:
        first = LLMResponseCache.key("local-fast", "1", "summary-text")
        second = LLMResponseCache.key("local-fast", "1", "summary-text")

        assert first == second
        assert len(first) == 64
        assert all(char in "0123456789abcdef" for char in first)

    def test_key_changes_with_parts(self) -> None:
        base = LLMResponseCache.key("local-fast", "1", "summary")
        assert LLMResponseCache.key("local-flagship", "1", "summary") != base
        assert LLMResponseCache.key("local-fast", "2", "summary") != base
        assert LLMResponseCache.key("local-fast", "1", "other") != base

    def test_key_separates_ambiguous_part_splittings(self) -> None:
        """Unit-Separator-Trennung: (ab, cd) != (a, bcd)."""
        assert LLMResponseCache.key("ab", "cd") != LLMResponseCache.key("a", "bcd")


class TestRoundtrip:
    """put/get im In-Memory-Dict und Datei-Roundtrip."""

    def test_get_miss_returns_none(self, tmp_path: Path) -> None:
        cache = LLMResponseCache(str(tmp_path / "cache.jsonl"))
        assert cache.get("missing") is None

    def test_put_get_roundtrip_creates_parent_dir(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "dir" / "cache.jsonl"
        cache = LLMResponseCache(str(path))

        cache.put("k", "v")

        assert cache.get("k") == "v"
        assert path.is_file()

    def test_unicode_values_roundtrip(self, tmp_path: Path) -> None:
        cache = LLMResponseCache(str(tmp_path / "cache.jsonl"))
        value = 'Decision: "BUY" — Trend bestätigt (42 %)'
        cache.put("k", value)
        assert cache.get("k") == value


class TestReload:
    """Neuinstanzierung liest die Datei; letzter Eintrag pro Key gewinnt."""

    def test_reload_from_disk_last_write_wins(self, tmp_path: Path) -> None:
        path = tmp_path / "cache.jsonl"
        cache = LLMResponseCache(str(path))
        cache.put("k", "first")
        cache.put("other", "value")
        cache.put("k", "second")

        reloaded = LLMResponseCache(str(path))

        assert reloaded.get("k") == "second"
        assert reloaded.get("other") == "value"
        assert reloaded.get("absent") is None

    def test_corrupt_lines_are_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "cache.jsonl"
        cache = LLMResponseCache(str(path))
        cache.put("good", "value")
        with path.open("a", encoding="utf-8") as handle:
            handle.write("not-json\n")
            handle.write("\n")
            handle.write('{"key": "bad", "value": 42}\n')

        reloaded = LLMResponseCache(str(path))

        assert reloaded.get("good") == "value"
        assert reloaded.get("bad") is None

    @pytest.mark.parametrize("corrupt", ["", "{}", "[]"])
    def test_empty_or_non_object_lines_are_ignored(
        self, tmp_path: Path, corrupt: str
    ) -> None:
        path = tmp_path / "cache.jsonl"
        path.write_text(corrupt, encoding="utf-8")

        cache = LLMResponseCache(str(path))

        assert cache.get("k") is None
