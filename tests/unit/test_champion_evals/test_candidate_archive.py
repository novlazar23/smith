"""Tests für das Kandidaten-Archiv (Stufe 2): Code-Hash-Dedup + Gate-Reflexion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from apps.champion_evals.candidate_archive import (
    ARCHIVE_FILENAME,
    code_hash,
    format_archive_digest,
    known_code_hashes,
    load_archive,
    recent_rejections,
    record_candidates,
)


def _entry(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "run_at": "2026-09-22T04:00:00",
        "name": "volume_drift",
        "persona": "trend",
        "claim": "Anhaltendes Übermaß an Up-Volumen.",
        "code_hash": code_hash("Code A"),
        "admitted": False,
        "score": 0.34,
        "oos_brier": 0.66,
        "reasons": ["OOS-Score 0.3400 < Zufalls-Basis"],
        "shadow_p": 1.0,
        "shadow_holm_rejected": False,
        "effective_margin": 0.02,
    }
    base.update(overrides)
    return base


class TestCodeHash:
    def test_deterministic(self) -> None:
        assert code_hash("def predict(): ...") == code_hash("def predict(): ...")

    def test_distinct_for_different_code(self) -> None:
        assert code_hash("Code A") != code_hash("Code B")

    def test_is_sha256_hexdigest(self) -> None:
        digest = code_hash("x")
        assert len(digest) == 64
        assert int(digest, 16) >= 0


class TestRecordAndLoad:
    def test_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / ARCHIVE_FILENAME
        entries = [_entry(), _entry(name="other_one", admitted=True)]
        assert record_candidates(path, entries) == path
        assert load_archive(path) == entries

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert load_archive(tmp_path / ARCHIVE_FILENAME) == []

    def test_corrupt_line_skipped_with_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        path = tmp_path / ARCHIVE_FILENAME
        path.write_text(
            json.dumps(_entry(), ensure_ascii=False) + "\n"
            + "{defekt" + "\n"
            + json.dumps(_entry(name="zweite"), ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        loaded = load_archive(path)
        assert [entry["name"] for entry in loaded] == ["volume_drift", "zweite"]
        assert any("übersprungen" in record.message for record in caplog.records)

    def test_non_object_line_skipped(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        path = tmp_path / ARCHIVE_FILENAME
        path.write_text("[1, 2]\n" + json.dumps(_entry(), ensure_ascii=False) + "\n", encoding="utf-8")
        loaded = load_archive(path)
        assert [entry["name"] for entry in loaded] == ["volume_drift"]
        assert any("übersprungen" in record.message for record in caplog.records)

    def test_empty_entries_leave_path_untouched(self, tmp_path: Path) -> None:
        path = tmp_path / ARCHIVE_FILENAME
        assert record_candidates(path, []) == path
        assert not path.exists()

    def test_append_extends_existing_file(self, tmp_path: Path) -> None:
        path = tmp_path / ARCHIVE_FILENAME
        record_candidates(path, [_entry()])
        record_candidates(path, [_entry(name="zweite")])
        assert [entry["name"] for entry in load_archive(path)] == ["volume_drift", "zweite"]


class TestKnownCodeHashes:
    def test_collects_hashes(self, tmp_path: Path) -> None:
        path = tmp_path / ARCHIVE_FILENAME
        hash_a, hash_b = code_hash("Code A"), code_hash("Code B")
        record_candidates(
            path,
            [
                _entry(code_hash=hash_a),
                _entry(name="other_one", code_hash=hash_b),
                _entry(name="ohne_hash", code_hash=""),
            ],
        )
        assert known_code_hashes(path) == {hash_a, hash_b}

    def test_missing_file_empty_set(self, tmp_path: Path) -> None:
        assert known_code_hashes(tmp_path / ARCHIVE_FILENAME) == set()


class TestRecentRejections:
    def test_only_rejected_newest_first_deduped(self, tmp_path: Path) -> None:
        path = tmp_path / ARCHIVE_FILENAME
        # Datei-Reihenfolge = chronologisch (alt → neu):
        record_candidates(
            path,
            [
                _entry(name="a", code_hash=code_hash("1"), score=0.30, run_at="2026-09-20T04:00:00"),
                _entry(name="b", code_hash=code_hash("2"), admitted=True),
                _entry(name="a", code_hash=code_hash("1"), score=0.31, run_at="2026-09-21T04:00:00"),
                _entry(name="a", code_hash=code_hash("3"), score=0.32, run_at="2026-09-22T04:00:00"),
                _entry(name="b", code_hash=code_hash("4"), run_at="2026-09-22T04:00:00"),
            ],
        )
        result = recent_rejections(path)
        # Zulassungen (b/Code-2) bleiben draußen; Duplikat (a/Code-1) →
        # neuestes Vorkommnis gewinnt; Reihenfolge: neueste zuerst.
        assert [entry["name"] for entry in result] == ["b", "a", "a"]
        assert [entry["score"] for entry in result if entry["name"] == "a"] == [0.32, 0.31]

    def test_limit_respected(self, tmp_path: Path) -> None:
        path = tmp_path / ARCHIVE_FILENAME
        record_candidates(
            path,
            [_entry(name=f"agent_{i:02d}", code_hash=code_hash(f"Code {i}")) for i in range(20)],
        )
        result = recent_rejections(path, limit=5)
        assert len(result) == 5
        assert result[0]["name"] == "agent_19"
        assert result[-1]["name"] == "agent_15"

    def test_missing_file_empty(self, tmp_path: Path) -> None:
        assert recent_rejections(tmp_path / ARCHIVE_FILENAME) == []


class TestFormatArchiveDigest:
    def test_line_format(self) -> None:
        digest = format_archive_digest(
            [_entry(score=0.3456, reasons=["Score zu niedrig", "LOO ≤ 0"], code_hash=code_hash("Code A"))]
        )
        line = digest.splitlines()[0]
        assert line.startswith("- volume_drift (trend, 2026-09-22):")
        assert "Score 0.3456" in line
        assert "Gate: Score zu niedrig; LOO ≤ 0" in line
        assert f"Code {code_hash('Code A')[:12]}" in line

    def test_none_score_and_empty_reasons(self) -> None:
        digest = format_archive_digest([_entry(score=None, reasons=[])])
        assert "Score n/a" in digest
        assert "Gate: (ohne Metriken)" in digest

    def test_none_persona_is_handgeschrieben(self) -> None:
        digest = format_archive_digest([_entry(persona=None)])
        assert "(handgeschrieben, 2026-09-22)" in digest

    def test_empty_entries_empty_string(self) -> None:
        assert format_archive_digest([]) == ""

    def test_limit_caps_lines(self) -> None:
        entries = [_entry(name=f"agent_{i:02d}", code_hash=code_hash(f"C{i}")) for i in range(20)]
        assert len(format_archive_digest(entries, limit=15).splitlines()) == 15
