"""Tests für die Seed-Ableitung der Evolutions-Mutationen.

Ohne expliziten Seed muss der tägliche Lauf neue Varianten erkunden
(Seed = UTC-Tag) statt denselben deterministischen Pfad zu wiederholen;
ein einzelner Tag bleibt dabei reproduzierbar.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime

import pytest
from apps.champion_evals import __main__ as champion_evals_main
from apps.champion_evals.__main__ import _evolve_seed, build_parser


def _args(*extra: str) -> argparse.Namespace:
    return build_parser().parse_args(["--output", "/tmp/champion_evals.json", *extra])


def _freeze(monkeypatch: pytest.MonkeyPatch, moment: datetime) -> None:
    class _FixedDatetime(datetime):
        @classmethod
        def now(cls, tz: datetime.tzinfo | None = None) -> datetime:
            del tz
            return moment

    monkeypatch.setattr(champion_evals_main, "datetime", _FixedDatetime)


class TestEvolveSeed:
    def test_parser_default_is_none(self) -> None:
        """--seed ist jetzt optional (Default None = Tag-Seed)."""
        assert _args().seed is None

    def test_explicit_seed_wins(self) -> None:
        assert _evolve_seed(_args("--seed", "7")) == 7

    def test_omitted_seed_derives_from_utc_day(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _freeze(monkeypatch, datetime(2026, 9, 14, 6, 30, tzinfo=UTC))
        assert _evolve_seed(_args()) == 20260914

    def test_day_seed_reproducible_within_same_moment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _freeze(monkeypatch, datetime(2026, 9, 14, 23, 59, tzinfo=UTC))
        assert _evolve_seed(_args()) == _evolve_seed(_args()) == 20260914
