"""Evolutions-State: JSONL-Dateien, append-only.

Dateien im State-Dir (Default s. ``default_state_dir``):

- ``hypotheses.jsonl`` — eine Zeile pro Hypothesen-Version; bei
  Statuswechsel wird eine neue Zeile angehängt, beim Lesen gilt die
  jeweils neueste Version pro ID (true append-only, kein Rewrite).
- ``graveyard.jsonl`` — eine Zeile pro abgelehntem/fehlgeschlagenem
  Kandidaten (Hypothese + Ablehnungsgründe + relevante Metriken).
- ``registry.json`` — promoted Kandidaten (bewusst small, wird rewritten).
- ``state.json`` — Meta: Baseline-Variante, Kosten-Doku, Familien-Zähler,
  Letzter Zyklus.

Der Grab-Eintrag ist der Schutz vor Retesting toter Enden: ``sweep`` und
``cycle`` vergleichen neue Vorschläge gegen den Grabinhalt (gleiche
Variante = nicht erneut testen).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .models import Hypothesis, Variant, variant_key

logger = logging.getLogger(__name__)

HYPOTHESES_FILE = "hypotheses.jsonl"
GRAVEYARD_FILE = "graveyard.jsonl"
REGISTRY_FILE = "registry.json"
META_FILE = "state.json"

DEFAULT_BASELINE: dict[str, Any] = {
    "strategy": "rsi_mean_reversion",
    "params": {"period": 30.0, "buy_below": 20.0, "sell_above": 80.0},
    "label": "D (12. Kalibrierungslauf): defensiver Mean-Reversion-Long",
    "costs": "0,15 %/Seite (Commission 0,1 % + Slippage 5 bps) = BacktestConfig-Default",
}


def default_state_dir() -> Path:
    """State-Dir: Env ``EVOLUTION_STATE_DIR``, sonst Container-Default, sonst ``./evolution``."""
    import os

    if os.environ.get("EVOLUTION_STATE_DIR"):
        return Path(os.environ["EVOLUTION_STATE_DIR"])
    if Path("/app/backtest_reports").is_dir():
        return Path("/app/backtest_reports/evolution")
    return Path("evolution")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Korrupte Zeile in %s übersprungen", path.name)
    return rows


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class EvolutionStore:
    """Dateibasierter Evolutions-State (alle Zugriffe synchron, keine Locks).

    ponytail: JSONL + letztes-Vorkommnis-Verhalten reicht für nächtliche
    Einzel-Cycle-Ausführung; bei parallelen Zyklen wäre ein Lock nötig.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ── Meta (state.json) ──────────────────────────────────────────────

    def meta(self) -> dict[str, Any]:
        path = self.root / META_FILE
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                logger.warning("state.json korrupt — Defaults werden verwendet")
        return {"baseline": dict(DEFAULT_BASELINE), "families": {}, "last_cycle": None}

    def set_meta(self, **updates: object) -> None:
        meta = self.meta()
        meta.update(updates)
        (self.root / META_FILE).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def baseline(self) -> Variant:
        """Aktuelle Referenz-Variante (Baseline), gegen die jeder Kandidat antreten muss."""
        base = self.meta().get("baseline", DEFAULT_BASELINE)
        params = {k: float(v) for k, v in base.get("params", {}).items()}
        return Variant(strategy=base["strategy"], params=params)

    def family_count(self, family: str) -> int:
        """Anzahl aller Hypothesen der Familie (jeder Status) — Deflations-Basis."""
        counter = self.meta().get("families", {})
        if isinstance(counter, dict):
            value = counter.get(family)
            if isinstance(value, int):
                return value
        return sum(1 for h in self.all_hypotheses() if h.family == family)

    def bump_family(self, family: str) -> int:
        meta = self.meta()
        counter = meta.get("families", {})
        if not isinstance(counter, dict):
            counter = {}
        counter[family] = int(counter.get(family, 0)) + 1
        meta["families"] = counter
        (self.root / META_FILE).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return counter[family]

    # ── Hypothesen ─────────────────────────────────────────────────────

    def all_hypotheses(self) -> list[Hypothesis]:
        """Alle Hypothesen, jeweils in der jeweils neuesten Version pro ID."""
        latest: dict[str, dict[str, Any]] = {}
        for row in _read_jsonl(self.root / HYPOTHESES_FILE):
            hid = row.get("id")
            if isinstance(hid, str):
                latest[hid] = row
        result: list[Hypothesis] = []
        for hid in sorted(latest):
            try:
                result.append(Hypothesis.model_validate(latest[hid]))
            except Exception as exc:
                logger.warning("Hypothese %s nicht validierbar (%s) — übersprungen", hid, exc)
        return result

    def hypothesis_ids(self) -> set[str]:
        return {h.id for h in self.all_hypotheses()}

    def add_hypothesis(self, hypothesis: Hypothesis) -> None:
        _append_jsonl(self.root / HYPOTHESES_FILE, hypothesis.model_dump())
        self.bump_family(hypothesis.family)

    def update_hypothesis(self, hypothesis: Hypothesis) -> None:
        """Neue Version anhängen (append-only; beim Lesen gewinnt die neueste Zeile)."""
        _append_jsonl(self.root / HYPOTHESES_FILE, hypothesis.model_dump())

    def find_duplicate(self, variant: Variant) -> Hypothesis | None:
        """Erste Hypothese mit identischer Variante (Strategie + Parameter)."""
        key = variant_key(variant)
        for hypothesis in self.all_hypotheses():
            if variant_key(hypothesis.variant) == key:
                return hypothesis
        return None

    # ── Grab (graveyard.jsonl) ─────────────────────────────────────────

    def graveyard(self) -> list[dict[str, Any]]:
        return _read_jsonl(self.root / GRAVEYARD_FILE)

    def add_graveyard(self, entry: dict[str, Any]) -> None:
        _append_jsonl(self.root / GRAVEYARD_FILE, entry)

    def in_graveyard(self, variant: Variant) -> dict[str, Any] | None:
        """Grab-Eintrag mit identischer Variante (Retesting-Schutz)."""
        key = variant_key(variant)
        for entry in self.graveyard():
            if entry.get("variant_key") == key:
                return entry
        return None

    # ── Registry (registry.json) ───────────────────────────────────────

    def registry(self) -> dict[str, Any]:
        path = self.root / REGISTRY_FILE
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                logger.warning("registry.json korrupt — leere Registry")
        return {"promoted": []}

    def promote(self, entry: dict[str, Any]) -> None:
        registry = self.registry()
        promoted = registry.setdefault("promoted", [])
        if not any(item.get("id") == entry.get("id") for item in promoted):
            promoted.append(entry)
        (self.root / REGISTRY_FILE).write_text(
            json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def promoted_strategy_names(self) -> set[str]:
        names: set[str] = set()
        for item in self.registry().get("promoted", []):
            variant = item.get("variant", {})
            if isinstance(variant, dict) and isinstance(variant.get("strategy"), str):
                names.add(variant["strategy"])
        return names
