"""Konfig-Varianten: Grid-Erzeugung + Preregistrierung mit Dedup.

Varianten werden nur dann vorgeschlagen, wenn sie (a) im
``param_specs``-Bounds der Strategie liegen, (b) nicht bereits als
Hypothese existieren und (c) nicht im Grab liegen (gleiche Strategie +
Parameter). Der Grid selbst ist **kein** Test — er erzeugt nur
preregistrierte Hypothesen; die Entscheidung fällt der Judge.
"""

from __future__ import annotations

import logging
from itertools import product
from typing import Any

from packages.strategies import describe, list_strategies

from .models import Hypothesis, Proposal, Variant, hypothesis_from_proposal, variant_key
from .state import EvolutionStore

logger = logging.getLogger(__name__)


def grid_combinations(grid: dict[str, list[float]]) -> list[dict[str, float]]:
    """Alle Kombinationen eines Parameter-Grids (leeres Grid → [{}])."""
    keys = list(grid)
    if not keys:
        return [{}]
    return [dict(zip(keys, values, strict=True)) for values in product(*(grid[key] for key in keys))]


def validate_grid(grid: dict[str, list[float]], family: str) -> list[str]:
    """Prüft Grid gegen das ``param_specs``-Manifest; liefert Ablehnungs-Gründe."""
    if family not in list_strategies():
        return [f"unbekannte Strategie {family!r}"]
    specs = describe(family)["params"]
    problems: list[str] = []
    for key, values in grid.items():
        if key not in specs:
            problems.append(f"Parameter {key!r} nicht in {family}-Manifest")
            continue
        lo, hi = float(specs[key]["min"]), float(specs[key]["max"])
        for value in values:
            if not lo <= float(value) <= hi:
                problems.append(f"{key}={value} außerhalb [{lo:g}, {hi:g}]")
    return problems


def propose_grid(
    store: EvolutionStore,
    family: str,
    grid: dict[str, list[float]],
    *,
    source: str = "sweep",
    max_variants: int = 25,
) -> tuple[list[Hypothesis], list[str]]:
    """Erzeugt Grid-Varianten und preregistriert die neuen (dedup'd).

    Returns:
        ``(neue Hypothesen, übersprungene Gründe)``.
    """
    problems = validate_grid(grid, family)
    if problems:
        return [], problems
    new: list[Hypothesis] = []
    skipped: list[str] = []
    existing_keys = {variant_key(h.variant) for h in store.all_hypotheses()}
    existing_ids = store.hypothesis_ids()
    for params in grid_combinations(grid)[:max_variants]:
        if not params:
            skipped.append(f"{family}::{params}: Baseline-Konfiguration (kein Parameter-Override)")
            continue
        variant = Variant(strategy=family, params={k: float(v) for k, v in params.items()})
        key = variant_key(variant)
        if key in existing_keys:
            skipped.append(f"{key}: bereits preregistriert")
            continue
        if store.in_graveyard(variant) is not None:
            skipped.append(f"{key}: liegt im Grab (nicht retesten)")
            continue
        proposal = Proposal(
            family=family,
            kind="config",
            claim=f"Grid-Variante von {family}: {key}",
            variant=variant,
        )
        hypothesis = hypothesis_from_proposal(proposal, existing_ids, source)
        store.add_hypothesis(hypothesis)
        existing_keys.add(key)
        existing_ids.add(hypothesis.id)
        new.append(hypothesis)
    return new, skipped


def sweep_report(store: EvolutionStore) -> dict[str, Any]:
    """Kurzbericht über den Zustand (für ``--status``)."""
    hypotheses = store.all_hypotheses()
    by_status: dict[str, int] = {}
    for hypothesis in hypotheses:
        by_status[hypothesis.status] = by_status.get(hypothesis.status, 0) + 1
    return {
        "n_hypotheses": len(hypotheses),
        "by_status": by_status,
        "families": sorted({h.family for h in hypotheses}),
        "n_graveyard": len(store.graveyard()),
        "n_promoted": len(store.registry().get("promoted", [])),
        "last_cycle": store.meta().get("last_cycle"),
    }
