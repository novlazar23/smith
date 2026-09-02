"""Marktrege-Szenarien (historische Presets) für den Backtest.

Jedes Szenario definiert einen Zeitfenster (``start``/``end`` als Kalendertag,
UTC) auf den der Kerzen-Feed gefiltert wird. ``full`` bedeutet: ohne
Zeitgrenzen über die gesamte verfügbare Historie.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class ScenarioPreset:
    """Ein benanntes Szenario mit Zeitfenster und Beschreibung."""

    label: str
    start: date | None
    end: date | None
    description: str


SCENARIOS: dict[str, ScenarioPreset] = {
    "crash-2021-05": ScenarioPreset(
        "crash-2021-05",
        date(2021, 5, 15),
        date(2021, 5, 25),
        "BTC-Crash Mai 2021 (ETF-Abwahl, -30 % binnen Tagen)",
    ),
    "pump-2021-11": ScenarioPreset(
        "pump-2021-11",
        date(2021, 11, 1),
        date(2021, 11, 10),
        "Bull-Run November 2021 (Anstieg Richtung Allzeithoch)",
    ),
    "crash-2022-06": ScenarioPreset(
        "crash-2022-06",
        date(2022, 6, 15),
        date(2022, 6, 25),
        "LUNA/UST-Crash Juni 2022 (marktbreiter Einbruch)",
    ),
    "range-2022-03": ScenarioPreset(
        "range-2022-03",
        date(2022, 3, 1),
        date(2022, 3, 31),
        "Range-Phase März 2022 (konsolidierender Markt)",
    ),
    "full": ScenarioPreset(
        "full",
        None,
        None,
        "Gesamthistorie ohne Zeitgrenzen",
    ),
}


def parse_scenarios(raw: str) -> list[tuple[str, date | None, date | None]]:
    """Parst eine Kommagetrennte-Liste von Szenario-Namen.

    Args:
        raw: Kommagetrennte Namen, z.B. ``"crash-2021-05,pump-2021-11"``.

    Returns:
        Liste von (label, start, end)-Tuples in der gegebenen Reihenfolge.

    Raises:
        ValueError: Bei leeren Namen oder unbekanntem Szenario.
    """
    names = [item.strip() for item in raw.split(",") if item.strip()]
    if not names:
        raise ValueError(f"Leere Szenario-Liste: {raw!r}")
    scenarios: list[tuple[str, date | None, date | None]] = []
    for name in names:
        preset = SCENARIOS.get(name)
        if preset is None:
            available = ", ".join(sorted(SCENARIOS))
            raise ValueError(f"Unbekanntes Szenario: {name!r} (verfügbar: {available})")
        scenarios.append((preset.label, preset.start, preset.end))
    return scenarios
