"""Trial-Ledger: kumulativer Trial-Count und steigende Zulassungs-Hurdle.

Beide Evolutionsstufen prüfen Kandidaten gegen dieselbe rollierende
OOS-Datenbasis (180-Tage-Fenster): Stufe 1 mutierte Parametersätze
pro Agent-Familie, Stufe 2 LLM-generierte Agenten-Logiken. Jeder
geprüfte Kandidat ist damit ein weiterer Test auf derselben
Datenbasis — bei fester Hurdle häufen sich falsche Zulassungen und
Promotions mit der Anzahl N getesteter Kandidaten (Multiple-Testing-
Problem durch Datenwiederverwendung).

Das Ledger zählt die kumulativ getesteten Kandidaten pro Stufe
persistenzfähig (eine JSON-Datei neben dem jeweiligen Artefakt), und
die Zulassungs-/Promotions-Margin steigt pro Verdopplung des
Suchraums:

    margin = base + per_doubling · max(0, log2(max(1, trials)))

Analogie zu Bonferroni/DSR, aber in Log-Skala: Der Preis pro
Verdopplung des Suchraums ist konstant (``per_doubling``), nicht
linear in N — das hält die Hurdle auch bei hunderten täglichen
Läufen handhabbar, während sie trotzdem monoton mit der
Suchraum-Expansion wächst.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from apps.champion_evals.agent_params import AGENT_TYPES
from apps.champion_evals.evolve import write_json_atomic

logger = logging.getLogger(__name__)

#: Ledger-Datei neben ``evolved_agents.json`` (Stufe 2).
TRIALS_FILENAME_EVOLVED = "evolved_agents_trials.json"
#: Ledger-Datei neben ``champion_configs.json`` (Stufe 1).
TRIALS_FILENAME_CHAMPION = "champion_trials.json"


def admission_margin(trials: int, base: float, per_doubling: float = 0.005) -> float:
    """Steigende Zulassungs-/Promotions-Hurdle nach kumulativem Trial-Count.

    Formel: ``base + per_doubling · max(0, log2(max(1, trials)))``.

    Rationale: Jeder Kandidat (Stufe 1: mutierter Parametersatz, Stufe
    2: LLM-Agenten-Logik) prüft dieselbe rollierende OOS-Datenbasis
    (180-Tage-Fenster). Bei fester Hurdle häufen sich falsche
    Zulassungen/Promotions mit der Anzahl N getesteter Kandidaten
    (Multiple-Testing-Problem, Datenwiederverwendung). Die Hurdle steigt
    daher pro Verdopplung des Suchraums um ``per_doubling`` — Analogie
    zu Bonferroni/DSR, aber in Log-Skala (der Preis einer Verdopplung
    ist konstant, nicht linear in N).

    Monoton nicht-fallend in ``trials``; ``trials ≤ 1`` → exakt ``base``
    (der erste Kandidat trägt noch keine Multi-Test-Korrektur).
    """
    return base + per_doubling * max(0.0, math.log2(max(1, trials)))


def load_trial_count(path: str | Path) -> int:
    """Lädt den kumulativen Trial-Count fail-soft.

    Fehlende Datei → ``0`` (erster Lauf). Defekte JSON, fehlendes
    ``trials``-Feld, nicht-ganzzahlige oder negative Werte → ``0`` mit
    ``logger.warning`` — ein defektes Ledger darf den Lauf nicht
    abbrechen (Fail-Soft, konsistent mit den anderen Artefakt-Loadern;
    die Hurdle fällt dann für diesen Lauf auf die Basis-Margin zurück).
    """
    file = Path(path)
    if not file.exists():
        return 0
    try:
        data: Any = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Trial-Ledger %s nicht lesbar (%s) — Trial-Count 0", file, exc)
        return 0
    if not isinstance(data, dict):
        logger.warning("Trial-Ledger %s erwartet ein JSON-Objekt — Trial-Count 0", file)
        return 0
    count = data.get("trials")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        logger.warning("Trial-Ledger %s: 'trials' ungültig (%r) — Trial-Count 0", file, count)
        return 0
    return count


def record_trial_count(path: str | Path, count: int) -> Path:
    """Persistiert den kumulativen Trial-Count atomar (tmp + rename).

    Schreibt ``{"trials": count, "updated_at": <UTC-ISO, Sekunden>}``
    über ``write_json_atomic`` (Reader sehen nie halbe Stände).
    """
    payload = {
        "trials": count,
        "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    return write_json_atomic(path, payload)


def stage1_batch_size(variants: int) -> int:
    """Trial-Batch-Größe eines Stufe-1-Laufs.

    Pro Agent-Familie (``AGENT_TYPES``) werden ``variants`` mutierte
    Parametersätze gegen das OOS-Fenster geprüft — jede Variante ist ein
    Trial. Batch = ``len(AGENT_TYPES) · variants``.
    """
    return len(AGENT_TYPES) * variants
