"""Shadow-Sequenztest: einseitige z-Tests als Kandidat-Ersatz für die log2-Hurdle.

Komplement zu ``trial_ledger`` (steigende fixe Hurdle) — hier ersetzt
ein inferenzbasierter Test die Hurdle:

- **Stufe 1** (Parameter): gepaarter einseitiger z-Test pro Variante auf
  der pro-Sample-Score-Differenz (Variante - Champion) über die OOS-
  Samples, H0: ``mu_diff <= 0.005``; Benjamini-Hochberg (``q=0.05``)
  kontrolliert die Falsch-Entdeckungs-Rate über den täglichen Batch
  (4 Familien x 8 Varianten = 32 Kandidaten).
- **Stufe 2** (Agenten-Logik): einseitiger z-Test (eine Stichprobe) pro
  Kandidat auf der pro-Sample-Differenz (Score - Zufalls-Basis 1/3),
  H0: ``mu <= 0.02``; Holm-Bonferroni (``alpha=0.05``) über den Batch.

Die Standardfehler tragen eine Newey-West (HAC, Bartlett)-Korrektur mit
Lag = 1 Kalendertag (in Samples gemessen). Vor-Messung an den
180-Tage-Produktionsdaten (21.09.2026, trend-Familie, N=4300): die
Autokorrelation der Differenzreihe zerfällt innerhalb eines Tages
(``rho(1d) <= 0.094``), und die HAC-SE ist im ungünstigsten Fall 1.55x
die naive — naive Standardfehler würden Signifikanz leicht
überschätzen, daher Korrektur.

**Shadow-Modus:** Diese Tests entscheiden NICHTS. Die Promotion/
Zulassung bleibt bei der bestehenden Trial-Ledger-Hurdle; die Test-
Entscheidungen werden parallel protokolliert (``champion_shadow.jsonl``
für Stufe 1, ``evolved_agents_last_run.json`` für Stufe 2), damit der
Wechsel nach 1-2 Wochen datengestützt erfolgen kann.

Beobachtetes Degenerat-Verhalten (4 von 8 Varianten der Vor-Messung):
regeldominierte Agenten liefern in benachbarten Parametersätzen
identische Prädiktionen → konstante Differenz, Varianz 0. Dann ist der
Effekt exakt da oder nicht: ``p = 0`` wenn ``mean > min_effect``,
sonst ``p = 1``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

#: Shadow-Log (Stufe 1): eine JSON-Zeile pro Lauf, neben ``champion_configs.json``.
SHADOW_LOG_FILENAME = "champion_shadow.jsonl"


def newey_west_factor(diffs: Sequence[float], lag: int) -> float:
    """HAC-Faktor: ``SE_newey_west / SE_naive`` für den Mittelwert (>= 1).

    Bartlett-Kern: ``1 + 2 * sum_k (1 - k/(L+1)) * rho_k``. Positive
    Autokorrelation inflatiert die Varianz des Mittelwerts; negative
    Autokorrelation (Faktor < 1) wird auf 1.0 gekappt — bei N ~ 4300 ist
    die naive SE dort die konservative Referenz. ``lag <= 0``, zu kleine
    Stichprobe oder Varianz 0 → 1.0 (naive).
    """
    n = len(diffs)
    if lag <= 0 or n <= lag or n < 3:
        return 1.0
    mean = sum(diffs) / n
    d = [x - mean for x in diffs]
    g0 = sum(x * x for x in d) / n
    if g0 <= 0:
        return 1.0
    total = g0
    for k in range(1, lag + 1):
        gk = sum(d[t] * d[t + k] for t in range(n - k)) / n
        total += 2.0 * (1.0 - k / (lag + 1)) * gk
    return max(1.0, math.sqrt(max(total, 0.0) / g0))


def one_sided_z_pvalue(diffs: Sequence[float], min_effect: float, nw_lag: int = 0) -> float:
    """Einseitiger z-Test (Normalnäherung): H0 ``mean(diffs) <= min_effect``.

    ``diffs``: pro-Sample-Differenzen auf derselben OOS-Stichprobe
    (z. B. Score_Kandidat - Score_Champion oder Score_Kandidat -
    Zufalls-Basis). ``nw_lag > 0``: Newey-West-Korrektur der
    Standardfehler (Lag in Samples; die Wiring übergibt 1 Kalendertag).

    Degeneration (``n < 2`` oder Varianz 0 — bei regeldominierten
    Agenten häufig, siehe Modul-Doku): kein Rauschen, der Effekt ist
    exakt vorhanden oder nicht → ``p = 0.0`` wenn ``mean > min_effect``,
    sonst ``p = 1.0``.
    """
    n = len(diffs)
    if n < 2:
        return 0.0 if n == 1 and diffs[0] > min_effect else 1.0
    mean = sum(diffs) / n
    var = sum((x - mean) ** 2 for x in diffs) / (n - 1)
    if var <= 0.0:
        return 0.0 if mean > min_effect else 1.0
    se = math.sqrt(var / n) * newey_west_factor(diffs, nw_lag)
    z = (mean - min_effect) / se
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def bh_reject(pvalues: Sequence[float], q: float = 0.05) -> frozenset[int]:
    """Benjamini-Hochberg (FDR <= q): Indexe der abgelehnten Null-Hypothesen.

    Größtes k mit ``p_(k) <= q * k / m``; alle Kandidaten bis und
    einschließlich Rank k werden abgelehnt.
    """
    m = len(pvalues)
    if m == 0:
        return frozenset()
    order = sorted(range(m), key=lambda i: pvalues[i])
    for rank in range(m, 0, -1):
        if pvalues[order[rank - 1]] <= q * rank / m:
            return frozenset(order[:rank])
    return frozenset()


def holm_reject(pvalues: Sequence[float], alpha: float = 0.05) -> frozenset[int]:
    """Holm-Bonferroni (FWER <= alpha): Indexe der abgelehnten Null-Hypothesen.

    Sortiert aufsteigend; abgelehnt, solange ``p_(i) <= alpha / (m - i + 1)``
    — beim ersten Verfehlen stoppt die Sequenz.
    """
    m = len(pvalues)
    if m == 0:
        return frozenset()
    order = sorted(range(m), key=lambda i: pvalues[i])
    rejected: set[int] = set()
    for pos, idx in enumerate(order):
        if pvalues[idx] <= alpha / (m - pos):
            rejected.add(idx)
        else:
            break
    return frozenset(rejected)


def daily_lag(as_ofs: Sequence[datetime]) -> int:
    """Newey-West-Lag = 1 Kalendertag, ausgedrückt in Samples.

    ``max(1, round(Anzahl_Samples / abgedeckte_Tage))`` — bei der
    Produktions-Kadenz (stündliche Bewertungen, 2 Instrumente auf einer
    Zeitachse) ergibt das 48 Samples/Tag. Ungültige/zu kurze Achse → 1.
    """
    if len(as_ofs) < 2:
        return 1
    span_days = max(1.0, (as_ofs[-1] - as_ofs[0]).total_seconds() / 86400.0)
    return max(1, round(len(as_ofs) / span_days))


def append_jsonl(path: str | Path, entry: Mapping[str, Any]) -> Path:
    """Appendet eine JSON-Zeile (Shadow-Log).

    Kein atomarer Schreibschutz nötig: der einzige Reader ist der Mensch
    (eine Zeile pro Lauf, mehrere hundert Bytes).
    """
    out = Path(path)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return out
