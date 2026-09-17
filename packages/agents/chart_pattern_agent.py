"""Chart-Pattern-Agent — Kerzen-, Chartmuster, Fibonacci und Elliott.

Perspektive: „Welche Chartstruktur steht unmittelbar vor dem Auslösen?"
Vier Evidenz-Familien mit ATR-relativen (nicht absoluten) Schwellen:
Kerzenmuster (Kerze 0.25), Chartmuster (0.35), Fibonacci-Goldene-Zone
(0.35) und vereinfachter Elliott-Impuls (0.20). Jede Familie liefert
höchstens ein Signal s in [-1, +1] (größtes |s|, bei Gleichstand das
jüngst gebildete); Kombination als gewichteter Mittelwert plus
Koverage-Faktor. Keine Lookahead, kein Zufall — reine NumPy-Logik,
deterministisch. Die Erkennungslogik und alle Schwellen sind
preregistriert und identisch mit dem Stage-2-Kandidaten
``apps/champion_evals/candidates/chart_pattern_fibonacci.py``.
"""

from __future__ import annotations

import datetime
import uuid
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray
from packages.schemas.agent_report import (
    AgentReport,
    EvidenceReference,
    InvalidationCondition,
)

from .base import AgentConfig, AgentType, BaseAgent

REQUIRED_KEYS = frozenset({"open", "high", "low", "close"})

#: Familien-Gewichte (Kerze, Chart, Fibonacci, Elliott) — preregistriert.
WEIGHTS: tuple[float, float, float, float] = (0.25, 0.35, 0.35, 0.20)

#: Familien-Labels in der Reihenfolge von WEIGHTS (für die Evidenz).
FAMILY_LABELS: tuple[str, str, str, str] = ("candle", "chart", "fib", "elliott")


def _direction_label(value: float) -> str:
    """Mappt einen signierten Wert auf die Evidenz-Richtung."""
    if value > 0.0:
        return "positive"
    if value < 0.0:
        return "negative"
    return "neutral"


def _atr(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
) -> float:
    """ATR14: Mittel der letzten 14 True Ranges (Flat-Window-Guard 1e-9)."""
    prev_close = close[:-1]
    true_range = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - prev_close), np.abs(low[1:] - prev_close)),
    )
    return max(float(true_range[-14:].mean()), 1e-9)


def _local_extrema(values: NDArray[np.float64], win: int, kind: str) -> list[int]:
    """Indizes, deren Wert Max/Min über einer +/-2-Kerzen-Nachbarschaft ist.

    Innerhalb der letzten ``win`` Kerzen; Kanten werden gestutzt.
    """
    n = len(values)
    start = max(0, n - win)
    out: list[int] = []
    for i in range(start, n):
        a = max(start, i - 2)
        b = min(n, i + 3)
        if (kind == "max" and values[i] >= np.max(values[a:b])) or (
            kind == "min" and values[i] <= np.min(values[a:b])
        ):
            out.append(i)
    return out


def _candle_signal(
    open_: NDArray[np.float64],
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    atr: float,
) -> tuple[float, int] | None:
    """Kerzenmuster der letzten 1-3 Kerzen; Signal (s, recency) oder None."""
    n = len(close)
    b = n - 1
    o1, c1 = float(open_[b]), float(close[b])
    h1, l1 = float(high[b]), float(low[b])
    body = abs(c1 - o1)
    rng = h1 - l1
    upper = h1 - max(o1, c1)
    lower = min(o1, c1) - l1
    drift5 = c1 - float(close[b - 5])
    cands: list[tuple[float, int]] = []
    if rng >= 0.25 * atr:
        if lower >= 2.0 * body and upper <= body and drift5 < -0.25 * atr:
            cands.append((0.7, b))  # Hammer nach Abwärtsdrift
        if upper >= 2.0 * body and lower <= body and drift5 > 0.25 * atr:
            cands.append((-0.7, b))  # Shooting Star nach Aufwärtsdrift
    o0, c0 = float(open_[b - 1]), float(close[b - 1])
    body0 = abs(c0 - o0)
    if body > 0.3 * rng and body0 > 0.0:
        if c1 > o1 and c0 < o0 and o1 <= c0 and c1 >= o0:
            cands.append((0.8, b))  # bullische Engulfing
        if c1 < o1 and c0 > o0 and o1 >= c0 and c1 <= o0:
            cands.append((-0.8, b))  # bärische Engulfing
    if b >= 2:
        idxs = (b - 2, b - 1, b)
        greens = all(float(close[i]) > float(open_[i]) for i in idxs)
        reds = all(float(close[i]) < float(open_[i]) for i in idxs)
        opens_ok = all(
            min(float(open_[i - 1]), float(close[i - 1])) <= float(open_[i])
            <= max(float(open_[i - 1]), float(close[i - 1]))
            for i in idxs[1:]
        )
        move = c1 - float(open_[b - 2])
        if greens and opens_ok and move > 2.0 * atr:
            cands.append((0.9, b))  # drei weiße Soldaten
        if reds and opens_ok and move < -2.0 * atr:
            cands.append((-0.9, b))  # drei schwarze Krähen
    if not cands:
        return None
    s, rec = max(cands, key=lambda t: (abs(t[0]), t[1]))
    return (float(s), rec)


def _chart_signal(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    atr: float,
) -> tuple[float, int] | None:
    """Chartmuster der letzten ~40 Kerzen; Signal (s, recency) oder None."""
    n = len(close)
    last = float(close[n - 1])
    tol = max(1.0 * atr, 0.005 * abs(last))
    maxima = _local_extrema(high, 40, "max")
    minima = _local_extrema(low, 40, "min")
    cands: list[tuple[float, int]] = []
    if len(maxima) >= 2:
        t1, t2 = maxima[-2], maxima[-1]
        if t2 > t1 + 1 and abs(float(high[t1]) - float(high[t2])) <= tol:
            trough = float(np.min(low[t1 + 1 : t2]))
            tops = min(float(high[t1]), float(high[t2]))
            if tops - trough >= 1.0 * atr:
                s = -1.0 if last < trough else -0.6
                cands.append((s, t2))  # Doppeltop: Nackenbruch / sich bildend
    if len(minima) >= 2:
        u1, u2 = minima[-2], minima[-1]
        if u2 > u1 + 1 and abs(float(low[u1]) - float(low[u2])) <= tol:
            peak = float(np.max(high[u1 + 1 : u2]))
            bottoms = max(float(low[u1]), float(low[u2]))
            if peak - bottoms >= 1.0 * atr:
                s = 1.0 if last > peak else 0.6
                cands.append((s, u2))  # Doppelboden: Nackenbruch / sich bildend
    if len(maxima) >= 3:
        i1, i2, i3 = maxima[-3], maxima[-2], maxima[-1]
        h1, h2, h3 = float(high[i1]), float(high[i2]), float(high[i3])
        if (
            h2 >= h1 + 0.5 * atr
            and h2 >= h3 + 0.5 * atr
            and abs(h1 - h3) <= 1.0 * atr
            and i2 > i1 + 1
            and i3 > i2 + 1
        ):
            neck = min(float(np.min(low[i1 + 1 : i2])), float(np.min(low[i2 + 1 : i3])))
            s = -1.0 if last < neck else -0.5
            cands.append((s, i3))  # Kopf-Schultern: Nackenbruch / sich bildend
    if len(minima) >= 3 and len(maxima) >= 2:
        m1, m2, m3 = minima[-3], minima[-2], minima[-1]
        if float(low[m2]) >= float(low[m1]) + 0.25 * atr and float(low[m3]) >= float(
            low[m2]
        ) + 0.25 * atr:
            p1, p2 = maxima[-2], maxima[-1]
            if abs(float(high[p1]) - float(high[p2])) <= tol:
                res = max(float(high[p1]), float(high[p2]))
                s = 1.0 if last > res else 0.5
                cands.append((s, m3))  # steigendes Dreieck: Breakout / sich bildend
    if len(maxima) >= 3 and len(minima) >= 2:
        x1, x2, x3 = maxima[-3], maxima[-2], maxima[-1]
        if float(high[x2]) <= float(high[x1]) - 0.25 * atr and float(high[x3]) <= float(
            high[x2]
        ) - 0.25 * atr:
            q1, q2 = minima[-2], minima[-1]
            if abs(float(low[q1]) - float(low[q2])) <= tol:
                sup = min(float(low[q1]), float(low[q2]))
                s = -1.0 if last < sup else -0.5
                cands.append((s, x3))  # fallendes Dreieck: Breakdown / sich bildend
    # Flag-Fortsetzung: 6-Kerzen-Impuls + 6-Kerzen-Tight-Retracement
    net = float(close[n - 7] - close[n - 12])
    if abs(net) >= 3.0 * atr:
        direction = 1.0 if net > 0.0 else -1.0
        band = float(np.max(high[n - 8 : n]) - np.min(low[n - 8 : n]))
        retrace = direction * (float(close[n - 7]) - last)
        if band <= 2.0 * atr and 0.0 <= retrace < 0.5 * abs(net):
            cands.append((0.8 * direction, n - 1))  # Flag-Fortsetzung
    if not cands:
        return None
    s, rec = max(cands, key=lambda t: (abs(t[0]), t[1]))
    return (float(s), rec)


def _fib_signal(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    atr: float,
) -> tuple[float, int] | None:
    """Goldene-Zone-Retracement des signifikanten 100-Kerzen-Swings."""
    n = len(close)
    last = float(close[n - 1])
    start = max(0, n - 100)
    seg_l = low[start:]
    seg_h = high[start:]
    li = start + int(np.argmin(seg_l))
    hi = start + int(np.argmax(seg_h))
    s_low = float(np.min(seg_l))
    s_high = float(np.max(seg_h))
    swing = s_high - s_low
    if swing < 2.0 * atr:
        return None
    if li < hi:  # Aufwärtsswing: Pullback-Retracement
        r = (s_high - last) / swing
        if 0.618 <= r <= 0.68:
            return (0.7, hi)
        if 0.382 <= r <= 0.5:
            return (0.3, hi)
    elif hi < li:  # Abwärtsswing: bärischer Bounce
        r = (last - s_low) / swing
        if 0.618 <= r <= 0.68:
            return (-0.7, li)
        if 0.382 <= r <= 0.5:
            return (-0.3, li)
    return None


def _zigzag(seg_h: NDArray[np.float64], seg_l: NDArray[np.float64], thr: float) -> list[tuple[int, float]]:
    """Alternierende Pivots (Index, Preis) mit ``thr``-Umkehrfilter."""
    m = len(seg_l)
    if m < 4:
        return []
    hi, lo = float(seg_h[0]), float(seg_l[0])
    hi_i = lo_i = 0
    for i in range(1, m):
        if float(seg_h[i]) > hi:
            hi, hi_i = float(seg_h[i]), i
        if float(seg_l[i]) < lo:
            lo, lo_i = float(seg_l[i]), i
        if hi - lo >= thr:
            break
    else:
        return []
    pivots: list[tuple[int, float]] = []
    if hi_i < lo_i:  # Hoch zuerst, erstes Leg abwärts
        pivots.append((hi_i, hi))
        direction = -1
        lo, lo_i = float(seg_l[hi_i]), hi_i
        i0 = hi_i
    else:  # Tief zuerst, erstes Leg aufwärts
        pivots.append((lo_i, lo))
        direction = 1
        hi, hi_i = float(seg_h[lo_i]), lo_i
        i0 = lo_i
    for i in range(i0 + 1, m):
        if direction == 1:
            if float(seg_h[i]) > hi:
                hi, hi_i = float(seg_h[i]), i
            if float(seg_l[i]) <= hi - thr:
                pivots.append((hi_i, hi))
                direction = -1
                lo, lo_i = float(seg_l[i]), i
        else:
            if float(seg_l[i]) < lo:
                lo, lo_i = float(seg_l[i]), i
            if float(seg_h[i]) >= lo + thr:
                pivots.append((lo_i, lo))
                direction = 1
                hi, hi_i = float(seg_h[i]), i
    return pivots


def _elliott_signal(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    atr: float,
    n: int,
) -> tuple[float, int] | None:
    """Vereinfachter 5-Swing-Impuls + Korrektur; die schwächste Familie."""
    start = max(0, n - 60)
    m = n - start
    piv = _zigzag(high[start:], low[start:], 1.5 * atr)
    swings: list[tuple[float, float, float, float, int]] = []
    for k in range(len(piv) - 1):
        _, p0 = piv[k]
        i1, p1 = piv[k + 1]
        d = 1.0 if p1 > p0 else -1.0
        swings.append((d, abs(p1 - p0), p0, p1, i1))
    if len(swings) < 5:
        return None
    recent = m - 10  # "gerade abgeschlossen": Pivot endet innerhalb 10 Kerzen
    cands: list[tuple[float, int]] = []
    five = swings[-5:]
    s1, l1, p1s, _, _ = five[0]
    s3, l3, _, _, _ = five[2]
    s5, l5, _, _, i5 = five[4]
    if (
        s5 == s3 == s1
        and l1 >= 1.5 * atr
        and l3 >= 1.5 * atr
        and l5 >= 1.5 * atr
        and l3 > l1
        and l3 > l5
        and i5 >= recent
    ):
        cands.append((-0.35 * s1, i5 + start))  # Impuls fertig, Korrektur erwartet
    if len(swings) >= 6:
        six = swings[-6:]
        s1b, l1b, p1s, _, _ = six[0]
        s3b, l3b, _, _, _ = six[2]
        s5b, l5b, _, e5b, _ = six[4]
        s6, _, _, e6, i6 = six[5]
        if (
            s5b == s3b == s1b
            and l1b >= 1.5 * atr
            and l3b >= 1.5 * atr
            and l5b >= 1.5 * atr
            and l3b > l1b
            and l3b > l5b
            and s6 == -s1b
            and i6 >= recent
        ):
            imp_range = abs(e5b - p1s)
            if imp_range > 0.0:
                back = (e5b - e6) if s1b > 0.0 else (e6 - e5b)
                r = back / imp_range
                if 0.382 <= r <= 0.618:
                    cands.append((0.5 * s1b, i6 + start))  # 618-Retracement, Fortsetzung
    if not cands:
        return None
    s, rec = max(cands, key=lambda t: (abs(t[0]), t[1]))
    return (float(s), rec)


def _combine(fired: list[tuple[float, float]]) -> tuple[float, float, float]:
    """Gewichteter Mittelwert der feuenden Familien → (p_up, p_down, p_range).

    Summe exakt 1.0 (nach Clipping auf [0.02, 0.95] renormiert); ohne
    feuende Familie bleibt die neutrale Verteilung (Range-Prior 0.40).
    """
    if not fired:
        return (0.35, 0.35, 0.40)
    cov = len(fired) / 4.0
    score = sum(w * s for w, s in fired) / sum(w for w, _ in fired)
    conviction = score * (0.5 + 0.5 * cov)
    p_range = 0.40 + 0.10 * (1.0 - cov)
    base = (1.0 - p_range) / 2.0
    p_up = base + conviction * base
    p_down = base - conviction * base

    def _clip(v: float) -> float:
        return min(0.95, max(0.02, v))

    p_up, p_down, p_range = _clip(p_up), _clip(p_down), _clip(p_range)
    total = p_up + p_down + p_range
    return (float(p_up / total), float(p_down / total), float(p_range / total))


@dataclass(frozen=True, slots=True)
class ChartPatternState:
    """Momentaufnahme der vier Evidenz-Familien (nur aktuelle/vergangene Kerzen)."""

    atr: float  # ATR14, mit Flat-Window-Guard
    candle: tuple[float, int] | None  # (Signal s, recency) oder None
    chart: tuple[float, int] | None
    fib: tuple[float, int] | None
    elliott: tuple[float, int] | None
    fired: tuple[tuple[float, float], ...]  # (Gewicht, s) der feuenden Familien
    score: float  # gewichteter Mittelwert der feuenden Familien, [-1, 1]
    probabilities: tuple[float, float, float]  # (p_up, p_down, p_range), Summe 1.0


class ChartPatternAgent(BaseAgent):
    """Chart-Pattern-Agent — vier Evidenz-Familien, ATR-relevante Schwellen, kein Lookahead."""

    MIN_BARS: int = 31

    def __init__(self, config: AgentConfig | None = None, params: None = None) -> None:
        if config is None:
            config = AgentConfig(
                agent_id="chart_pattern",
                agent_type=AgentType.PATTERN,
            )
        super().__init__(config)
        # params: nur der Ensemble-Signatur-Verträglichkeit; der Agent läuft
        # mit den preregistrierten Konstanten (kein PARAM_CLASSES-Eintrag).
        del params

    def analyze(self, data: dict[str, NDArray[np.float64]]) -> AgentReport:
        """Analysiert OHLCV-Daten auf auslösbare Chartstrukturen.

        Accepts a dict with OHLCV arrays (aufsteigend, älteste → neueste,
        Index -1 = aktuelle Kerze):
            open, high, low, close: NDArray[float64] (volume/timestamps optional, ignoriert)

        Returns:
            AgentReport mit up/down/range-Wahrscheinlichkeiten für die
            nächsten ~15 Minuten; Range trägt den Basis-Prior (0.40).

        Raises:
            ValueError: Wenn erforderliche Schlüssel fehlen.
        """
        missing = REQUIRED_KEYS - set(data.keys())
        if missing:
            raise ValueError(f"Missing required OHLCV keys: {sorted(missing)}")

        open_ = np.asarray(data["open"], dtype=np.float64)
        high = np.asarray(data["high"], dtype=np.float64)
        low = np.asarray(data["low"], dtype=np.float64)
        close = np.asarray(data["close"], dtype=np.float64)

        if len(close) < self.MIN_BARS:
            return self._short_data_report(len(close))

        state = self._compute_state(open_, high, low, close)
        probabilities, confidence = self._calibrate(state)

        return AgentReport(
            report_id=self._generate_report_id(),
            run_id=uuid.uuid4().hex,
            agent_id=self.agent_id,
            agent_version=self.config.agent_version,
            instrument=self.config.instrument,
            horizon=self.config.horizon,
            as_of=datetime.datetime.now(datetime.UTC),
            hypothesis=self._build_hypothesis(state),
            probabilities=probabilities,
            evidence=self._build_evidence(state),
            counter_evidence=self._build_counter_evidence(state),
            invalidations=self._build_invalidations(state),
            raw_confidence=confidence,
            status=self.config.status,
            expected_return=None,
            calibrated_confidence=0.0,
        )

    # ── Evidenz-Familien ────────────────────────────────────────────────────

    def _compute_state(
        self,
        open_: NDArray[np.float64],
        high: NDArray[np.float64],
        low: NDArray[np.float64],
        close: NDArray[np.float64],
    ) -> ChartPatternState:
        """Berechnet alle vier Familien-Signale und die Kombination."""
        atr = _atr(high, low, close)
        signals = (
            _candle_signal(open_, high, low, close, atr),
            _chart_signal(high, low, close, atr),
            _fib_signal(high, low, close, atr),
            _elliott_signal(high, low, atr, len(close)),
        )
        fired: list[tuple[float, float]] = []
        for weight, signal in zip(WEIGHTS, signals, strict=True):
            if signal is not None:
                fired.append((weight, signal[0]))
        score = sum(w * s for w, s in fired) / sum(w for w, _ in fired) if fired else 0.0
        return ChartPatternState(
            atr=atr,
            candle=signals[0],
            chart=signals[1],
            fib=signals[2],
            elliott=signals[3],
            fired=tuple(fired),
            score=score,
            probabilities=_combine(fired),
        )

    # ── Kalibrierung ────────────────────────────────────────────────────────

    def _calibrate(self, state: ChartPatternState) -> tuple[dict[str, float], float]:
        """up/down/range-Verteilung (Summe exakt 1.0) plus rohe Konfidenz.

        Der Roh-Triple wird renormiert (der neutrale Fall ohne feuende
        Familie ist (0.35, 0.35, 0.40) — Summe 1.10, wie im Sandbox-
        Kandidaten; downstream wird er dort ebenfalls renormiert).
        """
        p_up, p_down, p_range = state.probabilities
        total = p_up + p_down + p_range
        if total <= 0.0:  # Defensive: _combine liefert immer positive Summen
            return (
                {"up": 0.34, "down": 0.33, "range": 0.33},
                0.08,
            )
        p_up = round(p_up / total, 4)
        p_down = round(p_down / total, 4)
        probabilities = {
            "up": p_up,
            "down": p_down,
            "range": round(1.0 - p_up - p_down, 4),
        }
        return probabilities, self._raw_confidence(state)

    def _raw_confidence(self, state: ChartPatternState) -> float:
        """Roh-Konfidenz in [0.1, 0.9] — TrendAgent-Kalibrierung auf dieser Skala.

        Richtungs-Sieger auf dieser Agentskala liegen in [~0.30, 0.60]
        (0.35 → 0.85 Konfidenz); Range-dominierte Verteilungen in
        [0.40, 0.50] (0.10 → 0.30). Ohne feuende Familie: 0.10.
        """
        p_up, p_down, p_range = state.probabilities
        if not state.fired:
            return 0.10
        p_dom = max(p_up, p_down)
        if p_dom >= p_range:
            raw = 0.35 + 0.50 * min(1.0, max(0.0, (p_dom - 0.30) / 0.30))
        else:
            raw = 0.10 + 0.20 * min(1.0, max(0.0, (p_range - 0.40) / 0.10))
        return float(round(min(0.9, max(0.1, raw)), 4))

    # ── Berichtskomponenten ─────────────────────────────────────────────────

    def _build_hypothesis(self, state: ChartPatternState) -> str:
        """Einzeilige Zusammenfassung des Chartstruktur-Zustands."""
        p_up, p_down, p_range = state.probabilities
        if not state.fired:
            return (
                f"No chart pattern, Fibonacci or Elliott signal fired (0/4 "
                f"families); probabilities stay range-dominated "
                f"(ATR14 {state.atr:.4f})"
            )
        dominant = "up" if state.score > 0.0 else "down" if state.score < 0.0 else "range"
        return (
            f"Chart-pattern scan: {len(state.fired)}/4 evidence families fired, "
            f"weighted score {state.score:+.2f} → {dominant} lean "
            f"(up {p_up:.2f} / down {p_down:.2f} / range {p_range:.2f}, "
            f"ATR14 {state.atr:.4f})"
        )

    def _build_evidence(self, state: ChartPatternState) -> list[EvidenceReference]:
        """Evidenz — ATR-Basis plus je feuende Familie ein Signal-Referenz."""
        signals = (state.candle, state.chart, state.fib, state.elliott)
        evidence: list[EvidenceReference] = [
            self._make_evidence(
                "atr14",
                f"ATR14 {state.atr:.4f}",
                "neutral",
                0.3,
            )
        ]
        for label, weight, signal in zip(FAMILY_LABELS, WEIGHTS, signals, strict=True):
            if signal is None:
                continue
            s, _recency = signal
            evidence.append(
                self._make_evidence(
                    f"{label}_signal",
                    f"{label} pattern signal {s:+.2f} (weight {weight:.2f})",
                    _direction_label(s),
                    min(1.0, abs(s)),
                )
            )
        return evidence

    def _build_counter_evidence(self, state: ChartPatternState) -> list[EvidenceReference]:
        """Gegenhypothesen: Chartmuster können ausbleiben oder fehlschlagen."""
        if state.fired:
            value = (
                f"pattern may not follow through: {len(state.fired)}/4 families fired, "
                f"net score {state.score:+.2f} — breakout/breakdown often fails"
            )
        else:
            value = "no pattern structure fired — directional claim impossible by design"
        return [
            self._make_evidence(
                "counter_pattern_fail",
                value,
                "negative",
                0.3 if state.fired else 0.2,
            )
        ]

    def _build_invalidations(self, state: ChartPatternState) -> list[InvalidationCondition]:
        """Invalidierungsbedingungen für die Chartstruktur-Hypothese."""
        invalidations: list[InvalidationCondition] = [
            self._make_invalidations(
                condition="volatility regime shift — ATR14 more than doubles",
                indicator="atr14",
                threshold=round(2.0 * state.atr, 4),
                direction="above",
            ),
            self._make_invalidations(
                condition="sample size too small for stable ATR/pattern estimates",
                indicator="sample_size",
                threshold=float(self.MIN_BARS),
                direction="below",
            ),
        ]
        if state.score > 0.0:
            invalidations.append(
                self._make_invalidations(
                    condition="bullish lean invalidated: net weighted score flips below zero",
                    indicator="score",
                    threshold=0.0,
                    direction="below",
                )
            )
        elif state.score < 0.0:
            invalidations.append(
                self._make_invalidations(
                    condition="bearish lean invalidated: net weighted score flips above zero",
                    indicator="score",
                    threshold=0.0,
                    direction="above",
                )
            )
        else:
            invalidations.append(
                self._make_invalidations(
                    condition="range lean invalidated: a directional pattern fires (|score| > 0)",
                    indicator="score",
                    threshold=0.0,
                    direction="above",
                )
            )
        return invalidations

    _SHORT_DATA_COUNTERS: ClassVar[tuple[str, str]] = (
        "counter_insufficient",
        "no pattern claim possible with insufficient data — by design",
    )

    def _short_data_report(self, n: int) -> AgentReport:
        """Bericht bei unzureichenden Daten (weniger als MIN_BARS Kerzen)."""
        hypothesis = (
            f"Insufficient data for chart-pattern analysis: {n} bars available, "
            f"minimum {self.MIN_BARS} required. Probabilities stay neutral."
        )
        probabilities = {"up": 0.34, "down": 0.33, "range": 0.33}
        counter_feature, counter_value = self._SHORT_DATA_COUNTERS
        return AgentReport(
            report_id=self._generate_report_id(),
            run_id=uuid.uuid4().hex,
            agent_id=self.agent_id,
            agent_version=self.config.agent_version,
            instrument=self.config.instrument,
            horizon=self.config.horizon,
            as_of=datetime.datetime.now(datetime.UTC),
            hypothesis=hypothesis,
            probabilities=probabilities,
            evidence=[
                self._make_evidence(
                    "insufficient_data",
                    f"only {n} bars available, need at least {self.MIN_BARS}",
                    "neutral",
                    0.0,
                )
            ],
            counter_evidence=[
                self._make_evidence(counter_feature, counter_value, "negative", 0.0)
            ],
            invalidations=[
                self._make_invalidations(
                    condition="sample size too small for stable ATR/pattern estimates",
                    indicator="sample_size",
                    threshold=float(self.MIN_BARS),
                    direction="below",
                ),
                self._make_invalidations(
                    condition="data quality below threshold (gaps, out-of-range values)",
                    indicator="data_quality",
                    threshold=0.9,
                    direction="below",
                ),
            ],
            raw_confidence=0.08,
            status=self.config.status,
            expected_return=None,
            calibrated_confidence=0.0,
        )
