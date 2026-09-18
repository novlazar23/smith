"""Hand-written stage-2 candidate for the evolved-agent gates.

Crypto chart-pattern + Fibonacci + simplified-Elliott signal agent for
5m windows. NOT a ``BaseAgent``: sandbox contract per
``apps/champion_evals/agent_sandbox.py`` (single ``predict`` function,
numpy-only imports, no module-level state, no forbidden calls).

Deterministic and price/volume-driven (``timestamps`` unused, no
randomness, no lookahead). Evidence families and weights: candle 0.25,
chart 0.35, fib 0.35, elliott 0.20; each family contributes at most one
signal in [-1, +1] (largest |s| wins, ties resolved by recency). All
thresholds are ATR-relative, never absolute prices. Output:
(p_up, p_down, p_range) for the next ~15 minutes; base range prior
0.40, up to 0.50 when no family fires.
"""

import numpy as np


def predict(
    open: np.ndarray,  # noqa: A002  (signature is fixed by the sandbox contract)
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    timestamps: np.ndarray,
) -> tuple[float, float, float]:
    """3-class outcome (UP / DOWN / RANGE) of the next ~15 minutes.

    Production target: UP if realized return > +1%, DOWN if < -1%,
    else RANGE — so RANGE is the most likely class and carries the
    base prior.
    """
    n = len(close)
    if n < 31:  # warmup: too little structure for any family
        return (0.34, 0.33, 0.33)

    last = float(close[n - 1])

    def _atr() -> float:
        # 14-bar true-range mean; guarded against flat windows
        pc = close[:-1]
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(np.abs(high[1:] - pc), np.abs(low[1:] - pc)),
        )
        return max(float(np.mean(tr[-14:])), 1e-9)

    atr = _atr()

    def _extrema(win: int, kind: str) -> list[int]:
        # indices of bars whose high/low is the max/min over a +/-2-bar
        # neighborhood, inside the last `win` bars
        start = max(0, n - win)
        out: list[int] = []
        for i in range(start, n):
            a = max(start, i - 2)
            b = min(n, i + 3)
            if (kind == "max" and high[i] >= np.max(high[a:b])) or (
                kind == "min" and low[i] <= np.min(low[a:b])
            ):
                out.append(i)
        return out

    def _candle_signal() -> tuple[float, int] | None:
        # candlestick patterns on the last 1-3 bars; signal s in [-1, 1]
        b = n - 1
        o1, c1 = float(open[b]), last
        h1, l1 = float(high[b]), float(low[b])
        body = abs(c1 - o1)
        rng = h1 - l1
        upper = h1 - max(o1, c1)
        lower = min(o1, c1) - l1
        drift5 = c1 - float(close[b - 5])
        cands: list[tuple[float, int]] = []
        if rng >= 0.25 * atr:
            if lower >= 2.0 * body and upper <= body and drift5 < -0.25 * atr:
                cands.append((0.7, b))  # hammer after down-drift
            if upper >= 2.0 * body and lower <= body and drift5 > 0.25 * atr:
                cands.append((-0.7, b))  # shooting star after up-drift
        o0, c0 = float(open[b - 1]), float(close[b - 1])
        body0 = abs(c0 - o0)
        if body > 0.3 * rng and body0 > 0.0:
            if c1 > o1 and c0 < o0 and o1 <= c0 and c1 >= o0:
                cands.append((0.8, b))  # bullish engulfing
            if c1 < o1 and c0 > o0 and o1 >= c0 and c1 <= o0:
                cands.append((-0.8, b))  # bearish engulfing
        if b >= 2:
            idxs = (b - 2, b - 1, b)
            greens = all(float(close[i]) > float(open[i]) for i in idxs)
            reds = all(float(close[i]) < float(open[i]) for i in idxs)
            opens_ok = all(
                min(float(open[i - 1]), float(close[i - 1])) <= float(open[i])
                <= max(float(open[i - 1]), float(close[i - 1]))
                for i in idxs[1:]
            )
            move = c1 - float(open[b - 2])
            if greens and opens_ok and move > 2.0 * atr:
                cands.append((0.9, b))  # three white soldiers
            if reds and opens_ok and move < -2.0 * atr:
                cands.append((-0.9, b))  # three black crows
        if not cands:
            return None
        s, rec = max(cands, key=lambda t: (abs(t[0]), t[1]))
        return (float(s), rec)

    def _chart_signal() -> tuple[float, int] | None:
        # chart patterns over the last ~40 bars; signal s in [-1, 1]
        tol = max(1.0 * atr, 0.005 * abs(last))
        maxima = _extrema(40, "max")
        minima = _extrema(40, "min")
        cands: list[tuple[float, int]] = []
        if len(maxima) >= 2:
            t1, t2 = maxima[-2], maxima[-1]
            if t2 > t1 + 1 and abs(float(high[t1]) - float(high[t2])) <= tol:
                trough = float(np.min(low[t1 + 1 : t2]))
                tops = min(float(high[t1]), float(high[t2]))
                if tops - trough >= 1.0 * atr:
                    s = -1.0 if last < trough else -0.6
                    cands.append((s, t2))  # double top: break / forming
        if len(minima) >= 2:
            u1, u2 = minima[-2], minima[-1]
            if u2 > u1 + 1 and abs(float(low[u1]) - float(low[u2])) <= tol:
                peak = float(np.max(high[u1 + 1 : u2]))
                bottoms = max(float(low[u1]), float(low[u2]))
                if peak - bottoms >= 1.0 * atr:
                    s = 1.0 if last > peak else 0.6
                    cands.append((s, u2))  # double bottom: break / forming
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
                cands.append((s, i3))  # head & shoulders: break / forming
        if len(minima) >= 3 and len(maxima) >= 2:
            m1, m2, m3 = minima[-3], minima[-2], minima[-1]
            if float(low[m2]) >= float(low[m1]) + 0.25 * atr and float(
                low[m3]
            ) >= float(low[m2]) + 0.25 * atr:
                p1, p2 = maxima[-2], maxima[-1]
                if abs(float(high[p1]) - float(high[p2])) <= tol:
                    res = max(float(high[p1]), float(high[p2]))
                    s = 1.0 if last > res else 0.5
                    cands.append((s, m3))  # ascending triangle: breakout / forming
        if len(maxima) >= 3 and len(minima) >= 2:
            x1, x2, x3 = maxima[-3], maxima[-2], maxima[-1]
            if float(high[x2]) <= float(high[x1]) - 0.25 * atr and float(
                high[x3]
            ) <= float(high[x2]) - 0.25 * atr:
                q1, q2 = minima[-2], minima[-1]
                if abs(float(low[q1]) - float(low[q2])) <= tol:
                    sup = min(float(low[q1]), float(low[q2]))
                    s = -1.0 if last < sup else -0.5
                    cands.append((s, x3))  # descending triangle: breakdown / forming
        # flag continuation: 6-bar impulse + 6-bar tight retrace
        net = float(close[n - 7] - close[n - 12])
        if abs(net) >= 3.0 * atr:
            direction = 1.0 if net > 0.0 else -1.0
            band = float(np.max(high[n - 8 : n]) - np.min(low[n - 8 : n]))
            retrace = direction * (float(close[n - 7]) - last)
            if band <= 2.0 * atr and 0.0 <= retrace < 0.5 * abs(net):
                cands.append((0.8 * direction, n - 1))  # flag continuation
        if not cands:
            return None
        s, rec = max(cands, key=lambda t: (abs(t[0]), t[1]))
        return (float(s), rec)

    def _fib_signal() -> tuple[float, int] | None:
        # golden-pocket retracement of the significant 100-bar swing
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
        if li < hi:  # up-swing: pullback retracement
            r = (s_high - last) / swing
            if 0.618 <= r <= 0.68:
                return (0.7, hi)
            if 0.382 <= r <= 0.5:
                return (0.3, hi)
        elif hi < li:  # down-swing: bearish bounce
            r = (last - s_low) / swing
            if 0.618 <= r <= 0.68:
                return (-0.7, li)
            if 0.382 <= r <= 0.5:
                return (-0.3, li)
        return None

    def _zigzag(seg_h: np.ndarray, seg_l: np.ndarray, thr: float) -> list[tuple[int, float]]:
        # alternating pivots (index, price) with a `thr` reversal filter
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
        if hi_i < lo_i:  # high formed first, first leg down
            pivots.append((hi_i, hi))
            direction = -1
            lo, lo_i = float(seg_l[hi_i]), hi_i
            i0 = hi_i
        else:  # low formed first, first leg up
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

    def _elliott_signal() -> tuple[float, int] | None:
        # simplified 5-swing impulse + correction; the weakest family
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
        recent = m - 10  # "just completed": pivot must end within 10 bars
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
            cands.append((-0.35 * s1, i5 + start))  # impulse done, expect correction
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
                        cands.append((0.5 * s1b, i6 + start))  # 618-style retrace, continuation
        if not cands:
            return None
        s, rec = max(cands, key=lambda t: (abs(t[0]), t[1]))
        return (float(s), rec)

    # ── combination: weighted mean of the fired families ──
    weights = (0.25, 0.35, 0.35, 0.20)
    results = (
        _candle_signal(),
        _chart_signal(),
        _fib_signal(),
        _elliott_signal(),
    )
    fired: list[tuple[float, float]] = []
    for k in range(len(results)):
        if results[k] is not None:
            fired.append((weights[k], results[k][0]))
    if not fired:
        return (0.35, 0.35, 0.40)  # no evidence: neutral, range prior 0.40
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
