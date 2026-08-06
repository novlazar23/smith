"""Mustererkennung — BOS, CHoCH, Failed Breakout."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .base import ChartPattern, ChartStructureResult
from .swing import SwingDetector


class PatternDetector:
    """Erkennt Chart-Muster: BOS, CHoCH, Failed Breakout."""

    def __init__(
        self,
        confirmation: float = 0.01,
        reclaim_bars: int = 3,
    ) -> None:
        """Initialisiert den Muster-Detektor.

        Args:
            confirmation: Mindest-%-Breakout für Bestätigung.
            reclaim_bars: Anzahl der Bars zum Prüfen von Reclaims.
        """
        if confirmation <= 0:
            raise ValueError(f"confirmation muss > 0 sein, erhalten: {confirmation}")
        if reclaim_bars < 1:
            raise ValueError(f"reclaim_bars muss >= 1 sein, erhalten: {reclaim_bars}")
        self.confirmation = confirmation
        self.reclaim_bars = reclaim_bars

    def detect_bos(
        self,
        data: dict[str, NDArray[np.float64]],
    ) -> list[ChartPattern]:
        """Erkennt Break of Structure (BOS).

        Bullish BOS: Preis schließt über einem früheren Swing High um >= confirmation%.
        Bearish BOS: Preis schließt unter einem früheren Swing Low um >= confirmation%.

        Args:
            data: Dict mit 'high', 'low', 'close'.

        Returns:
            Liste von ChartPattern BOS-Einträgen.
        """
        swing_detector = SwingDetector(lookback=3)
        pivots = swing_detector.detect_swings(data)

        if len(pivots) < 2:
            return []

        close = data["close"]
        n = len(close)
        bos_patterns: list[ChartPattern] = []
        seen_bos = False

        for pivot in pivots:
            if pivot.time >= n - 1:
                continue

            if pivot.direction == "high":
                # Bullish BOS: check if later bars close above this swing high
                for bar_idx in range(pivot.time + 1, min(pivot.time + 20, n)):
                    if close[bar_idx] > pivot.price * (1 + self.confirmation):
                        if not seen_bos:
                            bos_patterns.append(ChartPattern.BOS)
                            seen_bos = True
                        break

            elif pivot.direction == "low":
                # Bearish BOS: check if later bars close below this swing low
                for bar_idx in range(pivot.time + 1, min(pivot.time + 20, n)):
                    if close[bar_idx] < pivot.price * (1 - self.confirmation):
                        if not seen_bos:
                            bos_patterns.append(ChartPattern.BOS)
                            seen_bos = True
                        break

        return bos_patterns

    def detect_choch(
        self,
        data: dict[str, NDArray[np.float64]],
    ) -> ChartPattern | None:
        """Erkennt Change of Character (CHoCH).

        Nach mindestens 2 Swing Lows in Folge (Abwärtstrend),
        wenn der Preis über dem letzten Swing High schließt → CHoCH.

        Args:
            data: Dict mit 'high', 'low', 'close'.

        Returns:
            ChartPattern.CHoCH oder None.
        """
        swing_detector = SwingDetector(lookback=3)
        pivots = swing_detector.detect_swings(data)

        # Need at least 2 swing lows for a downtrend
        swing_lows = [p for p in pivots if p.direction == "low"]
        swing_highs = [p for p in pivots if p.direction == "high"]

        if len(swing_lows) < 2:
            return None

        # Check for CHoCH: price closes above the most recent swing high
        # after having made at least 2 swing lows (downtrend)
        last_swing_high = swing_highs[-1] if swing_highs else None
        if last_swing_high is None:
            return None

        close = data["close"]
        n = len(close)

        # Check if any bar after the last swing high closes above it
        for i in range(last_swing_high.time + 1, n):
            if close[i] > last_swing_high.price * (1 + self.confirmation):
                return ChartPattern.CHoCH

        return None

    def detect_failed_breakout(
        self,
        data: dict[str, NDArray[np.float64]],
    ) -> list[ChartPattern]:
        """Erkennt Failed Breakouts.

        Wenn der Preis ein Swing-Pivot durchbricht, aber innerhalb von
        reclaim_bars Bars wieder auf die ursprüngliche Seite zurückkehrt.

        Args:
            data: Dict mit 'high', 'low', 'close'.

        Returns:
            Liste von ChartPattern FAILED_BREAKOUT-Einträgen.
        """
        swing_detector = SwingDetector(lookback=3)
        pivots = swing_detector.detect_swings(data)

        if len(pivots) == 0:
            return []

        close = data["close"]
        n = len(close)
        failed: list[ChartPattern] = []

        for pivot in pivots:
            if pivot.time >= n - 1:
                continue

            pivot_price = pivot.price

            if pivot.direction == "high":
                # Swing High: price broke above, but closed back below within reclaim_bars
                for i in range(pivot.time + 1, min(pivot.time + self.reclaim_bars + 1, n)):
                    if close[i] > pivot_price:
                        # Look ahead for reclaim
                        for j in range(i + 1, min(i + self.reclaim_bars, n)):
                            if close[j] < pivot_price:
                                failed.append(ChartPattern.FAILED_BREAKOUT)
                                break
                        if failed:
                            break
            elif pivot.direction == "low":
                # Swing Low: price broke below, but closed back above within reclaim_bars
                for i in range(pivot.time + 1, min(pivot.time + self.reclaim_bars + 1, n)):
                    if close[i] < pivot_price:
                        # Look ahead for reclaim
                        for j in range(i + 1, min(i + self.reclaim_bars, n)):
                            if close[j] > pivot_price:
                                failed.append(ChartPattern.FAILED_BREAKOUT)
                                break
                        if failed:
                            break

        return list(dict.fromkeys(failed))

    def detect_all_patterns(
        self,
        data: dict[str, NDArray[np.float64]],
    ) -> ChartStructureResult:
        """Führt alle Musterdetektoren aus und kombiniert das Ergebnis.

        Args:
            data: Dict mit 'high', 'low', 'close'.

        Returns:
            ChartStructureResult mit allen erkannten Mustern und Pivots.
        """
        swing_detector = SwingDetector(lookback=3)
        pivots = swing_detector.detect_swings(data)

        bos = self.detect_bos(data)
        choch = self.detect_choch(data)
        failed = self.detect_failed_breakout(data)

        all_patterns: list[ChartPattern] = []
        all_patterns.extend(bos)
        if choch is not None:
            all_patterns.append(choch)
        all_patterns.extend(failed)

        return ChartStructureResult(
            patterns=all_patterns,
            pivots=pivots,
            metadata={
                "bos_count": len(bos),
                "choch_detected": choch is not None,
                "failed_breakout_count": len(failed),
            },
        )
