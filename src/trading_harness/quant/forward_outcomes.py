"""Forward Outcome Statistics (Phase 6).

Berechnet was nach einem bestimmten Markt-Muster tatsächlich passiert ist:
Forward Returns, Hit Rate, Profit Factor, Expectancy über konfigurierbare Horizonte.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ForwardOutcome:
    """Ergebnis für einen einzelnen Forward-Return-Horizont."""
    horizon: int  # Anzahl Kerzen
    mean_return: float  # durchschnittlicher Forward Return
    median_return: float  # median Forward Return
    hit_rate: float  # Anteil positiver Returns (0.0–1.0)
    profit_factor: float  # sum(gains) / sum(losses), 0 wenn keine Verluste
    expectancy: float  # erwarteter Return pro Trade
    std_return: float  # Standardabweichung der Forward Returns
    sample_size: int  # Anzahl berechneter Forward Returns
    max_gain: float  # bester Forward Return
    max_loss: float  # schlechtester Forward Return


@dataclass
class ForwardOutcomeResult:
    """Gesamtergebnis der Forward-Outcome-Berechnung."""
    symbol: str
    timeframe: str
    pattern_length: int
    outcomes: dict[int, ForwardOutcome]  # horizon → outcome


class ForwardOutcomeEngine:
    """Berechnet Forward Outcome Statistics — nur stdlib."""

    def __init__(self, horizons: list[int] | None = None) -> None:
        self.horizons = horizons or [5, 10, 20, 50]

    def compute(
        self,
        candles: list[dict],
        pattern_length: int,
        symbol: str = "",
        timeframe: str = "1m",
    ) -> ForwardOutcomeResult:
        """Berechnet Forward Returns ab jedem Punkt nach pattern_length Kerzen.

        Args:
            candles: OHLCV-Kerzenreihe
            pattern_length: Länge des Referenz-Musters (Anzahl Kerzen)
            symbol: Symbol-Name (für Metadata)
            timeframe: Timeframe (für Metadata)

        Returns:
            ForwardOutcomeResult mit Forward-Statistics für jeden Horizont
        """
        closes = [c["close"] for c in candles]
        outcomes: dict[int, ForwardOutcome] = {}

        max_horizon = max(self.horizons) if self.horizons else 50
        # Wir brauchen genug Daten: pattern_length + max_horizon
        if len(closes) < pattern_length + 2:
            for h in self.horizons:
                outcomes[h] = ForwardOutcome(
                    horizon=h, mean_return=0.0, median_return=0.0,
                    hit_rate=0.0, profit_factor=0.0, expectancy=0.0,
                    std_return=0.0, sample_size=0, max_gain=0.0, max_loss=0.0,
                )
            return ForwardOutcomeResult(
                symbol=symbol, timeframe=timeframe,
                pattern_length=pattern_length, outcomes=outcomes,
            )

        # Forward Returns: für jeden Startpunkt i (0 bis len-pattern_length-max_horizon)
        # berechnen wir den Return nach horizon Kerzen
        max_start = len(closes) - max_horizon

        for horizon in self.horizons:
            forward_returns: list[float] = []
            for i in range(min(max_start, len(closes) - horizon)):
                entry_price = closes[i]
                exit_price = closes[i + horizon]
                if entry_price > 0:
                    ret = (exit_price - entry_price) / entry_price
                    forward_returns.append(ret)

            if not forward_returns:
                outcomes[horizon] = ForwardOutcome(
                    horizon=horizon, mean_return=0.0, median_return=0.0,
                    hit_rate=0.0, profit_factor=0.0, expectancy=0.0,
                    std_return=0.0, sample_size=0, max_gain=0.0, max_loss=0.0,
                )
                continue

            n = len(forward_returns)
            mean_ret = sum(forward_returns) / n
            sorted_ret = sorted(forward_returns)
            median_ret = sorted_ret[n // 2] if n % 2 == 1 else (sorted_ret[n // 2 - 1] + sorted_ret[n // 2]) / 2

            positives = [r for r in forward_returns if r > 0]
            negatives = [r for r in forward_returns if r < 0]
            hit_rate = len(positives) / n if n > 0 else 0.0

            sum_gains = sum(positives) if positives else 0.0
            sum_losses = abs(sum(negatives)) if negatives else 0.0
            profit_factor = sum_gains / sum_losses if sum_losses > 0 else (float("inf") if sum_gains > 0 else 0.0)
            # Cap profit_factor at reasonable value
            if profit_factor == float("inf"):
                profit_factor = 10.0

            expectancy = mean_ret
            variance = sum((r - mean_ret) ** 2 for r in forward_returns) / n
            std_ret = math.sqrt(variance)

            outcomes[horizon] = ForwardOutcome(
                horizon=horizon,
                mean_return=mean_ret,
                median_return=median_ret,
                hit_rate=hit_rate,
                profit_factor=profit_factor,
                expectancy=expectancy,
                std_return=std_ret,
                sample_size=n,
                max_gain=max(forward_returns),
                max_loss=min(forward_returns),
            )

        return ForwardOutcomeResult(
            symbol=symbol, timeframe=timeframe,
            pattern_length=pattern_length, outcomes=outcomes,
        )

    def compute_for_pattern(
        self,
        pattern: list[dict],
        history: list[dict],
        symbol: str = "",
        timeframe: str = "1m",
    ) -> ForwardOutcomeResult:
        """Berechnet Forward Returns für alle Vorkommen des Patterns in der Historie."""
        pattern_closes = [c["close"] for c in pattern]
        history_closes = [c["close"] for c in history]

        if len(pattern_closes) < 2 or len(history_closes) < len(pattern_closes) + 2:
            return ForwardOutcomeResult(
                symbol=symbol, timeframe=timeframe,
                pattern_length=len(pattern_closes), outcomes={},
            )

        # Find pattern occurrences and compute forward returns from each
        pattern_len = len(pattern_closes)
        max_horizon = max(self.horizons) if self.horizons else 50
        all_forward_returns: dict[int, list[float]] = {h: [] for h in self.horizons}

        for i in range(len(history_closes) - pattern_len - max_horizon):
            window = history_closes[i : i + pattern_len]
            if len(window) == pattern_len:
                entry = window[-1]
                for horizon in self.horizons:
                    exit_idx = i + pattern_len + horizon - 1
                    if exit_idx < len(history_closes):
                        exit_price = history_closes[exit_idx]
                        if entry > 0:
                            ret = (exit_price - entry) / entry
                            all_forward_returns[horizon].append(ret)

        outcomes: dict[int, ForwardOutcome] = {}
        for horizon in self.horizons:
            fwd = all_forward_returns[horizon]
            if not fwd:
                outcomes[horizon] = ForwardOutcome(
                    horizon=horizon, mean_return=0.0, median_return=0.0,
                    hit_rate=0.0, profit_factor=0.0, expectancy=0.0,
                    std_return=0.0, sample_size=0, max_gain=0.0, max_loss=0.0,
                )
                continue

            n = len(fwd)
            mean_ret = sum(fwd) / n
            sorted_f = sorted(fwd)
            median_ret = sorted_f[n // 2] if n % 2 == 1 else (sorted_f[n // 2 - 1] + sorted_f[n // 2]) / 2
            positives = [r for r in fwd if r > 0]
            negatives = [r for r in fwd if r < 0]
            hit_rate = len(positives) / n
            sum_gains = sum(positives) if positives else 0.0
            sum_losses = abs(sum(negatives)) if negatives else 0.0
            pf = min(sum_gains / sum_losses, 10.0) if sum_losses > 0 else (10.0 if sum_gains > 0 else 0.0)
            variance = sum((r - mean_ret) ** 2 for r in fwd) / n

            outcomes[horizon] = ForwardOutcome(
                horizon=horizon, mean_return=mean_ret, median_return=median_ret,
                hit_rate=hit_rate, profit_factor=pf, expectancy=mean_ret,
                std_return=math.sqrt(variance), sample_size=n,
                max_gain=max(fwd), max_loss=min(fwd),
            )

        return ForwardOutcomeResult(
            symbol=symbol, timeframe=timeframe,
            pattern_length=pattern_len, outcomes=outcomes,
        )
