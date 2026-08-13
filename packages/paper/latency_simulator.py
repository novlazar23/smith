"""Latency simulator — configurable processing delay for trade execution."""

from __future__ import annotations

import random
import time


class LatencySimulator:
    """Simulates configurable latency between order submission and fill.

    Uses a seed-based RNG for deterministic behaviour across repeated runs.
    Delay range: 10 ms - 5 s (configurable).
    """

    MIN_DELAY_MS: float = 10.0
    MAX_DELAY_MS: float = 5000.0

    def __init__(
        self,
        min_delay_ms: float = 10.0,
        max_delay_ms: float = 5000.0,
        default_delay_ms: float = 100.0,
        seed: int | None = None,
    ) -> None:
        """Initialise the latency simulator.

        Args:
            min_delay_ms: Minimum latency in milliseconds (default 10 ms).
            max_delay_ms: Maximum latency in milliseconds (default 5000 ms = 5 s).
            default_delay_ms: Default delay applied when no range is specified
                              (default 100 ms).
            seed: Optional RNG seed for deterministic behaviour.
        """
        self.min_delay_ms = max(min_delay_ms, self.MIN_DELAY_MS)
        self.max_delay_ms = max(max_delay_ms, self.min_delay_ms)
        self.default_delay_ms = default_delay_ms
        self._rng = random.Random(seed)

        # Metrics
        self.total_simulated_time: float = 0.0
        self.actual_trades_processed: int = 0

    def simulate_latency(self, delay_ms: float | None = None) -> float:
        """Simulate a processing delay by sleeping for the configured duration.

        Args:
            delay_ms: Optional delay in milliseconds. Falls back to
                      ``default_delay_ms`` when *None*.

        Returns:
            The actual elapsed time in seconds.
        """
        delay = delay_ms if delay_ms is not None else self.default_delay_ms
        delay = max(self.min_delay_ms, min(delay, self.max_delay_ms))

        # Convert ms → s and sleep
        delay_s = delay / 1000.0
        time.sleep(delay_s)

        self.total_simulated_time += delay_s
        self.actual_trades_processed += 1

        return delay_s

    def simulate_latency_for_order(self, order_type: str) -> float:
        """Return a latency value tailored to the order type.

        Args:
            order_type: String representation of the order type
                        (``"MARKET"``, ``"LIMIT"``, ``"STOP"``).

        Returns:
            Delay in milliseconds.
        """
        # Market orders are fast; limit/stop may need queue processing.
        if order_type == "MARKET":
            return self.simulate_latency()

        # LIMIT / STOP get a slightly longer queue delay.
        return self.simulate_latency(self._rng.uniform(50.0, 500.0))
