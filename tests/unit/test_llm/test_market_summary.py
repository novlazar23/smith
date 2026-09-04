"""Tests für packages.llm.market_summary (Determinismus, Werte, Short-Window)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from packages.llm.market_summary import summarize_window


def _flat(n: int, price: float = 100.0, volume: float = 1000.0) -> dict[str, np.ndarray]:
    """Komplett flache OHLCV-Serie (open=high=low=close)."""
    flat = np.full(n, price, dtype=np.float64)
    return {
        "open": flat.copy(),
        "high": flat.copy(),
        "low": flat.copy(),
        "close": flat.copy(),
        "volume": np.full(n, volume, dtype=np.float64),
    }


def _varying(n: int, seed: int = 42) -> dict[str, np.ndarray]:
    """Deterministisch variierende Serie (fixer Seed, kein Zufall im Test)."""
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0.0, 1.0, n))
    close = np.maximum(close, 1.0)
    high = close + np.abs(rng.normal(0.5, 0.2, n))
    low = close - np.abs(rng.normal(0.5, 0.2, n))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.full(n, 100.0, dtype=np.float64) * 1.2,
    }


class TestDeterminism:
    """Byte-Identität bei identischen Eingaben."""

    def test_flat_series_is_deterministic(self) -> None:
        kwargs = _flat(80)
        assert summarize_window(last_n=60, **kwargs) == summarize_window(last_n=60, **kwargs)

    def test_varying_series_is_deterministic(self) -> None:
        kwargs = _varying(250)
        assert summarize_window(**kwargs) == summarize_window(**kwargs)

    def test_different_series_differ(self) -> None:
        assert summarize_window(**_flat(80)) != summarize_window(**_varying(80))


class TestValues:
    """Bekannte berechnete Werte in der Ausgabe."""

    def test_flat_series_has_zero_returns_and_zero_dists(self) -> None:
        lines = summarize_window(last_n=60, **_flat(80)).splitlines()

        assert "last_close=100" in lines
        assert "return_1b=0" in lines
        assert "return_5b=0" in lines
        assert "return_20b=0" in lines
        assert "dist_from_high_lastN_pct=0" in lines
        assert "dist_from_low_lastN_pct=0" in lines
        assert "volume_trend=1" in lines

    def test_linear_series_has_known_return(self) -> None:
        n = 80
        close = np.arange(100.0, 100.0 + n, dtype=np.float64)
        kwargs: dict[str, Any] = {
            "open": close.copy(),
            "high": close.copy(),
            "low": close.copy(),
            "close": close,
            "volume": np.full(n, 500.0, dtype=np.float64),
        }
        lines = summarize_window(last_n=60, **kwargs).splitlines()

        # 179/178 - 1 = 0,5618 % auf 4 signifikante Ziffern
        assert "return_1b=0.5618" in lines

    def test_header_and_recent_bars_section(self) -> None:
        lines = summarize_window(last_n=60, **_flat(80)).splitlines()

        assert lines[0] == "BTC/USDT-style OHLCV snapshot (last 60 bars, 1m) — bar 59 of 60"
        assert "# recent bars (close, volume, ret_1b)" in lines
        # 30 Zeilen für die letzten 30 Kerzen, neueste = bar -1
        bar_lines = [line for line in lines if line.startswith("bar -")]
        assert len(bar_lines) == 30
        assert bar_lines[0].startswith("bar -30:")
        assert bar_lines[-1].startswith("bar -1:")
        assert bar_lines[-1] == "bar -1: close=100 vol=1000 ret=0"

    def test_snapshot_uses_last_n_bars_only(self) -> None:
        kwargs = _flat(200)
        long_text = summarize_window(last_n=60, **kwargs)
        short_text = summarize_window(last_n=200, **kwargs)
        assert "last 60 bars" in long_text.splitlines()[0]
        assert "last 200 bars" in short_text.splitlines()[0]
        assert long_text != short_text


class TestGuards:
    """Kurze Fenster dürfen nicht crashen."""

    def test_ten_bars_do_not_crash(self) -> None:
        lines = summarize_window(last_n=60, **_flat(10)).splitlines()

        assert lines[0] == "BTC/USDT-style OHLCV snapshot (last 10 bars, 1m) — bar 9 of 10"
        # nur 10 recent-bar-Zeilen, erste Zeile ohne Vor-Kerze → n/a
        bar_lines = [line for line in lines if line.startswith("bar -")]
        assert len(bar_lines) == 10
        assert bar_lines[0].endswith("ret=n/a")

    def test_single_bar_does_not_crash(self) -> None:
        text = summarize_window(last_n=60, **_flat(1))
        assert "last 1 bars" in text

    def test_empty_input_raises_value_error(self) -> None:
        kwargs = _flat(0)
        with pytest.raises(ValueError):
            summarize_window(last_n=60, **kwargs)

    def test_zero_volume_does_not_crash(self) -> None:
        kwargs = _flat(40, volume=0.0)
        text = summarize_window(last_n=60, **kwargs)
        assert "volume_trend=1" in text
