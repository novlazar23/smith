"""Tests für den CLI-Datenfluss: Instrument-Parsing, Stufe-1/Stufe-2-Serien-Trennung, Refresh-Union."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from apps.champion_evals import __main__ as cm
from packages.backtesting.core import Candle

MIN_CANDLES = 30
HORIZON_BARS = 3
N_CANDLES = 40  # >= min_candles + horizon_bars
Series = list[tuple[str, list[Candle]]]


def _args(**overrides: Any) -> SimpleNamespace:
    base = {
        "instrument": "BTC/USDT,ETH/USDT",
        "venue": "BINANCE_FUTURES",
        "resample": "5m",
        "days": None,
        "start": None,
        "end": None,
        "min_candles": MIN_CANDLES,
        "horizon_bars": HORIZON_BARS,
        "evolve": True,
        "evolve_agents": 8,
        "evolve_agents_instruments": None,
        "refresh_data": False,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _candle(symbol: str, i: int) -> Candle:
    return Candle(
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        symbol=symbol,
        open=1.0,
        high=1.1,
        low=0.9,
        close=1.0 + 0.01 * i,
        volume=1.0,
    )


def _make_feed(counts: dict[str, int]) -> type:
    class _FakeFeed:
        def __init__(
            self,
            engine: object,
            instrument: str,
            *,
            venue: str | None = None,
            start: str | None = None,
            end: str | None = None,
            resample: str | None = None,
        ) -> None:
            self.instrument = instrument

        def get_candles(self) -> list[Candle]:
            return [_candle(self.instrument, i) for i in range(counts.get(self.instrument, N_CANDLES))]

    return _FakeFeed


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[str]]:
    captured: dict[str, list[str]] = {}
    monkeypatch.setattr(cm, "_ch_engine", lambda args: object())
    monkeypatch.setattr("apps.backtest.ch_feed.ClickHouseDataFeed", _make_feed({}))

    def _fake_run_evolve(args: SimpleNamespace, series: Series) -> int:
        captured["stage1"] = [i for i, _ in series]
        return 0

    def _fake_run_agent_evolve(args: SimpleNamespace, series: Series) -> int:
        captured["stage2"] = [i for i, _ in series]
        return 0

    monkeypatch.setattr(cm, "_run_evolve", _fake_run_evolve)
    monkeypatch.setattr(cm, "_run_agent_evolve", _fake_run_agent_evolve)
    return captured


def test_parse_instruments_strips_and_dedupes() -> None:
    assert cm._parse_instruments(" BTC/USDT , ETH/USDT,BTC/USDT ") == ("BTC/USDT", "ETH/USDT")
    assert cm._parse_instruments(",,") == ()


def test_stage2_default_uses_stage1_series(patched: dict[str, list[str]]) -> None:
    assert cm._run_once(_args()) == 0
    assert patched["stage1"] == ["BTC/USDT", "ETH/USDT"]
    assert patched["stage2"] == ["BTC/USDT", "ETH/USDT"]


def test_stage2_own_instruments(patched: dict[str, list[str]]) -> None:
    assert cm._run_once(_args(evolve_agents_instruments="BTC/USDT,SOL/USDT,XRP/USDT")) == 0
    assert patched["stage1"] == ["BTC/USDT", "ETH/USDT"]
    assert patched["stage2"] == ["BTC/USDT", "SOL/USDT", "XRP/USDT"]


def test_stage2_same_list_skips_second_load(patched: dict[str, list[str]]) -> None:
    assert cm._run_once(_args(evolve_agents_instruments="BTC/USDT,ETH/USDT")) == 0
    assert patched["stage2"] == ["BTC/USDT", "ETH/USDT"]


def test_refresh_uses_union_of_both_lists(patched: dict[str, list[str]], monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, tuple[str, ...]] = {}

    def _fake_refresh(engine: object, instruments: tuple[str, ...], start: str, end: str) -> None:
        seen["instruments"] = instruments

    monkeypatch.setattr(cm, "_refresh_history", _fake_refresh)
    assert cm._run_once(_args(refresh_data=True, days=180, evolve_agents_instruments="SOL/USDT,ADA/USDT")) == 0
    assert seen["instruments"] == ("BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT")


def test_stage2_without_data_fails_run(patched: dict[str, list[str]], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("apps.backtest.ch_feed.ClickHouseDataFeed", _make_feed({"SOL/USDT": 5}))
    rc = cm._run_once(_args(evolve_agents_instruments="SOL/USDT"))
    assert rc == 1
    assert "stage1" in patched
    assert "stage2" not in patched
