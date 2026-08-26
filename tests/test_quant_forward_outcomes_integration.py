"""Integration-Tests für Forward Outcomes (Phase 6, P6-4).

Verbindet die ``ForwardOutcomeEngine`` (P6-1) mit dem Store-Schreib-/Lesepfad
für InfluxDB. Alle Tests sind vollständig gemockt — es werden keine echten
InfluxDB-Verbindungen aufgebaut.

Anmerkung: Das Modul ``quant/forward_outcomes_store.py`` (P6-2) ist noch in
Bearbeitung. Bis es committet ist, pinnt dieser Test eine lokale Stand-in-Klasse
``ForwardOutcomeStore``, die den für P6-2 geplanten Vertrag festlegt
(Muster: ``AnomalyStore`` / ``FeatureStore``):

- ``compute_and_store(...)`` — Engine berechnet die Outcomes; pro Horizont
  wird ein InfluxDB-Punkt über ``write_points`` geschrieben
  (Tags: ``symbol`` / ``timeframe`` / ``horizon``), nur wenn
  ``store.is_available`` True ist.
- ``load_outcomes(...)`` — Flux-Query über ``store.query``; jede Zeile wird
  wieder in ein ``ForwardOutcome``-Objekt rekonstruiert.

Wenn P6-2 committet ist, wird die Stand-in durch den echten Import ersetzt;
die Assertions bleiben der zu haltende Vertrag.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from trading_harness.quant.forward_outcomes import (
    ForwardOutcome,
    ForwardOutcomeEngine,
    ForwardOutcomeResult,
)

FORWARD_OUTCOMES_MEASUREMENT = "forward_outcomes"
SYMBOL = "BTCUSDT"
TIMEFRAME = "1m"
EXCHANGE = "binance"
BASE_TS = 1_700_000_000  # Epoch-Sekunden der ersten Kerze (1-Minuten-Abstand)


def _candle_time(index: int) -> str:
    """UTC-ISO-Zeichenkette für die Kerze an Position *index*."""
    moment = datetime.fromtimestamp(BASE_TS + index * 60, tz=UTC)
    return moment.isoformat().replace("+00:00", "Z")


def make_candles(prices: list[float]) -> list[dict]:
    """Synthetische OHLCV-Reihe aus einer Close-Preisliste."""
    return [
        {
            "time": _candle_time(i),
            "open": p,
            "high": p * 1.01,
            "low": p * 0.99,
            "close": p,
            "volume": 1000.0,
        }
        for i, p in enumerate(prices)
    ]


def make_mock_store(available: bool = True) -> MagicMock:
    """InfluxDBStore-Mock: Verfügbarkeits-Flag + async write/query + Bucket-Config."""
    store = MagicMock()
    store._bucket = "quant"
    store.is_available = available
    store.write_points = AsyncMock()
    store.query = AsyncMock(return_value=[])
    return store


class ForwardOutcomeStore:
    """Stand-in für das in Bearbeitung befindliche P6-2-Modul (siehe Modul-Docstring)."""

    def __init__(
        self,
        store: MagicMock,
        engine: ForwardOutcomeEngine | None = None,
    ) -> None:
        self._store = store
        self._engine = engine or ForwardOutcomeEngine()

    async def compute_and_store(
        self,
        candles: list[dict],
        pattern_length: int,
        symbol: str,
        timeframe: str,
        timestamp_ns: int,
    ) -> ForwardOutcomeResult:
        """Berechnet Outcomes und schreibt pro Horizont einen InfluxDB-Punkt.

        Bei nicht verfügbarem Store (``is_available=False``) wird nicht
        geschrieben — die Berechnung findet trotzdem statt.
        """
        result = self._engine.compute(
            candles, pattern_length, symbol=symbol, timeframe=timeframe
        )
        if self._store.is_available:
            for horizon in sorted(result.outcomes):
                outcome = result.outcomes[horizon]
                await self._store.write_points(
                    measurement=FORWARD_OUTCOMES_MEASUREMENT,
                    tags={
                        "symbol": symbol,
                        "exchange": EXCHANGE,
                        "timeframe": timeframe,
                        "horizon": str(horizon),
                    },
                    fields={
                        "mean_return": outcome.mean_return,
                        "median_return": outcome.median_return,
                        "hit_rate": outcome.hit_rate,
                        "profit_factor": outcome.profit_factor,
                        "expectancy": outcome.expectancy,
                        "std_return": outcome.std_return,
                        "sample_size": outcome.sample_size,
                        "max_gain": outcome.max_gain,
                        "max_loss": outcome.max_loss,
                        "pattern_length": pattern_length,
                    },
                    timestamp=timestamp_ns,
                )
        return result

    async def load_outcomes(self, symbol: str, timeframe: str) -> dict[int, ForwardOutcome]:
        """Liest Outcomes aus InfluxDB und rekonstruiert ``ForwardOutcome`` pro Horizont."""
        if not self._store.is_available:
            return {}
        flux = (
            f'from(bucket: "quant")'
            f' |> range(start: -30d)'
            f' |> filter(fn: (r) => r._measurement == "{FORWARD_OUTCOMES_MEASUREMENT}")'
            f' |> filter(fn: (r) => r.symbol == "{symbol}")'
            f' |> filter(fn: (r) => r.timeframe == "{timeframe}")'
        )
        rows = await self._store.query(flux)
        outcomes: dict[int, ForwardOutcome] = {}
        for record in rows:
            try:
                horizon = int(record.get("horizon"))
            except (TypeError, ValueError):
                continue
            outcomes[horizon] = ForwardOutcome(
                horizon=horizon,
                mean_return=float(record.get("mean_return", 0.0)),
                median_return=float(record.get("median_return", 0.0)),
                hit_rate=float(record.get("hit_rate", 0.0)),
                profit_factor=float(record.get("profit_factor", 0.0)),
                expectancy=float(record.get("expectancy", 0.0)),
                std_return=float(record.get("std_return", 0.0)),
                sample_size=int(record.get("sample_size", 0)),
                max_gain=float(record.get("max_gain", 0.0)),
                max_loss=float(record.get("max_loss", 0.0)),
            )
        return outcomes


# ----------------------------------------------------------------------
# 1. Engine: Uptrend erzeugt positive Outcomes
# ----------------------------------------------------------------------


def test_engine_uptrend_positive_outcomes() -> None:
    """Stetiger Uptrend → positive Mean/Median/Expectancy und Hit Rate 1.0 über alle Horizonte."""
    candles = make_candles([100.0 + i * 0.5 for i in range(120)])
    engine = ForwardOutcomeEngine(horizons=[5, 10, 20])
    result = engine.compute(candles, pattern_length=10, symbol=SYMBOL, timeframe=TIMEFRAME)

    assert isinstance(result, ForwardOutcomeResult)
    assert result.symbol == SYMBOL
    assert result.timeframe == TIMEFRAME
    assert result.pattern_length == 10
    assert set(result.outcomes) == {5, 10, 20}
    for horizon, outcome in result.outcomes.items():
        assert outcome.mean_return > 0.0, f"horizon {horizon}: mean_return nicht positiv"
        assert outcome.median_return > 0.0, f"horizon {horizon}: median_return nicht positiv"
        assert outcome.expectancy > 0.0, f"horizon {horizon}: expectancy nicht positiv"
        assert outcome.hit_rate == 1.0, f"horizon {horizon}: alle Forward Returns positiv"
        assert outcome.sample_size > 0, f"horizon {horizon}: keine Samples berechnet"


# ----------------------------------------------------------------------
# 2. Engine → Store Roundtrip (InfluxDB gemockt)
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_engine_to_store_roundtrip() -> None:
    """Engine-Resultat wird pro Horizont geschrieben; Lese-Pfad rekonstruiert identische Outcomes."""
    store = make_mock_store()
    outcomes_store = ForwardOutcomeStore(store, ForwardOutcomeEngine(horizons=[5, 10]))
    candles = make_candles([100.0 + i * 0.5 for i in range(120)])
    timestamp_ns = BASE_TS * 1_000_000_000

    result = await outcomes_store.compute_and_store(
        candles, pattern_length=10, symbol=SYMBOL, timeframe=TIMEFRAME, timestamp_ns=timestamp_ns
    )

    # Ein Punkt pro Horizont, mit korrektem Measurement, Tags und Fields.
    assert store.write_points.await_count == 2
    first = store.write_points.await_args_list[0]
    assert first.kwargs["measurement"] == FORWARD_OUTCOMES_MEASUREMENT
    assert first.kwargs["tags"] == {
        "symbol": SYMBOL,
        "exchange": EXCHANGE,
        "timeframe": TIMEFRAME,
        "horizon": "5",
    }
    assert first.kwargs["timestamp"] == timestamp_ns
    fields = first.kwargs["fields"]
    assert fields["mean_return"] == pytest.approx(result.outcomes[5].mean_return)
    assert fields["hit_rate"] == pytest.approx(result.outcomes[5].hit_rate)
    assert fields["sample_size"] == result.outcomes[5].sample_size
    assert fields["pattern_length"] == 10

    # Simulierte InfluxDB: die geschriebenen Punkte werden als Query-Zeilen zurückgegeben.
    records = [{**c.kwargs["tags"], **c.kwargs["fields"]} for c in store.write_points.await_args_list]
    store.query = AsyncMock(return_value=records)

    loaded = await outcomes_store.load_outcomes(SYMBOL, TIMEFRAME)

    assert store.query.await_count == 1
    flux = store.query.await_args.args[0]
    assert FORWARD_OUTCOMES_MEASUREMENT in flux
    assert set(loaded) == {5, 10}
    for horizon in (5, 10):
        original = result.outcomes[horizon]
        restored = loaded[horizon]
        assert restored.mean_return == pytest.approx(original.mean_return)
        assert restored.median_return == pytest.approx(original.median_return)
        assert restored.hit_rate == pytest.approx(original.hit_rate)
        assert restored.expectancy == pytest.approx(original.expectancy)
        assert restored.sample_size == original.sample_size
        assert restored.max_gain == pytest.approx(original.max_gain)
        assert restored.max_loss == pytest.approx(original.max_loss)


@pytest.mark.asyncio
async def test_store_unavailable_skips_writes() -> None:
    """is_available=False → Berechnung läuft, aber es werden keine Punkte geschrieben."""
    store = make_mock_store(available=False)
    outcomes_store = ForwardOutcomeStore(store, ForwardOutcomeEngine(horizons=[5, 10]))
    candles = make_candles([100.0 + i * 0.5 for i in range(120)])

    result = await outcomes_store.compute_and_store(
        candles,
        pattern_length=10,
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        timestamp_ns=BASE_TS * 1_000_000_000,
    )

    store.write_points.assert_not_awaited()
    assert store.query.await_count == 0
    # Die Berechnung selbst ist unbeeindruckt vom Store-Zustand.
    assert set(result.outcomes) == {5, 10}
    assert result.outcomes[5].mean_return > 0.0
    assert await outcomes_store.load_outcomes(SYMBOL, TIMEFRAME) == {}


# ----------------------------------------------------------------------
# 3. Hit Rate bleibt über alle Marktszenarien in [0.0, 1.0]
# ----------------------------------------------------------------------


def test_outcomes_hit_rate_bounded() -> None:
    """Hit Rate ist für Uptrend, Downtrend, Oszillation, Flat und Crash immer in [0.0, 1.0]."""
    scenarios: list[list[float]] = [
        [100.0 + i * 0.5 for i in range(120)],  # Uptrend
        [200.0 - i * 0.5 for i in range(120)],  # Downtrend
        [100.0 + math.sin(i * 0.3) * 5.0 for i in range(120)],  # Oszillation
        [100.0] * 120,  # Flat
        [100.0] * 60 + [100.0 - (i - 59) * 3.0 for i in range(60, 120)],  # Crash
    ]
    engine = ForwardOutcomeEngine(horizons=[5, 10, 20, 50])
    for prices in scenarios:
        result = engine.compute(make_candles(prices), pattern_length=10)
        for horizon, outcome in result.outcomes.items():
            assert 0.0 <= outcome.hit_rate <= 1.0, (
                f"hit_rate außerhalb [0, 1]: horizon {horizon} → {outcome.hit_rate}"
            )
            if outcome.sample_size > 0:
                assert outcome.max_gain >= outcome.max_loss


# ----------------------------------------------------------------------
# 4. Zu wenige Daten → sample_size 0 für alle Horizonte
# ----------------------------------------------------------------------


def test_insufficient_data_returns_zero_sample_size() -> None:
    """Leere oder zu kleine Kerzenreihen liefern für jeden Horizont sample_size=0 und Null-Statistiken."""
    engine = ForwardOutcomeEngine(horizons=[5, 10, 20, 50])
    insufficient = [
        [],  # keine Kerzen
        make_candles([100.0]),  # eine Kerze
        make_candles([100.0, 101.0]),  # zwei Kerzen
        make_candles([100.0, 100.5, 101.0, 101.5]),  # kürzer als pattern_length + 2
    ]
    for candles in insufficient:
        result = engine.compute(candles, pattern_length=10)
        assert set(result.outcomes) == {5, 10, 20, 50}
        for outcome in result.outcomes.values():
            assert outcome.sample_size == 0
            assert outcome.mean_return == 0.0
            assert outcome.median_return == 0.0
            assert outcome.hit_rate == 0.0
            assert outcome.profit_factor == 0.0
            assert outcome.expectancy == 0.0
            assert outcome.std_return == 0.0
            assert outcome.max_gain == 0.0
            assert outcome.max_loss == 0.0
