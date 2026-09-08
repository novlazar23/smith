"""TimesFM-Research-Spike: Feature-Generierung aus historischen Kerzen.

Bewusst klein und read-only:

- kein Backtest, kein Judge, keine Promotion,
- kein Import in generierten Mechanismus-Code,
- keine Default-Dependency auf ``timesfm`` (Provider wird injiziert),
- Leakage-Regel: eine Zeile wird nur aus dem Context bis ``t`` erzeugt
  (keine Future-Daten im Provider-Input).

Output (optional, ``out_dir``):

- ``features.parquet`` — eine Zeile pro Forecast-Startpunkt,
- ``report.json`` — Parameter, Count, Zeitfenster, ``cache_key``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from packages.backtesting.core import Candle
from packages.forecasting.timesfm import ForecastProvider, provider_label

FEATURE_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "instrument",
    "base_close",
    "tfm_median_return_pct",
    "tfm_q10_return_pct",
    "tfm_q90_return_pct",
    "tfm_interval_width_pct",
)


def safe_instrument(instrument: str) -> str:
    """Dateisystem-sichere Instrument-Kennung (``BTC/USDT`` → ``BTC-USDT``)."""
    return instrument.replace("/", "-")


def _return_pct(price: float, base: float) -> float:
    if base <= 0.0:
        raise ValueError("Base-Preis muss positiv sein")
    return (price / base - 1.0) * 100.0


def run_timesfm_spike(
    candles: Sequence[Candle],
    *,
    instrument: str,
    provider: ForecastProvider,
    horizon: int = 288,
    context: int = 512,
    step: int = 288,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Erzeugt TimesFM-Features aus Kerzen (``out_dir=None`` → nur Report).

    Pro Startpunkt ``t`` (Abstand ``step``, letzter Context-Index ``t``)
    wird der Forecast auf den Endhorizont ``t + horizon`` reduziert:
    ``tfm_*_return_pct`` = Rendite vom Close bei ``t`` auf den
    jeweiligen Quantil-Preis am Horizont.
    """
    if horizon < 1:
        raise ValueError("horizon muss >= 1 sein")
    if context < 2:
        raise ValueError("context muss >= 2 sein")
    if step < 1:
        raise ValueError("step muss >= 1 sein")
    if len(candles) < context:
        raise ValueError(f"Zu wenige Kerzen: {len(candles)} < context {context}")

    ordered = sorted(candles, key=lambda candle: candle.timestamp)
    rows: list[dict[str, Any]] = []
    for end_index in range(context - 1, len(ordered), step):
        context_candles = ordered[end_index - context + 1 : end_index + 1]
        context_array = np.array([candle.close for candle in context_candles], dtype=np.float64)
        forecast = provider.predict(context_array, horizon)
        if min(forecast.median.size, forecast.q10.size, forecast.q90.size) < horizon:
            raise ValueError("Forecast ist kürzer als der angeforderte horizon")
        base = float(context_array[-1])
        rows.append(
            {
                "timestamp": context_candles[-1].timestamp,
                "instrument": instrument,
                "base_close": base,
                "tfm_median_return_pct": round(_return_pct(float(forecast.median[-1]), base), 6),
                "tfm_q10_return_pct": round(_return_pct(float(forecast.q10[-1]), base), 6),
                "tfm_q90_return_pct": round(_return_pct(float(forecast.q90[-1]), base), 6),
                "tfm_interval_width_pct": round(
                    _return_pct(float(forecast.q90[-1]), base) - _return_pct(float(forecast.q10[-1]), base),
                    6,
                ),
            }
        )

    if not rows:
        raise ValueError("Keine Features erzeugt (Context/Step passen nicht zur Kerzenzahl)")

    frame = pd.DataFrame(rows)
    cache_payload = {
        "instrument": instrument,
        "provider": provider_label(provider),
        "context": context,
        "horizon": horizon,
        "step": step,
        "candles": len(ordered),
        "start": ordered[0].timestamp.isoformat(),
        "end": ordered[-1].timestamp.isoformat(),
    }
    report: dict[str, Any] = {
        "instrument": instrument,
        "provider": provider_label(provider),
        "n_candles": len(ordered),
        "n_features": len(rows),
        "start": ordered[0].timestamp.isoformat(),
        "end": ordered[-1].timestamp.isoformat(),
        "params": {"context": context, "horizon": horizon, "step": step},
        "cache_key": hashlib.sha256(json.dumps(cache_payload, sort_keys=True).encode("utf-8")).hexdigest(),
    }

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        features_path = out_dir / "features.parquet"
        report_path = out_dir / "report.json"
        frame.to_parquet(features_path, index=False)
        report["files"] = {"features": str(features_path), "report": str(report_path)}
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return report
