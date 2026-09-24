"""Prometheus-Metriken für den Demo-Trader.

Gauges für die Risiko-Alerts (TradingDrawdownHigh, TradingMaxPositions,
TradingConfidenceLow in infrastructure/prometheus/alerts.yml) und das
Grafana-Dashboard. Eigene CollectorRegistry (gleiches Muster wie
apps/api/metrics.py): Test-Neuimports lösen keine
Duplikat-Registrierungsfehler aus.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from prometheus_client import CollectorRegistry, Gauge

if TYPE_CHECKING:
    from apps.demo_trader.service import DemoTrader

#: HTTP-Port des Metrik-Endpunkts (Container-intern, kein Host-Mapping).
METRICS_PORT = 9464

REGISTRY = CollectorRegistry()

OPEN_POSITIONS = Gauge(
    "trading_open_positions",
    "Anzahl offener Paper-Positionen",
    registry=REGISTRY,
)
PORTFOLIO_DRAWDOWN = Gauge(
    "trading_portfolio_drawdown",
    "Drawdown (Portion) vom Equity-Hoch",
    registry=REGISTRY,
)
MAX_OPEN_POSITIONS = Gauge(
    "trading_config_max_open_positions",
    "Maximale Anzahl offener Positionen (eine pro Instrument)",
    registry=REGISTRY,
)
SIGNAL_CONFIDENCE = Gauge(
    "trading_signal_confidence",
    "Letzte Ensemble-Konsens-Konfidenz pro Instrument (0..1)",
    ["instrument"],
    registry=REGISTRY,
)


def update_metrics(trader: DemoTrader) -> None:
    """Erfrischt alle Gauges aus dem aktuellen Trader-Zustand (einmal pro Zyklus)."""
    account = trader.account
    peak = trader._equity_peak
    drawdown = (peak - account.equity) / peak if peak > 0 else 0.0
    OPEN_POSITIONS.set(len(account.positions))
    PORTFOLIO_DRAWDOWN.set(max(0.0, drawdown))
    MAX_OPEN_POSITIONS.set(len(trader.config.instruments))
    for instrument in trader.config.instruments:
        SIGNAL_CONFIDENCE.labels(instrument=instrument).set(
            trader._last_confidences.get(instrument, 0.0)
        )
