"""Dashboard-Router — Aggregiert Live-Daten für das Web-Dashboard.

Der Endpunkt ``GET /v1/dashboard`` bündelt in einer einzigen Antwort:

  - Systemzustand (aus den /status-Internals, inkl. Uptime und Zählern)
  - Marktdaten des letzten Kerzen-Fensters pro Instrument (ClickHouse)
  - Demo-Konto, offene Positionen und aktuelle Trades (PostgreSQL)
  - Letzte Shadow-Entscheidungen und News-Events (PostgreSQL)

Alle Quellen werden defensiv abgefragt: eine ausgefallene Quelle liefert
leere Listen bzw. ``None`` — der Endpunkt antwortet nie mit HTTP 500.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from apps.api.endpoints import status_endpoint
from fastapi import APIRouter
from packages.persistence.clickhouse.engine import ClickHouseConfig, create_ch_engine
from packages.persistence.sqlalchemy.engine import DatabaseConfig, SQLAlchemyEngine
from sqlalchemy import text

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["dashboard"])

# Instrumente des Dashboards
INSTRUMENTS: tuple[str, ...] = ("BTC/USDT", "ETH/USDT")

# Größe des Kerzen-Fensters pro Instrument
_CANDLE_WINDOW = 60

# Timeout für eine einzelne (blockierende) Datenquelle in Sekunden
_SOURCE_TIMEOUT_SECONDS = 5.0

# Zeitpunkt des Modul-Imports — Uptime-Fallback, falls der Status-Endpunkt
# selbst nicht antwortet (gleiche Semantik wie apps.api.endpoints._START).
_START = time.monotonic()


# ---------------------------------------------------------------------------
# Konfiguration (gleiche Env-Vars wie apps.api.endpoints / apps.db_init)
# ---------------------------------------------------------------------------


def _pg_config() -> DatabaseConfig:
    """Liest die PostgreSQL-Konfiguration aus der Umgebung."""
    return DatabaseConfig(
        host=os.environ.get("DB_HOST", "localhost"),
        port=int(os.environ.get("DB_PORT", "5432")),
        database=os.environ.get("DB_NAME", "trading"),
        user=os.environ.get("DB_USER", "orchestra"),
        password=os.environ.get("DB_PASSWORD", ""),
    )


def _ch_config() -> ClickHouseConfig:
    """Liest die ClickHouse-Konfiguration aus der Umgebung."""
    return ClickHouseConfig(
        host=os.environ.get("CH_HOST", "localhost"),
        port=int(os.environ.get("CH_PORT", "8123")),
        database=os.environ.get("CH_DB", "trading_events"),
        user=os.environ.get("CH_USER", "orchestra"),
        password=os.environ.get("CH_PASSWORD", ""),
    )


# ---------------------------------------------------------------------------
# Konvertierungs-Hilfen
# ---------------------------------------------------------------------------


def _iso(value: object) -> str | None:
    """Wandelt einen Datetime-Wert in ISO-8601 mit UTC-„Z" um."""
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    return None


def _ch_time(value: str) -> str:
    """Konvertiert ClickHouse-Datetimes („YYYY-MM-DD HH:MM:SS") nach ISO-8601 UTC."""
    return f"{value.replace(' ', 'T', 1)}Z"


def _float(value: object) -> float | None:
    """Sichere Float-Konvertierung (None bei Fehlern)."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: object) -> int | None:
    """Sichere Integer-Konvertierung (None bei Fehlern)."""
    number = _float(value)
    return None if number is None else int(number)


# ---------------------------------------------------------------------------
# Datenquellen (blockierend — immer über asyncio.to_thread aufrufen)
# ---------------------------------------------------------------------------


def _pg_rows(sql: str) -> list[dict[str, Any]]:
    """Führt eine Read-Only-Query in PostgreSQL aus und liefert Dict-Zeilen."""
    engine = SQLAlchemyEngine(_pg_config()).engine
    with engine.connect() as conn:
        return [dict(row._mapping) for row in conn.execute(text(sql))]


def _ch_query(sql: str) -> tuple[list[str], list[list[str]]]:
    """Führt eine ClickHouse-SELECT-Query über die Engine aus."""
    return create_ch_engine(_ch_config()).query(sql)


def _fetch_data_source() -> str:
    """Ermittelt die Datenquelle: „live" bei Binance-Kerzen, sonst „synthetic"."""
    _, rows = _ch_query("SELECT upper(venue) AS venue FROM candles ORDER BY open_time DESC LIMIT 1")
    if rows and rows[0] and str(rows[0][0]).startswith("BINANCE"):
        return "live"
    return "synthetic"


def _fetch_market() -> list[dict[str, Any]]:
    """Liefert das Kerzen-Fenster pro Instrument aus ClickHouse."""
    market: list[dict[str, Any]] = []
    for instrument in INSTRUMENTS:
        sql = (
            "SELECT open_time, argMax(close, ingestion_time) AS close "
            f"FROM candles WHERE instrument = '{instrument}' "
            f"GROUP BY open_time ORDER BY open_time DESC LIMIT {_CANDLE_WINDOW}"
        )
        _, rows = _ch_query(sql)
        candles: list[tuple[str, float]] = []
        for row in reversed(rows):
            try:
                candles.append((str(row[0]), float(row[1])))
            except (IndexError, ValueError):
                continue
        market.append(_build_market_entry(instrument, candles))
    return market


def _build_market_entry(instrument: str, candles: list[tuple[str, float]]) -> dict[str, Any]:
    """Baut den Markt-Eintrag eines Instruments aus einem Kerzen-Fenster."""
    last_price: float | None = candles[-1][1] if candles else None
    change_pct: float | None = None
    if len(candles) >= 2 and candles[0][1] > 0:
        change_pct = round((candles[-1][1] / candles[0][1] - 1.0) * 100.0, 2)
    return {
        "instrument": instrument,
        "last_price": last_price,
        "change_pct": change_pct,
        "candles": [{"t": _ch_time(t), "c": c} for t, c in candles],
    }


def _fetch_demo_account() -> dict[str, Any] | None:
    """Liest das Demo-Konto (account_id='demo'); None, falls es noch nicht existiert."""
    rows = _pg_rows(
        "SELECT cash, equity, initial_cash, total_pnl, total_trades, "
        "total_commission, updated_at FROM demo_account WHERE account_id = 'demo'"
    )
    if not rows:
        return None
    row = rows[0]
    total_pnl = _float(row.get("total_pnl")) or 0.0
    initial_cash = _float(row.get("initial_cash")) or 0.0
    return {
        "cash": _float(row.get("cash")),
        "equity": _float(row.get("equity")),
        "initial_cash": _float(row.get("initial_cash")),
        "total_pnl": _float(row.get("total_pnl")),
        "pnl_pct": round(total_pnl / initial_cash * 100.0, 2) if initial_cash else None,
        "total_trades": _int(row.get("total_trades")),
        "total_commission": _float(row.get("total_commission")),
        "updated_at": _iso(row.get("updated_at")),
    }


def _fetch_recent_trades() -> list[dict[str, Any]]:
    """Liefert die letzten 20 Demo-Trades (neueste zuerst)."""
    rows = _pg_rows(
        "SELECT created_at, instrument, direction, quantity, price, commission, status "
        "FROM demo_trades ORDER BY created_at DESC LIMIT 20"
    )
    return [
        {
            "ts": _iso(row.get("created_at")),
            "instrument": str(row.get("instrument", "")),
            "direction": str(row.get("direction", "")),
            "quantity": _float(row.get("quantity")),
            "price": _float(row.get("price")),
            "commission": _float(row.get("commission")),
            "status": str(row.get("status", "")),
        }
        for row in rows
    ]


def _fetch_positions(market: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Leitet offene Positionen aus den Demo-Trades ab.

    Offene Position = Netto-Menge (BUY minus SELL) pro Instrument; der
    Ø-Preis ist der gewichtete Kaufpreis. Marktpreis und unrealisierter
    P&L kommen aus dem aktuellsten Close in ClickHouse (übergebenes
    Markt-Fenster).
    """
    last_prices = {entry["instrument"]: entry["last_price"] for entry in market}
    rows = _pg_rows(
        "SELECT created_at, instrument, direction, quantity, price "
        "FROM demo_trades ORDER BY created_at ASC"
    )
    open_positions: dict[str, dict[str, Any]] = {}
    for row in rows:
        try:
            instrument = str(row["instrument"])
            quantity = float(row["quantity"])
            price = float(row["price"])
        except (KeyError, TypeError, ValueError):
            continue
        entry = open_positions.setdefault(
            instrument, {"qty": 0.0, "buy_qty": 0.0, "cost": 0.0, "opened_at": None}
        )
        if str(row.get("direction", "")).upper() == "BUY":
            entry["qty"] += quantity
            entry["buy_qty"] += quantity
            entry["cost"] += quantity * price
            if entry["opened_at"] is None:
                entry["opened_at"] = _iso(row.get("created_at"))
        else:
            entry["qty"] -= quantity

    positions: list[dict[str, Any]] = []
    for instrument, entry in open_positions.items():
        if entry["qty"] <= 0 or entry["buy_qty"] <= 0:
            continue
        avg_price = entry["cost"] / entry["buy_qty"]
        market_price = last_prices.get(instrument)
        unrealized = (
            round((market_price - avg_price) * entry["qty"], 2)
            if market_price is not None
            else None
        )
        positions.append(
            {
                "instrument": instrument,
                "quantity": round(entry["qty"], 8),
                "avg_price": round(avg_price, 2),
                "market_price": market_price,
                "unrealized_pnl": unrealized,
                "opened_at": entry["opened_at"],
            }
        )
    positions.sort(key=lambda position: position["instrument"])
    return positions


def _fetch_recent_decisions() -> list[dict[str, Any]]:
    """Liefert die letzten 10 Shadow-Entscheidungen (neueste zuerst)."""
    rows = _pg_rows(
        "SELECT created_at, instrument, decision, confidence, reason "
        "FROM shadow_decisions ORDER BY created_at DESC LIMIT 10"
    )
    return [
        {
            "ts": _iso(row.get("created_at")),
            "instrument": str(row.get("instrument", "")),
            "decision": str(row.get("decision", "")),
            "confidence": _float(row.get("confidence")),
            "reason": str(row.get("reason") or ""),
        }
        for row in rows
    ]


def _fetch_recent_news() -> list[dict[str, Any]]:
    """Liefert die letzten 10 News-Events (neueste zuerst)."""
    rows = _pg_rows(
        "SELECT created_at, source_name, title, status "
        "FROM news_events ORDER BY created_at DESC LIMIT 10"
    )
    return [
        {
            "ts": _iso(row.get("created_at")),
            "source": str(row.get("source_name", "")),
            "title": str(row.get("title", "")),
            "category": str(row.get("status", "")),
        }
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Endpunkt
# ---------------------------------------------------------------------------


async def _status_or_fallback() -> dict[str, Any]:
    """Ruft den /status-Endpunkt ab; degradiert bei Fehlern statt 500."""
    try:
        return await status_endpoint()
    except Exception:
        logger.warning("dashboard: Status-Endpunkt ausgefallen", exc_info=True)
        return {"status": "degraded", "uptime_seconds": round(time.monotonic() - _START, 2)}


def _system_block(status_data: dict[str, Any], data_source: str) -> dict[str, Any]:
    """Baut den System-Block aus der /status-Antwort."""
    streaming = status_data.get("streaming") or {}
    flags = status_data.get("feature_flags") or {}
    return {
        "status": status_data.get("status", "degraded"),
        "uptime_seconds": status_data.get("uptime_seconds"),
        "data_source": data_source,
        "live_trading_enabled": bool(flags.get("live_trading_enabled", False)),
        "candles_total": streaming.get("candles_1h"),
        "news_events_total": streaming.get("news_events_total"),
    }


async def _run_source[T](fn: Callable[..., T], default: T, *args: object) -> T:
    """Führt eine blockierende Datenquellen-Funktion im Thread-Pool aus.

    Bei Timeout oder unerwartetem Fehler wird der Default-Wert geliefert,
    damit der Endpunkt nie hängt oder mit 500 antwortet.
    """
    try:
        started = asyncio.to_thread(fn, *args)
        return await asyncio.wait_for(started, timeout=_SOURCE_TIMEOUT_SECONDS)
    except Exception:
        logger.warning("dashboard: Datenquelle %s ausgefallen", fn.__name__, exc_info=True)
        return default


@router.get("/dashboard")
async def dashboard() -> dict[str, Any]:
    """Liefert alle Daten für das Live-Web-Dashboard in einer Antwort.

    Aggregiert Systemzustand, Marktdaten (ClickHouse) sowie Konto-,
    Positions-, Entscheidungs- und News-Daten (PostgreSQL). Ausfallende
    Quellen degradieren zu leeren Werten — nie ein HTTP-Fehler.
    """
    (
        status_data,
        data_source,
        market,
        demo_account,
        trades,
        decisions,
        news,
    ) = await asyncio.gather(
        _status_or_fallback(),
        _run_source(_fetch_data_source, "synthetic"),
        _run_source(_fetch_market, []),
        _run_source(_fetch_demo_account, None),
        _run_source(_fetch_recent_trades, []),
        _run_source(_fetch_recent_decisions, []),
        _run_source(_fetch_recent_news, []),
    )
    positions = await _run_source(_fetch_positions, [], market)

    return {
        "system": _system_block(status_data, data_source),
        "market": market,
        "demo_account": demo_account,
        "positions": positions,
        "recent_trades": trades,
        "recent_decisions": decisions,
        "recent_news": news,
    }
