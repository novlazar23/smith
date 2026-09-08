"""Evidenz-Digest: kompakte Markdown-Zusammenfassung des Evolutions-Zustands.

Der Digest ist die gemeinsame Gesprächsgrundlage: er wird (a) im Zyklus
protokolliert und (b) den LLM-Personas als Kontext übergeben. Er enthält
nur Fakten aus dem State (keine neuen Entscheidungen) — Baseline, Registry,
Familien-Zählung (Deflations-Basis), letzte Verdicts, Grab, Live-Paper.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from sqlalchemy import text

from .models import utcnow_iso
from .state import EvolutionStore

logger = logging.getLogger(__name__)

GRAVEYARD_DIGEST_LIMIT = 10


def fetch_live_paper() -> dict[str, Any] | None:
    """Demo-Trader-Zustand aus PostgreSQL (graceful degradation → None).

    Gelesen werden ``demo_account`` (letzter Snapshot) und die letzten
    5 ``demo_trades``. Fehlende Umgebung/Verbindung ist kein Fehler —
    der Digest läuft auch ohne Live-Paper.
    """
    try:
        from packages.persistence.sqlalchemy.engine import DatabaseConfig, SQLAlchemyEngine

        config = DatabaseConfig(
            host=os.environ.get("DB_HOST", "postgres"),
            port=int(os.environ.get("DB_PORT", "5432")),
            database=os.environ.get("DB_NAME", "trading"),
            user=os.environ.get("DB_USER", "orchestra"),
            password=os.environ.get("DB_PASSWORD", ""),
        )
        engine = SQLAlchemyEngine(config)
        with engine.engine.connect() as connection:
            account = connection.execute(
                text(
                    "SELECT cash, equity, initial_cash, total_pnl, total_commission, "
                    "total_trades, positions, updated_at FROM demo_account "
                    "ORDER BY updated_at DESC LIMIT 1"
                )
            ).mappings().first()
            trades = connection.execute(
                text(
                    "SELECT instrument, direction, quantity, filled_price, status, created_at "
                    "FROM demo_trades ORDER BY created_at DESC LIMIT 5"
                )
            ).mappings().all()
        if account is None:
            return None
        return {
            "account": dict(account),
            "recent_trades": [dict(row) for row in trades],
        }
    except Exception as exc:
        logger.info("Live-Paper nicht abrufbar (%s: %s) — Digest ohne Live-Sektion", type(exc).__name__, exc)
        return None


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:+.2f} %"


def _baseline_section(store: EvolutionStore) -> list[str]:
    base = store.meta().get("baseline", {})
    params = ", ".join(f"{k}={v:g}" for k, v in base.get("params", {}).items())
    return [
        "## Baseline (Referenz, gegen die jeder Kandidat antreten muss)",
        "",
        f"- **{base.get('label', '—')}**: `{base.get('strategy', '—')}` {params}",
        f"- Kosten: {base.get('costs', '—')}",
        "",
    ]


def _registry_section(store: EvolutionStore) -> list[str]:
    lines = ["## Registry (promoted)", ""]
    promoted = store.registry().get("promoted", [])
    if not promoted:
        lines.append("- (noch keine Promotionen — der Judge hat bisher nichts durchgewunken)")
    for entry in promoted:
        variant = entry.get("variant", {})
        params = ", ".join(f"{k}={v:g}" for k, v in variant.get("params", {}).items()) or "Default"
        lines.append(
            f"- **{entry.get('id', '—')}** (`{variant.get('strategy', '—')}` {params}) — "
            f"{entry.get('claim', '')} _[{entry.get('promoted_at', '—')}]_"
        )
    lines.append("")
    return lines


def _families_section(store: EvolutionStore) -> list[str]:
    hypotheses = store.all_hypotheses()
    counts: dict[str, dict[str, int]] = {}
    for hypothesis in hypotheses:
        row = counts.setdefault(hypothesis.family, {"tested": 0, "passed": 0, "rejected": 0, "other": 0})
        if hypothesis.status == "passed":
            row["passed"] += 1
        elif hypothesis.status == "rejected":
            row["rejected"] += 1
        elif hypothesis.status == "testing":
            row["other"] += 1
        else:  # proposed / pending / error
            row["tested"] += 1
    lines = ["## Familien (Deflations-Basis: ≥ 10 Hypothesen → OOS-Marge x2)", ""]
    if not counts:
        lines.append("- (noch keine Familien)")
    else:
        lines.append("| Familie | getestet | promoted | rejected | in Arbeit |")
        lines.append("|---|---:|---:|---:|---:|")
        for family in sorted(counts):
            row = counts[family]
            lines.append(
                f"| {family} | {row['tested']} | {row['passed']} | {row['rejected']} | {row['other']} |"
            )
    lines.append("")
    return lines


def _last_cycle_section(store: EvolutionStore) -> list[str]:
    last = store.meta().get("last_cycle")
    lines = ["## Letzter Zyklus", ""]
    if not last:
        lines.append("- (noch keiner)")
    else:
        lines.append(
            f"- {last.get('finished_at', '—')}: {last.get('n_tested', 0)} getestet, "
            f"{last.get('n_promoted', 0)} promoted, {last.get('n_rejected', 0)} rejected, "
            f"{last.get('n_pending', 0)} pending, {last.get('n_error', 0)} Fehler"
        )
        for verdict in last.get("verdicts", [])[-5:]:
            lines.append(
                f"  - {verdict.get('id', '—')}: **{verdict.get('decision', '—')}** "
                f"({verdict.get('delta_pp') or 0:+.2f} pp OOS vs Baseline)"
            )
    lines.append("")
    return lines


def _graveyard_section(store: EvolutionStore) -> list[str]:
    lines = ["## Grab (letzte Ablehnungen — keine Retests derselben Variante)", ""]
    entries = store.graveyard()[-GRAVEYARD_DIGEST_LIMIT:]
    if not entries:
        lines.append("- (leer)")
    for entry in entries:
        reasons = "; ".join(entry.get("reasons", [])) or "—"
        lines.append(f"- `{entry.get('variant_key', entry.get('id', '—'))}`: {reasons}")
    lines.append("")
    return lines


def _live_section(live: dict[str, Any] | None) -> list[str]:
    lines = ["## Live-Paper (Demo-Trader — die dauerhafte OOS-Quelle)", ""]
    if live is None:
        lines.append("- (nicht abrufbar in dieser Umgebung)")
        lines.append("")
        return lines
    account = live["account"]
    initial = float(account.get("initial_cash") or 0.0)
    equity = float(account.get("equity") or 0.0)
    return_pct = (equity / initial - 1.0) * 100.0 if initial > 0 else 0.0
    lines.append(
        f"- Equity {equity:,.0f} $ ({_fmt_pct(return_pct)}), PnL {account.get('total_pnl', 0):+,.0f} $, "
        f"{account.get('total_trades', 0)} Trades, Stand {account.get('updated_at', '—')}"
    )
    for trade in live["recent_trades"]:
        lines.append(
            f"  - {trade.get('created_at', '—')} {trade.get('direction', '—')} "
            f"{trade.get('instrument', '—')} @ {trade.get('filled_price', 0)} ({trade.get('status', '—')})"
        )
    lines.append("")
    return lines


def _timesfm_section(store: EvolutionStore) -> list[str]:
    """TimesFM-Research-Reports aus ``<state>/timesfm/`` (nur Evidenz)."""
    lines = ["## TimesFM-Research (nur Evidenz — keine Promotion, kein Live-Entscheider)", ""]
    timesfm_dir = store.root / "timesfm"
    reports: list[tuple[float, str, dict[str, Any]]] = []
    if timesfm_dir.is_dir():
        for report_path in timesfm_dir.glob("*/report.json"):
            try:
                data = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.info("TimesFM-Report %s übersprungen: %s", report_path, exc)
                continue
            if not isinstance(data, dict):
                continue
            try:
                mtime = report_path.stat().st_mtime
            except OSError:
                mtime = 0.0
            reports.append((mtime, report_path.parent.name, data))
    if not reports:
        lines.append("- (noch keine TimesFM-Reports — `python -m apps.evolution --timesfm-spike --timesfm-fake`)")
        lines.append("")
        return lines
    for _mtime, instrument, data in sorted(reports, key=lambda item: item[0], reverse=True):
        params = data.get("params", {})
        if not isinstance(params, dict):
            params = {}
        cache_key = str(data.get("cache_key", "—"))
        lines.append(
            f"- **{instrument}**: provider={data.get('provider', '—')}, "
            f"n_features={data.get('n_features', '—')}, "
            f"context={params.get('context', '—')}, horizon={params.get('horizon', '—')}, "
            f"step={params.get('step', '—')}, cache={cache_key[:12]}"
        )
    lines.append("")
    return lines


def build_digest(store: EvolutionStore, *, live: dict[str, Any] | None = None) -> str:
    """Baut den Evidenz-Digest (Markdown) aus dem State."""
    lines = [f"# Evolutions-Digest — {utcnow_iso()}", ""]
    lines += _baseline_section(store)
    lines += _registry_section(store)
    lines += _families_section(store)
    lines += _last_cycle_section(store)
    lines += _graveyard_section(store)
    lines += _timesfm_section(store)
    lines += _live_section(live)
    return "\n".join(lines)


def write_digest(store: EvolutionStore, digest: str, *, subdir: str = "digests") -> Path | None:
    """Legt den Digest unter ``<state>/digests/`` ab (None, wenn nicht schreibbar)."""
    try:
        target = store.root / subdir
        target.mkdir(parents=True, exist_ok=True)
        path = target / f"cycle-{utcnow_iso().replace(':', '')}.md"
        path.write_text(digest, encoding="utf-8")
        return path
    except OSError as exc:
        logger.warning("Digest-Datei nicht schreibbar (%s)", exc)
        return None
