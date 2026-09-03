"""Markdown-Report und Artefakte (JSON/CSV) für Backtest-Runs.

``render_markdown`` rendert Summary (pro Szenario), Per-Agent-Tabellen,
Confidence-Bucket-Tabellen und — falls vorhanden — die Gate-Sweep-Tabelle.
``write_artifacts`` schreibt pro Szenario ``report.json``,
``equity_curve.csv`` und ``evaluations.json`` in ein Verzeichnis.
"""

from __future__ import annotations

import csv
import json
import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from packages.backtesting.core import BacktestResult

logger = logging.getLogger(__name__)


def _fmt_pct(value: int | float | None, digits: int = 4) -> str:
    """Formatiert einen Prozentwert (oder '—' bei None)."""
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}%"


def _fmt_num(value: int | float | None, digits: int = 4) -> str:
    """Formatiert eine Kennzahl (oder '—' bei None)."""
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _summary_row(label: str, result: BacktestResult, extra: dict[str, Any]) -> str:
    """Eine Summary-Zeile (Metriken aus result.metrics + extra)."""
    metrics = result.metrics
    final_equity = result.metadata.get("final_equity")
    return (
        f"| {label} "
        f"| {_fmt_num(final_equity, 2)} "
        f"| {_fmt_pct(metrics.get('total_return_pct'))} "
        f"| {_fmt_num(metrics.get('sharpe_ratio'))} "
        f"| {_fmt_pct(metrics.get('max_drawdown_pct'))} "
        f"| {_fmt_pct(metrics.get('win_rate_pct'), 2)} "
        f"| {metrics.get('total_trades', 0)} "
        f"| {_fmt_num(metrics.get('profit_factor'))} "
        f"| {_fmt_num(extra.get('gate_pass_rate'))} |"
    )


def _per_agent_table(extra: dict[str, Any]) -> list[str]:
    """Per-Agent-Tabelle (Evaluations, mittlere Konfidenz, Richtungen)."""
    per_agent: dict[str, dict[str, Any]] = extra.get("per_agent", {})
    lines = ["### Per-Agent", "", "| agent | evaluations | mean_confidence | directions |", "|---|---:|---:|---|"]
    if not per_agent:
        lines.append("| _keine Agenten-Details (leere oder Stub-Reports)_ | | | |")
        return lines
    for agent_id in sorted(per_agent):
        stats = per_agent[agent_id]
        directions = ", ".join(
            f"{direction}: {count}" for direction, count in sorted(stats.get("directions", {}).items())
        ) or "—"
        lines.append(
            f"| {agent_id} | {stats['evaluations']} | {stats['mean_confidence']:.4f} | {directions} |"
        )
    return lines


def _bucket_table(buckets: list[dict[str, Any]]) -> list[str]:
    """Confidence-Bucket-Tabelle (Evaluations + Trade-Stats pro Bucket)."""
    lines = [
        "### Confidence-Buckets",
        "",
        "| bucket | n_evaluations | n_trades | win_rate | avg_pnl |",
        "|---|---:|---:|---:|---:|",
    ]
    for bucket in buckets:
        win_rate = "—" if bucket["win_rate"] is None else f"{bucket['win_rate']:.2%}"
        avg_pnl = "—" if bucket["avg_pnl"] is None else f"{bucket['avg_pnl']:.2f}"
        lines.append(
            f"| {bucket['bucket']} | {bucket['n_evaluations']} | {bucket['n_trades']} "
            f"| {win_rate} | {avg_pnl} |"
        )
    return lines


def _exit_table(extra: dict[str, Any]) -> list[str]:
    """Exit-Verteilung (Round-Trips pro Exit-Grund mit PnL-Statistik)."""
    round_trips: list[dict[str, Any]] = extra.get("round_trips", [])
    lines = [
        "### Exit-Verteilung",
        "",
        "| exit_reason | n_round_trips | win_rate | avg_pnl | total_pnl |",
        "|---|---:|---:|---:|---:|",
    ]
    if not round_trips:
        lines.append("| _keine Round-Trips_ | | | | |")
        return lines
    by_reason: dict[str, list[dict[str, Any]]] = {}
    for rt in round_trips:
        by_reason.setdefault(str(rt.get("exit_reason", "unknown")), []).append(rt)
    for reason in sorted(by_reason):
        trips = by_reason[reason]
        pnls = [float(rt.get("pnl", 0.0)) for rt in trips]
        wins = sum(1 for pnl in pnls if pnl > 0)
        win_rate = f"{wins / len(pnls):.2%}" if pnls else "—"
        avg_pnl = f"{sum(pnls) / len(pnls):.2f}" if pnls else "—"
        lines.append(f"| {reason} | {len(trips)} | {win_rate} | {avg_pnl} | {sum(pnls):.2f} |")
    return lines


def _sweep_table(rows: list[dict[str, Any]]) -> list[str]:
    """Gate-Sweep-Tabelle (gate → trades, win_rate, Return, Sharpe, DD, Pass-Rate)."""
    lines = [
        "## Gate-Sweep",
        "",
        "| gate | trades | win_rate | total_return | sharpe | max_drawdown | gate_pass_rate |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        win_rate = "—" if row["win_rate"] is None else f"{row['win_rate']:.2%}"
        lines.append(
            f"| {row['gate']:.2f} | {row['trades']} | {win_rate} "
            f"| {_fmt_pct(row.get('total_return_pct'))} "
            f"| {_fmt_num(row.get('sharpe_ratio'))} "
            f"| {_fmt_pct(row.get('max_drawdown_pct'))} "
            f"| {_fmt_num(row.get('gate_pass_rate'))} |"
        )
    return lines


def render_markdown(
    runs: list[tuple[str, BacktestResult, dict[str, Any]]],
    sweep_rows: list[dict[str, Any]] | None = None,
) -> str:
    """Rendert den kompletten Markdown-Report.

    Args:
        runs: (label, BacktestResult, extra) pro Szenario.
        sweep_rows: Optionale Gate-Sweep-Zeilen (dann zusätzliche Tabelle).

    Returns:
        Markdown-String mit Summary, Per-Agent- und Bucket-Tabellen je Run
        sowie (optional) der Gate-Sweep-Tabelle.
    """
    lines: list[str] = ["# Backtest-Report", ""]
    if not runs:
        lines.append("_Kein Backtest ausgeführt (keine Daten im Zeitraum?)._")
        lines.append("")
    if runs:
        lines.extend(
            [
                "## Summary",
                "",
                "| Szenario | final_equity | total_return | sharpe_ratio | max_drawdown "
                "| win_rate | total_trades | profit_factor | gate_pass_rate |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        lines.extend(_summary_row(label, result, extra) for label, result, extra in runs)
    for label, _result, extra in runs:
        lines.extend(["", f"## Szenario: {label}", ""])
        lines.extend(_per_agent_table(extra))
        lines.extend(["", *_bucket_table(extra.get("buckets", []))])
        lines.extend(["", *_exit_table(extra)])
    if sweep_rows:
        lines.extend(["", *_sweep_table(sweep_rows)])
    return "\n".join(lines) + "\n"


def _write_equity_csv(path: Path, result: BacktestResult, extra: dict[str, Any]) -> None:
    """Schreibt die Equity-Kurve als CSV (Initialzeile + je Handels-Bar)."""
    equity_curve: list[float] = result.metadata.get("equity_curve", [])
    warmup = int(extra.get("warmup_bars", result.config.warmup_bars))
    candles = result.candles
    rows: list[tuple[int, str, float]] = []
    if equity_curve:
        first_trading_ts = (
            candles[warmup].timestamp.isoformat()
            if warmup < len(candles)
            else (candles[0].timestamp.isoformat() if candles else "")
        )
        rows.append((0, first_trading_ts, equity_curve[0]))
        for i in range(warmup, min(len(candles), warmup + len(equity_curve) - 1)):
            rows.append((i - warmup + 1, candles[i].timestamp.isoformat(), equity_curve[i - warmup + 1]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "timestamp", "equity"])
        for index, timestamp, equity in rows:
            writer.writerow([index, timestamp, f"{equity:.4f}"])


def resolve_output_dir(outdir: Path) -> Path:
    """Liefert ``outdir`` (anglegt) oder — wenn das nicht schreibbar ist —
    ein schreibbares Temporär-Verzeichnis.

    Motivation: In Docker ist ``./backtest_reports`` ein Bind-Mount, dessen
    Host-Verzeichnis ggf. root-eigen ist, während der Container als
    nicht-privilegiertes ``appuser`` läuft. Dann schlägt ``mkdir``/Schreiben
    fehl; statt den gesamten (teuren) Backtest-Run zu verlieren, fallen wir
    auf ein Temp-Verzeichnis zurück und loggen eine Warnung.
    """
    try:
        outdir.mkdir(parents=True, exist_ok=True)
        probe = outdir / ".backtest_write_probe"
        probe.touch()
        probe.unlink()
        return outdir
    except OSError as exc:
        fallback = Path(tempfile.mkdtemp(prefix="backtest_reports_"))
        logger.warning(
            "Artefakt-Verzeichnis %s nicht schreibbar (%s) — verwende %s",
            outdir,
            exc,
            fallback,
        )
        return fallback


def write_artifacts(
    outdir: Path,
    label: str,
    result: BacktestResult,
    extra: dict[str, Any],
) -> dict[str, Path]:
    """Schreibt report.json, equity_curve.csv und evaluations.json.

    Args:
        outdir: Zielfverzeichnis (wird angelegt; bei fehlender Schreibrecht
            wird auf ein Temp-Verzeichnis umgeschaltet, s.
            ``_resolve_writable_dir``).
        label: Szenario-Label (landet in report.json).
        result: BacktestResult (Metriken, Metadata inkl. Equity-Kurve).
        extra: Analytics-Dict (gate, buckets, per_agent, evaluations, ...).

    Returns:
        Mapping Artefaktname → Pfad.
    """
    outdir = resolve_output_dir(outdir)
    report_path = outdir / "report.json"
    equity_path = outdir / "equity_curve.csv"
    evaluations_path = outdir / "evaluations.json"

    report = {
        "label": label,
        "config": {
            "symbol": result.config.symbol,
            "timeframe": result.config.timeframe,
            "initial_capital": result.config.initial_capital,
            "commission_rate": result.config.commission_rate,
            "slippage_bps": result.config.slippage_bps,
            "warmup_bars": result.config.warmup_bars,
        },
        "metrics": result.metrics,
        "metadata": {key: value for key, value in result.metadata.items() if key != "strategy"},
        "extra": extra,
    }
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _write_equity_csv(equity_path, result, extra)
    evaluations_path.write_text(
        json.dumps(extra.get("evaluations", []), indent=2, default=str), encoding="utf-8"
    )
    return {
        "report": report_path,
        "equity_curve": equity_path,
        "evaluations": evaluations_path,
    }
