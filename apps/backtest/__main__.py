"""CLI-Einstiegspunkt für den Backtest (``python -m apps.backtest``).

Führt die PRODUKTIONS-Entscheidungslogik des Demo-Traders
(ACTIVE-Ensemble → OrchestratorPipeline → Konsens → Konfidenz-Gate) auf
historischen ClickHouse-Kerzen aus — pro Szenario ein Backtest, optional
ein Gate-Sweep auf den Daten des letzten Szenarios (mit gecachtem Konsens,
ohne Pipeline-Rekomputation). Output: Markdown-Report auf der Konsole plus
Artefakte (report.json, equity_curve.csv, evaluations.json) pro Szenario.

Beispiele:
  python -m apps.backtest --instrument BTC/USDT --scenario crash-2021-05 --gate 0.3
  python -m apps.backtest --scenarios crash-2021-05,pump-2021-11,crash-2022-06,range-2022-03
  python -m apps.backtest --scenario full --resample 5m --sweep-gates 0.2,0.3,0.4,0.5,0.6,0.7

MLflow-Logging ist opt-in (Env ``MLFLOW_ENABLED=true``, Experiment
``backtest``); fehlende ClickHouse-Daten für ein Szenario führen zu einer
Warnung und werden übersprungen (nicht fatal).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from packages.backtesting.core import BacktestConfig
from packages.persistence.clickhouse.engine import (
    ClickHouseConfig,
    ClickHouseEngine,
    create_ch_engine,
)

from .agent_strategy import AgentEnsembleStrategy
from .ch_feed import ClickHouseDataFeed
from .mlflow_report import log_backtest_to_mlflow
from .report import render_markdown, resolve_output_dir, write_artifacts
from .runner import confidence_buckets, extra_metrics, gate_sweep, run_backtest
from .scenarios import parse_scenarios

logger = logging.getLogger(__name__)

DEFAULT_INSTRUMENT = "BTC/USDT"
DEFAULT_VENUE = "BINANCE_FUTURES"


def build_parser() -> argparse.ArgumentParser:
    """Baut den Argument-Parser für den Backtest-CLI."""
    parser = argparse.ArgumentParser(
        prog="apps.backtest",
        description="Backtest der Produktions-Agenten-Ensemble-Logik auf historischen Kerzen.",
    )
    parser.add_argument("--instrument", default=DEFAULT_INSTRUMENT, help="Handelspaar (Default: BTC/USDT)")
    parser.add_argument("--venue", default=DEFAULT_VENUE, help="Venue-Filter (Default: BINANCE_FUTURES)")
    parser.add_argument("--scenario", default=None, help="Einzelnes Szenario (Default: full)")
    parser.add_argument("--scenarios", default=None, help="Kommagetrennte Szenario-Liste")
    parser.add_argument("--from", dest="from_date", default=None, help="Explizites Startdatum (überschlägt Szenarien)")
    parser.add_argument("--to", dest="to_date", default=None, help="Explizites Endedatum (überschlägt Szenarien)")
    parser.add_argument("--gate", type=float, default=0.3, help="Konfidenz-Gate (Default: 0.3)")
    parser.add_argument("--sweep-gates", default=None, help="Kommagetrennte Gate-Liste für den Sweep")
    parser.add_argument("--candle-limit", type=int, default=200, help="Fenstergröße in Kerzen (Default: 200)")
    parser.add_argument("--min-candles", type=int, default=30, help="Mindestkerzen vor Evaluation (Default: 30)")
    parser.add_argument("--evaluate-every", type=int, default=5, help="Evaluation alle N Bars (Default: 5)")
    parser.add_argument("--resample", default=None, help="Resampling ('5m' für 1m→5m, sonst leer)")
    parser.add_argument("--horizon", default="15m", help="Analyse-Horizont (Default: 15m)")
    parser.add_argument("--trade-notional", type=float, default=2000.0, help="Nominal pro BUY (Default: 2000)")
    parser.add_argument("--initial-capital", type=float, default=100_000.0, help="Startkapital (Default: 100000)")
    parser.add_argument(
        "--stop-loss",
        type=float,
        default=None,
        help="Stop-Loss als Anteil unterhalb des Positions-Durchschnittspreises (z.B. 0.08 = 8 %%; Default: aus)",
    )
    parser.add_argument(
        "--max-holding-bars",
        type=int,
        default=None,
        help="Maximale Haltezeit in Bars, danach wird die Position geschlossen (Default: aus)",
    )
    parser.add_argument("--output", default="./backtest_reports", help="Artefakt-Verzeichnis (Default: ./backtest_reports)")
    parser.add_argument("--ch-host", default=None, help="ClickHouse-Host (Env CH_HOST, Default: clickhouse)")
    parser.add_argument("--ch-port", type=int, default=None, help="ClickHouse-Port (Env CH_PORT, Default: 8123)")
    parser.add_argument("--ch-db", default=None, help="ClickHouse-DB (Env CH_DB, Default: trading_events)")
    parser.add_argument("--ch-password", default=None, help="ClickHouse-Passwort (Env CH_PASSWORD)")
    return parser


def parse_bound(value: str | None) -> datetime | None:
    """Parst ein CLI-Datum (-YYYY-MM-DD → 00:00 UTC; ISO-Datetime wird beibehalten)."""
    if value is None:
        return None
    try:
        day = date.fromisoformat(value)
    except ValueError:
        return datetime.fromisoformat(value)
    return datetime(day.year, day.month, day.day, tzinfo=UTC)


def _iso(value: datetime) -> str:
    """Datetime → ISO-String (für JSON-Artefakte)."""
    return value.isoformat()


def build_ch_engine(args: argparse.Namespace) -> ClickHouseEngine:
    """Erzeugt die ClickHouse-Engine aus CLI-Argumenten bzw. Env-Defaults."""
    return create_ch_engine(
        ClickHouseConfig(
            host=args.ch_host if args.ch_host is not None else os.environ.get("CH_HOST", "clickhouse"),
            port=args.ch_port if args.ch_port is not None else int(os.environ.get("CH_PORT", "8123")),
            database=args.ch_db if args.ch_db is not None else os.environ.get("CH_DB", "trading_events"),
            user="orchestra",
            password=args.ch_password if args.ch_password is not None else os.environ.get("CH_PASSWORD", ""),
        )
    )


def make_strategy(args: argparse.Namespace) -> AgentEnsembleStrategy:
    """Erzeugt die Agent-Ensemble-Strategie aus den CLI-Argumenten."""
    return AgentEnsembleStrategy(
        instrument=args.instrument,
        horizon=args.horizon,
        candle_limit=args.candle_limit,
        min_candles=args.min_candles,
        evaluate_every=args.evaluate_every,
        min_confidence=args.gate,
        trade_notional=args.trade_notional,
        initial_capital=args.initial_capital,
    )


def backtest_config(args: argparse.Namespace) -> BacktestConfig:
    """BacktestConfig für den Run (Symbol/Timeframe; Defaults aus runner)."""
    return BacktestConfig(
        symbol=args.instrument,
        timeframe="5m" if args.resample == "5m" else "1m",
        stop_loss_pct=args.stop_loss,
        max_holding_bars=args.max_holding_bars,
    )


def run_scenario(
    args: argparse.Namespace,
    ch_engine: ClickHouseEngine,
    label: str,
    start: datetime | date | None,
    end: datetime | date | None,
) -> tuple[str, Any, Any, dict[str, Any]] | None:
    """Führt ein Szenario aus (Feed → Backtest → Analytics → Artefakte).

    Returns:
        (label, Feed, BacktestResult, extra) oder None, wenn keine Daten
        verfügbar sind (Warnung, nicht fatal).
    """
    feed = ClickHouseDataFeed(
        ch_engine,
        args.instrument,
        venue=args.venue,
        start=start,
        end=end,
        resample=args.resample,
    )
    try:
        candles = feed.get_candles()
    except Exception as exc:
        logger.warning(
            "Szenario %s übersprungen — Kerzen nicht abrufbar (%s → %s): %s", label, start, end, exc
        )
        return None
    if not candles:
        logger.warning("Szenario %s übersprungen — keine Kerzen im Zeitraum (%s → %s)", label, start, end)
        return None
    if len(candles) <= args.min_candles:
        logger.warning(
            "Szenario %s übersprungen — nur %d Kerzen verfügbar (min. %d erforderlich)",
            label,
            len(candles),
            args.min_candles,
        )
        return None

    strategy = make_strategy(args)
    result = run_backtest(feed, lambda: strategy, config=backtest_config(args), label=label)
    extra = extra_metrics(result, strategy)
    extra["buckets"] = confidence_buckets(result, strategy)
    extra["evaluations"] = strategy.evaluations_to_dicts()
    extra["strategy"] = strategy.to_dict()
    extra["warmup_bars"] = args.candle_limit
    extra["round_trips"] = [
        {
            "entry_time": _iso(rt["entry_time"]),
            "exit_time": _iso(rt["exit_time"]),
            "entry_price": rt["entry_price"],
            "exit_price": rt["exit_price"],
            "quantity": rt["quantity"],
            "pnl": rt["pnl"],
            "holding_bars": rt["holding_bars"],
            "exit_reason": rt["exit_reason"],
        }
        for rt in result.metadata.get("round_trips", [])
    ]
    for pos in result.metadata.get("open_positions", []):
        extra["round_trips"].append(
            {
                "entry_time": None,
                "exit_time": None,
                "entry_price": pos["avg_price"],
                "exit_price": None,
                "quantity": pos["quantity"],
                "pnl": pos["unrealized_pnl"],
                "holding_bars": pos["holding_bars"],
                "exit_reason": "open",
            }
        )
    extra["params"] = {
        "instrument": args.instrument,
        "venue": args.venue,
        "scenario": label,
        "timeframe": "5m" if args.resample == "5m" else "1m",
        "candle_limit": args.candle_limit,
        "evaluate_every": args.evaluate_every,
        "min_confidence": args.gate,
        "stop_loss_pct": args.stop_loss,
        "max_holding_bars": args.max_holding_bars,
        "trade_notional": args.trade_notional,
        "initial_capital": args.initial_capital,
        "resample": args.resample or "none",
        "data_start": candles[0].timestamp.isoformat(),
        "data_end": candles[-1].timestamp.isoformat(),
    }
    outdir = Path(args.output) / label
    paths = write_artifacts(outdir, label, result, extra)
    logger.info(
        "Szenario %s fertig: %d Kerzen, %d Evaluations, gate=%.2f → %s",
        label,
        len(candles),
        extra["n_evaluations"],
        args.gate,
        {name: str(path) for name, path in paths.items()},
    )
    return label, feed, result, extra


def run_sweep(args: argparse.Namespace, last: tuple[str, Any, AgentEnsembleStrategy, dict[str, Any]]) -> list[dict[str, Any]]:
    """Führt den Gate-Sweep auf den Daten des letzten Szenarios aus.

    Der bereits gelaufene warm_strategy überträgt seinen gefüllten
    consensus_cache — die Pipeline/Agenten werden nicht erneut ausgeführt.
    """
    label, feed, warm_strategy, _extra = last
    gates = [float(item) for item in args.sweep_gates.split(",") if item.strip()]
    rows = gate_sweep(
        feed,
        lambda: make_strategy(args),
        gates,
        config=backtest_config(args),
        warm_strategy=warm_strategy,
    )
    sweep_path = Path(args.output) / label / "sweep.json"
    sweep_path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    logger.info("Gate-Sweep für %s fertig: %d Gates → %s", label, len(gates), sweep_path)
    return rows


def main(argv: list[str] | None = None) -> int:
    """CLI-Hauptfunktion. Returns: 0 (Erfolg), 2 (ungültige Argumente)."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    try:
        if args.from_date is not None or args.to_date is not None:
            scenarios = [("custom", parse_bound(args.from_date), parse_bound(args.to_date))]
        else:
            scenarios = parse_scenarios(args.scenarios or args.scenario or "full")
    except ValueError as exc:
        logger.error("Ungültige Szenario-Angabe: %s", exc)
        return 2

    ch_engine = build_ch_engine(args)
    # Einmalig auflösen: root-eigener Bind-Mount (Docker) → Temp-Fallback,
    # damit ein lauffähiger Backtest nicht am Artefakt-Schreiben scheitert.
    args.output = str(resolve_output_dir(Path(args.output)))
    runs: list[tuple[str, Any, dict[str, Any]]] = []
    last: tuple[str, Any, AgentEnsembleStrategy, dict[str, Any]] | None = None
    for label, start, end in scenarios:
        outcome = run_scenario(args, ch_engine, label, start, end)
        if outcome is None:
            continue
        outcome_label, feed, result, extra = outcome
        runs.append((outcome_label, result, extra))
        strategy = result.metadata["strategy"]
        last = (outcome_label, feed, strategy, extra)

    sweep_rows: list[dict[str, Any]] = []
    if args.sweep_gates:
        if last is None:
            logger.warning("--sweep-gates übersprungen — kein Szenario mit Daten ausgeführt")
        else:
            sweep_rows = run_sweep(args, last)

    if runs:
        print(render_markdown(runs, sweep_rows if sweep_rows else None))

    for label, result, extra in runs:
        log_backtest_to_mlflow(label, result, extra, backtest_config(args))
    if sweep_rows and last is not None:
        label, _feed, _strategy, extra = last
        sweep_extra = dict(extra)
        sweep_extra["sweep"] = sweep_rows
        sweep_extra.pop("evaluations", None)
        log_backtest_to_mlflow(
            f"{label}-sweep",
            None,
            sweep_extra,
            backtest_config(args),
            extra_tags={"sweep": "true"},
        )
    if not runs:
        logger.warning("Kein Szenario ausgeführt (keine Daten?) — Report entfällt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
