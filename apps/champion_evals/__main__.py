"""CLI: reale pro-Agent-Evaluationsdaten aus ClickHouse-Kerzen erzeugen.

Lädt ``candles_history`` (``apps.backtest.ch_feed``), führt das ACTIVE-
4-Agenten-Ensemble rückwärts darüber, bewertet pro Agent Kalibrierung vs.
OOS (Brier), Stabilität (Hit-Rate) und den marginalen Beitrag (Leave-One-Out)
und schreibt das ``champion_evals.json``-Artefakt für den Orchestrator-Feed.

Mehrere Instrumente (``--instrument BTC/USDT,ETH/USDT``) werden auf einer
gemeinsamen Zeitachse gepoolt (``score_window`` sortiert die Samples nach
``as_of``); das Artefakt bleibt pro Agent (keine pro-Instrument-Aufspaltung).
``--days N`` setzt ein rollierendes Fenster (letzte N Tage bis heute) statt
expliziter ``--start``/``--end``; ``--loop N`` wiederholt den Lauf alle N
Sekunden (Dauerbetrieb als Scheduled-Service).

Beispiele (Docker-Compose-Profil on-demand):
    docker compose --profile on-demand run --rm backtest python -m apps.champion_evals \
      --instrument BTC/USDT --start 2026-03-02 --end 2026-09-02 --resample 5m \
      --output /app/backtest_reports/champion_evals.json
    docker compose run --rm champion-evals   # täglicher Lauf, 180 Tage, BTC+ETH
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from packages.persistence.clickhouse.engine import (
    ClickHouseConfig,
    ClickHouseEngine,
    create_ch_engine,
)

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pro-Agent OOS-Evaluationsdaten aus ClickHouse-Kerzen erzeugen.")
    parser.add_argument("--instrument", default="BTC/USDT", help="Instrument oder Komma-Liste (z. B. BTC/USDT,ETH/USDT)")
    parser.add_argument("--venue", default=None, help="Venue-Filter (Default Env CANDLE_VENUE / BINANCE_FUTURES)")
    parser.add_argument("--horizon", default="15m", help="Analyse-Horizont der Agenten")
    parser.add_argument("--days", type=int, default=None, help="Rollierendes Fenster: letzte N Tage bis heute (überschreibt --start/--end)")
    parser.add_argument("--start", default=None, help="Untere Zeitgrenze (ISO/Datum, UTC)")
    parser.add_argument("--end", default=None, help="Obere Zeitgrenze (ISO/Datum, UTC)")
    parser.add_argument("--resample", default="5m", help="Kerzen-Auflösung (Default 5m; None = 1m)")
    parser.add_argument("--candle-limit", type=int, default=200, help="Fenstergröße in Kerzen")
    parser.add_argument("--min-candles", type=int, default=30, help="Warmup: min. Kerzen vor der ersten Bewertung")
    parser.add_argument("--evaluate-every", type=int, default=5, help="Bewertung alle N Kerzen")
    parser.add_argument("--horizon-bars", type=int, default=3, help="Kerzen für die Outcome-Realisierung")
    parser.add_argument("--calibration-ratio", type=float, default=0.5, help="Anteil früherer Zeitachse = Kalibrierung")
    parser.add_argument("--up-threshold", type=float, default=0.01, help="UP-Schwelle für das Real-Outcome")
    parser.add_argument("--down-threshold", type=float, default=-0.01, help="DOWN-Schwelle für das Real-Outcome")
    parser.add_argument("--min-samples", type=int, default=20, help="Min. OOS-Samples pro Agent, sonst weggelassen")
    parser.add_argument("--loop", type=int, default=None, help="Dauerbetrieb: Lauf alle N Sekunden wiederholen (Default = Einzellauf)")
    parser.add_argument("--version", default="current", help="Versions-Label für das Artefakt")
    parser.add_argument("--output", required=True, help="Zielpfad des JSON-Artefakts")
    parser.add_argument("--ch-host", default=None, help="ClickHouse-Host (Env CH_HOST, Default clickhouse)")
    parser.add_argument("--ch-port", type=int, default=None, help="ClickHouse-Port (Env CH_PORT, Default 8123)")
    parser.add_argument("--ch-db", default=None, help="ClickHouse-DB (Env CH_DB, Default trading_events)")
    parser.add_argument("--ch-password", default=None, help="ClickHouse-Passwort (Env CH_PASSWORD)")
    parser.add_argument("--quiet", action="store_true", help="nur ERROR-Logs")
    return parser


def _ch_engine(args: argparse.Namespace) -> ClickHouseEngine:
    return create_ch_engine(
        ClickHouseConfig(
            host=args.ch_host or os.environ.get("CH_HOST", "clickhouse"),
            port=args.ch_port if args.ch_port is not None else int(os.environ.get("CH_PORT", "8123")),
            database=args.ch_db or os.environ.get("CH_DB", "trading_events"),
            user="orchestra",
            password=args.ch_password or os.environ.get("CH_PASSWORD", ""),
        )
    )


def _print_summary(scored: dict, args: argparse.Namespace, path: Path) -> None:
    print(f"\nChampion-Evaluations: {args.instrument} horizon={args.horizon} resample={args.resample}")
    print(f"{'Agent':22} {'cal':>4} {'oos':>4} {'OOS-Brier':>11} {'OOS-Hit':>9} {'LOO-Marg':>10}")
    for agent_id, m in sorted(scored.items()):
        print(
            f"{agent_id:22} {m.cal_samples:>4} {m.oos_samples:>4} "
            f"{m.oos_brier:>11.4f} {m.oos_stability:>9.3f} {m.oos_marginal:>10.4f}"
        )
    print(f"Artefakt: {path} ({len(scored)} Agenten)")


def _window(args: argparse.Namespace) -> tuple[str | None, str | None]:
    """Zeitfenster: ``--days N`` (rollierend bis heute) oder explizit ``--start``/``--end``."""
    if args.days is not None:
        today = datetime.now(UTC).date()
        return str(today - timedelta(days=args.days)), str(today)
    return args.start, args.end


def _run_once(args: argparse.Namespace) -> int:
    """Ein kompletter Evaluationslauf über alle angebenen Instrumente."""
    from apps.backtest.ch_feed import ClickHouseDataFeed
    from apps.champion_evals.score import EvalSample, replay_ensemble, score_window, write_artifact
    from packages.validation.target_variables import TargetConfig

    instruments = tuple(item.strip() for item in args.instrument.split(",") if item.strip())
    venue = args.venue or os.environ.get("CANDLE_VENUE", "BINANCE_FUTURES")
    engine = _ch_engine(args)
    start, end = _window(args)

    # Samples aller Instrumente auf einer gemeinsamen Zeitachse poolen
    # (score_window sortiert nach as_of; ein Instrument mit zu wenigen
    # Kerzen wird übersprungen, nicht der gesamte Lauf gescheitert).
    samples: list[EvalSample] = []
    for instrument in instruments:
        feed = ClickHouseDataFeed(engine, instrument, venue=venue, start=start, end=end, resample=args.resample)
        candles = feed.get_candles()
        if len(candles) < args.min_candles + args.horizon_bars:
            logger.error(
                "Instrument %s übersprungen: nur %d Kerzen (min-candles=%d + horizon-bars=%d)",
                instrument,
                len(candles),
                args.min_candles,
                args.horizon_bars,
            )
            continue
        target = TargetConfig(up_threshold=args.up_threshold, down_threshold=args.down_threshold, horizon=args.horizon)
        samples.extend(
            replay_ensemble(
                candles,
                instrument,
                args.horizon,
                candle_limit=args.candle_limit,
                min_candles=args.min_candles,
                evaluate_every=args.evaluate_every,
                horizon_bars=args.horizon_bars,
                target_config=target,
            )
        )
    if not samples:
        logger.error("Keine Bewertungsschritte erzeugt (zu wenige Kerzen oder Warmup zu groß)")
        return 1

    scored = {a: m for a, m in score_window(samples, calibration_ratio=args.calibration_ratio).items() if m.oos_samples >= args.min_samples}
    if not scored:
        logger.error("Kein Agent erfüllt --min-samples=%d (Fenster zu klein?)", args.min_samples)
        return 1

    path = write_artifact(args.output, scored, version=args.version)
    _print_summary(scored, args, path)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO)
    if not args.loop:
        return _run_once(args)
    logger.info("Champion-Evals-Dauerbetrieb: Lauf alle %d s", args.loop)
    while True:
        try:
            _run_once(args)
        except Exception:
            logger.exception("Champion-Evals-Lauf fehlgeschlagen (nächster Lauf in %d s)", args.loop)
        time.sleep(args.loop)


if __name__ == "__main__":
    sys.exit(main())
