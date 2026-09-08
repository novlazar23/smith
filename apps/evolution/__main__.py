"""CLI-Einstiegspunkt der Evolutions-Pipeline (``python -m apps.evolution``).

Kommandos:

- ``--digest``: Evidenz-Digest auf die Konsole.
- ``--status``: Kurzbericht (Hypothesen, Grab, Registry, letzter Zyklus).
- ``--cycle``: einen Zyklus fahren (``--with-llm`` schaltet die
  Persona-Diskussion ein; Budget via ``--max-proposals``/``--max-runs``/
  ``--max-minutes``).
- ``--sweep-family NAME --sweep "k=v1,v2,..." [--sweep "k2=..."]``:
  Grid-Varianten preregistrieren (ohne Testen; der nächste ``--cycle``
  testet sie) — kombiniert mit ``--cycle`` testet derselbe Aufruf.
- ``--propose DATEI``: Hypothese aus einer JSON-Datei preregistrieren
  (Schema = ``models.Proposal``; für Mechanismen inkl. ``variant.code``).
- ``--export DIR``: State + Mechanismus-Code + Registry-Patch exportieren.
- ``--timesfm-spike``: optionaler TimesFM-Research-Spike (Feature-Generierung,
  keine Promotion, kein Jail, kein Live-Entscheider; ``--timesfm-fake`` für
  synthetischen Provider).

State-Dir: Env ``EVOLUTION_STATE_DIR`` (Container-Default:
``/app/backtest_reports/evolution``). Kerzen-Lader: ClickHouse
(``CH_HOST``/``CH_PORT``/``CH_DB``, wie beim Backtest-Service).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from packages.backtesting.core import Candle
from packages.persistence.clickhouse.engine import (
    ClickHouseConfig,
    ClickHouseEngine,
    create_ch_engine,
)

from .digest import build_digest, fetch_live_paper
from .evaluate import FeedFactory
from .models import Proposal
from .state import EvolutionStore, default_state_dir
from .sweep import propose_grid, sweep_report

logger = logging.getLogger(__name__)

DEFAULT_INSTRUMENTS = "BTC/USDT,ETH/USDT"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="apps.evolution",
        description="Evolutions-Pipeline: autonome Strategie-Entwicklung mit Data-Snooping-Firewall.",
    )
    parser.add_argument("--digest", action="store_true", help="Evidenz-Digest auf die Konsole")
    parser.add_argument("--status", action="store_true", help="Kurzbericht des Evolutions-States")
    parser.add_argument("--cycle", action="store_true", help="einen Evolutions-Zyklus fahren")
    parser.add_argument("--with-llm", action="store_true", help="Persona-Diskussion (LLM) im Zyklus")
    parser.add_argument("--model", default=None, help="Modell-Override für die Personas")
    parser.add_argument("--max-proposals", type=int, default=3, help="Max. neue Vorschläge pro Zyklus (Default: 3)")
    parser.add_argument("--max-runs", type=int, default=40, help="Max. Engine-Runs pro Zyklus (Default: 40)")
    parser.add_argument("--max-minutes", type=float, default=120.0, help="Wanduhr-Budget in Minuten (Default: 120)")
    parser.add_argument(
        "--sweep-family",
        default=None,
        help="Zoo-Strategie für den Grid-Sweep (mit --sweep)",
    )
    parser.add_argument(
        "--sweep",
        action="append",
        default=None,
        metavar="K=V1,V2",
        help="Grid-Achse, wiederholbar (z.B. --sweep buy_below=15,20,25 --sweep sell_above=70,80)",
    )
    parser.add_argument("--propose", default=None, metavar="DATEI", help="Hypothese aus JSON-Datei preregistrieren")
    parser.add_argument("--export", default=None, metavar="DIR", help="State exportieren nach DIR")
    parser.add_argument(
        "--instruments",
        default=DEFAULT_INSTRUMENTS,
        help=f"Instrumente für neue Testpläne (Default: {DEFAULT_INSTRUMENTS})",
    )
    parser.add_argument(
        "--state-dir",
        default=None,
        help="State-Dir (Default: Env EVOLUTION_STATE_DIR oder auto)",
    )
    parser.add_argument("--ch-host", default=None, help="ClickHouse-Host (Env CH_HOST, Default: clickhouse)")
    parser.add_argument("--ch-port", type=int, default=None, help="ClickHouse-Port (Env CH_PORT, Default: 8123)")
    parser.add_argument("--ch-db", default=None, help="ClickHouse-DB (Env CH_DB, Default: trading_events)")
    parser.add_argument("--ch-password", default=None, help="ClickHouse-Passwort (Env CH_PASSWORD)")
    parser.add_argument(
        "--timesfm-spike",
        action="store_true",
        help="Optionaler TimesFM-Research-Spike (Feature-Generierung, keine Promotion)",
    )
    parser.add_argument("--timesfm-instrument", default="BTC/USDT", help="Instrument für den TimesFM-Spike")
    parser.add_argument("--timesfm-start", default=None, help="Start (ISO) für den TimesFM-Spike")
    parser.add_argument("--timesfm-end", default=None, help="Ende (ISO) für den TimesFM-Spike")
    parser.add_argument("--timesfm-resample", default="5m", help="Kerzen-Auflösung für den TimesFM-Spike (Default: 5m)")
    parser.add_argument("--timesfm-context", type=int, default=512, help="TimesFM-Context in Kerzen (Default: 512)")
    parser.add_argument("--timesfm-horizon", type=int, default=288, help="TimesFM-Horizont in Kerzen (Default: 288 = 24h bei 5m)")
    parser.add_argument("--timesfm-step", type=int, default=288, help="Abstand zwischen TimesFM-Features in Kerzen (Default: 288)")
    parser.add_argument("--timesfm-out", default=None, help="Output-Dir für den TimesFM-Spike (Default: State-Dir/timesfm/INSTRUMENT)")
    parser.add_argument("--timesfm-fake", action="store_true", help="Synthetischen Fake-Provider nutzen (kein echtes Modell)")
    parser.add_argument("--quiet", action="store_true", help="nur ERROR-Logs")
    return parser


def make_store(args: argparse.Namespace) -> EvolutionStore:
    root = Path(args.state_dir) if args.state_dir else default_state_dir()
    return EvolutionStore(root)


def ch_feed_factory(args: argparse.Namespace) -> FeedFactory:
    """CH-basierter Feed-Lader: (instrument, start, end, resample) → Kerzen."""
    from apps.backtest.ch_feed import ClickHouseDataFeed

    engine: ClickHouseEngine | None = None

    def feed_factory(instrument: str, start: str | None, end: str | None, resample: str | None) -> list[Candle]:
        nonlocal engine
        if engine is None:
            engine = create_ch_engine(
                ClickHouseConfig(
                    host=args.ch_host if args.ch_host is not None else os.environ.get("CH_HOST", "clickhouse"),
                    port=args.ch_port if args.ch_port is not None else int(os.environ.get("CH_PORT", "8123")),
                    database=args.ch_db if args.ch_db is not None else os.environ.get("CH_DB", "trading_events"),
                    user="orchestra",
                    password=args.ch_password if args.ch_password is not None else os.environ.get("CH_PASSWORD", ""),
                )
            )
        assert engine is not None
        feed = ClickHouseDataFeed(
            engine,
            instrument,
            venue=os.environ.get("CANDLE_VENUE", "BINANCE_FUTURES"),
            start=start,
            end=end,
            resample=resample,
        )
        return feed.get_candles()

    return feed_factory


def parse_sweep_axes(raw_axes: list[str] | None) -> dict[str, list[float]]:
    """``["k=v1,v2", ...]`` → ``{"k": [v1, v2]}`` (Fehler → argparse-Fehler)."""
    grid: dict[str, list[float]] = {}
    for raw in raw_axes or []:
        key, sep, values = raw.partition("=")
        if not sep:
            raise ValueError(f"--sweep-Achse {raw!r} im Format K=V1,V2 erwartet")
        values = [v.strip() for v in values.split(",") if v.strip()]
        try:
            grid[key.strip()] = [float(v) for v in values]
        except ValueError as exc:
            raise ValueError(f"--sweep-Achse {raw!r}: {exc}") from None
    return grid


def load_proposal_file(path: str, instruments: str) -> Proposal:
    """Liest einen Vorschlag aus einer JSON-Datei (Schema = models.Proposal)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    plan = data.get("test_plan")
    if not isinstance(plan, dict):
        plan = {}
        data["test_plan"] = plan
    if "instruments" not in plan:
        plan["instruments"] = tuple(i.strip() for i in instruments.split(",") if i.strip())
    return Proposal.model_validate(data)


def run_digest(store: EvolutionStore) -> int:
    print(build_digest(store, live=fetch_live_paper()))
    return 0


def run_status(store: EvolutionStore) -> int:
    print(json.dumps(sweep_report(store), ensure_ascii=False, indent=2, default=str))
    return 0


def run_timesfm_spike_cli(args: argparse.Namespace) -> int:
    from packages.forecasting.timesfm import FakeProvider, load_timesfm_provider

    from .timesfm_spike import run_timesfm_spike, safe_instrument

    provider = FakeProvider() if args.timesfm_fake else load_timesfm_provider()
    candles = ch_feed_factory(args)(
        args.timesfm_instrument,
        args.timesfm_start,
        args.timesfm_end,
        args.timesfm_resample,
    )
    root = Path(args.state_dir) if args.state_dir else default_state_dir()
    out_dir = Path(args.timesfm_out) if args.timesfm_out else root / "timesfm" / safe_instrument(args.timesfm_instrument)
    report = run_timesfm_spike(
        candles,
        instrument=args.timesfm_instrument,
        provider=provider,
        horizon=args.timesfm_horizon,
        context=args.timesfm_context,
        step=args.timesfm_step,
        out_dir=out_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0


def run_cycle_cli(args: argparse.Namespace, store: EvolutionStore) -> int:
    from .cycle import CycleBudget, run_cycle

    proposals = None
    if args.propose:
        instruments = args.instruments
        proposals = [load_proposal_file(args.propose, instruments)]
    if args.sweep_family and args.sweep:
        grid = parse_sweep_axes(args.sweep)
        new, skipped = propose_grid(
            store,
            args.sweep_family,
            grid,
            source="cli:sweep",
        )
        for reason in skipped:
            logger.info("Sweep übersprungen: %s", reason)
        if new:
            logger.info("Sweep: %d Varianten preregistriert", len(new))
        if not args.cycle:
            print(f"{len(new)} neue Varianten preregistriert (nächster --cycle testet sie)")
            return 0

    budget = CycleBudget(
        max_proposals=args.max_proposals,
        max_runs=args.max_runs,
        max_minutes=args.max_minutes,
    )
    export_dir = Path(args.export) if args.export else _auto_export_dir()
    summary = run_cycle(
        store,
        ch_feed_factory(args),
        with_llm=args.with_llm,
        model=args.model,
        proposals=proposals,
        budget=budget,
        export_dir=export_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


def _auto_export_dir() -> Path | None:
    """Auto-Export auf das backtest_reports-Volume (nur im Container)."""
    if os.environ.get("EVOLUTION_EXPORT_DIR"):
        return Path(os.environ["EVOLUTION_EXPORT_DIR"])
    candidate = Path("/app/backtest_reports/evolution_export")
    if candidate.parent.is_dir():
        return candidate
    return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        if args.timesfm_spike:
            return run_timesfm_spike_cli(args)
        store = make_store(args)
        if args.digest:
            return run_digest(store)
        if args.status:
            return run_status(store)
        if args.cycle or args.propose or (args.sweep_family and args.sweep) or args.export:
            if args.export and not (args.cycle or args.propose or args.sweep_family):
                from .export import export_state

                written = export_state(store, Path(args.export))
                print(json.dumps([str(p) for p in written], ensure_ascii=False, indent=2))
                return 0
            return run_cycle_cli(args, store)
        parser.print_help()
        return 2
    except ValueError as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    sys.exit(main())
