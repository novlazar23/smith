"""Tests für Markdown-Report und Artefakt-Schreibung."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from apps.backtest.report import render_markdown, resolve_output_dir, write_artifacts
from packages.backtesting.core import BacktestConfig, BacktestResult
from tests.unit.test_backtest.conftest import BTC, make_candles


def make_result(
    final_equity: float = 101_000.0,
    total_return: float = 1.0,
    n_candles: int = 0,
) -> BacktestResult:
    """Minimaler BacktestResult (Manuell, kein Engine-Run)."""
    candles = make_candles(n_candles, step=500.0) if n_candles else []
    return BacktestResult(
        config=BacktestConfig(symbol=BTC, warmup_bars=0),
        candles=candles,
        snapshots=[],
        trades=[],
        metrics={
            "total_return_pct": total_return,
            "sharpe_ratio": 0.75,
            "sortino_ratio": 0.6,
            "max_drawdown_pct": 2.0,
            "win_rate_pct": 50.0,
            "profit_factor": 1.25,
            "total_trades": 4,
        },
        metadata={
            "final_equity": final_equity,
            "equity_curve": [100_000.0, 100_500.0, 101_000.0],
        },
    )


def make_extra(label: str) -> dict[str, Any]:
    return {
        "gate": 0.3,
        "gate_pass_rate": 0.65,
        "n_evaluations": 40,
        "decision_distribution": {"LONG_BIAS": 25, "NO_TRADE": 15},
        "mean_confidence": 0.55,
        "per_agent": {
            "anomaly": {"evaluations": 40, "mean_confidence": 0.6, "directions": {"LONG": 30, "RANGE": 10}},
            "chart": {"evaluations": 40, "mean_confidence": 0.4, "directions": {"SHORT": 12}},
        },
        "buckets": [
            {"bucket": "[0.3,0.4)", "low": 0.3, "high": 0.4, "n_evaluations": 10, "n_trades": 4,
             "win_rate": 0.5, "avg_pnl": 12.5},
            {"bucket": "[0.8,1.0]", "low": 0.8, "high": 1.01, "n_evaluations": 5, "n_trades": 2,
             "win_rate": None, "avg_pnl": None},
        ],
        "evaluations": [{"timestamp": "2021-05-15T00:05:00+00:00", "decision": "LONG_BIAS",
                         "confidence": 0.55, "per_agent": {}, "signal_emitted": True}],
        "warmup_bars": 0,
    }


def make_sweep_rows() -> list[dict[str, Any]]:
    return [
        {"gate": 0.2, "trades": 30, "win_rate": 0.4, "total_return_pct": 2.0,
         "sharpe_ratio": 0.8, "max_drawdown_pct": 3.0, "gate_pass_rate": 1.0},
        {"gate": 0.5, "trades": 10, "win_rate": None, "total_return_pct": 1.0,
         "sharpe_ratio": 0.4, "max_drawdown_pct": 1.5, "gate_pass_rate": 0.5},
    ]


class TestRenderMarkdown:
    """Der Report enthält alle Szenarien, Metriken und Tabellen."""

    def test_contains_labels_and_metric_names(self) -> None:
        runs = [
            ("crash-2021-05", make_result(101_000.0, 1.0), make_extra("crash-2021-05")),
            ("pump-2021-11", make_result(99_000.0, -1.0), make_extra("pump-2021-11")),
        ]
        markdown = render_markdown(runs, make_sweep_rows())
        for fragment in (
            "crash-2021-05",
            "pump-2021-11",
            "final_equity",
            "total_return",
            "sharpe_ratio",
            "max_drawdown",
            "win_rate",
            "total_trades",
            "profit_factor",
            "gate_pass_rate",
            "Per-Agent",
            "Confidence-Buckets",
            "Gate-Sweep",
        ):
            assert fragment in markdown
        # beide Szenarien landen als Summary-Zeile
        assert markdown.count("| crash-2021-05 ") == 1
        assert markdown.count("| pump-2021-11 ") == 1
        # Per-Agent-Tabelleninhalte
        assert "anomaly" in markdown
        assert "chart" in markdown
        # Bucket-Tabelle (inkl. None-Formatting)
        assert "[0.3,0.4)" in markdown
        assert "[0.8,1.0]" in markdown

    def test_comparison_table_across_scenarios(self) -> None:
        runs = [
            ("a", make_result(101_000.0, 1.0), make_extra("a")),
            ("b", make_result(102_000.0, 2.0), make_extra("b")),
        ]
        markdown = render_markdown(runs)
        summary = markdown.split("## Summary")[1].split("## Szenario")[0]
        assert "| a " in summary
        assert "| b " in summary
        assert "101000.00" in summary
        assert "102000.00" in summary

    def test_sweep_table_values(self) -> None:
        runs = [("a", make_result(), make_extra("a"))]
        markdown = render_markdown(runs, make_sweep_rows())
        sweep = markdown.split("## Gate-Sweep")[1]
        assert "0.20" in sweep
        assert "0.50" in sweep
        assert "| 30 |" in sweep
        assert "| 10 |" in sweep

    def test_empty_runs_placeholder(self) -> None:
        markdown = render_markdown([])
        assert "Kein Backtest ausgeführt" in markdown


class TestWriteArtifacts:
    """write_artifacts legt die drei Dateien an; JSON round-trippt."""

    def test_creates_three_files(self, tmp_path: Path) -> None:
        paths = write_artifacts(tmp_path / "crash-2021-05", "crash-2021-05", make_result(), make_extra("x"))
        assert set(paths) == {"report", "equity_curve", "evaluations"}
        for path in paths.values():
            assert path.is_file()
        assert paths["report"] == tmp_path / "crash-2021-05" / "report.json"

    def test_report_json_round_trips(self, tmp_path: Path) -> None:
        result = make_result()
        extra = make_extra("x")
        paths = write_artifacts(tmp_path / "x", "x", result, extra)
        loaded = json.loads(paths["report"].read_text(encoding="utf-8"))
        assert loaded["label"] == "x"
        assert loaded["metrics"]["total_return_pct"] == 1.0
        assert loaded["metadata"]["final_equity"] == 101_000.0
        assert loaded["extra"]["buckets"][0]["bucket"] == "[0.3,0.4)"
        assert loaded["config"]["symbol"] == BTC
        # Strategie-Objekt darf nicht im Report landen
        assert "strategy" not in loaded["metadata"]

    def test_equity_csv_rows(self, tmp_path: Path) -> None:
        result = make_result(n_candles=3)
        paths = write_artifacts(tmp_path / "x", "x", result, make_extra("x"))
        lines = paths["equity_curve"].read_text(encoding="utf-8").strip().splitlines()
        assert lines[0] == "index,timestamp,equity"
        # 3 Equity-Werte → 3 Zeilen (Initialzeile + 2 Bars)
        assert len(lines) == 4
        assert "100000.0000" in lines[1]
        assert "101000.0000" in lines[3]

    def test_evaluations_json_round_trips(self, tmp_path: Path) -> None:
        extra = make_extra("x")
        paths = write_artifacts(tmp_path / "x", "x", make_result(), extra)
        loaded = json.loads(paths["evaluations"].read_text(encoding="utf-8"))
        assert loaded[0]["decision"] == "LONG_BIAS"
        assert loaded[0]["confidence"] == 0.55


class TestResolveOutputDir:
    """``resolve_output_dir``: lauffähiger Backtest scheitert nie am Artefakt-
    Schreiben (z. B. root-eigener Docker-Bind-Mount, nicht-privilegierter
    Container-User)."""

    def test_writable_dir_returned_as_is(self, tmp_path: Path) -> None:
        target = tmp_path / "reports"
        assert resolve_output_dir(target) == target
        assert target.is_dir()

    def test_unwritable_parent_falls_back_to_temp(self, tmp_path: Path) -> None:
        # Parent ist eine DATEI → mkdir schlägt fehl (NotADirectoryError),
        # auch bei root → deterministischer Fallback-Pfad.
        blocker = tmp_path / "blocker"
        blocker.write_text("x", encoding="utf-8")
        fallback = resolve_output_dir(blocker / "reports")
        assert fallback != blocker / "reports"
        assert fallback.is_dir()

    def test_write_artifacts_survives_unwritable_dir(self, tmp_path: Path) -> None:
        blocker = tmp_path / "blocker"
        blocker.write_text("x", encoding="utf-8")
        paths = write_artifacts(blocker / "reports", "x", make_result(), make_extra("x"))
        assert paths["report"].exists()
        assert paths["report"].parent != blocker / "reports"
