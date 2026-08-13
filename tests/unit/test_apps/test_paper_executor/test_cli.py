"""Tests for paper executor CLI module."""

from __future__ import annotations

import pytest
from apps.paper_executor.cli import (
    main,
)


class TestMainCLI:
    """Tests for the main CLI entry point."""

    def test_main_no_command_exits_with_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Running CLI without a command prints help and exits 1."""
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "usage:" in captured.out.lower() or "paper trading" in captured.out.lower()

    def test_main_unknown_subcommand_exits_with_error(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Unknown subcommand prints help and exits 2 (argparse convention)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["bogus-command"])
        assert exc_info.value.code == 2

    def test_main_create_account_subcommand(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Create-account subcommand runs successfully."""
        main(["create-account", "testacct"])
        captured = capsys.readouterr()
        assert "Account 'testacct' created" in captured.out
        assert "100,000.00" in captured.out  # default initial cash

    def test_main_simulate_subcommand(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Simulate trade subcommand runs successfully."""
        main([
            "simulate",
            "testacct",
            "--instrument", "AAPL",
            "--direction", "buy",
            "--quantity", "10",
            "--price", "150.0",
            "--order-type", "MARKET",
        ])
        captured = capsys.readouterr()
        assert "Trade executed" in captured.out
        assert "Account cash after trade" in captured.out

    def test_main_show_subcommand(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Show account subcommand runs successfully."""
        main(["show", "testacct"])
        captured = capsys.readouterr()
        assert "Account: testacct" in captured.out
        assert "Cash:" in captured.out
        assert "Equity:" in captured.out
        assert "Total PnL:" in captured.out

    def test_main_simulate_sell(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Sell trade without prior buy is rejected."""
        with pytest.raises(SystemExit) as exc_info:
            main([
                "simulate",
                "sellacct",
                "--instrument", "GOOG",
                "--direction", "sell",
                "--quantity", "10",
                "--price", "2800.0",
            ])
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Order rejected" in captured.err
        assert "No position to sell" in captured.err

    def test_main_simulate_limit_order(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Simulate a LIMIT order."""
        main([
            "simulate",
            "limitacct",
            "--instrument", "TSLA",
            "--direction", "buy",
            "--quantity", "5",
            "--price", "200.0",
            "--order-type", "LIMIT",
        ])
        captured = capsys.readouterr()
        assert "Trade executed" in captured.out
        assert "TSLA" in captured.out

    def test_main_custom_initial_cash(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Custom initial cash value is respected."""
        main(["create-account", "richacct", "--initial-cash", "500000"])
        captured = capsys.readouterr()
        assert "Account 'richacct' created" in captured.out

    def test_main_custom_slippage_and_commission(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Custom slippage and commission values are accepted."""
        main([
            "simulate",
            "testacct",
            "--instrument", "SPY",
            "--direction", "buy",
            "--quantity", "1",
            "--price", "400.0",
            "--slippage", "0.005",
            "--commission", "0.002",
        ])
        captured = capsys.readouterr()
        assert "Trade executed" in captured.out


class TestCLIValidation:
    """Tests for CLI argument validation."""

    def test_simulate_missing_required_args_exits(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Missing required trade args causes argparse error."""
        with pytest.raises(SystemExit):
            main(["simulate", "acct"])

    def test_simulate_invalid_direction_exits(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Invalid direction argument is rejected."""
        with pytest.raises(SystemExit):
            main([
                "simulate", "acct",
                "--instrument", "AAPL",
                "--direction", "hold",
                "--quantity", "10",
                "--price", "150.0",
            ])

    def test_simulate_invalid_order_type_exits(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Invalid order type is rejected."""
        with pytest.raises(SystemExit):
            main([
                "simulate", "acct",
                "--instrument", "AAPL",
                "--direction", "buy",
                "--quantity", "10",
                "--price", "150.0",
                "--order-type", "STOP_LIMIT",
            ])


class TestCLIErrorHandling:
    """Tests for error paths in CLI commands."""

    def test_simulate_insufficient_cash(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Buying more than available cash is rejected."""
        with pytest.raises(SystemExit) as exc_info:
            main([
                "simulate",
                "pooracct",
                "--instrument", "AAPL",
                "--direction", "buy",
                "--quantity", "1000000",
                "--price", "500.0",
            ])
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Order rejected" in captured.err
        assert "Insufficient cash" in captured.err

    def test_simulate_no_position_to_sell(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Selling an instrument with no position is rejected."""
        with pytest.raises(SystemExit) as exc_info:
            main([
                "simulate",
                "novelacct",
                "--instrument", "NOVEL",
                "--direction", "sell",
                "--quantity", "10",
                "--price", "100.0",
            ])
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Order rejected" in captured.err
        assert "No position to sell" in captured.err


class TestCLIOutputFormat:
    """Tests for CLI output formatting."""

    def test_create_account_output_format(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Create account output follows expected format."""
        main(["create-account", "fmtacct", "--initial-cash", "25000"])
        captured = capsys.readouterr()
        assert "Account 'fmtacct' created with cash" in captured.out
        assert "25,000.00" in captured.out

    def test_simulate_trade_output_contains_slippage(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Trade output includes slippage percentage."""
        main([
            "simulate", "slipacct",
            "--instrument", "AAPL",
            "--direction", "buy",
            "--quantity", "10",
            "--price", "150.0",
            "--slippage", "0.005",
        ])
        captured = capsys.readouterr()
        assert "slippage:" in captured.out
        assert "commission:" in captured.out

    def test_show_account_output_format(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Show account displays all expected fields."""
        main(["show", "showacct"])
        captured = capsys.readouterr()
        assert "Account: showacct" in captured.out
        assert "Cash:" in captured.out
        assert "Equity:" in captured.out
        assert "Total PnL:" in captured.out
        assert "Realized PnL:" in captured.out
        assert "Unrealized PnL:" in captured.out
        assert "Trades:" in captured.out
        assert "Commission:" in captured.out
        assert "Positions:" in captured.out


class TestCLIWithCustomParams:
    """Tests for CLI with non-default parameters."""

    def test_create_account_custom_commission(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Account created with custom commission rate."""
        main(["create-account", "commacct", "--commission", "0.005"])
        captured = capsys.readouterr()
        assert "Account 'commacct' created" in captured.out

    def test_simulate_high_slippage(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Trade with high slippage completes without error."""
        main([
            "simulate", "slipacct",
            "--instrument", "BTC",
            "--direction", "buy",
            "--quantity", "1",
            "--price", "50000.0",
            "--slippage", "0.02",
        ])
        captured = capsys.readouterr()
        assert "Trade executed" in captured.out

    def test_show_zero_equity_account(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Show account with zero trades shows zero PnL."""
        main(["show", "emptyacct"])
        captured = capsys.readouterr()
        assert "Total PnL:     0.00" in captured.out
        assert "Trades:        0" in captured.out
        assert "Positions:     0" in captured.out
