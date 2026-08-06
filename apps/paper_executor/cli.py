"""Command-line interface for paper trading simulation."""

from __future__ import annotations

import argparse
import sys

from packages.paper import OrderType, PaperExecutor, TradeDirection


def create_paper_trading_account(args: argparse.Namespace) -> None:
    """Create a new paper trading account."""
    executor = PaperExecutor(
        initial_cash=args.initial_cash,
        default_slippage_pct=args.slippage,
        default_commission_pct=args.commission,
    )
    account = executor.create_account(args.account_id)
    print(f"Account '{account.account_id}' created with cash {account.cash:,.2f}")


def simulate_trade(args: argparse.Namespace) -> None:
    """Simulate a single trade with given parameters."""
    executor = PaperExecutor(
        initial_cash=args.initial_cash,
        default_slippage_pct=args.slippage,
        default_commission_pct=args.commission,
    )
    account = executor.create_account(args.account_id)

    try:
        trade = executor.submit_order(
            account=account,
            instrument=args.instrument,
            direction=TradeDirection(args.direction.upper()),
            quantity=args.quantity,
            price=args.price,
            order_type=OrderType(args.order_type.upper()),
        )
        print(
            f"Trade executed: {trade.direction.value} {trade.filled_quantity:.4f} "
            f"{trade.instrument} @ {trade.filled_price:.4f} "
            f"(slippage: {trade.slippage:.4%}, commission: {trade.commission:.2f})"
        )
        print(f"Account cash after trade: {account.cash:,.2f}")
    except ValueError as exc:
        print(f"Order rejected: {exc}", file=sys.stderr)
        sys.exit(1)


def show_account(args: argparse.Namespace) -> None:
    """Show account summary."""
    executor = PaperExecutor(
        initial_cash=args.initial_cash,
        default_slippage_pct=args.slippage,
        default_commission_pct=args.commission,
    )
    account = executor.create_account(args.account_id)
    summary = executor.get_account_summary(account)
    print(f"Account: {summary['account_id']}")
    print(f"  Cash:          {summary['cash']:,.2f}")
    print(f"  Equity:        {summary['equity']:,.2f}")
    print(f"  Total PnL:     {summary['total_pnl']:,.2f}")
    print(f"  Realized PnL:  {summary['realized_pnl']:,.2f}")
    print(f"  Unrealized PnL:{summary['unrealized_pnl']:,.2f}")
    print(f"  Trades:        {summary['total_trades']}")
    print(f"  Commission:    {summary['total_commission']:,.2f}")
    print(f"  Positions:     {summary['num_positions']}")


def main(argv: list[str] | None = None) -> None:
    """Main entry point for the paper executor CLI."""
    parser = argparse.ArgumentParser(
        description="Paper trading executor CLI"
    )
    subparsers = parser.add_subparsers(dest="command")

    # create
    create_parser = subparsers.add_parser("create-account", help="Create a paper account")
    create_parser.add_argument("account_id", help="Account identifier")
    create_parser.add_argument("--initial-cash", type=float, default=100000.0)
    create_parser.add_argument("--slippage", type=float, default=0.001)
    create_parser.add_argument("--commission", type=float, default=0.001)

    # trade
    trade_parser = subparsers.add_parser("simulate", help="Simulate a trade")
    trade_parser.add_argument("account_id", help="Account identifier")
    trade_parser.add_argument("--instrument", required=True, help="Symbol to trade")
    trade_parser.add_argument(
        "--direction", required=True, choices=["buy", "sell"], help="Trade direction"
    )
    trade_parser.add_argument("--quantity", required=True, type=float, help="Quantity")
    trade_parser.add_argument("--price", required=True, type=float, help="Market price")
    trade_parser.add_argument(
        "--order-type", default="MARKET", choices=["MARKET", "LIMIT", "STOP"]
    )
    trade_parser.add_argument("--initial-cash", type=float, default=100000.0)
    trade_parser.add_argument("--slippage", type=float, default=0.001)
    trade_parser.add_argument("--commission", type=float, default=0.001)

    # show
    show_parser = subparsers.add_parser("show", help="Show account summary")
    show_parser.add_argument("account_id", help="Account identifier")
    show_parser.add_argument("--initial-cash", type=float, default=100000.0)
    show_parser.add_argument("--slippage", type=float, default=0.001)
    show_parser.add_argument("--commission", type=float, default=0.001)

    args = parser.parse_args(argv)

    commands = {
        "create-account": create_paper_trading_account,
        "simulate": simulate_trade,
        "show": show_account,
    }

    handler = commands.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    handler(args)


if __name__ == "__main__":  # pragma: no cover
    main()
