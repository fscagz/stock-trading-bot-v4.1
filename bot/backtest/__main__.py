from __future__ import annotations
import argparse
import csv
import logging
import os
from datetime import date, timedelta
from pathlib import Path
from typing import List

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env", override=True)

import bot.broker_alpaca as broker
from bot.backtest.backtest_metrics import compute_metrics
from bot.backtest.bar_fetcher import BarFetcher
from bot.backtest.candidate_screener import CandidateScreener
from bot.backtest.simulator import Simulator
from bot.config import V4Config
from bot.data.daily_loader import get_daily
from bot.intraday.types import TradeRecord

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _trading_days(start: date, end: date) -> List[date]:
    days = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def _write_trades_csv(trades: List[TradeRecord], path: Path) -> None:
    fields = [
        "ticker", "direction", "entry_time", "entry_price", "shares",
        "stop_price", "target_price", "exit_time", "exit_price",
        "pnl", "exit_reason", "portfolio_heat_at_entry", "signals",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for t in trades:
            writer.writerow({
                "ticker": t.ticker,
                "direction": t.direction,
                "entry_time": t.entry_time.isoformat(),
                "entry_price": t.entry_price,
                "shares": t.shares,
                "stop_price": t.stop_price,
                "target_price": t.target_price,
                "exit_time": t.exit_time.isoformat() if t.exit_time else "",
                "exit_price": t.exit_price if t.exit_price is not None else "",
                "pnl": round(t.pnl, 2) if t.pnl is not None else "",
                "exit_reason": t.exit_reason or "",
                "portfolio_heat_at_entry": round(t.portfolio_heat_at_entry, 4),
                "signals": "|".join(t.signals),
            })


def _write_summary_csv(metrics: dict, path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in metrics.items():
            writer.writerow([k, v])


def main() -> None:
    parser = argparse.ArgumentParser(description="V4 Momentum Bot Backtester")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--start", help="Start date YYYY-MM-DD (statistical mode)")
    mode.add_argument("--symbol", help="Single symbol (targeted mode)")
    parser.add_argument("--end", help="End date YYYY-MM-DD (required with --start)")
    parser.add_argument("--date", dest="target_date", help="Date YYYY-MM-DD (required with --symbol)")
    parser.add_argument("--slippage", type=float, default=0.001, help="Slippage fraction (default 0.001)")
    args = parser.parse_args()

    api_key = os.environ["APCA_API_KEY_ID"]
    secret_key = os.environ["APCA_API_SECRET_KEY"]
    base_url = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

    config = V4Config()
    account = broker.get_account_info()
    initial_equity = account["portfolio_value"]
    logger.info("Initial equity: $%.2f", initial_equity)

    out_dir = Path("backtest_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    screener = CandidateScreener(config, api_key, secret_key, base_url)
    fetcher = BarFetcher(api_key, secret_key)
    simulator = Simulator(config, initial_equity, slippage_pct=args.slippage)

    if args.symbol:
        if not args.target_date:
            parser.error("--symbol requires --date")
        trade_date = date.fromisoformat(args.target_date)
        days = [trade_date]
        prefix = f"{args.symbol}_{trade_date}"
    else:
        if not args.end:
            parser.error("--start requires --end")
        days = _trading_days(date.fromisoformat(args.start), date.fromisoformat(args.end))
        prefix = f"{args.start}_{args.end}"

    all_trades: List[TradeRecord] = []

    for d in days:
        candidates = [args.symbol] if args.symbol else screener.candidates_for_date(d)
        if not candidates:
            logger.info("No candidates for %s — skipped", d)
            continue
        logger.info("%s: %d candidates", d, len(candidates))

        bars_by_symbol = {}
        baseline_volumes = {}
        for sym in candidates:
            bars = fetcher.fetch(sym, d)
            if not bars:
                logger.debug("No IEX bars for %s on %s", sym, d)
                continue
            bars_by_symbol[sym] = bars
            try:
                df = get_daily(sym, period="1mo")
                baseline_volumes[sym] = (
                    df["volume"].tail(20).mean() / 390 if not df.empty else 0.0
                )
            except Exception:
                baseline_volumes[sym] = 0.0

        if not bars_by_symbol:
            logger.info("No bar data for %s — skipped", d)
            continue

        result = simulator.run_day(d, bars_by_symbol, baseline_volumes)
        all_trades.extend(result.trades)
        logger.info("%s: %d trades closed", d, len(result.trades))

    metrics = compute_metrics(all_trades, initial_equity)
    trades_path = out_dir / f"trades_{prefix}.csv"
    summary_path = out_dir / f"summary_{prefix}.csv"
    _write_trades_csv(all_trades, trades_path)
    _write_summary_csv(metrics, summary_path)

    print(f"\n=== Backtest Results ({prefix}) ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"\nTrades written to: {trades_path}")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
