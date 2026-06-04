from __future__ import annotations
import argparse
import copy
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
from bot.backtest.news_filter import NewsFilter
from bot.backtest.simulator import Simulator
from bot.config import V4Config, make_long_config
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
    parser.add_argument("--long", action="store_true",
                        help="Use make_long_config() — catalyst momentum longs (default: V4Config)")
    parser.add_argument("--risk-scale", type=float, default=1.0,
                        help="Scale risk_per_trade and max_portfolio_heat (default 1.0)")
    parser.add_argument("--regime", action="store_true", default=False,
                        help="Enable SPY 20-day MA regime filter: skip long entries in downtrend")
    parser.add_argument("--no-overnight", action="store_true", default=False,
                        help="Always close at EOD — no overnight holds (matches live runner)")
    parser.add_argument("--market-fill", action="store_true", default=False,
                        help="Fill at signal bar close instead of next bar open (matches live runner)")
    parser.add_argument("--news-mode", default="auto",
                        choices=["auto", "require", "exclude", "ignore"],
                        help="auto: require for --long, ignore otherwise; require: catalyst required; "
                             "exclude: no-catalyst only; ignore: no filter")
    args = parser.parse_args()

    api_key = os.environ["APCA_API_KEY_ID"]
    secret_key = os.environ["APCA_API_SECRET_KEY"]
    base_url = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

    news_mode = args.news_mode
    if news_mode == "auto":
        news_mode = "require" if args.long else "ignore"

    config = make_long_config() if args.long else V4Config()

    if args.risk_scale != 1.0:
        config.risk_per_trade = round(config.risk_per_trade * args.risk_scale, 6)
        config.max_portfolio_heat = min(round(config.max_portfolio_heat * args.risk_scale, 4), 1.0)
        logger.info("Risk scale %.2f×: risk_per_trade=%.4f max_portfolio_heat=%.4f",
                    args.risk_scale, config.risk_per_trade, config.max_portfolio_heat)

    news_filter = None
    if news_mode in ("require", "exclude"):
        logger.info("News filter: mode=%s", news_mode)
        news_filter = NewsFilter(api_key, secret_key, cache_only=False)

    account = broker.get_account_info()
    initial_equity = account["portfolio_value"]
    logger.info("Initial equity: $%.2f", initial_equity)

    # Pre-load SPY regime flags if requested
    regime_flags: dict = {}
    if args.regime and not args.symbol:
        import pandas as pd
        logger.info("Regime filter enabled: loading SPY data...")
        from bot.data.regime import RegimeFilter
        rf = RegimeFilter()

    out_dir = Path("backtest_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    screener = CandidateScreener(config, api_key, secret_key, base_url)
    fetcher = BarFetcher(api_key, secret_key)
    simulator = Simulator(
        config, initial_equity,
        slippage_pct=args.slippage,
        overnight_holds=not args.no_overnight,
        market_order_fill=args.market_fill,
        news_filter=news_filter,
        news_mode=news_mode,
    )

    if args.symbol:
        if not args.target_date:
            parser.error("--symbol requires --date")
        trade_date = date.fromisoformat(args.target_date)
        days = [trade_date]
        prefix = f"{args.symbol}_{trade_date}"
    else:
        if not args.end:
            parser.error("--start requires --end")
        start_date = date.fromisoformat(args.start)
        end_date = date.fromisoformat(args.end)
        days = _trading_days(start_date, end_date)
        prefix = f"{args.start}_{args.end}"
        screener.preload(start_date, end_date)

    if args.long:
        prefix = f"long_{prefix}"
    if news_mode != "ignore":
        prefix = f"{prefix}_news{news_mode}"
    if args.risk_scale != 1.0:
        prefix = f"{prefix}_scale{args.risk_scale}"
    if args.regime:
        prefix = f"{prefix}_regime"
    if args.no_overnight:
        prefix = f"{prefix}_noonight"
    if args.market_fill:
        prefix = f"{prefix}_mktfill"

    all_trades: List[TradeRecord] = []

    for d in days:
        # Regime filter: skip long entries on downtrend days
        if args.regime:
            from bot.data.regime import RegimeFilter
            if not RegimeFilter().is_uptrend(d):
                logger.debug("Regime filter: skipping %s (SPY downtrend)", d)
                continue

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
                continue
            bars_by_symbol[sym] = bars
            # Baseline from screener daily cache; fallback to yfinance
            df = screener._daily_cache.get(sym)
            if df is not None and not df.empty:
                past = df[df.index < __import__('pandas').Timestamp(d)]
                baseline = float(past["volume"].tail(20).mean()) / 390 if not past.empty else 0.0
            else:
                try:
                    df2 = get_daily(sym, period="1mo")
                    baseline = df2["volume"].tail(20).mean() / 390 if not df2.empty else 0.0
                except Exception:
                    baseline = 0.0
            baseline_volumes[sym] = baseline

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
