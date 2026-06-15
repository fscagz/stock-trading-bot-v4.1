"""
MA period sweep for the SPY regime filter.

Tests MA periods: none (off), 5, 10, 15, 20 (current), 30, 50
across all three backtest periods with live-bot settings
(dynamic equity, no overnights, market-order fill, news + tier-4 bypass).

Key metrics: Net PnL, MaxDD%, win rate, and how many trading days
each MA period blocks — so we can see the protection/opportunity tradeoff.
"""
from __future__ import annotations
import copy, os, warnings, logging
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from typing import Dict, List, Optional

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

import pandas as pd
import bot.broker_alpaca as broker
from bot.backtest.backtest_metrics import compute_metrics
from bot.backtest.bar_fetcher import BarFetcher
from bot.backtest.candidate_screener import CandidateScreener
from bot.backtest.news_filter import NewsFilter
from bot.backtest.simulator import Simulator
from bot.config import make_gap_hold_config
from bot.data.daily_loader import get_daily
from bot.intraday.types import TradeRecord

PERIODS = {
    "2022 bear":    (date(2022, 1,  3), date(2022, 12, 30)),
    "2023 recovery":(date(2023, 1,  3), date(2023, 12, 29)),
    "Jan-Apr 2025": (date(2025, 1,  3), date(2025,  4, 20)),
    "2025-26 bull": (date(2025, 6,  1), date(2026,  5, 28)),
}

MA_PERIODS = [None, 5, 10, 15, 20, 30, 50]  # None = no regime filter

_spy_cache: Optional[pd.DataFrame] = None

def _get_spy() -> pd.DataFrame:
    global _spy_cache
    if _spy_cache is None:
        _spy_cache = get_daily("SPY", start="2021-06-01", end="2026-06-10")
    return _spy_cache

def spy_uptrend(d: date, ma_period: int) -> bool:
    spy = _get_spy()
    past = spy[spy.index < pd.Timestamp(d)]
    if len(past) < ma_period:
        return True
    ma = float(past["close"].tail(ma_period).mean())
    return float(past.iloc[-1]["close"]) >= ma

def trading_days(start: date, end: date) -> List[date]:
    days, d = [], start
    while d <= end:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days

def baseline_vol(screener: CandidateScreener, sym: str, as_of: date) -> float:
    df = screener._daily_cache.get(sym)
    if df is None or df.empty:
        return 0.0
    past = df[df.index < pd.Timestamp(as_of)]
    return float(past["volume"].tail(20).mean()) / 390.0 if not past.empty else 0.0

def extended_metrics(trades: List[TradeRecord], initial_equity: float) -> dict:
    m = compute_metrics(trades, initial_equity)
    m["total_return_pct"] = round(m["total_pnl"] / initial_equity * 100, 1)
    eq, peak = initial_equity, initial_equity
    for t in trades:
        if t.pnl is not None:
            eq += t.pnl
            peak = max(peak, eq)
    m["max_drawdown_pct"] = round(m["max_drawdown"] / peak * 100, 1) if peak > 0 else 0.0
    return m


def run(
    cfg,
    days: List[date],
    screener: CandidateScreener,
    fetcher: BarFetcher,
    initial_equity: float,
    news_filter: NewsFilter,
    ma_period: Optional[int],
) -> dict:
    trades: List[TradeRecord] = []
    running_eq = initial_equity
    cache_dir = fetcher._cache_dir
    days_blocked = 0

    for d in days:
        if ma_period is not None and not spy_uptrend(d, ma_period):
            days_blocked += 1
            continue

        sim = Simulator(
            cfg, running_eq,
            slippage_pct=0.001,
            overnight_holds=False,
            market_order_fill=True,
            news_filter=news_filter,
            news_mode="require",
            news_tier_bypass=4,
        )

        cands  = screener.candidates_for_date(d)
        cached = [s for s in cands if (cache_dir / f"{s}_{d}.json").exists()]
        if not cached:
            continue

        bars_by_sym: Dict = {}
        baselines:   Dict = {}
        with ThreadPoolExecutor(max_workers=16) as pool:
            for sym, bars, bl in pool.map(
                lambda s: (s, fetcher.fetch(s, d), baseline_vol(screener, s, d)),
                cached,
            ):
                if bars and bl > 0:
                    bars_by_sym[sym] = bars
                    baselines[sym]   = bl

        if not bars_by_sym:
            continue

        result = sim.run_day(d, bars_by_sym, baselines)
        trades.extend(result.trades)
        day_pnl    = sum(t.pnl for t in result.trades if t.pnl is not None)
        running_eq = max(running_eq + day_pnl, 1.0)

    m = extended_metrics(trades, initial_equity)
    m["days_blocked"] = days_blocked
    return m


def main():
    api_key    = os.environ["APCA_API_KEY_ID"]
    secret_key = os.environ["APCA_API_SECRET_KEY"]
    base_url   = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

    initial_equity = broker.get_account_info()["portfolio_value"]
    print(f"Starting equity: ${initial_equity:,.0f}\n")

    fetcher     = BarFetcher(api_key, secret_key)
    news_filter = NewsFilter(api_key, secret_key, cache_only=False)

    hdr = (f"  {'MA':>5}  {'Trades':>7} {'WR':>6} {'Avg W':>8} {'Avg L':>8} "
           f"{'Net PnL':>11} {'Return':>8} {'MaxDD':>9} {'MaxDD%':>7} {'Blocked':>8}")
    sep = "  " + "-" * 101

    # period_totals[ma_label] = [pnl_period1, pnl_period2, ...]
    ma_labels = ["off" if p is None else str(p) for p in MA_PERIODS]
    period_totals: Dict[str, List[float]] = {lbl: [] for lbl in ma_labels}

    for period_label, (start, end) in PERIODS.items():
        cfg      = make_gap_hold_config()
        screener = CandidateScreener(copy.copy(cfg), api_key, secret_key, base_url)
        print(f"\n{'='*105}")
        print(f"  {period_label}  ({start} → {end})")
        print(f"{'='*105}")
        print(f"  Loading screener...", flush=True)
        screener.preload(start, end)
        days = trading_days(start, end)
        print(f"  {len(screener._daily_cache):,} symbols, {len(days)} trading days.\n")
        print(hdr)
        print(sep)

        for ma_period, lbl in zip(MA_PERIODS, ma_labels):
            m = run(copy.copy(cfg), days, screener, fetcher, initial_equity,
                    news_filter, ma_period)
            period_totals[lbl].append(m["total_pnl"])
            marker = " ← current" if ma_period == 20 else ""
            print(
                f"  {lbl:>5}  {m['total_trades']:>7,} {m['win_rate']:>5.1%} "
                f"${m['avg_winner']:>7,.0f} ${m['avg_loser']:>7,.0f} "
                f"${m['total_pnl']:>10,.0f} {m['total_return_pct']:>7.1f}% "
                f"${m['max_drawdown']:>8,.0f} {m['max_drawdown_pct']:>6.1f}% "
                f"{m['days_blocked']:>8}{marker}",
                flush=True,
            )

    print(f"\n{'='*105}")
    print("  COMBINED PnL ACROSS ALL THREE PERIODS")
    print(f"{'='*105}")
    period_names = list(PERIODS.keys())
    print(f"  {'MA':>5}  {period_names[0]:>12} {period_names[1]:>16} {period_names[2]:>14} {period_names[3]:>14} {'TOTAL':>12}")
    print("  " + "-" * 75)
    for lbl in ma_labels:
        vals  = period_totals[lbl]
        total = sum(vals)
        marker = " ← current" if lbl == "20" else ""
        print(f"  {lbl:>5}  ${vals[0]:>10,.0f} ${vals[1]:>14,.0f} ${vals[2]:>12,.0f} ${vals[3]:>12,.0f} ${total:>10,.0f}{marker}")

    print("\nDone.")


if __name__ == "__main__":
    main()
