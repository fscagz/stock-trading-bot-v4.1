"""
Threshold experiments for the conditional news filter.

Baseline anchors:
  A. No news filter
  B. News always required

Conditional variants (news required when condition is true):
  1. SPY 10d return <= 0%        (current)
  2. SPY 10d return <= 1.5%      (higher bar — catches slow tops)
  3. SPY  5d return <= 0%        (faster signal)
  4. SPY  5d <= 0% OR 10d <= 1.5% (combined — either weakness triggers)
"""
from __future__ import annotations
import copy, os, warnings, logging
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from typing import Callable, Dict, List, Optional

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
    "2022 bear":         (date(2022, 1,  3),  date(2022, 12, 30)),
    "Jan–Apr 2025":      (date(2025, 1,  3),  date(2025, 4,  20)),
    "Jun 2025–May 2026": (date(2025, 6,  1),  date(2026, 5,  28)),
}

_spy: pd.DataFrame | None = None

def _load_spy() -> pd.DataFrame:
    global _spy
    if _spy is None:
        df = get_daily("SPY", start="2021-06-01", end="2026-06-01")
        df["ma20"] = df["close"].rolling(20).mean()
        _spy = df
    return _spy

def spy_uptrend_20(d: date) -> bool:
    df = _load_spy()
    past = df[df.index < pd.Timestamp(d)].dropna(subset=["ma20"])
    if past.empty:
        return True
    r = past.iloc[-1]
    return float(r["close"]) >= float(r["ma20"])

def spy_return(d: date, sessions: int) -> float:
    df = _load_spy()
    past = df[df.index < pd.Timestamp(d)]
    if len(past) < sessions + 1:
        return 0.0
    tail = past["close"].iloc[-(sessions + 1):]
    return (float(tail.iloc[-1]) - float(tail.iloc[0])) / float(tail.iloc[0])

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
    news_filter: Optional[NewsFilter],
    bear_signal: Optional[Callable[[date], bool]] = None,
) -> dict:
    """
    bear_signal(date) -> True  means: activate news filter today.
    If None and news_filter set: always active.
    If None and no news_filter: never active.
    """
    trades: List[TradeRecord] = []
    running_eq = initial_equity
    cache_dir  = fetcher._cache_dir

    for d in days:
        if not spy_uptrend_20(d):
            continue

        if bear_signal is not None:
            in_bear   = bear_signal(d)
            active_nf = news_filter if in_bear else None
            mode      = "require"   if in_bear else "ignore"
        else:
            active_nf = news_filter
            mode      = "require" if news_filter else "ignore"

        sim = Simulator(
            cfg, running_eq,
            slippage_pct=0.001,
            overnight_holds=False,
            market_order_fill=True,
            news_filter=active_nf,
            news_mode=mode,
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

        result     = sim.run_day(d, bars_by_sym, baselines)
        trades.extend(result.trades)
        day_pnl    = sum(t.pnl for t in result.trades if t.pnl is not None)
        running_eq = max(running_eq + day_pnl, 1.0)

    return extended_metrics(trades, initial_equity)


def main():
    api_key    = os.environ["APCA_API_KEY_ID"]
    secret_key = os.environ["APCA_API_SECRET_KEY"]
    base_url   = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

    initial_equity = broker.get_account_info()["portfolio_value"]
    print(f"Starting equity: ${initial_equity:,.0f}\n")

    fetcher     = BarFetcher(api_key, secret_key)
    news_filter = NewsFilter(api_key, secret_key, cache_only=False)

    configs = [
        # label,                         news_filter,  bear_signal
        ("A. No news filter",            None,         None),
        ("B. News always",               news_filter,  None),
        ("1. 10d <= 0%   (current)",     news_filter,  lambda d: spy_return(d, 10) <= 0.000),
        ("2. 10d <= 1.5%",               news_filter,  lambda d: spy_return(d, 10) <= 0.015),
        ("3.  5d <= 0%",                 news_filter,  lambda d: spy_return(d,  5) <= 0.000),
        ("4.  5d<=0% OR 10d<=1.5%",      news_filter,  lambda d: spy_return(d,  5) <= 0.000
                                                               or spy_return(d, 10) <= 0.015),
    ]

    hdr = (f"  {'Config':<30} {'Trades':>7} {'WR':>7} {'Avg W':>8} {'Avg L':>8} "
           f"{'Net PnL':>10} {'Return':>8} {'MaxDD':>10} {'MaxDD%':>7}")
    sep = "  " + "-" * 104

    period_totals: Dict[str, List[float]] = {label: [] for label, *_ in configs}

    for period_label, (start, end) in PERIODS.items():
        cfg      = make_gap_hold_config()
        screener = CandidateScreener(copy.copy(cfg), api_key, secret_key, base_url)
        print(f"\n{'='*108}")
        print(f"  {period_label}  ({start} → {end})")
        print(f"{'='*108}")
        print(f"  Loading screener...", flush=True)
        screener.preload(start, end)
        print(f"  {len(screener._daily_cache):,} symbols loaded.\n")
        print(hdr)
        print(sep)

        days = trading_days(start, end)

        for label, nf, signal in configs:
            m = run(copy.copy(cfg), days, screener, fetcher, initial_equity, nf, signal)
            period_totals[label].append(m["total_pnl"])
            print(
                f"  {label:<30} {m['total_trades']:>7,} {m['win_rate']:>6.1%} "
                f"${m['avg_winner']:>7,.0f} ${m['avg_loser']:>7,.0f} "
                f"${m['total_pnl']:>9,.0f} {m['total_return_pct']:>7.1f}% "
                f"${m['max_drawdown']:>9,.0f} {m['max_drawdown_pct']:>6.1f}%",
                flush=True,
            )

    # Combined totals
    print(f"\n{'='*108}")
    print("  COMBINED PnL ACROSS ALL THREE PERIODS")
    print(f"{'='*108}")
    print(f"  {'Config':<30} {'2022':>12} {'Jan-Apr 25':>12} {'Jun25-May26':>13} {'TOTAL':>12}")
    print("  " + "-" * 60)
    for label, *_ in configs:
        vals = period_totals[label]
        total = sum(vals)
        print(f"  {label:<30} ${vals[0]:>10,.0f} ${vals[1]:>10,.0f} ${vals[2]:>11,.0f} ${total:>10,.0f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
