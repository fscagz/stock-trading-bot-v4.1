"""
Hard-stop avoidance experiments — 2025-26 bull period.

Tests all combinations of entry filters designed to avoid entering positions
that will get hard-stopped:

  1. Baseline (live-equivalent): market fill, no vwap filter, no day-high filter
  2. require_above_vwap_at_entry=True
  3. stage2_min_dist_from_day_high_pct=0.02  (2% from day high)
  4. stage2_min_dist_from_day_high_pct=0.05  (5% from day high)
  5. market_order_fill=False  (limit at next bar open)
  6. vwap + day_high_2pct
  7. vwap + day_high_5pct
  8. vwap + limit_order
  9. day_high_2pct + limit_order
 10. day_high_5pct + limit_order
 11. vwap + day_high_2pct + limit_order
 12. vwap + day_high_5pct + limit_order

All configs: dynamic equity, news_mode=require, news_tier_bypass=4,
             slippage=0.001, no overnight holds (live-equivalent).
"""
from __future__ import annotations
import copy, os, warnings, logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

import pandas as pd
import bot.broker_alpaca as broker
from bot.backtest.bar_fetcher import BarFetcher
from bot.backtest.candidate_screener import CandidateScreener
from bot.backtest.news_filter import NewsFilter
from bot.backtest.simulator import Simulator
from bot.config import make_gap_hold_config
from bot.data.daily_loader import get_daily
from bot.intraday.types import TradeRecord

START = date(2025, 6, 1)
END   = date(2026, 5, 28)

_spy: Optional[pd.DataFrame] = None
def spy_uptrend(d: date) -> bool:
    global _spy
    if _spy is None:
        _spy = get_daily("SPY", start="2021-11-01", end="2026-06-10")
        _spy["ma20"] = _spy["close"].rolling(20).mean()
    past = _spy[_spy.index < pd.Timestamp(d)].dropna(subset=["ma20"])
    if past.empty:
        return True
    return float(past.iloc[-1]["close"]) >= float(past.iloc[-1]["ma20"])

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


def run_config(
    cfg,
    days: List[date],
    screener: CandidateScreener,
    fetcher: BarFetcher,
    initial_equity: float,
    news_filter: NewsFilter,
    market_order_fill: bool,
    require_above_vwap: bool,
    day_high_dist: float,
) -> dict:
    """Run one config dynamically, return summary dict."""
    running_eq = initial_equity
    all_trades: List[TradeRecord] = []
    cache_dir = fetcher._cache_dir

    sim = Simulator(
        cfg, initial_equity,
        slippage_pct=0.001,
        overnight_holds=False,
        market_order_fill=market_order_fill,
        news_filter=news_filter,
        news_mode="require",
        news_tier_bypass=4,
        require_above_vwap_at_entry=require_above_vwap,
        stage2_min_dist_from_day_high_pct=day_high_dist,
    )

    for d in days:
        if not spy_uptrend(d):
            continue

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

        sim._initial_equity = running_eq
        result = sim.run_day(d, bars_by_sym, baselines)
        all_trades.extend(result.trades)
        day_pnl = sum(t.pnl for t in result.trades if t.pnl is not None)
        running_eq = max(running_eq + day_pnl, 1.0)

    # Summarise
    trades = all_trades
    if not trades:
        return {"trades": 0, "wr": 0.0, "net_pnl": 0.0, "ret_pct": 0.0,
                "hard_stops": 0, "hs_total": 0.0, "hs_avg": 0.0,
                "targets": 0, "tgt_total": 0.0}

    wins   = [t for t in trades if t.pnl and t.pnl > 0]
    stops  = [t for t in trades if t.exit_reason == "hard_stop"]
    tgts   = [t for t in trades if t.exit_reason == "target"]
    net    = sum(t.pnl for t in trades if t.pnl is not None)

    return {
        "trades":    len(trades),
        "wr":        len(wins) / len(trades) * 100,
        "net_pnl":   net,
        "ret_pct":   net / initial_equity * 100,
        "hard_stops": len(stops),
        "hs_total":  sum(t.pnl for t in stops if t.pnl is not None),
        "hs_avg":    sum(t.pnl for t in stops if t.pnl is not None) / len(stops) if stops else 0.0,
        "targets":   len(tgts),
        "tgt_total": sum(t.pnl for t in tgts if t.pnl is not None),
    }


def main():
    api_key    = os.environ["APCA_API_KEY_ID"]
    secret_key = os.environ["APCA_API_SECRET_KEY"]
    base_url   = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

    initial_equity = broker.get_account_info()["portfolio_value"]
    print(f"Starting equity: ${initial_equity:,.0f}")
    print(f"Period: {START} → {END}\n")

    cfg      = make_gap_hold_config()
    screener = CandidateScreener(copy.copy(cfg), api_key, secret_key, base_url)
    print("Loading screener...", flush=True)
    screener.preload(START, END)
    print(f"  {len(screener._daily_cache):,} symbols loaded.\n", flush=True)

    fetcher     = BarFetcher(api_key, secret_key)
    news_filter = NewsFilter(api_key, secret_key, cache_only=False)
    days        = trading_days(START, END)

    # (label, market_order_fill, require_vwap, day_high_dist)
    experiments = [
        ("Baseline (live-equiv)",            True,  False, 0.00),
        ("above_vwap",                       True,  True,  0.00),
        ("day_high_2pct",                    True,  False, 0.02),
        ("day_high_5pct",                    True,  False, 0.05),
        ("limit_order",                      False, False, 0.00),
        ("vwap + day_high_2pct",             True,  True,  0.02),
        ("vwap + day_high_5pct",             True,  True,  0.05),
        ("vwap + limit_order",               False, True,  0.00),
        ("day_high_2pct + limit_order",      False, False, 0.02),
        ("day_high_5pct + limit_order",      False, False, 0.05),
        ("vwap + day_high_2pct + limit_ord", False, True,  0.02),
        ("vwap + day_high_5pct + limit_ord", False, True,  0.05),
    ]

    hdr = (f"  {'Config':<34} {'Trades':>7} {'WR':>6} {'Net PnL':>11} {'Ret%':>7} "
           f"{'HStops':>7} {'HS Total':>11} {'HS Avg':>9} {'Targets':>8} {'Tgt Total':>11}")
    print(hdr)
    print("  " + "-" * 120)

    for label, mof, vwap, dhd in experiments:
        print(f"  {label:<34}  running...", end="\r", flush=True)
        m = run_config(
            copy.copy(cfg), days, screener, fetcher, initial_equity,
            news_filter, mof, vwap, dhd,
        )
        print(
            f"  {label:<34} {m['trades']:>7,} {m['wr']:>5.1f}% "
            f"${m['net_pnl']:>10,.0f} {m['ret_pct']:>6.1f}% "
            f"{m['hard_stops']:>7,} ${m['hs_total']:>10,.0f} ${m['hs_avg']:>8,.0f} "
            f"{m['targets']:>8,} ${m['tgt_total']:>10,.0f}",
            flush=True,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
