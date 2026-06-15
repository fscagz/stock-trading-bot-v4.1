"""
News filter A/B test: Jan 3, 2025 → Apr 20, 2025.

Configs (all dynamic equity, live-bot settings):
  1. No news filter
  2. News filter required
  3. News filter only when SPY < 200-day MA (conditional bear-regime filter)
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

START = date(2025, 1, 3)
END   = date(2025, 4, 20)

_spy: pd.DataFrame | None = None

def _load_spy() -> pd.DataFrame:
    global _spy
    if _spy is None:
        df = get_daily("SPY", start="2024-06-01", end="2025-05-01")
        df["ma20"]  = df["close"].rolling(20).mean()
        df["ma200"] = df["close"].rolling(200).mean()
        _spy = df
    return _spy

def spy_uptrend_20(d: date) -> bool:
    df = _load_spy()
    past = df[df.index < pd.Timestamp(d)].dropna(subset=["ma20"])
    if past.empty:
        return True
    r = past.iloc[-1]
    return float(r["close"]) >= float(r["ma20"])

def spy_below_200(d: date) -> bool:
    df = _load_spy()
    past = df[df.index < pd.Timestamp(d)].dropna(subset=["ma200"])
    if past.empty:
        return False
    r = past.iloc[-1]
    return float(r["close"]) < float(r["ma200"])

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
    m["final_equity"] = round(initial_equity + m["total_pnl"], 2)
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
    conditional_news: bool = False,  # True → news only when SPY < 200d MA
) -> dict:
    trades: List[TradeRecord] = []
    running_eq = initial_equity
    cache_dir = fetcher._cache_dir

    for d in days:
        if not spy_uptrend_20(d):
            continue

        # Conditional news: only require catalyst on bear-regime days
        if conditional_news:
            active_nf   = news_filter if spy_below_200(d) else None
            active_mode = "require"   if spy_below_200(d) else "ignore"
        else:
            active_nf   = news_filter
            active_mode = "require" if news_filter else "ignore"

        sim = Simulator(
            cfg, running_eq,
            slippage_pct=0.001,
            overnight_holds=False,
            market_order_fill=True,
            news_filter=active_nf,
            news_mode=active_mode,
        )

        cands  = screener.candidates_for_date(d)
        cached = [s for s in cands if (cache_dir / f"{s}_{d}.json").exists()]
        if not cached:
            continue

        bars_by_sym: Dict = {}
        baselines: Dict   = {}
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
        day_pnl   = sum(t.pnl for t in result.trades if t.pnl is not None)
        running_eq = max(running_eq + day_pnl, 1.0)

    return extended_metrics(trades, initial_equity)


def main():
    api_key    = os.environ["APCA_API_KEY_ID"]
    secret_key = os.environ["APCA_API_SECRET_KEY"]
    base_url   = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

    initial_equity = broker.get_account_info()["portfolio_value"]
    print(f"Starting equity: ${initial_equity:,.0f}")
    print(f"Period: {START} → {END}\n")

    fetcher     = BarFetcher(api_key, secret_key)
    news_filter = NewsFilter(api_key, secret_key, cache_only=False)
    cfg         = make_gap_hold_config()

    screener = CandidateScreener(copy.copy(cfg), api_key, secret_key, base_url)
    print("Loading screener data...", flush=True)
    screener.preload(START, END)
    print(f"Loaded {len(screener._daily_cache):,} symbols.\n", flush=True)

    days = trading_days(START, END)

    configs = [
        ("Dynamic, no news filter",           False, None,        False),
        ("Dynamic, news required (always)",   False, news_filter, False),
        ("Dynamic, news on bear days only",   True,  news_filter, True),
    ]

    hdr = (f"  {'Config':<40} {'Trades':>7} {'WR':>7} {'Avg W':>8} {'Avg L':>8} "
           f"{'Net PnL':>10} {'Return':>8} {'MaxDD':>10} {'MaxDD%':>7}")
    print(hdr)
    print("  " + "-" * 108)

    for label, conditional, nf, cond_flag in configs:
        m = run(copy.copy(cfg), days, screener, fetcher, initial_equity, nf, conditional_news=cond_flag)
        print(
            f"  {label:<40} {m['total_trades']:>7,} {m['win_rate']:>6.1%} "
            f"${m['avg_winner']:>7,.0f} ${m['avg_loser']:>7,.0f} "
            f"${m['total_pnl']:>9,.0f} {m['total_return_pct']:>7.1f}% "
            f"${m['max_drawdown']:>9,.0f} {m['max_drawdown_pct']:>6.1f}%",
            flush=True,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
