"""
Definitive pre-deployment backtest.

Runs four configurations so you can see each change's isolated effect:
  1. Fixed equity,   no news filter   (what we previously ran)
  2. Dynamic equity, no news filter   (compounding effect in isolation)
  3. Fixed equity,   news filter      (news filter effect in isolation)
  4. Dynamic equity, news filter      (closest to live-bot behaviour)

Live-bot-equivalent settings throughout:
  - No overnight holds
  - Market-order fill (bar close)
  - SPY 20-day MA regime filter
  - make_gap_hold_config() — $500k DV, 1×/2×/3×/4× confidence tiers
"""
from __future__ import annotations
import copy, os, sys, warnings, logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from typing import Dict, List, Tuple

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
    "2025-26 bull": (date(2025, 6, 1),  date(2026, 5, 28)),
    "2022 bear":    (date(2022, 1, 3),  date(2022, 12, 30)),
}

_spy: pd.DataFrame | None = None
def spy_uptrend(d: date) -> bool:
    global _spy
    if _spy is None:
        _spy = get_daily("SPY", start="2021-11-01", end="2026-06-01")
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

def baseline_from_cache(screener: CandidateScreener, sym: str, as_of: date) -> float:
    df = screener._daily_cache.get(sym)
    if df is None or df.empty:
        return 0.0
    past = df[df.index < pd.Timestamp(as_of)]
    return float(past["volume"].tail(20).mean()) / 390.0 if not past.empty else 0.0


def extended_metrics(trades: List[TradeRecord], initial_equity: float) -> dict:
    """compute_metrics + final equity + return % + max drawdown %."""
    m = compute_metrics(trades, initial_equity)
    final_equity = initial_equity + m["total_pnl"]
    m["final_equity"] = round(final_equity, 2)
    m["total_return_pct"] = round(m["total_pnl"] / initial_equity * 100, 1)

    # Re-derive peak for max_drawdown_pct
    eq = initial_equity
    peak = eq
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
    dynamic: bool,
    news_filter: NewsFilter | None,
) -> dict:
    sim = Simulator(
        cfg, initial_equity,
        slippage_pct=0.001,
        overnight_holds=False,
        market_order_fill=True,
        news_filter=news_filter,
        news_mode="require" if news_filter else "ignore",
        news_tier_bypass=4 if news_filter else 0,    # tier-4 signals bypass news gate
    )

    trades: List[TradeRecord] = []
    cache_dir = fetcher._cache_dir
    running_eq = initial_equity

    for d in days:
        if not spy_uptrend(d):
            continue

        cands = screener.candidates_for_date(d)
        cached = [s for s in cands if (cache_dir / f"{s}_{d}.json").exists()]
        if not cached:
            continue

        def _fetch(sym):
            bars = fetcher.fetch(sym, d)
            bl = baseline_from_cache(screener, sym, d)
            return sym, bars, bl

        bars_by_sym: Dict = {}
        baselines: Dict = {}
        with ThreadPoolExecutor(max_workers=16) as pool:
            for sym, bars, bl in pool.map(_fetch, cached):
                if bars and bl > 0:
                    bars_by_sym[sym] = bars
                    baselines[sym] = bl

        if not bars_by_sym:
            continue

        if dynamic:
            sim._initial_equity = running_eq

        result = sim.run_day(d, bars_by_sym, baselines)
        trades.extend(result.trades)

        if dynamic:
            day_pnl = sum(t.pnl for t in result.trades if t.pnl is not None)
            running_eq = max(running_eq + day_pnl, 1.0)

    return extended_metrics(trades, initial_equity)


def main():
    api_key = os.environ["APCA_API_KEY_ID"]
    secret_key = os.environ["APCA_API_SECRET_KEY"]
    base_url = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

    initial_equity = broker.get_account_info()["portfolio_value"]
    print(f"Starting equity: ${initial_equity:,.0f}\n")

    fetcher = BarFetcher(api_key, secret_key)
    news_filter = NewsFilter(api_key, secret_key, cache_only=False)

    configs = [
        ("Fixed,  no news  (baseline)",       False, None),
        ("Dynamic, no news",                  True,  None),
        ("Fixed,  news + tier-4 bypass",      False, news_filter),
        ("Dynamic, news + tier-4 bypass (live)", True, news_filter),
    ]

    hdr = (f"  {'Config':<34} {'Trades':>7} {'WR':>7} {'Avg W':>8} {'Avg L':>8} "
           f"{'Net PnL':>11} {'Return':>8} {'MaxDD':>10} {'MaxDD%':>8} {'Hold':>6}")
    sep = "  " + "-" * 110

    for period_label, (start, end) in PERIODS.items():
        print(f"\n{'='*114}")
        print(f"  {period_label}  ({start} → {end})")
        print(f"{'='*114}")
        print(hdr)
        print(sep)

        cfg = make_gap_hold_config()
        screener = CandidateScreener(copy.copy(cfg), api_key, secret_key, base_url)
        print(f"  Loading screener data...", flush=True)
        screener.preload(start, end)
        print(f"  Loaded {len(screener._daily_cache):,} symbols. Running...\n", flush=True)

        days = trading_days(start, end)

        for label, dynamic, nf in configs:
            m = run(copy.copy(cfg), days, screener, fetcher, initial_equity, dynamic, nf)
            print(
                f"  {label:<34} {m['total_trades']:>7,} {m['win_rate']:>6.1%} "
                f"${m['avg_winner']:>7,.0f} ${m['avg_loser']:>7,.0f} "
                f"${m['total_pnl']:>10,.0f} {m['total_return_pct']:>7.1f}% "
                f"${m['max_drawdown']:>9,.0f} {m['max_drawdown_pct']:>7.1f}% "
                f"{m['avg_hold_minutes']:>5.1f}m",
                flush=True,
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
