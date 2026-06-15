"""
Tier-4 regime bypass test.

Question: do tier-4 confidence signals perform well even when the broad
market (SPY 20-day MA) is in a downtrend, justifying a regime bypass?

Three configs, all with dynamic equity + live-bot settings (no overnights,
market-order fill, news filter + tier-4 news bypass):

  A. Regime ON   — skip all longs on regime-down days (current behaviour)
  B. Tier-4 bypass — regime-down days: allow tier-4 entries only
                    (tier-4 score bypasses BOTH regime and news gates)
  C. Regime OFF  — ignore regime entirely (upper bound / sanity check)

Periods: 2022 bear + 2025-26 bull
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
    "2022 bear":         (date(2022, 1,  3), date(2022, 12, 30)),
    "2025-26 bull":      (date(2025, 6,  1), date(2026, 5,  28)),
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
    # Count regime-down days that actually produced trades
    m["trades_list"] = trades
    return m


def run(
    cfg,
    days: List[date],
    screener: CandidateScreener,
    fetcher: BarFetcher,
    initial_equity: float,
    news_filter: NewsFilter,
    *,
    regime_mode: str,   # "on" | "bypass" | "off"
) -> dict:
    """
    regime_mode:
      "on"     — skip regime-down days entirely (current behaviour)
      "bypass" — regime-down days: tier-4 only (min_entry_tier=4, tier-4 news bypass)
      "off"    — no regime filter (trade every day normally)
    """
    trades: List[TradeRecord] = []
    running_eq = initial_equity
    cache_dir = fetcher._cache_dir
    regime_down_days = 0
    regime_down_trades = 0

    for d in days:
        uptrend = spy_uptrend_20(d)

        if regime_mode == "on" and not uptrend:
            continue

        # Build the simulator for this day
        if not uptrend and regime_mode == "bypass":
            # Regime-down day: only tier-4 entries; tier-4 also bypasses news gate
            sim = Simulator(
                cfg, running_eq,
                slippage_pct=0.001,
                overnight_holds=False,
                market_order_fill=True,
                news_filter=news_filter,
                news_mode="require",
                news_tier_bypass=4,
                min_entry_tier=4,
            )
            regime_down_days += 1
        else:
            # Normal day (regime up, or regime_mode=="off")
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
        day_trades = result.trades
        trades.extend(day_trades)

        if not uptrend and regime_mode == "bypass":
            regime_down_trades += len(day_trades)

        day_pnl    = sum(t.pnl for t in day_trades if t.pnl is not None)
        running_eq = max(running_eq + day_pnl, 1.0)

    m = extended_metrics(trades, initial_equity)
    m["regime_down_days_traded"] = regime_down_days
    m["regime_down_trades"]      = regime_down_trades
    return m


def main():
    api_key    = os.environ["APCA_API_KEY_ID"]
    secret_key = os.environ["APCA_API_SECRET_KEY"]
    base_url   = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

    initial_equity = broker.get_account_info()["portfolio_value"]
    print(f"Starting equity: ${initial_equity:,.0f}\n")

    fetcher     = BarFetcher(api_key, secret_key)
    news_filter = NewsFilter(api_key, secret_key, cache_only=False)

    configs = [
        ("A. Regime ON  (current)",   "on"),
        ("B. Tier-4 bypass",          "bypass"),
        ("C. Regime OFF (no filter)", "off"),
    ]

    hdr = (f"  {'Config':<26} {'Trades':>7} {'WR':>7} {'Avg W':>8} {'Avg L':>8} "
           f"{'Net PnL':>10} {'Return':>8} {'MaxDD':>9} {'MaxDD%':>7} "
           f"{'↓Days':>6} {'↓Trades':>8}")
    sep = "  " + "-" * 116

    period_totals: Dict[str, List[float]] = {label: [] for label, _ in configs}

    for period_label, (start, end) in PERIODS.items():
        cfg      = make_gap_hold_config()
        screener = CandidateScreener(copy.copy(cfg), api_key, secret_key, base_url)
        print(f"\n{'='*120}")
        print(f"  {period_label}  ({start} → {end})")
        print(f"{'='*120}")
        print(f"  Loading screener...", flush=True)
        screener.preload(start, end)
        print(f"  {len(screener._daily_cache):,} symbols loaded.\n")
        print(hdr)
        print(sep)

        days = trading_days(start, end)
        regime_down_count = sum(1 for d in days if not spy_uptrend_20(d))
        print(f"  ({len(days)} trading days, {regime_down_count} regime-down)\n")

        for label, mode in configs:
            m = run(copy.copy(cfg), days, screener, fetcher, initial_equity,
                    news_filter, regime_mode=mode)
            period_totals[label].append(m["total_pnl"])
            down_days   = m.get("regime_down_days_traded", 0)
            down_trades = m.get("regime_down_trades", 0)
            print(
                f"  {label:<26} {m['total_trades']:>7,} {m['win_rate']:>6.1%} "
                f"${m['avg_winner']:>7,.0f} ${m['avg_loser']:>7,.0f} "
                f"${m['total_pnl']:>9,.0f} {m['total_return_pct']:>7.1f}% "
                f"${m['max_drawdown']:>8,.0f} {m['max_drawdown_pct']:>6.1f}% "
                f"{down_days:>6} {down_trades:>8}",
                flush=True,
            )

    print(f"\n{'='*120}")
    print("  COMBINED PnL ACROSS BOTH PERIODS")
    print(f"{'='*120}")
    print(f"  {'Config':<26} {'2022 bear':>12} {'2025-26 bull':>14} {'TOTAL':>12}")
    print("  " + "-" * 65)
    for label, _ in configs:
        vals  = period_totals[label]
        total = sum(vals)
        print(f"  {label:<26} ${vals[0]:>10,.0f} ${vals[1]:>12,.0f} ${total:>10,.0f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
