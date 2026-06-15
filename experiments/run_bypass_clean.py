"""
Clean bypass experiment — fixed equity + daily dollar-volume position cap.

Fixed equity removes the compounding explosion so we can see whether the
bypass itself adds alpha.  DV cap (20% of avg daily $ volume) prevents the
simulator from pretending we can deploy $500k into a $2M/day small-cap.

Configs (all news_mode="require" baseline, live-bot settings):
  A.  News always required          (baseline)
  1.  Tier-3 bypass                 (conf mult ≥ 3× bypasses gate)
  2.  Tier-4 bypass                 (conf mult ≥ 4× bypasses gate)
  3.  Rel-vol bypass  40×           (exceptional volume bypasses gate)
  4.  Rel-vol bypass  60×
  5.  Tier-3 + rel-vol 40×          (either condition bypasses)
  6.  Tier-4 + rel-vol 60×          (strictest combination)
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
    "2022 bear":         (date(2022, 1,  3),  date(2022, 12, 30)),
    "Jan–Apr 2025":      (date(2025, 1,  3),  date(2025, 4,  20)),
    "Jun 2025–May 2026": (date(2025, 6,  1),  date(2026, 5,  28)),
}

# Position capped at 20% of avg daily dollar volume — keeps us out of stocks
# we can't actually move real size through.
MAX_POSITION_DV_PCT = 0.20

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
    return m


def run(
    cfg,
    days: List[date],
    screener: CandidateScreener,
    fetcher: BarFetcher,
    initial_equity: float,
    news_filter: Optional[NewsFilter],
    news_relvol_bypass: float = 0.0,
    news_tier_bypass: int = 0,
) -> dict:
    trades: List[TradeRecord] = []
    cache_dir  = fetcher._cache_dir
    running_eq = initial_equity

    for d in days:
        if not spy_uptrend_20(d):
            continue

        sim = Simulator(
            cfg, running_eq,
            slippage_pct=0.001,
            overnight_holds=False,
            market_order_fill=True,
            news_filter=news_filter,
            news_mode="require" if news_filter else "ignore",
            news_relvol_bypass=news_relvol_bypass,
            news_tier_bypass=news_tier_bypass,
            max_position_dv_pct=MAX_POSITION_DV_PCT,
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

    return extended_metrics(trades, initial_equity)


def main():
    api_key    = os.environ["APCA_API_KEY_ID"]
    secret_key = os.environ["APCA_API_SECRET_KEY"]
    base_url   = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

    initial_equity = broker.get_account_info()["portfolio_value"]
    print(f"Starting equity (dynamic): ${initial_equity:,.0f}")
    print(f"Position DV cap: {MAX_POSITION_DV_PCT:.0%} of avg daily dollar volume\n")

    fetcher     = BarFetcher(api_key, secret_key)
    news_filter = NewsFilter(api_key, secret_key, cache_only=False)

    # (label, relvol_bypass, tier_bypass)
    configs = [
        ("A. News always (baseline)",  0.0,  0),
        ("1. Tier-3 bypass",           0.0,  3),
        ("2. Tier-4 bypass",           0.0,  4),
        ("3. Rel-vol 40× bypass",      40.0, 0),
        ("4. Rel-vol 60× bypass",      60.0, 0),
        ("5. Tier-3 + rel-vol 40×",    40.0, 3),
        ("6. Tier-4 + rel-vol 60×",    60.0, 4),
    ]

    hdr = (f"  {'Config':<28} {'Trades':>7} {'WR':>7} {'Avg W':>8} {'Avg L':>8} "
           f"{'Net PnL':>10} {'Return':>8} {'MaxDD':>10} {'MaxDD%':>7}")
    sep = "  " + "-" * 102

    period_totals: Dict[str, List[float]] = {label: [] for label, *_ in configs}

    for period_label, (start, end) in PERIODS.items():
        cfg      = make_gap_hold_config()
        screener = CandidateScreener(copy.copy(cfg), api_key, secret_key, base_url)
        print(f"\n{'='*106}")
        print(f"  {period_label}  ({start} → {end})")
        print(f"{'='*106}")
        print(f"  Loading screener...", flush=True)
        screener.preload(start, end)
        print(f"  {len(screener._daily_cache):,} symbols loaded.\n")
        print(hdr)
        print(sep)

        days = trading_days(start, end)

        for label, rvb, tb in configs:
            m = run(copy.copy(cfg), days, screener, fetcher, initial_equity,
                    news_filter, news_relvol_bypass=rvb, news_tier_bypass=tb)
            period_totals[label].append(m["total_pnl"])
            print(
                f"  {label:<28} {m['total_trades']:>7,} {m['win_rate']:>6.1%} "
                f"${m['avg_winner']:>7,.0f} ${m['avg_loser']:>7,.0f} "
                f"${m['total_pnl']:>9,.0f} {m['total_return_pct']:>7.1f}% "
                f"${m['max_drawdown']:>9,.0f} {m['max_drawdown_pct']:>6.1f}%",
                flush=True,
            )

    print(f"\n{'='*106}")
    print("  COMBINED PnL  (dynamic equity, 20% DV cap)")
    print(f"{'='*106}")
    print(f"  {'Config':<28} {'2022':>12} {'Jan-Apr 25':>12} {'Jun25-May26':>13} {'TOTAL':>12}")
    print("  " + "-" * 68)
    for label, *_ in configs:
        vals  = period_totals[label]
        total = sum(vals)
        print(f"  {label:<28} ${vals[0]:>10,.0f} ${vals[1]:>10,.0f} ${vals[2]:>11,.0f} ${total:>10,.0f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
