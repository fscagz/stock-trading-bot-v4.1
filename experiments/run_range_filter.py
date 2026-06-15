"""
Range-position filter sweep.

Tests whether rejecting entries near the top of the day's intraday range
improves performance. For each threshold T, an entry is skipped when:

    (bar.close - day_low) / (day_high - day_low) > T

Sweeps T = [none, 0.60, 0.70, 0.75, 0.80, 0.90] using the H+ config
(hybrid tier-4 + MA10 narrow-day filter) across all available periods.
"""
from __future__ import annotations
import copy, os, warnings, logging
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from typing import Dict, List, Optional, Set

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

BREADTH_THRESHOLD = 35
PB_ATR, PB_TTL, PB_TARGET = 2.0, 60, 2.5
CHASE_TARGET = 4.0
DV_CAP = 0.20

RANGE_THRESHOLDS = [0.0, 0.60, 0.70, 0.75, 0.80, 0.90]

PERIODS = [
    ("2022 bear",    date(2022, 1, 3), date(2022, 12, 30), -19.0),
    ("2023 mixed",   date(2023, 1, 3), date(2023, 12, 29), +26.0),
    ("OOS H1-2025",  date(2025, 1, 2), date(2025, 5, 28),  +0.5),
    ("2025-26 bull", date(2025, 6, 1), date(2026, 5, 28),  +30.0),
]

_spy: Optional[pd.DataFrame] = None

def spy_uptrend(d: date) -> bool:
    global _spy
    if _spy is None:
        _spy = get_daily("SPY", start="2021-06-01", end="2026-06-10")
        _spy["ma20"] = _spy["close"].rolling(20).mean()
    past = _spy[_spy.index < pd.Timestamp(d)].dropna(subset=["ma20"])
    if past.empty:
        return True
    return float(past.iloc[-1]["close"]) >= float(past.iloc[-1]["ma20"])


def trading_days(s: date, e: date) -> List[date]:
    days, d = [], s
    while d <= e:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def baseline_vol(screener, sym, as_of):
    df = screener._daily_cache.get(sym)
    if df is None or df.empty:
        return 0.0
    past = df[df.index < pd.Timestamp(as_of)]
    return float(past["volume"].tail(20).mean()) / 390.0 if not past.empty else 0.0


def stocks_above_daily_ma(screener, candidates, as_of, ma_period) -> Set[str]:
    result = set()
    for sym in candidates:
        df = screener._daily_cache.get(sym)
        if df is None or df.empty:
            result.add(sym)
            continue
        past = df[df.index < pd.Timestamp(as_of)]
        if len(past) < ma_period:
            result.add(sym)
            continue
        if float(past["close"].iloc[-1]) >= float(past["close"].iloc[-ma_period:].mean()):
            result.add(sym)
    return result


def run_period(cfg, start, end, screener, fetcher, initial_equity, news_filter,
               range_pct: float) -> dict:
    running_eq = initial_equity
    peak = initial_equity
    max_dd_pct = 0.0
    all_trades: List[TradeRecord] = []
    cache_dir = fetcher._cache_dir

    sim = Simulator(
        cfg, initial_equity,
        slippage_pct=0.001,
        overnight_holds=False,
        market_order_fill=True,
        news_filter=news_filter,
        news_mode="require",
        news_tier_bypass=4,
        stage2_min_dist_from_day_high_pct=0.05,
        max_position_dv_pct=DV_CAP,
        pullback_entry_atr=PB_ATR,
        pullback_entry_ttl_bars=PB_TTL,
        pullback_target_atr=PB_TARGET,
        pullback_chase_tier=4,
        max_range_position_pct=range_pct,
    )

    for d in trading_days(start, end):
        if not spy_uptrend(d):
            continue
        cands = screener.candidates_for_date(d)
        eligible = cands
        if len(cands) < BREADTH_THRESHOLD:
            eligible = [c for c in cands
                        if c in stocks_above_daily_ma(screener, cands, d, 10)]

        cached = [s for s in eligible if (cache_dir / f"{s}_{d}.json").exists()]
        if not cached:
            continue
        bars_by_sym: Dict = {}
        baselines: Dict = {}
        with ThreadPoolExecutor(max_workers=16) as pool:
            for sym, bars, bl in pool.map(
                lambda s: (s, fetcher.fetch(s, d), baseline_vol(screener, s, d)),
                cached,
            ):
                if bars and bl > 0:
                    bars_by_sym[sym] = bars
                    baselines[sym] = bl
        if not bars_by_sym:
            continue
        sim._initial_equity = running_eq
        result = sim.run_day(d, bars_by_sym, baselines)
        all_trades.extend(result.trades)
        day_pnl = sum(t.pnl for t in result.trades if t.pnl is not None)
        running_eq = max(running_eq + day_pnl, 1.0)
        peak = max(peak, running_eq)
        if peak > 0:
            max_dd_pct = max(max_dd_pct, (peak - running_eq) / peak * 100)

    trades = all_trades
    if not trades:
        return {"trades": 0, "wr": 0.0, "ret_pct": 0.0, "max_dd": 0.0}
    wins = [t for t in trades if t.pnl and t.pnl > 0]
    net = running_eq - initial_equity
    pbs = [t for t in trades if "momentum_pullback" in (t.signals or [])]
    return {
        "trades": len(trades),
        "wr": len(wins) / len(trades) * 100,
        "ret_pct": net / initial_equity * 100,
        "max_dd": max_dd_pct,
        "pullbacks": len(pbs),
        "chases": len(trades) - len(pbs),
    }


def main():
    api_key    = os.environ["APCA_API_KEY_ID"]
    secret_key = os.environ["APCA_API_SECRET_KEY"]
    base_url   = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

    initial_equity = broker.get_account_info()["portfolio_value"]
    print(f"Starting equity: ${initial_equity:,.0f}")
    print(f"H+ config (hybrid tier-4 + MA10 narrow-day filter)")
    print(f"Range filter: skip entries where close > T × day range\n")

    fetcher     = BarFetcher(api_key, secret_key)
    news_filter = NewsFilter(api_key, secret_key, cache_only=False)

    threshold_labels = {0.0: "none (baseline)", 0.60: "≤60%", 0.70: "≤70%",
                        0.75: "≤75%", 0.80: "≤80%", 0.90: "≤90%"}

    for period_label, start, end, spy_ret in PERIODS:
        base = make_gap_hold_config()
        base.stage2_min_vol_vs_prev_bar = 0.80
        base.target_atr_multiple = CHASE_TARGET

        screener = CandidateScreener(copy.copy(base), api_key, secret_key, base_url)
        print(f"Loading {period_label}...", flush=True)
        screener.preload(start, end)

        print(f"{'='*82}")
        print(f"  {period_label}  ({start}->{end})  S&P: {spy_ret:+.1f}%")
        print(f"{'='*82}")
        print(f"  {'Range filter':<20} {'Trades':>6} {'WR':>6} {'Return':>8} "
              f"{'MaxDD':>7} {'vs S&P':>8} {'Chase':>6} {'PB':>5}")
        print("  " + "-" * 70)

        for t in RANGE_THRESHOLDS:
            cfg = copy.copy(base)
            m = run_period(cfg, start, end, screener, fetcher,
                           initial_equity, news_filter, t)
            vs = m["ret_pct"] - spy_ret
            flag = " *" if m["ret_pct"] > 0 else "  "
            label = threshold_labels[t]
            print(f"  {label:<20} {m['trades']:>6,} {m['wr']:>5.1f}% "
                  f"{m['ret_pct']:>+7.1f}%{flag} {m['max_dd']:>6.1f}% {vs:>+7.1f}% "
                  f"{m['chases']:>6} {m['pullbacks']:>5}", flush=True)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
