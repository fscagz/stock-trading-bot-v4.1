"""
Volume acceleration filter experiment — all three periods.

Sweeps stage2_min_vol_vs_prev_bar: 0.0 (off) / 0.5 / 0.7 / 0.8 / 0.9 / 1.0

Base: top-50 cap, day_high_5pct, target 4×ATR, news tier-4, dynamic equity.

Signal analysis from 2023 found:
  - Losers  median vol_vs_prev = 0.69 (volume declining at entry)
  - Winners median vol_vs_prev = 1.10 (volume still building)
  Threshold of 0.7-0.9 should filter losing entries without touching winners.
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
from bot.backtest.bar_fetcher import BarFetcher
from bot.backtest.candidate_screener import CandidateScreener
from bot.backtest.news_filter import NewsFilter
from bot.backtest.simulator import Simulator
from bot.config import make_gap_hold_config
from bot.data.daily_loader import get_daily
from bot.intraday.types import TradeRecord

PERIODS = [
    ("2022 bear",    date(2022, 1,  3),  date(2022, 12, 30), -19.0),
    ("2023 mixed",   date(2023, 1,  3),  date(2023, 12, 29), +26.0),
    ("2025-26 bull", date(2025, 6,  1),  date(2026,  5, 28), +30.0),
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
) -> dict:
    running_eq  = initial_equity
    all_trades: List[TradeRecord] = []
    cache_dir   = fetcher._cache_dir

    sim = Simulator(
        cfg, initial_equity,
        slippage_pct=0.001,
        overnight_holds=False,
        market_order_fill=True,
        news_filter=news_filter,
        news_mode="require",
        news_tier_bypass=4,
        stage2_min_dist_from_day_high_pct=0.05,
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
        result     = sim.run_day(d, bars_by_sym, baselines)
        all_trades.extend(result.trades)
        day_pnl    = sum(t.pnl for t in result.trades if t.pnl is not None)
        running_eq = max(running_eq + day_pnl, 1.0)

    trades = all_trades
    if not trades:
        return {"trades": 0, "wr": 0.0, "ret_pct": 0.0, "hard_stops": 0, "targets": 0}

    wins  = [t for t in trades if t.pnl and t.pnl > 0]
    stops = [t for t in trades if t.exit_reason == "hard_stop"]
    tgts  = [t for t in trades if t.exit_reason == "target"]
    net   = sum(t.pnl for t in trades if t.pnl is not None)

    return {
        "trades":     len(trades),
        "wr":         len(wins) / len(trades) * 100,
        "ret_pct":    net / initial_equity * 100,
        "hard_stops": len(stops),
        "targets":    len(tgts),
    }


def main():
    api_key    = os.environ["APCA_API_KEY_ID"]
    secret_key = os.environ["APCA_API_SECRET_KEY"]
    base_url   = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

    initial_equity = broker.get_account_info()["portfolio_value"]
    print(f"Starting equity: ${initial_equity:,.0f}")
    print(f"Base: top-50 cap, day_high_5pct, target=4×ATR, tier-4 bypass, dynamic equity\n")
    print(f"Signal analysis baseline: 2023 losers median vol_vs_prev=0.69, winners=1.10\n")

    fetcher     = BarFetcher(api_key, secret_key)
    news_filter = NewsFilter(api_key, secret_key, cache_only=False)

    # vol_vs_prev thresholds to test
    thresholds = [
        ("no filter  (0.00)", 0.00),
        ("vol≥50%    (0.50)", 0.50),
        ("vol≥70%    (0.70)", 0.70),
        ("vol≥80%    (0.80)", 0.80),
        ("vol≥90%    (0.90)", 0.90),
        ("vol≥100%   (1.00)", 1.00),
    ]

    for period_label, start, end, spy_ret in PERIODS:
        base_cfg = make_gap_hold_config()
        base_cfg.target_atr_multiple = 4.0

        screener = CandidateScreener(copy.copy(base_cfg), api_key, secret_key, base_url)
        print(f"Loading {period_label}...", flush=True)
        screener.preload(start, end)
        print(f"  {len(screener._daily_cache):,} symbols loaded\n", flush=True)

        days = trading_days(start, end)

        print(f"{'='*80}")
        print(f"  {period_label}  ({start}→{end})  S&P benchmark: {spy_ret:+.0f}%")
        print(f"{'='*80}")
        hdr = (f"  {'Config':<24} {'Trades':>7} {'WR':>6} {'Return':>8} "
               f"{'vs S&P':>8} {'HStops':>7} {'Targets':>8}")
        print(hdr)
        print("  " + "-" * 72)

        for label, threshold in thresholds:
            cfg = copy.copy(base_cfg)
            cfg.stage2_min_vol_vs_prev_bar = threshold

            m      = run_config(cfg, days, screener, fetcher, initial_equity, news_filter)
            vs_spy = m["ret_pct"] - spy_ret
            flag   = " ✓" if vs_spy >= 0 else "  "

            print(
                f"  {label:<24} {m['trades']:>7,} {m['wr']:>5.1f}% "
                f"{m['ret_pct']:>+7.1f}% {vs_spy:>+7.1f}%{flag} "
                f"{m['hard_stops']:>7} {m['targets']:>8}",
                flush=True,
            )
        print()

    print("Done.")


if __name__ == "__main__":
    main()
