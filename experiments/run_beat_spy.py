"""
Find configs that beat S&P +30% on the 2025-26 bull period.

Base: day_high_5pct (best from hardstop experiments) + dynamic equity + news tier-4.

Sweeps:
  A) min_entry_tier: 1 (all) / 2 (tier-2+) / 3 (tier-3+)
  B) target_atr_multiple: 3× / 4× / 5×  (current is 3×)
  C) stop_atr_multiple:   1.5× (current) / 1.0×

Full grid: 3 tiers × 3 targets × 2 stops = 18 configs.
All use: day_high_5pct, dynamic equity, news_mode=require, tier-4 bypass, slippage=0.001.
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

START = date(2025, 6, 1)
END   = date(2026, 5, 28)
SPY_RETURN = 30.0  # benchmark %

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
    min_tier: int,
    stop_mult: float,
    target_mult: float,
) -> dict:
    cfg.stop_atr_multiple   = stop_mult
    cfg.target_atr_multiple = target_mult

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
        min_entry_tier=min_tier,
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
        day_pnl    = sum(t.pnl for t in result.trades if t.pnl is not None)
        running_eq = max(running_eq + day_pnl, 1.0)

    trades = all_trades
    if not trades:
        return {"trades": 0, "wr": 0.0, "ret_pct": 0.0,
                "hard_stops": 0, "targets": 0, "avg_w": 0.0, "avg_l": 0.0}

    wins   = [t for t in trades if t.pnl and t.pnl > 0]
    losses = [t for t in trades if t.pnl and t.pnl <= 0]
    stops  = [t for t in trades if t.exit_reason == "hard_stop"]
    tgts   = [t for t in trades if t.exit_reason == "target"]
    net    = sum(t.pnl for t in trades if t.pnl is not None)

    return {
        "trades":     len(trades),
        "wr":         len(wins) / len(trades) * 100,
        "ret_pct":    net / initial_equity * 100,
        "hard_stops": len(stops),
        "targets":    len(tgts),
        "avg_w":      sum(t.pnl for t in wins) / len(wins) if wins else 0.0,
        "avg_l":      sum(t.pnl for t in losses) / len(losses) if losses else 0.0,
    }


def main():
    api_key    = os.environ["APCA_API_KEY_ID"]
    secret_key = os.environ["APCA_API_SECRET_KEY"]
    base_url   = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

    initial_equity = broker.get_account_info()["portfolio_value"]
    print(f"Starting equity: ${initial_equity:,.0f}   |   Benchmark: S&P +{SPY_RETURN}%  ({+SPY_RETURN/100*initial_equity:,.0f})")
    print(f"Period: {START} → {END}   |   Base filter: day_high_5pct\n")

    cfg      = make_gap_hold_config()
    screener = CandidateScreener(copy.copy(cfg), api_key, secret_key, base_url)
    print("Loading screener...", flush=True)
    screener.preload(START, END)
    print(f"  {len(screener._daily_cache):,} symbols loaded.\n", flush=True)

    fetcher     = BarFetcher(api_key, secret_key)
    news_filter = NewsFilter(api_key, secret_key, cache_only=False)
    days        = trading_days(START, END)

    min_tiers    = [1, 2, 3]
    stop_mults   = [1.5, 1.0]
    target_mults = [3.0, 4.0, 5.0]

    hdr = (f"  {'Tier':>4} {'Stop':>5} {'Tgt':>4} | "
           f"{'Trades':>7} {'WR':>6} {'Ret%':>7} {'Beat?':>6} | "
           f"{'HStops':>7} {'Targets':>8} {'Avg W':>9} {'Avg L':>9}")
    print(hdr)
    print("  " + "-" * 102)

    results = []
    for tier in min_tiers:
        for stop in stop_mults:
            for tgt in target_mults:
                label = f"T{tier} S{stop:.1f}× Tgt{tgt:.0f}×"
                print(f"  {label:<20}  running...", end="\r", flush=True)
                m = run_config(
                    copy.copy(cfg), days, screener, fetcher, initial_equity,
                    news_filter, tier, stop, tgt,
                )
                beat = "YES ✓" if m["ret_pct"] >= SPY_RETURN else "     "
                print(
                    f"  {tier:>4}  {stop:.1f}×  {tgt:.0f}× | "
                    f"{m['trades']:>7,} {m['wr']:>5.1f}% {m['ret_pct']:>6.1f}% {beat:>6} | "
                    f"{m['hard_stops']:>7,} {m['targets']:>8,} "
                    f"${m['avg_w']:>8,.0f} ${m['avg_l']:>8,.0f}",
                    flush=True,
                )
                results.append((m["ret_pct"], tier, stop, tgt, m))

    print()
    winners = [(r, t, s, tg, m) for r, t, s, tg, m in results if r >= SPY_RETURN]
    if winners:
        print(f"  {len(winners)} config(s) beat S&P +{SPY_RETURN}%:")
        for r, t, s, tg, m in sorted(winners, reverse=True):
            print(f"    Tier={t} Stop={s:.1f}× Target={tg:.0f}×  →  {r:+.1f}%  ({m['trades']} trades, {m['wr']:.1f}% WR)")
    else:
        best = max(results, key=lambda x: x[0])
        print(f"  No config beat {SPY_RETURN}%. Best: Tier={best[1]} Stop={best[2]:.1f}× Target={best[3]:.0f}×  →  {best[0]:+.1f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
