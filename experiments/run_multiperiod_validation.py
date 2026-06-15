"""
Multi-period validation of the 18-config grid from run_beat_spy.py.

Tests all combinations of:
  min_entry_tier:    1 / 2 / 3
  stop_atr_multiple: 1.5× / 1.0×
  target_atr_multiple: 3× / 4× / 5×

Across three market regimes:
  2022     — bear   (S&P ≈ -19%)
  2023     — mixed  (S&P ≈ +26%)
  2025-26  — bull   (S&P ≈ +30%)

All configs: day_high_5pct, dynamic equity, news tier-4, slippage=0.001.
"""
from __future__ import annotations
import copy, os, warnings, logging
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

PERIODS = [
    ("2022 bear",   date(2022, 1,  3),  date(2022, 12, 30), -19.0),
    ("2023 mixed",  date(2023, 1,  3),  date(2023, 12, 29), +26.0),
    ("2025-26 bull",date(2025, 6,  1),  date(2026,  5, 28), +30.0),
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
        result     = sim.run_day(d, bars_by_sym, baselines)
        all_trades.extend(result.trades)
        day_pnl    = sum(t.pnl for t in result.trades if t.pnl is not None)
        running_eq = max(running_eq + day_pnl, 1.0)

    trades = all_trades
    if not trades:
        return {"trades": 0, "wr": 0.0, "ret_pct": 0.0}

    wins = [t for t in trades if t.pnl and t.pnl > 0]
    net  = sum(t.pnl for t in trades if t.pnl is not None)
    return {
        "trades":  len(trades),
        "wr":      len(wins) / len(trades) * 100,
        "ret_pct": net / initial_equity * 100,
    }


def main():
    api_key    = os.environ["APCA_API_KEY_ID"]
    secret_key = os.environ["APCA_API_SECRET_KEY"]
    base_url   = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

    initial_equity = broker.get_account_info()["portfolio_value"]
    print(f"Starting equity: ${initial_equity:,.0f}\n")

    fetcher     = BarFetcher(api_key, secret_key)
    news_filter = NewsFilter(api_key, secret_key, cache_only=False)

    min_tiers    = [1, 2, 3]
    stop_mults   = [1.5, 1.0]
    target_mults = [3.0, 4.0, 5.0]

    # Collect all results: key=(tier,stop,tgt) → {period_label: ret_pct}
    all_results: Dict[Tuple, Dict[str, dict]] = {}

    for period_label, start, end, spy_ret in PERIODS:
        print(f"\n{'='*70}")
        print(f"  {period_label}  ({start} → {end})  |  S&P benchmark: {spy_ret:+.0f}%")
        print(f"{'='*70}")

        cfg      = make_gap_hold_config()
        screener = CandidateScreener(copy.copy(cfg), api_key, secret_key, base_url)
        print(f"  Loading screener...", flush=True)
        screener.preload(start, end)
        print(f"  {len(screener._daily_cache):,} symbols. Running configs...\n", flush=True)

        days = trading_days(start, end)

        hdr = (f"  {'Tier':>4} {'Stop':>5} {'Tgt':>4} | "
               f"{'Trades':>7} {'WR':>6} {'Ret%':>7} {'vs SPY':>8}")
        print(hdr)
        print("  " + "-" * 55)

        for tier in min_tiers:
            for stop in stop_mults:
                for tgt in target_mults:
                    key = (tier, stop, tgt)
                    m = run_config(
                        copy.copy(cfg), days, screener, fetcher, initial_equity,
                        news_filter, tier, stop, tgt,
                    )
                    vs = m["ret_pct"] - spy_ret
                    flag = " ✓" if vs >= 0 else "  "
                    print(
                        f"  {tier:>4}  {stop:.1f}×  {tgt:.0f}× | "
                        f"{m['trades']:>7,} {m['wr']:>5.1f}% {m['ret_pct']:>6.1f}% "
                        f"{vs:>+7.1f}%{flag}",
                        flush=True,
                    )
                    if key not in all_results:
                        all_results[key] = {}
                    all_results[key][period_label] = m

    # ── Summary table across all periods ───────────────────────────────────
    period_labels = [p[0] for p in PERIODS]
    spy_rets      = {p[0]: p[3] for p in PERIODS}

    print(f"\n\n{'='*90}")
    print("  SUMMARY — Return % across all three periods")
    print(f"{'='*90}")
    print(f"  {'Tier':>4} {'Stop':>5} {'Tgt':>4} | ", end="")
    for lbl in period_labels:
        print(f"  {lbl[:12]:>12}", end="")
    print("  | Combined  Notes")
    print("  " + "-" * 88)

    ranked = []
    for key, period_data in all_results.items():
        tier, stop, tgt = key
        rets = [period_data.get(lbl, {}).get("ret_pct", float("nan")) for lbl in period_labels]
        combined = sum(r for r in rets if r == r)  # ignore nan
        beats_all = all(
            r >= spy_rets[lbl]
            for lbl, r in zip(period_labels, rets)
            if r == r
        )
        ranked.append((combined, beats_all, key, rets))

    for combined, beats_all, key, rets in sorted(ranked, reverse=True):
        tier, stop, tgt = key
        row = f"  {tier:>4}  {stop:.1f}×  {tgt:.0f}× | "
        for lbl, r in zip(period_labels, rets):
            marker = "✓" if r >= spy_rets[lbl] else " "
            row += f"  {r:>+10.1f}%{marker}"
        row += f"  | {combined:>+7.1f}%"
        if beats_all:
            row += "  ← beats SPY all 3"
        print(row)

    print("\nDone.")


if __name__ == "__main__":
    main()
