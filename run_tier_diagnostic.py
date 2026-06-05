"""
Diagnostic: compare confidence tier distribution before and after recalibration,
and show full PnL impact for 2025-26 and 2022.

Old tiers: 1x/2x/4x/8x, roc_range=3x, vol_range=3x, no min DV
New tiers: 1x/2x/3x/4x, roc_range=4x, vol_range=7x, min DV $500k
"""
from __future__ import annotations
import copy, os, sys, warnings, logging
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

import pandas as pd
import bot.broker_alpaca as broker
from bot.backtest.backtest_metrics import compute_metrics
from bot.backtest.bar_fetcher import BarFetcher
from bot.backtest.candidate_screener import CandidateScreener
from bot.backtest.simulator import Simulator
from bot.config import make_long_config
from bot.data.daily_loader import get_daily
from bot.intraday.types import TradeRecord
from bot.momentum.validator import MomentumValidator

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
    if past.empty: return True
    return float(past.iloc[-1]["close"]) >= float(past.iloc[-1]["ma20"])

def trading_days(start: date, end: date) -> List[date]:
    days, d = [], start
    while d <= end:
        if d.weekday() < 5: days.append(d)
        d += timedelta(days=1)
    return days

def baseline_from_cache(screener, sym, as_of):
    df = screener._daily_cache.get(sym)
    if df is None or df.empty: return 0.0
    past = df[df.index < pd.Timestamp(as_of)]
    return float(past["volume"].tail(20).mean()) / 390.0 if not past.empty else 0.0


class TierTrackingSimulator(Simulator):
    """Simulator that also records which confidence tier each entry hit."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tier_counts: Counter = Counter()
        self.tier_pnl: Dict[str, float] = defaultdict(float)

    def run_day(self, trade_date, bars_by_symbol, baseline_volumes):
        result = super().run_day(trade_date, bars_by_symbol, baseline_volumes)
        return result

    # Override confidence_multiplier to track tier assignments
    # We do this by patching the config object's method after the fact
    # via a post-processing pass over trades.


def run_with_tier_tracking(cfg, days, screener, fetcher, eq):
    """Run backtest and also report tier distribution by scanning all signal bars."""
    sim = Simulator(cfg, eq, slippage_pct=0.001,
                    overnight_holds=False, market_order_fill=True)
    validator = MomentumValidator(cfg)
    trades: List[TradeRecord] = []
    cache_dir = fetcher._cache_dir
    tier_counts: Counter = Counter()
    tier_pnl: Dict[str, list] = defaultdict(list)

    for d in days:
        if not spy_uptrend(d): continue
        cands = screener.candidates_for_date(d)
        cached = [s for s in cands if (cache_dir / f"{s}_{d}.json").exists()]
        if not cached: continue

        def _fetch(sym):
            bars = fetcher.fetch(sym, d)
            bl = baseline_from_cache(screener, sym, d)
            return sym, bars, bl

        bars_by_sym, baselines = {}, {}
        with ThreadPoolExecutor(max_workers=16) as pool:
            for sym, bars, bl in pool.map(_fetch, cached):
                if bars and bl > 0:
                    bars_by_sym[sym] = bars
                    baselines[sym] = bl

        if not bars_by_sym: continue

        # Track tier for each entry by replaying the bar stream
        local_validator = MomentumValidator(cfg)
        for sym, bars in bars_by_sym.items():
            bl = baselines[sym]
            for bar in bars:
                if local_validator.validate(bar, bl):
                    score = local_validator.confidence_score(bar, bl)
                    mult = cfg.confidence_multiplier(score)
                    tier = {
                        cfg.confidence_tier1_multiplier: "tier1",
                        cfg.confidence_tier2_multiplier: "tier2",
                        cfg.confidence_tier3_multiplier: "tier3",
                        cfg.confidence_tier4_multiplier: "tier4",
                    }.get(mult, f"tier?({mult}x)")
                    tier_counts[tier] += 1
                    break  # only first signal per sym per day

        result = sim.run_day(d, bars_by_sym, baselines)
        for t in result.trades:
            trades.append(t)

    m = compute_metrics(trades, eq)
    return m, tier_counts


def make_old_config():
    """Pre-change config: 1x/2x/4x/8x, roc=3x, vol=3x, no min DV."""
    cfg = make_long_config()
    cfg.confidence_tier1_multiplier = 1.0
    cfg.confidence_tier2_multiplier = 2.0
    cfg.confidence_tier3_multiplier = 4.0
    cfg.confidence_tier4_multiplier = 8.0
    cfg.confidence_score_roc_range_mult = 3.0
    cfg.confidence_score_vol_range_mult = 3.0
    cfg.min_avg_dollar_volume = 0.0
    return cfg


def main():
    api_key = os.environ["APCA_API_KEY_ID"]
    secret_key = os.environ["APCA_API_SECRET_KEY"]
    base_url = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
    eq = broker.get_account_info()["portfolio_value"]
    print(f"Equity: ${eq:,.0f}\n")
    fetcher = BarFetcher(api_key, secret_key)

    for period_label, (start, end) in PERIODS.items():
        print(f"\n{'='*90}")
        print(f"  {period_label}  ({start} → {end})")
        print(f"{'='*90}")

        old_cfg = make_old_config()
        new_cfg = make_long_config()  # uses the updated make_long_config

        # Load screeners
        scr_old = CandidateScreener(copy.copy(old_cfg), api_key, secret_key, base_url)
        print("  Loading screener (no DV filter)...", flush=True)
        scr_old.preload(start, end)

        scr_new = CandidateScreener(copy.copy(new_cfg), api_key, secret_key, base_url)
        scr_new._daily_cache = scr_old._daily_cache
        scr_new._universe = scr_old._universe
        scr_new._build_candidates_index(start, end)

        days = trading_days(start, end)

        print("  Running OLD config (1×/2×/4×/8×, no min DV)...", flush=True)
        m_old, tiers_old = run_with_tier_tracking(old_cfg, days, scr_old, fetcher, eq)

        print("  Running NEW config (1×/2×/3×/4×, $500k min DV)...", flush=True)
        m_new, tiers_new = run_with_tier_tracking(new_cfg, days, scr_new, fetcher, eq)

        # Print results
        hdr = f"  {'Config':<30} {'Trades':>7} {'WR':>7} {'PnL':>11} {'MaxDD':>10} {'Hold':>7}"
        print(hdr)
        print("  " + "-"*80)

        def row(lbl, m):
            print(f"  {lbl:<30} {m['total_trades']:>7d} {m['win_rate']:>6.1%} "
                  f"${m['total_pnl']:>10,.0f} ${m['max_drawdown']:>9,.0f} "
                  f"{m['avg_hold_minutes']:>5.1f}m", flush=True)

        row("OLD (1×/2×/4×/8×, no DV)", m_old)
        row("NEW (1×/2×/3×/4×, $500k DV)", m_new)

        # Tier distributions
        total_old = sum(tiers_old.values()) or 1
        total_new = sum(tiers_new.values()) or 1
        print()
        print(f"  {'Tier distribution':}")
        print(f"  {'Tier':<10} {'OLD count':>10} {'OLD %':>8} {'NEW count':>10} {'NEW %':>8}")
        print("  " + "-"*50)
        for tier in ["tier1", "tier2", "tier3", "tier4"]:
            c_old = tiers_old.get(tier, 0)
            c_new = tiers_new.get(tier, 0)
            print(f"  {tier:<10} {c_old:>10d} {c_old/total_old:>7.1%} "
                  f"{c_new:>10d} {c_new/total_new:>7.1%}")
        print(f"  {'TOTAL':<10} {total_old:>10d} {'100%':>8} {total_new:>10d} {'100%':>8}")

    print("\nDone.")


if __name__ == "__main__":
    main()
