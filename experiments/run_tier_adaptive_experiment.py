"""
Tier-gating experiment — all three periods.

Forensics finding (2023, base config):
  - Hard stops account for the ENTIRE 2023 loss (-$41.5k of -$40.1k net).
  - All 64 news-catalyst trades were tier 1-3 at entry: net -$40.7k, 20% WR.
  - All 11 tier-4 entries (vol/ROC extremes) broke even: +$646, 45.5% WR.
  - Mid-strength setups (tier 2-3, day move 30-50%) are the losers.

Hypothesis: in narrow-breadth regimes only extreme-confirmation entries carry
follow-through. min_entry_tier delays ALL entries (news included) until the
entry bar's confidence hits the tier floor.

Configs:
  A baseline                 (no MA, no tier gate)
  B adaptive MA              (narrow day -> 10d MA)        [current best]
  C adaptive tier>=3         (narrow day -> tier 3 floor)
  D adaptive tier>=4         (narrow day -> tier 4 floor)
  E adaptive MA + tier>=3    (narrow day -> both)
  F adaptive MA + tier>=4    (narrow day -> both)
  G always tier>=4           (every day, sanity check vs 2025-26)

Narrow day = < 35 candidates (breadth threshold from prior experiment).
Base: top-50 cap, day_high_5pct, vol>=80%, target 4xATR, news tier-4 bypass.
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

PERIODS = [
    ("2022 bear",    date(2022, 1, 3), date(2022, 12, 30), -19.0),
    ("2023 mixed",   date(2023, 1, 3), date(2023, 12, 29), +26.0),
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

def baseline_vol(screener: CandidateScreener, sym: str, as_of: date) -> float:
    df = screener._daily_cache.get(sym)
    if df is None or df.empty:
        return 0.0
    past = df[df.index < pd.Timestamp(as_of)]
    return float(past["volume"].tail(20).mean()) / 390.0 if not past.empty else 0.0

def stocks_above_daily_ma(screener, candidates, as_of: date, ma_period: int) -> Set[str]:
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
        ma = float(past["close"].iloc[-ma_period:].mean())
        if float(past["close"].iloc[-1]) >= ma:
            result.add(sym)
    return result


def run_period(
    cfg, start: date, end: date,
    screener: CandidateScreener, fetcher: BarFetcher,
    initial_equity: float, news_filter: NewsFilter,
    narrow_ma_period: int,    # 0 = MA filter off
    narrow_min_tier: int,     # 0 = tier gate off on narrow days
    always_min_tier: int,     # 0 = no global tier gate
) -> dict:
    running_eq = initial_equity
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
    )

    for d in trading_days(start, end):
        if not spy_uptrend(d):
            continue
        cands = screener.candidates_for_date(d)
        narrow = len(cands) < BREADTH_THRESHOLD

        eligible = cands
        if narrow and narrow_ma_period > 0:
            eligible = [c for c in cands
                        if c in stocks_above_daily_ma(screener, cands, d, narrow_ma_period)]

        # Per-day tier floor: global floor always applies; narrow floor on top.
        sim._min_entry_tier = max(always_min_tier, narrow_min_tier if narrow else 0)

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

    trades = all_trades
    if not trades:
        return {"trades": 0, "wr": 0.0, "ret_pct": 0.0, "hard_stops": 0, "targets": 0}
    wins  = [t for t in trades if t.pnl and t.pnl > 0]
    stops = [t for t in trades if t.exit_reason == "hard_stop"]
    tgts  = [t for t in trades if t.exit_reason == "target"]
    net   = sum(t.pnl for t in trades if t.pnl is not None)
    return {
        "trades": len(trades),
        "wr": len(wins) / len(trades) * 100,
        "ret_pct": net / initial_equity * 100,
        "hard_stops": len(stops),
        "targets": len(tgts),
    }


def main():
    api_key    = os.environ["APCA_API_KEY_ID"]
    secret_key = os.environ["APCA_API_SECRET_KEY"]
    base_url   = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

    initial_equity = broker.get_account_info()["portfolio_value"]
    print(f"Starting equity: ${initial_equity:,.0f}")
    print(f"Narrow day = <{BREADTH_THRESHOLD} candidates\n")

    fetcher     = BarFetcher(api_key, secret_key)
    news_filter = NewsFilter(api_key, secret_key, cache_only=False)

    # (label, narrow_ma_period, narrow_min_tier, always_min_tier)
    experiments = [
        ("A baseline",                 0, 0, 0),
        ("B adaptive MA (current)",   10, 0, 0),
        ("C adaptive tier>=3",         0, 3, 0),
        ("D adaptive tier>=4",         0, 4, 0),
        ("E adaptive MA + tier>=3",   10, 3, 0),
        ("F adaptive MA + tier>=4",   10, 4, 0),
        ("G always tier>=4",           0, 0, 4),
    ]

    for period_label, start, end, spy_ret in PERIODS:
        base_cfg = make_gap_hold_config()
        base_cfg.target_atr_multiple = 4.0
        base_cfg.stage2_min_vol_vs_prev_bar = 0.80

        screener = CandidateScreener(copy.copy(base_cfg), api_key, secret_key, base_url)
        print(f"Loading {period_label}...", flush=True)
        screener.preload(start, end)

        print(f"{'='*84}")
        print(f"  {period_label}  ({start}->{end})  S&P: {spy_ret:+.0f}%")
        print(f"{'='*84}")
        print(f"  {'Config':<28} {'Trades':>7} {'WR':>6} {'Return':>8} "
              f"{'vs S&P':>8} {'HStops':>7} {'Tgts':>6}")
        print("  " + "-" * 80)

        for label, ma_p, ntier, atier in experiments:
            cfg = copy.copy(base_cfg)
            m = run_period(cfg, start, end, screener, fetcher,
                           initial_equity, news_filter, ma_p, ntier, atier)
            vs_spy = m["ret_pct"] - spy_ret
            flag = " *" if m["ret_pct"] > 0 else "  "
            print(
                f"  {label:<28} {m['trades']:>7,} {m['wr']:>5.1f}% "
                f"{m['ret_pct']:>+7.1f}% {vs_spy:>+7.1f}%{flag} "
                f"{m['hard_stops']:>7} {m['targets']:>6}",
                flush=True,
            )
        print()

    print("Done.")


if __name__ == "__main__":
    main()
