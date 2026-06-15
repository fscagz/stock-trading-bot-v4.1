"""
Out-of-sample validation: Jan 2 - May 28, 2025.

This window was NOT used in any tier/MA/breadth experiment (tuning used
2022, 2023, and Jun 2025 - May 2026), and the earlier 18-config grid only
tested min_entry_tier 1-3. Tier-4 gating has never seen this data.

Configs:
  A baseline            (no gate)
  B adaptive MA         (narrow day -> 10d MA)
  G always tier>=4
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

START, END = date(2025, 1, 2), date(2025, 5, 28)
BREADTH_THRESHOLD = 35

_spy: Optional[pd.DataFrame] = None
def _load_spy():
    global _spy
    if _spy is None:
        _spy = get_daily("SPY", start="2024-06-01", end="2025-06-05")
        _spy["ma20"] = _spy["close"].rolling(20).mean()
    return _spy

def spy_uptrend(d: date) -> bool:
    spy = _load_spy()
    past = spy[spy.index < pd.Timestamp(d)].dropna(subset=["ma20"])
    if past.empty:
        return True
    return float(past.iloc[-1]["close"]) >= float(past.iloc[-1]["ma20"])

def spy_period_return() -> float:
    spy = _load_spy()
    window = spy[(spy.index >= pd.Timestamp(START)) & (spy.index <= pd.Timestamp(END))]
    if window.empty:
        return 0.0
    return (float(window.iloc[-1]["close"]) / float(window.iloc[0]["close"]) - 1) * 100

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


def run_period(cfg, screener, fetcher, initial_equity, news_filter,
               narrow_ma_period: int, always_min_tier: int) -> dict:
    running_eq = initial_equity
    all_trades: List[TradeRecord] = []
    cache_dir = fetcher._cache_dir
    skipped_regime = 0

    sim = Simulator(
        cfg, initial_equity,
        slippage_pct=0.001,
        overnight_holds=False,
        market_order_fill=True,
        news_filter=news_filter,
        news_mode="require",
        news_tier_bypass=4,
        stage2_min_dist_from_day_high_pct=0.05,
        min_entry_tier=always_min_tier,
    )

    for d in trading_days(START, END):
        if not spy_uptrend(d):
            skipped_regime += 1
            continue
        cands = screener.candidates_for_date(d)
        eligible = cands
        if narrow_ma_period > 0 and len(cands) < BREADTH_THRESHOLD:
            eligible = [c for c in cands
                        if c in stocks_above_daily_ma(screener, cands, d, narrow_ma_period)]

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
        return {"trades": 0, "wr": 0.0, "ret_pct": 0.0,
                "hard_stops": 0, "targets": 0, "skipped_regime": skipped_regime}
    wins = [t for t in trades if t.pnl and t.pnl > 0]
    net = sum(t.pnl for t in trades if t.pnl is not None)
    return {
        "trades": len(trades),
        "wr": len(wins) / len(trades) * 100,
        "ret_pct": net / initial_equity * 100,
        "hard_stops": sum(1 for t in trades if t.exit_reason == "hard_stop"),
        "targets": sum(1 for t in trades if t.exit_reason == "target"),
        "skipped_regime": skipped_regime,
    }


def main():
    api_key    = os.environ["APCA_API_KEY_ID"]
    secret_key = os.environ["APCA_API_SECRET_KEY"]
    base_url   = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

    initial_equity = broker.get_account_info()["portfolio_value"]
    spy_ret = spy_period_return()
    print(f"Starting equity: ${initial_equity:,.0f}")
    print(f"OOS window: {START} -> {END}   SPY: {spy_ret:+.1f}%\n")

    cfg = make_gap_hold_config()
    cfg.target_atr_multiple = 4.0
    cfg.stage2_min_vol_vs_prev_bar = 0.80

    screener = CandidateScreener(copy.copy(cfg), api_key, secret_key, base_url)
    print("Loading screener data...", flush=True)
    screener.preload(START, END)
    print(f"  {len(screener._daily_cache):,} symbols loaded\n", flush=True)

    fetcher     = BarFetcher(api_key, secret_key)
    news_filter = NewsFilter(api_key, secret_key, cache_only=False)

    experiments = [
        ("A baseline",          0, 0),
        ("B adaptive MA",      10, 0),
        ("G always tier>=4",    0, 4),
    ]

    print(f"  {'Config':<24} {'Trades':>7} {'WR':>6} {'Return':>8} "
          f"{'vs SPY':>8} {'HStops':>7} {'Tgts':>6} {'RegimeSkipD':>12}")
    print("  " + "-" * 84)
    for label, ma_p, atier in experiments:
        m = run_period(copy.copy(cfg), screener, fetcher,
                       initial_equity, news_filter, ma_p, atier)
        vs = m["ret_pct"] - spy_ret
        flag = " *" if m["ret_pct"] > 0 else "  "
        print(f"  {label:<24} {m['trades']:>7,} {m['wr']:>5.1f}% "
              f"{m['ret_pct']:>+7.1f}% {vs:>+7.1f}%{flag} "
              f"{m['hard_stops']:>7} {m['targets']:>6} {m['skipped_regime']:>12}",
              flush=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
