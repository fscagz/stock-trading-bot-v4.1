"""
Combined backtest: old baseline vs new configuration (rv=6 + gap-hold ≥5%).

Old baseline: rv=10, bp=0.85, no gap-hold (what was live before this session).
New config:   rv=6,  bp=0.85, gap-hold ≥5% (what was just wired into the runner).

H config, $10M DV, all 5 periods.
"""
from __future__ import annotations
import copy, os, sys, warnings, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
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

PB_ATR, PB_TTL, PB_TARGET = 2.0, 60, 2.5
CHASE_TARGET = 4.0
DV_CAP = 0.20
DV = 10_000_000
SLIPPAGE = 0.002   # 0.2% per-side
GAP_HOLD_BARS = 15
GAP_HOLD_TOLERANCE = 0.02

# (label, rv, gap_hold_on, gap_min_pct)
CONFIGS = [
    ("Old baseline (rv=10, no gap)", 10.0, False, 0.0),
    ("New config  (rv=6 + gap ≥5%)",  6.0, True,  0.05),
]

PERIODS = [
    ("2022 bear",       date(2022, 1, 3),  date(2022, 12, 30), -19.0),
    ("2023 mixed",      date(2023, 1, 3),  date(2023, 12, 29), +26.0),
    ("2024 validation", date(2024, 1, 2),  date(2024, 12, 31), +25.0),
    ("OOS H1-2025",     date(2025, 1, 2),  date(2025, 5, 28),   +0.5),
    ("2025-26 bull",    date(2025, 6, 1),  date(2026, 5, 28),  +30.0),
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


def prev_close_for(screener, sym, as_of: date) -> float:
    df = screener._daily_cache.get(sym)
    if df is None or df.empty:
        return 0.0
    past = df[df.index < pd.Timestamp(as_of)]
    return float(past.iloc[-1]["close"]) if not past.empty else 0.0


def run_period(cfg, start, end, screener, fetcher, initial_equity, news_filter,
               gap_hold_on: bool, gap_min_pct: float) -> dict:
    running_eq = initial_equity
    peak = initial_equity
    max_dd_pct = 0.0
    all_trades: List[TradeRecord] = []
    cache_dir = fetcher._cache_dir

    sim = Simulator(
        cfg, initial_equity,
        slippage_pct=SLIPPAGE,
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
        gap_hold_entry=gap_hold_on,
        gap_hold_min_pct=gap_min_pct,
        gap_hold_bars=GAP_HOLD_BARS,
        gap_hold_tolerance=GAP_HOLD_TOLERANCE,
    )

    for d in trading_days(start, end):
        if not spy_uptrend(d):
            continue
        cands = screener.candidates_for_date(d)
        cached = [s for s in cands if (cache_dir / f"{s}_{d}.json").exists()]
        if not cached:
            continue
        bars_by_sym: Dict = {}
        baselines: Dict = {}
        prev_closes: Dict = {}
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
        if gap_hold_on:
            for sym in bars_by_sym:
                pc = prev_close_for(screener, sym, d)
                if pc > 0:
                    prev_closes[sym] = pc
        sim._initial_equity = running_eq
        result = sim.run_day(d, bars_by_sym, baselines,
                             prev_closes=prev_closes if gap_hold_on else None)
        all_trades.extend(result.trades)
        day_pnl = sum(t.pnl for t in result.trades if t.pnl is not None)
        running_eq = max(running_eq + day_pnl, 1.0)
        peak = max(peak, running_eq)
        if peak > 0:
            max_dd_pct = max(max_dd_pct, (peak - running_eq) / peak * 100)

    trades = all_trades
    if not trades:
        return {"trades": 0, "gap_trades": 0, "mom_trades": 0,
                "wr": 0.0, "ret_pct": 0.0, "max_dd": 0.0,
                "hard_stops": 0, "profit_factor": 0.0}
    wins   = [t for t in trades if t.pnl and t.pnl > 0]
    losses = [t for t in trades if t.pnl and t.pnl <= 0]
    net    = running_eq - initial_equity
    total_gain = sum(t.pnl for t in wins)
    total_loss = abs(sum(t.pnl for t in losses)) if losses else 0.0
    gap_trades = [t for t in trades if "gap_hold" in (t.signals or [])]
    return {
        "trades":     len(trades),
        "gap_trades": len(gap_trades),
        "mom_trades": len(trades) - len(gap_trades),
        "wr":         len(wins) / len(trades) * 100,
        "ret_pct":    net / initial_equity * 100,
        "max_dd":     max_dd_pct,
        "hard_stops": sum(1 for t in trades if t.exit_reason == "hard_stop"),
        "profit_factor": total_gain / total_loss if total_loss > 0 else float("inf"),
    }


def main():
    api_key    = os.environ["APCA_API_KEY_ID"]
    secret_key = os.environ["APCA_API_SECRET_KEY"]
    base_url   = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

    initial_equity = broker.get_account_info()["portfolio_value"]
    print(f"Starting equity: ${initial_equity:,.0f}")
    print(f"Combined backtest: old baseline vs rv=6 + gap-hold ≥5%")
    print(f"H config, $10M DV, all 5 periods\n")

    fetcher     = BarFetcher(api_key, secret_key)
    news_filter = NewsFilter(api_key, secret_key, cache_only=False)

    HDR = (f"  {'Config':<34} {'Trades':>6} {'Mom':>5} {'Gap':>5} "
           f"{'WR':>6} {'Return':>8} {'MaxDD':>7} {'vs S&P':>8} "
           f"{'PF':>5} {'HStop':>6}")
    SEP = "  " + "-" * (len(HDR) - 2)

    for period_label, start, end, spy_ret in PERIODS:
        print(f"Loading {period_label}...", flush=True)

        print(f"\n{'=' * (len(HDR) - 2)}")
        print(f"  {period_label}  ({start} → {end})  S&P: {spy_ret:+.1f}%")
        print(HDR); print(SEP)

        for label, rv, gap_on, gap_pct in CONFIGS:
            base = make_gap_hold_config()
            base.min_avg_dollar_volume = DV
            base.stage2_min_vol_vs_prev_bar = 0.80
            base.target_atr_multiple = CHASE_TARGET
            base.stage2_min_relative_volume = rv   # override to test old vs new

            screener = CandidateScreener(copy.copy(base), api_key, secret_key, base_url)
            screener.preload(start, end)

            cfg = copy.copy(base)
            m = run_period(cfg, start, end, screener, fetcher,
                           initial_equity, news_filter, gap_on, gap_pct)
            vs   = m["ret_pct"] - spy_ret
            flag = " *" if m["ret_pct"] > 0 else "  "
            pf   = f"{m['profit_factor']:.2f}" if m["profit_factor"] != float("inf") else " inf"
            print(f"  {label:<34} {m['trades']:>6,} {m['mom_trades']:>5} {m['gap_trades']:>5} "
                  f"{m['wr']:>5.1f}% {m['ret_pct']:>+7.1f}%{flag} {m['max_dd']:>6.1f}% "
                  f"{vs:>+7.1f}% {pf:>5} {m['hard_stops']:>6}", flush=True)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
