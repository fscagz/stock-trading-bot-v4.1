"""
Regime filter variants.

Compares three versions of the SPY uptrend gate:
  none    — always allow entries (no regime filter)
  ma50    — SPY must be above its 50-day MA (more stable, fewer false blocks)
  ma20    — SPY must be above its 20-day MA (current baseline, reactive)

H config, $10M DV, all 5 backtest periods.
"""
from __future__ import annotations
import copy, os, sys, warnings, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from typing import Callable, Dict, List, Optional

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

PERIODS = [
    ("2022 bear",       date(2022, 1, 3),  date(2022, 12, 30), -19.0),
    ("2023 mixed",      date(2023, 1, 3),  date(2023, 12, 29), +26.0),
    ("2024 validation", date(2024, 1, 2),  date(2024, 12, 31), +25.0),
    ("OOS H1-2025",     date(2025, 1, 2),  date(2025, 5, 28),   +0.5),
    ("2025-26 bull",    date(2025, 6, 1),  date(2026, 5, 28),  +30.0),
]

_spy_df: Optional[pd.DataFrame] = None

def _load_spy() -> pd.DataFrame:
    global _spy_df
    if _spy_df is None:
        raw = get_daily("SPY", start="2021-06-01", end="2026-06-10")
        _spy_df = pd.DataFrame({
            "close": raw["close"],
            "ma20":  raw["close"].rolling(20).mean(),
            "ma50":  raw["close"].rolling(50).mean(),
        })
    return _spy_df

def uptrend_none(d: date) -> bool:
    return True

def uptrend_ma20(d: date) -> bool:
    df = _load_spy()
    past = df[df.index < pd.Timestamp(d)].dropna(subset=["ma20"])
    if past.empty:
        return True
    r = past.iloc[-1]
    return float(r["close"]) >= float(r["ma20"])

def uptrend_ma50(d: date) -> bool:
    df = _load_spy()
    past = df[df.index < pd.Timestamp(d)].dropna(subset=["ma50"])
    if past.empty:
        return True
    r = past.iloc[-1]
    return float(r["close"]) >= float(r["ma50"])

REGIMES: List[tuple] = [
    ("No filter (always trade)", uptrend_none),
    ("50-day MA filter",         uptrend_ma50),
    ("20-day MA filter (current)", uptrend_ma20),
]


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


def run_period(cfg, start, end, screener, fetcher, initial_equity, news_filter,
               uptrend_fn: Callable[[date], bool]) -> dict:
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
    )

    days_allowed = 0
    days_blocked = 0
    for d in trading_days(start, end):
        if not uptrend_fn(d):
            days_blocked += 1
            continue
        days_allowed += 1
        cands = screener.candidates_for_date(d)
        cached = [s for s in cands if (cache_dir / f"{s}_{d}.json").exists()]
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
    total_days = days_allowed + days_blocked
    if not trades:
        return {"trades": 0, "wr": 0.0, "ret_pct": 0.0, "max_dd": 0.0,
                "hard_stops": 0, "profit_factor": 0.0,
                "days_blocked_pct": days_blocked / total_days * 100 if total_days else 0.0}
    wins   = [t for t in trades if t.pnl and t.pnl > 0]
    losses = [t for t in trades if t.pnl and t.pnl <= 0]
    net    = running_eq - initial_equity
    total_gain = sum(t.pnl for t in wins)
    total_loss = abs(sum(t.pnl for t in losses)) if losses else 0.0
    return {
        "trades": len(trades),
        "wr": len(wins) / len(trades) * 100,
        "ret_pct": net / initial_equity * 100,
        "max_dd": max_dd_pct,
        "hard_stops": sum(1 for t in trades if t.exit_reason == "hard_stop"),
        "profit_factor": total_gain / total_loss if total_loss > 0 else float("inf"),
        "days_blocked_pct": days_blocked / total_days * 100 if total_days else 0.0,
    }


def main():
    api_key    = os.environ["APCA_API_KEY_ID"]
    secret_key = os.environ["APCA_API_SECRET_KEY"]
    base_url   = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

    initial_equity = broker.get_account_info()["portfolio_value"]
    print(f"Starting equity: ${initial_equity:,.0f}")
    print(f"H config, $10M DV — regime filter variants\n")

    fetcher     = BarFetcher(api_key, secret_key)
    news_filter = NewsFilter(api_key, secret_key, cache_only=False)

    HDR = (f"  {'Regime filter':<28} {'Trades':>6} {'WR':>6} {'Return':>8} "
           f"{'MaxDD':>7} {'vs S&P':>8} {'PF':>5} {'HStop':>6} {'Blocked%':>9}")
    SEP = "  " + "-" * (len(HDR) - 2)

    for period_label, start, end, spy_ret in PERIODS:
        base = make_gap_hold_config()
        base.min_avg_dollar_volume = DV
        base.stage2_min_vol_vs_prev_bar = 0.80
        base.target_atr_multiple = CHASE_TARGET

        screener = CandidateScreener(copy.copy(base), api_key, secret_key, base_url)
        print(f"Loading {period_label}...", flush=True)
        screener.preload(start, end)

        print(f"\n{'=' * (len(HDR) - 2)}")
        print(f"  {period_label}  ({start} → {end})  S&P: {spy_ret:+.1f}%")
        print(HDR); print(SEP)

        for label, uptrend_fn in REGIMES:
            cfg = copy.copy(base)
            m = run_period(cfg, start, end, screener, fetcher,
                           initial_equity, news_filter, uptrend_fn)
            vs   = m["ret_pct"] - spy_ret
            flag = " *" if m["ret_pct"] > 0 else "  "
            pf   = f"{m['profit_factor']:.2f}" if m["profit_factor"] != float("inf") else " inf"
            print(f"  {label:<28} {m['trades']:>6,} {m['wr']:>5.1f}% "
                  f"{m['ret_pct']:>+7.1f}%{flag} {m['max_dd']:>6.1f}% {vs:>+7.1f}% "
                  f"{pf:>5} {m['hard_stops']:>6} {m['days_blocked_pct']:>8.0f}%", flush=True)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
