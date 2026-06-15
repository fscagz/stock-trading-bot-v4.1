"""
Hybrid entry validation — the final candidate.

Established so far (all four periods):
  - B pullback always (2xATR limit, ttl 60, target 2.5x): positive everywhere
    (+0.9 / +0.3 / +9.6 OOS / +41.9) but gives up half the bull year.
  - G chase tier-4-only (market entry, target 4x): +100.9% bull, ~flat elsewhere.
    Extreme signals work as chases in every regime; they don't pull back.

Hybrid: tier-4 signals enter at market with the 4x target (G behavior);
all lower-tier signals arm a 2xATR pullback limit with a 2.5x target (B
behavior). One rule set, no regime switch, no breadth dependence.

Configs:
  B pullback always          (reference)
  G chase tier4-only         (reference)
  H hybrid                   (proposed)

Periods: 2022, 2023, OOS H1-2025, 2025-26.
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
    ("2022 bear",    date(2022, 1, 3), date(2022, 12, 30), -19.0),
    ("2023 mixed",   date(2023, 1, 3), date(2023, 12, 29), +26.0),
    ("OOS H1-2025",  date(2025, 1, 2), date(2025, 5, 28),  +0.5),
    ("2025-26 bull", date(2025, 6, 1), date(2026, 5, 28),  +30.0),
]

PB_ATR, PB_TTL, PB_TARGET = 2.0, 60, 2.5
CHASE_TARGET = 4.0

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


def run_period(cfg, start, end, screener, fetcher, initial_equity, news_filter,
               sim_kwargs: dict) -> dict:
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
        **sim_kwargs,
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
        return {"trades": 0, "wr": 0.0, "ret_pct": 0.0, "hard_stops": 0,
                "targets": 0, "chases": 0, "pullbacks": 0}
    wins = [t for t in trades if t.pnl and t.pnl > 0]
    net = sum(t.pnl for t in trades if t.pnl is not None)
    pbs = [t for t in trades if "momentum_pullback" in (t.signals or [])]
    return {
        "trades": len(trades),
        "wr": len(wins) / len(trades) * 100,
        "ret_pct": net / initial_equity * 100,
        "hard_stops": sum(1 for t in trades if t.exit_reason == "hard_stop"),
        "targets": sum(1 for t in trades if t.exit_reason == "target"),
        "chases": len(trades) - len(pbs),
        "pullbacks": len(pbs),
    }


def main():
    api_key    = os.environ["APCA_API_KEY_ID"]
    secret_key = os.environ["APCA_API_SECRET_KEY"]
    base_url   = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

    initial_equity = broker.get_account_info()["portfolio_value"]
    print(f"Starting equity: ${initial_equity:,.0f}\n")

    fetcher     = BarFetcher(api_key, secret_key)
    news_filter = NewsFilter(api_key, secret_key, cache_only=False)

    configs = [
        ("B pullback always t2.5", dict(
            pullback_entry_atr=PB_ATR, pullback_entry_ttl_bars=PB_TTL,
            pullback_target_atr=PB_TARGET)),
        ("G chase tier4-only t4", dict(min_entry_tier=4)),
        ("H hybrid t4-chase/pb t2.5", dict(
            pullback_entry_atr=PB_ATR, pullback_entry_ttl_bars=PB_TTL,
            pullback_target_atr=PB_TARGET, pullback_chase_tier=4)),
    ]

    for period_label, start, end, spy_ret in PERIODS:
        base = make_gap_hold_config()
        base.stage2_min_vol_vs_prev_bar = 0.80
        base.target_atr_multiple = CHASE_TARGET

        screener = CandidateScreener(copy.copy(base), api_key, secret_key, base_url)
        print(f"Loading {period_label}...", flush=True)
        screener.preload(start, end)

        print(f"{'='*86}")
        print(f"  {period_label}  ({start}->{end})  S&P: {spy_ret:+.1f}%")
        print(f"{'='*86}")
        print(f"  {'Config':<28} {'Trades':>6} {'WR':>6} {'Return':>8} "
              f"{'vs S&P':>8} {'HStop':>6} {'Tgts':>5} {'Chase':>6} {'PB':>5}")
        print("  " + "-" * 82)
        for label, kw in configs:
            cfg = copy.copy(base)
            m = run_period(cfg, start, end, screener, fetcher,
                           initial_equity, news_filter, kw)
            vs = m["ret_pct"] - spy_ret
            flag = " *" if m["ret_pct"] > 0 else "  "
            print(f"  {label:<28} {m['trades']:>6,} {m['wr']:>5.1f}% "
                  f"{m['ret_pct']:>+7.1f}%{flag} {vs:>+7.1f}% "
                  f"{m['hard_stops']:>6} {m['targets']:>5} "
                  f"{m['chases']:>6} {m['pullbacks']:>5}", flush=True)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
