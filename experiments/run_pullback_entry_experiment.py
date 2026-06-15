"""
Pullback entry sweep — phase 1: 2023 only.

Excursion analysis showed 2023's signature path: signal at spike top ->
flush ~1.3-1.5xATR -> recovery to +1.19xATR (median, even for trades the bot
lost). Chasing the spike at market buys the local top; exit redesign alone
peaked at -29%. This sweep arms a LIMIT order K x ATR below the signal close
(TTL-bounded), so the flush becomes our fill instead of our stop-out.

Live equivalent: a plain limit buy + TTL cancel. No fantasy fills: the limit
fills only when bar.low touches it; gap-below fills at the (better) open.

Grid: K (depth) x TTL x stop width. Sizing risk-normalized as always.
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

START, END = date(2023, 1, 3), date(2023, 12, 29)

_spy: Optional[pd.DataFrame] = None
def spy_uptrend(d: date) -> bool:
    global _spy
    if _spy is None:
        _spy = get_daily("SPY", start="2022-06-01", end="2024-01-15")
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


def run_2023(cfg, screener, fetcher, initial_equity, news_filter,
             pullback_atr: float, ttl: int) -> dict:
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
        pullback_entry_atr=pullback_atr,
        pullback_entry_ttl_bars=ttl,
    )

    for d in trading_days(START, END):
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
                "targets": 0, "eods": 0}
    wins = [t for t in trades if t.pnl and t.pnl > 0]
    net = sum(t.pnl for t in trades if t.pnl is not None)
    return {
        "trades": len(trades),
        "wr": len(wins) / len(trades) * 100,
        "ret_pct": net / initial_equity * 100,
        "hard_stops": sum(1 for t in trades if t.exit_reason == "hard_stop"),
        "targets": sum(1 for t in trades if t.exit_reason == "target"),
        "eods": sum(1 for t in trades if t.exit_reason in ("eod", "eod_no_bar")),
    }


def main():
    api_key    = os.environ["APCA_API_KEY_ID"]
    secret_key = os.environ["APCA_API_SECRET_KEY"]
    base_url   = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

    initial_equity = broker.get_account_info()["portfolio_value"]
    print(f"Starting equity: ${initial_equity:,.0f}   period: 2023\n")

    base = make_gap_hold_config()
    base.stage2_min_vol_vs_prev_bar = 0.80
    base.target_atr_multiple = 4.0

    screener = CandidateScreener(copy.copy(base), api_key, secret_key, base_url)
    print("Loading 2023 screener data...", flush=True)
    screener.preload(START, END)

    fetcher     = BarFetcher(api_key, secret_key)
    news_filter = NewsFilter(api_key, secret_key, cache_only=False)

    # (label, pullback_atr, ttl_bars, stop_x)
    grid = [
        ("chase (baseline)",        0.0,  0, 1.5),
        ("pb 0.5xATR ttl30",        0.5, 30, 1.5),
        ("pb 1.0xATR ttl30",        1.0, 30, 1.5),
        ("pb 1.5xATR ttl30",        1.5, 30, 1.5),
        ("pb 1.0xATR ttl60",        1.0, 60, 1.5),
        ("pb 1.5xATR ttl60",        1.5, 60, 1.5),
        ("pb 1.0xATR ttl30 stop2.5", 1.0, 30, 2.5),
        ("pb 1.5xATR ttl30 stop2.5", 1.5, 30, 2.5),
        ("pb 1.0xATR ttl60 stop2.5", 1.0, 60, 2.5),
        ("pb 2.0xATR ttl60",        2.0, 60, 1.5),
    ]

    print(f"\n  {'Config':<28} {'Trades':>7} {'WR':>6} {'Return':>8} "
          f"{'HStops':>7} {'Tgts':>6} {'EODs':>6}")
    print("  " + "-" * 74)
    for label, pb, ttl, stop_x in grid:
        cfg = copy.copy(base)
        cfg.stop_atr_multiple = stop_x
        m = run_2023(cfg, screener, fetcher, initial_equity, news_filter, pb, ttl)
        flag = " *" if m["ret_pct"] > 0 else "  "
        print(f"  {label:<28} {m['trades']:>7,} {m['wr']:>5.1f}% "
              f"{m['ret_pct']:>+7.1f}%{flag} {m['hard_stops']:>7} "
              f"{m['targets']:>6} {m['eods']:>6}", flush=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
