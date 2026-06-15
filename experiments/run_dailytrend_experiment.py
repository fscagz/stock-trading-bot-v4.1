"""
Daily trend alignment filter experiment — all three periods.

Hypothesis: in 2023, many intraday momentum spikes occur on stocks that are
in long-term daily downtrends (below 20-day daily MA). These are "gap and fail"
patterns — the catalyst causes a spike but the underlying downtrend resumes.

Filter: only allow entries when the stock's previous-day close is above its
20-day daily simple MA. Requires the stock to be in a daily uptrend.

Base: top-50 cap, day_high_5pct, vol≥80%, target 4×ATR, news tier-4.
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


def stocks_in_daily_uptrend(screener: CandidateScreener, candidates: List[str], as_of: date, ma_period: int) -> Set[str]:
    """Return set of symbols where previous-day close is above N-day daily MA."""
    result = set()
    for sym in candidates:
        df = screener._daily_cache.get(sym)
        if df is None or df.empty:
            result.add(sym)  # no data = don't filter out (benefit of the doubt)
            continue
        past = df[df.index < pd.Timestamp(as_of)]
        if len(past) < ma_period:
            result.add(sym)  # insufficient history = don't filter
            continue
        ma = float(past["close"].iloc[-ma_period:].mean())
        prev_close = float(past["close"].iloc[-1])
        if prev_close >= ma:
            result.add(sym)
    return result


def run_period(
    cfg,
    start: date,
    end: date,
    screener: CandidateScreener,
    fetcher: BarFetcher,
    initial_equity: float,
    news_filter: NewsFilter,
    daily_ma_period: int,  # 0 = disabled
) -> dict:
    running_eq  = initial_equity
    all_trades: List[TradeRecord] = []
    cache_dir   = fetcher._cache_dir
    filtered_out_count = 0

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
        cands  = screener.candidates_for_date(d)

        # Apply daily MA trend filter
        if daily_ma_period > 0:
            trending_up = stocks_in_daily_uptrend(screener, cands, d, daily_ma_period)
            filtered_out = len(cands) - len([c for c in cands if c in trending_up])
            filtered_out_count += filtered_out
            cands = [c for c in cands if c in trending_up]

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
        return {"trades": 0, "wr": 0.0, "ret_pct": 0.0, "hard_stops": 0, "targets": 0, "filtered": filtered_out_count}

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
        "filtered":   filtered_out_count,
    }


def main():
    api_key    = os.environ["APCA_API_KEY_ID"]
    secret_key = os.environ["APCA_API_SECRET_KEY"]
    base_url   = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

    initial_equity = broker.get_account_info()["portfolio_value"]
    print(f"Starting equity: ${initial_equity:,.0f}")
    print(f"Base: top-50 cap, day_high_5pct, vol≥80%, target=4×ATR, tier-4 bypass\n")

    fetcher     = BarFetcher(api_key, secret_key)
    news_filter = NewsFilter(api_key, secret_key, cache_only=False)

    # MA periods to test (0 = no filter)
    ma_periods = [
        ("No daily trend filter", 0),
        ("5-day daily MA",       5),
        ("10-day daily MA",     10),
        ("20-day daily MA",     20),
        ("50-day daily MA",     50),
    ]

    for period_label, start, end, spy_ret in PERIODS:
        base_cfg = make_gap_hold_config()
        base_cfg.target_atr_multiple = 4.0
        base_cfg.stage2_min_vol_vs_prev_bar = 0.80

        screener = CandidateScreener(copy.copy(base_cfg), api_key, secret_key, base_url)
        print(f"Loading {period_label}...", flush=True)
        screener.preload(start, end)
        print(f"  {len(screener._daily_cache):,} symbols loaded\n", flush=True)

        print(f"{'='*82}")
        print(f"  {period_label}  ({start}→{end})  S&P: {spy_ret:+.0f}%")
        print(f"{'='*82}")
        hdr = (f"  {'Config':<26} {'Trades':>7} {'WR':>6} {'Return':>8} "
               f"{'vs S&P':>8} {'HStops':>7} {'Tgts':>6} {'Filtered':>9}")
        print(hdr)
        print("  " + "-" * 78)

        for label, ma_period in ma_periods:
            cfg = copy.copy(base_cfg)
            m = run_period(cfg, start, end, screener, fetcher, initial_equity, news_filter, ma_period)
            vs_spy = m["ret_pct"] - spy_ret
            flag   = " ✓" if vs_spy >= 0 else "  "

            print(
                f"  {label:<26} {m['trades']:>7,} {m['wr']:>5.1f}% "
                f"{m['ret_pct']:>+7.1f}% {vs_spy:>+7.1f}%{flag} "
                f"{m['hard_stops']:>7} {m['targets']:>6} {m['filtered']:>9,}",
                flush=True,
            )
        print()

    print("Done.")


if __name__ == "__main__":
    main()
