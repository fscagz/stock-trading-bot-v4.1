"""
Short strategy experiment across three market periods.

Runs a grid of short entry filter combinations alongside the standard long book,
then prints a comparison table showing long-only vs combined results.

Periods tested:
  2022        — bear year (the core hedge-thesis test)
  2023        — choppy / sideways year
  2025-26     — recent bull run

For each period:
  - Long-only baseline (make_gap_hold_config)
  - Combined with each short filter variant

Usage:
    python experiments/run_short_backtest.py [--period 2022|2023|2025|all]
"""
from __future__ import annotations
import argparse
import copy
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.WARNING)

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

import bot.broker_alpaca as broker
from bot.backtest.backtest_metrics import compute_metrics
from bot.backtest.bar_fetcher import BarFetcher
from bot.backtest.candidate_screener import CandidateScreener
from bot.backtest.simulator import Simulator
from bot.config import (
    make_gap_hold_config,
    make_long_and_short_config,
    make_standard_config,
    CombinedConfig,
    V4Config,
)
from bot.intraday.types import TradeRecord

PERIODS = {
    "2022": (date(2022, 1, 3),  date(2022, 12, 30)),
    "2023": (date(2023, 1, 3),  date(2023, 12, 29)),
    "2025": (date(2025, 6, 1),  date(2026, 5, 28)),
}

# Experiment grid: (label, short_kwargs)
# Each row runs make_long_and_short_config(**short_kwargs) alongside the long-only baseline.
SHORT_EXPERIMENTS: List[Tuple[str, dict]] = [
    ("baseline",               {}),
    ("sell_pressure<0.4",      {"short_selling_pressure_max": 0.40}),
    ("red_bar",                {"short_require_red_bar": True}),
    ("exhaustion_0.85",        {"short_volume_exhaustion_ratio": 0.85}),
    ("exhaustion_0.75",        {"short_volume_exhaustion_ratio": 0.75}),
    ("red+exhaust_0.85",       {"short_require_red_bar": True, "short_volume_exhaustion_ratio": 0.85}),
    ("pressure+red",           {"short_selling_pressure_max": 0.40, "short_require_red_bar": True}),
    ("all_filters",            {"short_selling_pressure_max": 0.40, "short_require_red_bar": True,
                                "short_volume_exhaustion_ratio": 0.85}),
]


def trading_days(start: date, end: date) -> List[date]:
    days, d = [], start
    while d <= end:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def baseline_from_cache(screener: CandidateScreener, sym: str, as_of: date) -> float:
    df = screener._daily_cache.get(sym)
    if df is None or df.empty:
        return 0.0
    import pandas as pd
    past = df[df.index < pd.Timestamp(as_of)]
    return float(past["volume"].tail(20).mean()) / 390.0 if not past.empty else 0.0


def extended_metrics(trades: List[TradeRecord], initial_equity: float) -> dict:
    m = compute_metrics(trades, initial_equity)
    final_eq = initial_equity + m["total_pnl"]
    m["final_equity"] = round(final_eq, 2)
    m["total_return_pct"] = round(m["total_pnl"] / initial_equity * 100, 1)
    eq, peak = initial_equity, initial_equity
    for t in trades:
        if t.pnl is not None:
            eq += t.pnl
            peak = max(peak, eq)
    m["max_drawdown_pct"] = round(m["max_drawdown"] / peak * 100, 1) if peak > 0 else 0.0
    return m


def run_period(
    long_cfg: V4Config,
    short_cfg: Optional[V4Config],
    days: List[date],
    screener: CandidateScreener,
    fetcher: BarFetcher,
    initial_equity: float,
) -> dict:
    sim = Simulator(
        config=long_cfg,
        initial_equity=initial_equity,
        slippage_pct=0.001,
        overnight_holds=False,
        market_order_fill=True,
        short_config=short_cfg,
        etb_set=None,   # no ETB gate — upper-bound estimate; add later once concept validated
    )

    trades: List[TradeRecord] = []
    cache_dir = fetcher._cache_dir

    for d in days:
        cands = screener.candidates_for_date(d)
        cached = [s for s in cands if (cache_dir / f"{s}_{d}.json").exists()]
        if not cached:
            continue

        bars_by_sym: Dict = {}
        baselines: Dict = {}

        def _fetch(sym):
            bars = fetcher.fetch(sym, d)
            bl = baseline_from_cache(screener, sym, d)
            return sym, bars, bl

        with ThreadPoolExecutor(max_workers=16) as pool:
            for sym, bars, bl in pool.map(_fetch, cached):
                if bars and bl > 0:
                    bars_by_sym[sym] = bars
                    baselines[sym] = bl

        if not bars_by_sym:
            continue

        result = sim.run_day(d, bars_by_sym, baselines)
        trades.extend(result.trades)

    return extended_metrics(trades, initial_equity)


def print_table(
    period_label: str,
    long_only_metrics: dict,
    short_results: List[Tuple[str, dict]],
) -> None:
    hdr = (
        f"  {'Config':<28} {'Dir':>5} {'Trades':>7} {'WR':>7} "
        f"{'AvgW':>8} {'AvgL':>8} {'PnL':>10} {'Ret%':>7} {'MaxDD%':>8} {'Hold':>6}"
    )
    sep = "  " + "-" * 98
    print(f"\n{'='*102}")
    print(f"  {period_label}")
    print(f"{'='*102}")
    print(hdr)
    print(sep)

    def _row(label: str, m: dict, direction_filter: Optional[str] = None) -> str:
        dir_label = direction_filter or "all"
        return (
            f"  {label:<28} {dir_label:>5} {m['total_trades']:>7,} {m['win_rate']:>6.1%} "
            f"${m['avg_winner']:>7,.0f} ${m['avg_loser']:>7,.0f} "
            f"${m['total_pnl']:>9,.0f} {m['total_return_pct']:>6.1f}% "
            f"{m['max_drawdown_pct']:>7.1f}% {m['avg_hold_minutes']:>5.1f}m"
        )

    print(_row("long-only (gap-hold)", long_only_metrics, "long"))
    print(sep)

    for exp_label, combined_metrics in short_results:
        print(_row(f"+ {exp_label}", combined_metrics, "all"))


def split_by_direction(trades: List[TradeRecord]) -> Tuple[List[TradeRecord], List[TradeRecord]]:
    longs = [t for t in trades if t.direction == "long"]
    shorts = [t for t in trades if t.direction == "short"]
    return longs, shorts


def run_with_split(
    long_cfg: V4Config,
    short_cfg: Optional[V4Config],
    days: List[date],
    screener: CandidateScreener,
    fetcher: BarFetcher,
    initial_equity: float,
) -> Tuple[dict, dict, dict]:
    """Return (all_metrics, long_metrics, short_metrics) for a run."""
    sim = Simulator(
        config=long_cfg,
        initial_equity=initial_equity,
        slippage_pct=0.001,
        overnight_holds=False,
        market_order_fill=True,
        short_config=short_cfg,
        etb_set=None,
    )
    trades: List[TradeRecord] = []
    cache_dir = fetcher._cache_dir

    for d in days:
        cands = screener.candidates_for_date(d)
        cached = [s for s in cands if (cache_dir / f"{s}_{d}.json").exists()]
        if not cached:
            continue

        bars_by_sym: Dict = {}
        baselines: Dict = {}

        def _fetch(sym):
            bars = fetcher.fetch(sym, d)
            bl = baseline_from_cache(screener, sym, d)
            return sym, bars, bl

        with ThreadPoolExecutor(max_workers=16) as pool:
            for sym, bars, bl in pool.map(_fetch, cached):
                if bars and bl > 0:
                    bars_by_sym[sym] = bars
                    baselines[sym] = bl

        if not bars_by_sym:
            continue

        result = sim.run_day(d, bars_by_sym, baselines)
        trades.extend(result.trades)

    longs, shorts = split_by_direction(trades)
    all_m = extended_metrics(trades, initial_equity)
    long_m = extended_metrics(longs, initial_equity) if longs else {"total_trades": 0, "win_rate": 0, "avg_winner": 0, "avg_loser": 0, "total_pnl": 0, "total_return_pct": 0, "max_drawdown_pct": 0, "avg_hold_minutes": 0}
    short_m = extended_metrics(shorts, initial_equity) if shorts else {"total_trades": 0, "win_rate": 0, "avg_winner": 0, "avg_loser": 0, "total_pnl": 0, "total_return_pct": 0, "max_drawdown_pct": 0, "avg_hold_minutes": 0}
    return all_m, long_m, short_m


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", choices=["2022", "2023", "2025", "all"], default="all")
    args = parser.parse_args()

    api_key = os.environ["APCA_API_KEY_ID"]
    secret_key = os.environ["APCA_API_SECRET_KEY"]
    base_url = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

    initial_equity = broker.get_account_info()["portfolio_value"]
    print(f"Starting equity: ${initial_equity:,.0f}\n")

    fetcher = BarFetcher(api_key, secret_key)
    long_cfg = make_gap_hold_config()

    selected = list(PERIODS.keys()) if args.period == "all" else [args.period]

    hdr = (
        f"  {'Experiment':<28} {'Side':>5} {'Trades':>7} {'WR':>7} "
        f"{'AvgW':>8} {'AvgL':>8} {'PnL':>10} {'Ret%':>7} {'MaxDD%':>8} {'Hold':>6}"
    )

    for period_key in selected:
        start, end = PERIODS[period_key]
        days = trading_days(start, end)
        period_label = f"{period_key}  ({start} → {end})"

        screener = CandidateScreener(copy.copy(long_cfg), api_key, secret_key, base_url)
        print(f"Loading screener data for {period_label}...", flush=True)
        screener.preload(start, end)
        print(f"Loaded {len(screener._daily_cache):,} symbols. Running experiments...\n", flush=True)

        sep = "  " + "-" * 98
        print(f"\n{'='*102}")
        print(f"  {period_label}")
        print(f"{'='*102}")
        print(hdr)
        print(sep)

        def _row(label: str, side: str, m: dict) -> str:
            return (
                f"  {label:<28} {side:>5} {m['total_trades']:>7,} {m['win_rate']:>6.1%} "
                f"${m['avg_winner']:>7,.0f} ${m['avg_loser']:>7,.0f} "
                f"${m['total_pnl']:>9,.0f} {m['total_return_pct']:>6.1f}% "
                f"{m['max_drawdown_pct']:>7.1f}% {m['avg_hold_minutes']:>5.1f}m"
            )

        # Long-only baseline
        all_m, long_m, _ = run_with_split(
            copy.copy(long_cfg), None, days, screener, fetcher, initial_equity
        )
        print(_row("long-only (gap-hold)", "long", long_m))
        print(sep)

        for exp_label, short_kwargs in SHORT_EXPERIMENTS:
            scfg = make_long_and_short_config(long_cfg=copy.copy(long_cfg), **short_kwargs).short
            assert scfg is not None
            all_m, long_m, short_m = run_with_split(
                copy.copy(long_cfg), copy.copy(scfg), days, screener, fetcher, initial_equity
            )
            print(_row(exp_label, "long", long_m))
            if short_m["total_trades"] > 0:
                print(_row("", "short", short_m))
            print(_row("", "TOTAL", all_m))
            print(sep)

        print(flush=True)


if __name__ == "__main__":
    main()
