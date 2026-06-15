"""
Experiment: compare gap-up fade (B), spike age gate (A), and both (A+B)
against the current baseline (selling pressure + red bar).
Uses Jun 2025 – May 2026 cached bar data.
"""
from __future__ import annotations
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from typing import List, Dict

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

import bot.broker_alpaca as broker
from bot.backtest.backtest_metrics import compute_metrics
from bot.backtest.bar_fetcher import BarFetcher
from bot.backtest.candidate_screener import CandidateScreener
from bot.backtest.news_filter import NewsFilter
from bot.backtest.simulator import Simulator
from bot.config import make_short_config
from bot.intraday.types import TradeRecord

logging.basicConfig(level=logging.WARNING)

START = date(2025, 6, 1)
END   = date(2026, 5, 28)
RISK_SCALE = 2.0

# Each entry: (label, extra kwargs for make_short_config() overrides applied after)
EXPERIMENTS = [
    ("baseline",         {}),
    ("spike_age_10",     {"stage2_max_spike_age_bars": 10}),
    ("spike_age_5",      {"stage2_max_spike_age_bars": 5}),
    ("gap10_45min",      {"stage2_min_gap_pct": 0.10, "stage2_max_gap_entry_minutes": 45}),
    ("gap10_60min",      {"stage2_min_gap_pct": 0.10, "stage2_max_gap_entry_minutes": 60}),
    ("gap15_45min",      {"stage2_min_gap_pct": 0.15, "stage2_max_gap_entry_minutes": 45}),
    ("A+B age10_gap10",  {"stage2_max_spike_age_bars": 10, "stage2_min_gap_pct": 0.10, "stage2_max_gap_entry_minutes": 45}),
    ("A+B age5_gap10",   {"stage2_max_spike_age_bars": 5,  "stage2_min_gap_pct": 0.10, "stage2_max_gap_entry_minutes": 45}),
]


def trading_days(start: date, end: date) -> List[date]:
    days, cur = [], start
    while cur <= end:
        if cur.weekday() < 5:
            days.append(cur)
        cur += timedelta(days=1)
    return days


def run_experiment(name, overrides, initial_equity, bars_cache, baseline_cache,
                   prior_close_cache, etb_set, news_filter) -> dict:
    cfg = make_short_config()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    cfg.risk_per_trade       = round(cfg.risk_per_trade * RISK_SCALE, 6)
    cfg.max_position_pct     = min(round(cfg.max_position_pct * RISK_SCALE, 4), 1.0)
    cfg.max_portfolio_heat   = min(round(cfg.max_portfolio_heat * RISK_SCALE, 4), 1.0)

    sim = Simulator(cfg, initial_equity, slippage_pct=0.001,
                    news_filter=news_filter, short_mode=True,
                    etb_set=etb_set, news_mode="exclude")

    all_trades: List[TradeRecord] = []
    carry_over = {}
    for d in trading_days(START, END):
        if d not in bars_cache:
            continue
        result = sim.run_day(
            d, bars_cache[d], baseline_cache[d],
            carry_in=carry_over,
            prior_closes=prior_close_cache.get(d),
        )
        all_trades.extend(result.trades)
        carry_over = result.carry_over

    m = compute_metrics(all_trades, initial_equity)
    return {"name": name, **m}


def main():
    api_key    = os.environ["APCA_API_KEY_ID"]
    secret_key = os.environ["APCA_API_SECRET_KEY"]
    base_url   = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

    account        = broker.get_account_info()
    initial_equity = account["portfolio_value"]

    base_cfg  = make_short_config()
    screener  = CandidateScreener(base_cfg, api_key, secret_key, base_url)
    fetcher   = BarFetcher(api_key, secret_key)
    nf        = NewsFilter(api_key, secret_key)
    etb_set   = broker.get_etb_set()

    print("Preloading screener cache...")
    screener.preload(START, END)

    days = trading_days(START, END)
    print(f"Fetching bars + prior closes for {len(days)} days...")

    bars_cache:        Dict[date, dict] = {}
    baseline_cache:    Dict[date, dict] = {}
    prior_close_cache: Dict[date, dict] = {}

    for d in days:
        candidates = screener.candidates_for_date(d, etb_set=etb_set)
        if not candidates:
            continue

        def _fetch(sym):
            return sym, fetcher.fetch(sym, d), screener.baseline_volume(sym, d), screener.prior_close(sym, d)

        all_cached = all(fetcher.is_cached(sym, d) for sym in candidates)
        workers = 16 if all_cached else 1
        bars_d, baselines_d, prior_closes_d = {}, {}, {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for sym, bars, baseline, pc in pool.map(_fetch, candidates):
                if bars:
                    bars_d[sym]        = bars
                    baselines_d[sym]   = baseline
                    prior_closes_d[sym] = pc

        if bars_d:
            bars_cache[d]        = bars_d
            baseline_cache[d]    = baselines_d
            prior_close_cache[d] = prior_closes_d

    print(f"Data ready for {len(bars_cache)} days. Running {len(EXPERIMENTS)} experiments...\n")

    results = []
    with ThreadPoolExecutor(max_workers=len(EXPERIMENTS)) as pool:
        futures = {
            pool.submit(run_experiment, name, overrides, initial_equity,
                        bars_cache, baseline_cache, prior_close_cache, etb_set, nf): name
            for name, overrides in EXPERIMENTS
        }
        for fut in as_completed(futures):
            results.append(fut.result())

    order = {name: i for i, (name, _) in enumerate(EXPERIMENTS)}
    results.sort(key=lambda r: order[r["name"]])

    hdr = f"{'Experiment':<22} {'Trades':>6} {'WinRate':>7} {'PnL':>9} {'AvgWin':>8} {'AvgLoss':>9} {'MaxDD':>9}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(
            f"{r['name']:<22} {r['total_trades']:>6} {r['win_rate']:>7.1%} "
            f"{r['total_pnl']:>9.0f} {r['avg_winner']:>8.0f} {r['avg_loser']:>9.0f} "
            f"{r['max_drawdown']:>9.0f}"
        )


if __name__ == "__main__":
    main()
