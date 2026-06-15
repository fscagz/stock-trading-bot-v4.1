"""
Day-high pullback filter experiments.
Tests stage2_min_dist_from_day_high_pct thresholds on both 2022 (bear) and
2025-2026 (bull) to find a value that improves 2022 without hurting 2025-2026.
"""
from __future__ import annotations
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from typing import List, Dict, Tuple

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

import bot.broker_alpaca as broker
from bot.backtest.backtest_metrics import compute_metrics
from bot.backtest.bar_fetcher import BarFetcher
from bot.backtest.candidate_screener import CandidateScreener
from bot.backtest.news_filter import NewsFilter
from bot.backtest.simulator import Simulator
from bot.config import make_gap_hold_config
from bot.intraday.types import TradeRecord

logging.basicConfig(level=logging.WARNING)

RISK_SCALE = 2.0

PERIODS = {
    "2022": (date(2022, 1, 3),  date(2022, 12, 30)),
    "2025": (date(2025, 6, 1),  date(2026, 5, 28)),
}

# Thresholds to test (fraction below day high required at entry)
THRESHOLDS = [0.0, 0.02, 0.03, 0.05, 0.07, 0.10]


def trading_days(start: date, end: date) -> List[date]:
    days, current = [], start
    while current <= end:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def make_config(dist_pct: float):
    cfg = make_gap_hold_config()
    cfg.stage2_min_dist_from_day_high_pct = dist_pct
    cfg.risk_per_trade = round(cfg.risk_per_trade * RISK_SCALE, 6)
    cfg.max_portfolio_heat = min(round(cfg.max_portfolio_heat * RISK_SCALE, 4), 1.0)
    return cfg


def load_bars(period_label: str, start: date, end: date,
              screener: CandidateScreener, fetcher: BarFetcher) -> Tuple[Dict, Dict]:
    days = trading_days(start, end)
    bars_cache: Dict[date, dict] = {}
    baseline_cache: Dict[date, dict] = {}
    for d in days:
        candidates = screener.candidates_for_date(d)
        if not candidates:
            continue

        def _fetch(sym):
            return sym, fetcher.fetch(sym, d), screener.baseline_volume(sym, d)

        all_cached = all(fetcher.is_cached(sym, d) for sym in candidates)
        workers = 16 if all_cached else 1
        bars_by_sym, baselines = {}, {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for sym, bars, baseline in pool.map(_fetch, candidates):
                if bars:
                    bars_by_sym[sym] = bars
                    baselines[sym] = baseline
        if bars_by_sym:
            bars_cache[d] = bars_by_sym
            baseline_cache[d] = baselines
    return bars_cache, baseline_cache


def run_experiment(dist_pct: float, initial_equity: float,
                   bars_cache: Dict, baseline_cache: Dict,
                   start: date, end: date, news_filter) -> dict:
    cfg = make_config(dist_pct)
    sim = Simulator(cfg, initial_equity, slippage_pct=0.001,
                    news_filter=news_filter, short_mode=False,
                    etb_set=None, news_mode="require")
    all_trades: List[TradeRecord] = []
    carry_over = {}
    for d in trading_days(start, end):
        if d not in bars_cache:
            continue
        result = sim.run_day(d, bars_cache[d], baseline_cache[d], carry_in=carry_over)
        all_trades.extend(result.trades)
        carry_over = result.carry_over
    return compute_metrics(all_trades, initial_equity)


def main():
    api_key = os.environ["APCA_API_KEY_ID"]
    secret_key = os.environ["APCA_API_SECRET_KEY"]
    base_url = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

    account = broker.get_account_info()
    initial_equity = account["portfolio_value"]

    base_cfg = make_gap_hold_config()
    news_filter = NewsFilter(api_key, secret_key)
    fetcher = BarFetcher(api_key, secret_key)

    period_data = {}
    for label, (start, end) in PERIODS.items():
        screener = CandidateScreener(base_cfg, api_key, secret_key, base_url)
        print(f"Preloading screener for {label}...")
        screener.preload(start, end)
        print(f"Fetching bars for {label}...")
        bars_cache, baseline_cache = load_bars(label, start, end, screener, fetcher)
        period_data[label] = (start, end, bars_cache, baseline_cache)
        print(f"  {label}: {len(bars_cache)} days with bar data")

    # Build all (period, threshold) experiment pairs
    jobs = [(label, dist) for label in PERIODS for dist in THRESHOLDS]

    print(f"\nRunning {len(jobs)} experiments ({len(THRESHOLDS)} thresholds × {len(PERIODS)} periods)...\n")

    results: Dict[Tuple[str, float], dict] = {}
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {
            pool.submit(
                run_experiment, dist, initial_equity,
                period_data[label][2], period_data[label][3],
                period_data[label][0], period_data[label][1],
                news_filter,
            ): (label, dist)
            for label, dist in jobs
        }
        for future in as_completed(futures):
            key = futures[future]
            results[key] = future.result()

    # Print side-by-side table
    header = f"{'Dist%':>6}  {'2022 Trades':>11} {'2022 WR':>8} {'2022 PnL':>10} {'2022 DD':>9}  |  {'2025 Trades':>11} {'2025 WR':>8} {'2025 PnL':>10} {'2025 DD':>9}"
    print(header)
    print("-" * len(header))
    for dist in THRESHOLDS:
        r22 = results.get(("2022", dist), {})
        r25 = results.get(("2025", dist), {})
        print(
            f"{dist*100:>5.0f}%  "
            f"{r22.get('total_trades',0):>11} {r22.get('win_rate',0):>8.1%} "
            f"{r22.get('total_pnl',0):>10.0f} {r22.get('max_drawdown',0):>9.0f}  |  "
            f"{r25.get('total_trades',0):>11} {r25.get('win_rate',0):>8.1%} "
            f"{r25.get('total_pnl',0):>10.0f} {r25.get('max_drawdown',0):>9.0f}"
        )


if __name__ == "__main__":
    main()
