"""Run short strategy experiments with different entry filters and print a comparison table."""
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

logging.basicConfig(level=logging.WARNING)  # suppress INFO noise during experiments

START = date(2025, 6, 1)
END = date(2026, 5, 28)
RISK_SCALE = 2.0
ETB_SET_PATH = None  # populated at runtime

EXPERIMENTS = [
    ("baseline",        dict()),
    ("vol_decline",     dict(require_volume_decline=True)),
    ("red_bar",         dict(require_red_bar=True)),
    ("vol_decline+red", dict(require_volume_decline=True, require_red_bar=True)),
    ("exhaustion_0.90", dict(volume_exhaustion_ratio=0.90)),
    ("exhaustion_0.85", dict(volume_exhaustion_ratio=0.85)),
    ("exhaust+decline", dict(volume_exhaustion_ratio=0.90, require_volume_decline=True)),
    ("exhaust+red",     dict(volume_exhaustion_ratio=0.90, require_red_bar=True)),
]


def trading_days(start: date, end: date) -> List[date]:
    days, current = [], start
    while current <= end:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def run_experiment(name: str, cfg_kwargs: dict, initial_equity: float,
                   bars_cache: Dict, baseline_cache: Dict, etb_set, news_filter) -> dict:
    cfg = make_short_config(**cfg_kwargs)
    cfg.risk_per_trade = round(cfg.risk_per_trade * RISK_SCALE, 6)
    cfg.max_position_pct = min(round(cfg.max_position_pct * RISK_SCALE, 4), 1.0)
    cfg.max_portfolio_heat = min(round(cfg.max_portfolio_heat * RISK_SCALE, 4), 1.0)

    sim = Simulator(cfg, initial_equity, slippage_pct=0.001,
                    news_filter=news_filter, short_mode=True,
                    etb_set=etb_set, news_mode="exclude")

    all_trades: List[TradeRecord] = []
    carry_over = {}
    for d in trading_days(START, END):
        if d not in bars_cache:
            continue
        result = sim.run_day(d, bars_cache[d], baseline_cache[d], carry_in=carry_over)
        all_trades.extend(result.trades)
        carry_over = result.carry_over

    metrics = compute_metrics(all_trades, initial_equity)
    return {"name": name, **metrics}


def main():
    api_key = os.environ["APCA_API_KEY_ID"]
    secret_key = os.environ["APCA_API_SECRET_KEY"]
    base_url = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

    account = broker.get_account_info()
    initial_equity = account["portfolio_value"]

    # Use baseline config just for screening/fetching
    base_cfg = make_short_config()
    screener = CandidateScreener(base_cfg, api_key, secret_key, base_url)
    fetcher = BarFetcher(api_key, secret_key)
    news_filter = NewsFilter(api_key, secret_key)
    etb_set = broker.get_etb_set()

    print("Preloading screener cache...")
    screener.preload(START, END)

    days = trading_days(START, END)
    print(f"Fetching bars for {len(days)} trading days...")

    bars_cache: Dict[date, dict] = {}
    baseline_cache: Dict[date, dict] = {}

    for d in days:
        candidates = screener.candidates_for_date(d, etb_set=etb_set)
        if not candidates:
            continue
        bars_by_sym, baselines = {}, {}

        def _fetch(sym):
            return sym, fetcher.fetch(sym, d), screener.baseline_volume(sym, d)

        all_cached = all(fetcher.is_cached(sym, d) for sym in candidates)
        workers = 16 if all_cached else 1
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for sym, bars, baseline in pool.map(_fetch, candidates):
                if bars:
                    bars_by_sym[sym] = bars
                    baselines[sym] = baseline

        if bars_by_sym:
            bars_cache[d] = bars_by_sym
            baseline_cache[d] = baselines

    print(f"Bars loaded for {len(bars_cache)} days. Running {len(EXPERIMENTS)} experiments...\n")

    results = []
    with ThreadPoolExecutor(max_workers=len(EXPERIMENTS)) as pool:
        futures = {
            pool.submit(run_experiment, name, kwargs, initial_equity,
                        bars_cache, baseline_cache, etb_set, news_filter): name
            for name, kwargs in EXPERIMENTS
        }
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda r: EXPERIMENTS.index(next(e for e in EXPERIMENTS if e[0] == r["name"])))

    # Print comparison table
    header = f"{'Experiment':<22} {'Trades':>6} {'WinRate':>7} {'PnL':>9} {'AvgWin':>8} {'AvgLoss':>9} {'MaxDD':>9}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['name']:<22} {r['total_trades']:>6} {r['win_rate']:>7.1%} "
            f"{r['total_pnl']:>9.0f} {r['avg_winner']:>8.0f} {r['avg_loser']:>9.0f} "
            f"{r['max_drawdown']:>9.0f}"
        )


if __name__ == "__main__":
    main()
