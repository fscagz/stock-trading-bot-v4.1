"""
Improvement experiments for stock-trading-bot-v4.1.

Tests parameter changes to address the two known loss drivers in 2025-26:
  1. VWAP break exits firing too easily (-$60,499 from 85 trades)
  2. No dollar-volume floor admitting weak small-cap setups

Experiments (all use live-bot-equivalent mode: no overnight, market fill, SPY regime):
  A  Baseline                  current live-equivalent config
  B  VWAP@entry guard          only enter when close > VWAP at signal bar
  C  Raise vwap_vol_ratio x5   fire vwap_break only at 10× baseline (vs 2× now)
  D  Min DV $250k              20-day avg dollar volume >= $250k
  E  Min DV $500k              20-day avg dollar volume >= $500k
  F  B + D                     VWAP entry guard + $250k min DV
  G  C + D                     high vol ratio + $250k min DV
  H  B + C                     VWAP entry guard + high vol ratio
  I  B + C + D                 all three combined
"""
from __future__ import annotations
import copy, os, sys, warnings, logging
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from typing import Dict, List

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

import pandas as pd
import bot.broker_alpaca as broker
from bot.backtest.backtest_metrics import compute_metrics
from bot.backtest.bar_fetcher import BarFetcher
from bot.backtest.candidate_screener import CandidateScreener
from bot.backtest.simulator import Simulator
from bot.config import make_gap_hold_config
from bot.data.daily_loader import get_daily
from bot.intraday.types import TradeRecord

# --- periods to test ---
PERIODS = {
    "2025-26 bull": (date(2025, 6, 1),  date(2026, 5, 28)),
    "2022 bear":    (date(2022, 1, 3),  date(2022, 12, 30)),
}

INITIAL_EQUITY = None  # loaded from account below

# --- SPY regime ---
_spy: pd.DataFrame | None = None
def spy_uptrend(d: date) -> bool:
    global _spy
    if _spy is None:
        _spy = get_daily("SPY", start="2021-11-01", end="2026-06-01")
        _spy["ma20"] = _spy["close"].rolling(20).mean()
    past = _spy[_spy.index < pd.Timestamp(d)].dropna(subset=["ma20"])
    if past.empty:
        return True
    return float(past.iloc[-1]["close"]) >= float(past.iloc[-1]["ma20"])

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
    past = df[df.index < pd.Timestamp(as_of)]
    if past.empty:
        return 0.0
    return float(past["volume"].tail(20).mean()) / 390.0

def run_experiment(
    label: str,
    cfg,
    days: List[date],
    screener: CandidateScreener,
    fetcher: BarFetcher,
    eq: float,
    require_above_vwap_at_entry: bool = False,
) -> dict:
    sim = Simulator(
        cfg, eq,
        slippage_pct=0.001,
        overnight_holds=False,
        market_order_fill=True,
        require_above_vwap_at_entry=require_above_vwap_at_entry,
    )
    trades: List[TradeRecord] = []
    cache_dir = fetcher._cache_dir

    for d in days:
        if not spy_uptrend(d):
            continue
        cands = screener.candidates_for_date(d)
        cached = [s for s in cands if (cache_dir / f"{s}_{d}.json").exists()]
        if not cached:
            continue

        def _fetch(sym):
            bars = fetcher.fetch(sym, d)
            baseline = baseline_from_cache(screener, sym, d)
            return sym, bars, baseline

        bars_by_sym: Dict = {}
        baselines: Dict = {}
        with ThreadPoolExecutor(max_workers=16) as pool:
            for sym, bars, bl in pool.map(_fetch, cached):
                if bars and bl > 0:
                    bars_by_sym[sym] = bars
                    baselines[sym] = bl

        if not bars_by_sym:
            continue
        result = sim.run_day(d, bars_by_sym, baselines)
        trades.extend(result.trades)

    m = compute_metrics(trades, eq)
    # add exit breakdown
    from collections import Counter
    reasons = Counter(t.exit_reason for t in trades if t.exit_reason)
    m["exit_reasons"] = dict(reasons)
    return m


def main():
    global INITIAL_EQUITY
    api_key = os.environ["APCA_API_KEY_ID"]
    secret_key = os.environ["APCA_API_SECRET_KEY"]
    base_url = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

    acct = broker.get_account_info()
    eq = INITIAL_EQUITY or acct["portfolio_value"]
    print(f"Equity: ${eq:,.0f}\n")

    fetcher = BarFetcher(api_key, secret_key)

    hdr = (f"{'Experiment':<34} {'Trades':>7} {'WR':>7} {'PnL':>11} "
           f"{'MaxDD':>10} {'Hold':>7}  Exit breakdown")
    sep = "-" * 110

    for period_label, (start, end) in PERIODS.items():
        print(f"\n{'='*110}")
        print(f"  {period_label}  ({start} → {end})")
        print(f"{'='*110}")
        print(hdr)
        print(sep)

        # Load screener once per period (baseline config — no min DV).
        # For min-DV experiments we pass a different config to the screener
        # so it can rebuild the candidates index.
        base_cfg = make_gap_hold_config()
        screener_base = CandidateScreener(copy.copy(base_cfg), api_key, secret_key, base_url)
        print(f"  Loading daily data for {period_label}...", flush=True)
        screener_base.preload(start, end)
        total_syms = len(screener_base._daily_cache)
        print(f"  Loaded {total_syms} symbols. Running experiments...", flush=True)
        print()

        days = trading_days(start, end)

        # Screener with $250k min DV (different candidates index)
        cfg_250k = copy.copy(base_cfg)
        cfg_250k.min_avg_dollar_volume = 250_000
        screener_250k = CandidateScreener(copy.copy(cfg_250k), api_key, secret_key, base_url)
        screener_250k._daily_cache = screener_base._daily_cache  # reuse data
        screener_250k._universe = screener_base._universe
        screener_250k._build_candidates_index(start, end)

        # Screener with $500k min DV
        cfg_500k = copy.copy(base_cfg)
        cfg_500k.min_avg_dollar_volume = 500_000
        screener_500k = CandidateScreener(copy.copy(cfg_500k), api_key, secret_key, base_url)
        screener_500k._daily_cache = screener_base._daily_cache
        screener_500k._universe = screener_base._universe
        screener_500k._build_candidates_index(start, end)

        def row(lbl, m):
            exits = m.get("exit_reasons", {})
            breakdown = "  ".join(
                f"{k}:{v}" for k, v in sorted(exits.items(), key=lambda x: -x[1])
            )
            print(f"{lbl:<34} {m['total_trades']:>7d} {m['win_rate']:>6.1%} "
                  f"${m['total_pnl']:>10,.0f} ${m['max_drawdown']:>9,.0f} "
                  f"{m['avg_hold_minutes']:>5.1f}m  {breakdown}", flush=True)

        # --- A: Baseline ---
        cfg_A = copy.copy(base_cfg)
        row("A  Baseline",
            run_experiment("A", cfg_A, days, screener_base, fetcher, eq))

        # --- B: VWAP@entry guard ---
        cfg_B = copy.copy(base_cfg)
        row("B  VWAP@entry guard",
            run_experiment("B", cfg_B, days, screener_base, fetcher, eq,
                           require_above_vwap_at_entry=True))

        # --- C: vwap_break_vol_ratio = 10.0 ---
        cfg_C = copy.copy(base_cfg)
        cfg_C.vwap_break_volume_ratio = 10.0
        row("C  vwap_vol_ratio=10x",
            run_experiment("C", cfg_C, days, screener_base, fetcher, eq))

        # --- D: Min DV $250k ---
        cfg_D = copy.copy(cfg_250k)
        row("D  Min DV $250k",
            run_experiment("D", cfg_D, days, screener_250k, fetcher, eq))

        # --- E: Min DV $500k ---
        cfg_E = copy.copy(cfg_500k)
        row("E  Min DV $500k",
            run_experiment("E", cfg_E, days, screener_500k, fetcher, eq))

        print()

        # --- F: B + D ---
        cfg_F = copy.copy(cfg_250k)
        row("F  VWAP@entry + $250k DV",
            run_experiment("F", cfg_F, days, screener_250k, fetcher, eq,
                           require_above_vwap_at_entry=True))

        # --- G: C + D ---
        cfg_G = copy.copy(cfg_250k)
        cfg_G.vwap_break_volume_ratio = 10.0
        row("G  vol_ratio=10x + $250k DV",
            run_experiment("G", cfg_G, days, screener_250k, fetcher, eq))

        # --- H: B + C ---
        cfg_H = copy.copy(base_cfg)
        cfg_H.vwap_break_volume_ratio = 10.0
        row("H  VWAP@entry + vol_ratio=10x",
            run_experiment("H", cfg_H, days, screener_base, fetcher, eq,
                           require_above_vwap_at_entry=True))

        # --- I: B + C + D ---
        cfg_I = copy.copy(cfg_250k)
        cfg_I.vwap_break_volume_ratio = 10.0
        row("I  VWAP@entry + vol=10x + $250k DV",
            run_experiment("I", cfg_I, days, screener_250k, fetcher, eq,
                           require_above_vwap_at_entry=True))

        print()

    print("Done.")


if __name__ == "__main__":
    main()
