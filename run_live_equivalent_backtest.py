"""
Backtest comparing original model vs live-bot-equivalent model.

Live bot differences from original backtest:
  1. No overnight holds (force-closes at EOD every day)
  2. Market order fills (fills at signal bar close, not next bar open)
  3. Regime filter (SPY 20-day MA — skips longs in downtrend)

Runs 2022 and 2025-26 to show the delta.
"""
from __future__ import annotations
import copy, os, warnings, logging
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

import pandas as pd
import bot.broker_alpaca as broker
from bot.backtest.backtest_metrics import compute_metrics
from bot.backtest.bar_fetcher import BarFetcher
from bot.backtest.candidate_screener import CandidateScreener
from bot.backtest.simulator import Simulator
from bot.config import make_long_config
from bot.data.daily_loader import get_daily
from bot.intraday.types import TradeRecord

RISK_SCALE = 2.0
PERIODS = {
    "2022 bear":   (date(2022, 1, 3),  date(2022, 12, 30)),
    "2025-26 bull": (date(2025, 6, 1),  date(2026, 5, 28)),
}

# --- SPY regime helper ---
_spy: pd.DataFrame | None = None
def spy_uptrend(d: date) -> bool:
    global _spy
    if _spy is None:
        _spy = get_daily("SPY", start="2021-11-01", end="2026-06-01")
        _spy["ma20"] = _spy["close"].rolling(20).mean()
    past = _spy[_spy.index < pd.Timestamp(d)].dropna(subset=["ma20"])
    if past.empty: return True
    return float(past.iloc[-1]["close"]) >= float(past.iloc[-1]["ma20"])

def trading_days(start: date, end: date) -> List[date]:
    days, d = [], start
    while d <= end:
        if d.weekday() < 5: days.append(d)
        d += timedelta(days=1)
    return days

def baseline_from_cache(screener: CandidateScreener, sym: str, as_of: date) -> float:
    """Compute 20-day avg per-minute volume from screener's daily cache."""
    df = screener._daily_cache.get(sym)
    if df is None or df.empty:
        return 0.0
    past = df[df.index < pd.Timestamp(as_of)]
    if past.empty:
        return 0.0
    return float(past["volume"].tail(20).mean()) / 390.0


def run(label, config, days, screener, fetcher, eq,
        overnight_holds=True, market_order_fill=False, use_regime=False) -> dict:
    sim = Simulator(config, eq, slippage_pct=0.001,
                    overnight_holds=overnight_holds,
                    market_order_fill=market_order_fill)
    trades: List[TradeRecord] = []
    cache_dir = fetcher._cache_dir

    for d in days:
        if use_regime and not spy_uptrend(d):
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

    return compute_metrics(trades, eq)


def main():
    api_key = os.environ["APCA_API_KEY_ID"]
    secret_key = os.environ["APCA_API_SECRET_KEY"]
    base_url = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

    eq = broker.get_account_info()["portfolio_value"]
    print(f"Equity: ${eq:,.0f}\n")

    cfg = make_long_config()
    cfg.risk_per_trade    = round(cfg.risk_per_trade    * RISK_SCALE, 6)
    cfg.max_portfolio_heat = min(round(cfg.max_portfolio_heat * RISK_SCALE, 4), 1.0)

    fetcher = BarFetcher(api_key, secret_key)

    hdr = f"{'Config':<28} {'Trades':>7} {'WR':>7} {'PnL':>11} {'MaxDD':>10} {'Hold':>8}"
    sep = "-" * 75

    for period_label, (start, end) in PERIODS.items():
        print(f"{'='*75}")
        print(f"  {period_label}  ({start} → {end})")
        print(f"{'='*75}")
        print(hdr); print(sep)

        screener = CandidateScreener(copy.copy(cfg), api_key, secret_key, base_url)
        print("  Loading daily data...", flush=True)
        screener.preload(start, end)
        print(f"  Loaded {len(screener._daily_cache)} symbols. Running experiments...", flush=True)

        days = trading_days(start, end)
        args = dict(screener=screener, fetcher=fetcher, eq=eq)

        def row(lbl, m):
            print(f"{lbl:<28} {m['total_trades']:>7d} {m['win_rate']:>6.1%} "
                  f"${m['total_pnl']:>10,.0f} ${m['max_drawdown']:>9,.0f} "
                  f"{m['avg_hold_minutes']:>6.1f}m", flush=True)

        row("A: Original",
            run("A", copy.copy(cfg), days,
                overnight_holds=True,  market_order_fill=False, use_regime=False, **args))

        row("B: +Regime filter",
            run("B", copy.copy(cfg), days,
                overnight_holds=True,  market_order_fill=False, use_regime=True, **args))

        row("C: +Market fill",
            run("C", copy.copy(cfg), days,
                overnight_holds=True,  market_order_fill=True,  use_regime=False, **args))

        row("D: +No overnight",
            run("D", copy.copy(cfg), days,
                overnight_holds=False, market_order_fill=False, use_regime=False, **args))

        row("E: Live-bot equiv (all)",
            run("E", copy.copy(cfg), days,
                overnight_holds=False, market_order_fill=True,  use_regime=True,  **args))
        print()

    print("Done.")


if __name__ == "__main__":
    main()
