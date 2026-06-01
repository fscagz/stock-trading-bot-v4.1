"""
Comprehensive experiment runner for V4 strategy improvements.

Pre-loads bar data ONCE per period, then runs all experiments on the cached data.
This avoids re-reading files per-experiment for ~9x speedup.

Experiments:
  E0  Baseline long (make_long_config, news=require)
  E1  Long + SPY regime skip (0× risk / skip longs when SPY below 20-day MA)
  E2  Long + SPY regime half (0.5× risk in downtrend)
  E3  Long + min_spike_age=5
  E4  Long + min_spike_age=10
  E5  Short improved (make_short_config)
  E6  Short baseline (V4Config defaults)
  E7  FRD only (First Red Day shorts, no Stage-2 entries)
  E8  Long (regime-skip) + FRD combined
"""
from __future__ import annotations
import copy
import csv
import logging
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

import pandas as pd
import bot.broker_alpaca as broker
from bot.backtest.backtest_metrics import compute_metrics
from bot.backtest.bar_fetcher import BarFetcher
from bot.backtest.candidate_screener import CandidateScreener
from bot.backtest.news_filter import NewsFilter
from bot.backtest.simulator import Simulator
from bot.config import V4Config, make_long_config, make_short_config
from bot.data.daily_loader import get_daily
from bot.intraday.types import Bar, TradeRecord

RISK_SCALE = 2.0
PERIODS = {
    "2022 bear": (date(2022, 1, 3), date(2022, 12, 30)),
    "2025-26 bull": (date(2025, 6, 1), date(2026, 5, 28)),
}


# ── Regime helper ──────────────────────────────────────────────────────────

_spy_cache: Optional[pd.DataFrame] = None

def _get_spy() -> pd.DataFrame:
    global _spy_cache
    if _spy_cache is None:
        df = get_daily("SPY", start="2021-11-01", end="2026-06-01")
        df["ma20"] = df["close"].rolling(20).mean()
        _spy_cache = df
    return _spy_cache

def spy_uptrend(trade_date: date) -> bool:
    spy = _get_spy()
    target_ts = pd.Timestamp(trade_date)
    past = spy[spy.index < target_ts].dropna(subset=["ma20"])
    if past.empty:
        return True
    last = past.iloc[-1]
    return float(last["close"]) >= float(last["ma20"])

def trading_days(start: date, end: date) -> List[date]:
    days, d = [], start
    while d <= end:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


# ── Pre-load bar data (once per period) ───────────────────────────────────

BarCache = Dict[date, Dict[str, List[Bar]]]
BaselineCache = Dict[date, Dict[str, float]]
PriorCloseCache = Dict[date, Dict[str, float]]


def preload_bars(
    start: date,
    end: date,
    screener: CandidateScreener,
    fetcher: BarFetcher,
    config: V4Config,
) -> Tuple[BarCache, BaselineCache, PriorCloseCache]:
    """Load bars + baselines + prior_closes for all days in the period."""
    days = trading_days(start, end)
    bars_cache: BarCache = {}
    baseline_cache: BaselineCache = {}
    prior_close_cache: PriorCloseCache = {}

    for i, d in enumerate(days):
        if i % 50 == 0:
            print(f"    Loading bars: day {i+1}/{len(days)} ({d})", flush=True)
        candidates = screener.candidates_for_date(d, config=config)
        if not candidates:
            continue

        # Only process symbols that are already cached — avoids slow API fallback
        cached_candidates = [s for s in candidates if fetcher.is_cached(s, d)]
        if not cached_candidates:
            continue

        def _fetch(sym: str):
            bars = fetcher.fetch(sym, d)
            baseline = screener.baseline_volume(sym, d)
            prior_close = screener.prior_close(sym, d)
            return sym, bars, baseline, prior_close

        bars_by_sym: Dict = {}
        baselines: Dict = {}
        prior_closes: Dict = {}
        with ThreadPoolExecutor(max_workers=16) as pool:
            for sym, bars, baseline, prior_close in pool.map(_fetch, cached_candidates):
                if bars:
                    bars_by_sym[sym] = bars
                    baselines[sym] = baseline
                    prior_closes[sym] = prior_close
        if bars_by_sym:
            bars_cache[d] = bars_by_sym
            baseline_cache[d] = baselines
            prior_close_cache[d] = prior_closes

    return bars_cache, baseline_cache, prior_close_cache


# ── Run one experiment on pre-loaded data ─────────────────────────────────

@dataclass
class ExperimentResult:
    label: str
    period: str
    trades: int
    win_rate: float
    total_pnl: float
    avg_pnl: float
    max_dd: float
    avg_hold: float
    exit_reasons: dict


def run_on_preloaded(
    label: str,
    period: str,
    config: V4Config,
    initial_equity: float,
    days: List[date],
    bars_cache: BarCache,
    baseline_cache: BaselineCache,
    prior_close_cache: PriorCloseCache,
    carry_syms_cache: Dict[date, set],   # carry-over symbols per day (from prior run)
    news_filter: NewsFilter,
    short_mode: bool,
    news_mode: str,
    etb_set: Optional[set],
    regime_mode: str = "off",
    enable_frd: bool = False,
    frd_only: bool = False,
    fetcher: Optional[BarFetcher] = None,
    screener: Optional[CandidateScreener] = None,
) -> ExperimentResult:
    # For FRD-only mode, create a config that makes Stage-2 impossible to satisfy
    # (so only FRD shorts can enter). Use stage2_roc_min_pct=999 — no bar will ever
    # achieve a 99900% rate-of-change, so validator.validate() is always False.
    run_config = config
    if frd_only:
        run_config = copy.copy(config)
        run_config.stage2_roc_min_pct = 999.0

    sim = Simulator(
        run_config, initial_equity,
        slippage_pct=0.001,
        news_filter=news_filter,
        short_mode=short_mode,
        etb_set=etb_set,
        news_mode=news_mode,
    )
    all_trades: List[TradeRecord] = []
    carry_over = {}
    frd_watch: Dict = {}

    for d in days:
        day_risk_scale = 1.0
        if regime_mode == "skip" and not spy_uptrend(d):
            day_risk_scale = 0.0
        elif regime_mode == "half" and not spy_uptrend(d):
            day_risk_scale = 0.5

        # Use pre-loaded bars; add carry-over and FRD watch symbols if not in cache
        bars_by_sym = dict(bars_cache.get(d, {}))
        baselines = dict(baseline_cache.get(d, {}))
        prior_closes = dict(prior_close_cache.get(d, {}))

        # Add carry-over / FRD watch symbols not already loaded (only if cached)
        extra_syms = (set(carry_over.keys()) | set(frd_watch.keys())) - set(bars_by_sym.keys())
        if extra_syms and fetcher and screener:
            for sym in extra_syms:
                if fetcher.is_cached(sym, d):
                    bars = fetcher.fetch(sym, d)
                    if bars:
                        bars_by_sym[sym] = bars
                        baselines[sym] = screener.baseline_volume(sym, d)
                        prior_closes[sym] = screener.prior_close(sym, d)

        if not bars_by_sym:
            continue

        result = sim.run_day(
            d, bars_by_sym, baselines,
            carry_in=carry_over,
            prior_closes=prior_closes,
            day_risk_scale=day_risk_scale,
            frd_watch=frd_watch if (enable_frd or frd_only) else None,
        )
        all_trades.extend(result.trades)
        carry_over = result.carry_over
        if enable_frd or frd_only:
            frd_watch = result.frd_candidates

    m = compute_metrics(all_trades, initial_equity)
    return ExperimentResult(
        label=label, period=period,
        trades=m["total_trades"], win_rate=m["win_rate"],
        total_pnl=m["total_pnl"], avg_pnl=m["avg_pnl_per_trade"],
        max_dd=m["max_drawdown"], avg_hold=m["avg_hold_minutes"],
        exit_reasons=m["exit_reasons"],
    )


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    api_key = os.environ["APCA_API_KEY_ID"]
    secret_key = os.environ["APCA_API_SECRET_KEY"]
    base_url = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

    account = broker.get_account_info()
    initial_equity = account["portfolio_value"]
    print(f"Initial equity: ${initial_equity:,.2f}")

    long_cfg = make_long_config()
    short_cfg = make_short_config()
    base_short_cfg = V4Config()

    for cfg in (long_cfg, short_cfg, base_short_cfg):
        cfg.risk_per_trade = round(cfg.risk_per_trade * RISK_SCALE, 6)
        cfg.max_portfolio_heat = min(round(cfg.max_portfolio_heat * RISK_SCALE, 4), 1.0)

    fetcher = BarFetcher(api_key, secret_key)
    news_filter = NewsFilter(api_key, secret_key)

    all_results: List[ExperimentResult] = []

    for period_label, (start, end) in PERIODS.items():
        print(f"\n{'='*60}")
        print(f"Period: {period_label}  ({start} → {end})")
        print(f"{'='*60}")

        # Load screener (shared daily bar data for the period)
        long_screener = CandidateScreener(long_cfg, api_key, secret_key, base_url)
        print("  Loading screener...", flush=True)
        long_screener.preload(start, end)

        short_screener = CandidateScreener(short_cfg, api_key, secret_key, base_url)
        short_screener._daily_cache = long_screener._daily_cache
        short_screener._universe = long_screener._universe
        print("  Screener loaded.", flush=True)

        days = trading_days(start, end)

        # Pre-load bar data for long experiments (once)
        print("  Pre-loading long bars...", flush=True)
        long_bars, long_baselines, long_prior_closes = preload_bars(
            start, end, long_screener, fetcher, long_cfg
        )
        print(f"  Long bar cache: {len(long_bars)} days", flush=True)

        # Pre-load bar data for short experiments (may differ from long due to different screener threshold)
        print("  Pre-loading short bars...", flush=True)
        short_bars, short_baselines, short_prior_closes = preload_bars(
            start, end, short_screener, fetcher, short_cfg
        )
        print(f"  Short bar cache: {len(short_bars)} days", flush=True)

        common_long = dict(
            initial_equity=initial_equity, days=days,
            bars_cache=long_bars, baseline_cache=long_baselines, prior_close_cache=long_prior_closes,
            carry_syms_cache={}, news_filter=news_filter, etb_set=None,
            period=period_label, fetcher=fetcher, screener=long_screener,
        )
        common_short = dict(
            initial_equity=initial_equity, days=days,
            bars_cache=short_bars, baseline_cache=short_baselines, prior_close_cache=short_prior_closes,
            carry_syms_cache={}, news_filter=news_filter, etb_set=None,
            period=period_label, fetcher=fetcher, screener=short_screener,
        )

        def _print(e: ExperimentResult) -> None:
            reasons = sorted(e.exit_reasons.items(), key=lambda x: -x[1])
            rstr = "  ".join(f"{k}={v}" for k, v in reasons[:4])
            print(f"     → {e.trades} trades, {e.win_rate:.1%} WR, ${e.total_pnl:,.0f} PnL, ${e.max_dd:,.0f} DD  [{rstr}]", flush=True)

        print("\n  Running experiments...", flush=True)

        # E0: Baseline long
        print("  E0 Long baseline...", flush=True)
        e = run_on_preloaded("E0 Long baseline", config=long_cfg,
                             short_mode=False, news_mode="require",
                             regime_mode="off", enable_frd=False, frd_only=False, **common_long)
        all_results.append(e); _print(e)

        # E1: Long + regime skip
        print("  E1 Long+regime skip...", flush=True)
        e = run_on_preloaded("E1 Long+regime skip", config=long_cfg,
                             short_mode=False, news_mode="require",
                             regime_mode="skip", enable_frd=False, frd_only=False, **common_long)
        all_results.append(e); _print(e)

        # E2: Long + regime half
        print("  E2 Long+regime half...", flush=True)
        e = run_on_preloaded("E2 Long+regime half", config=long_cfg,
                             short_mode=False, news_mode="require",
                             regime_mode="half", enable_frd=False, frd_only=False, **common_long)
        all_results.append(e); _print(e)

        # E3: Long + spike age >= 5
        cfg_e3 = copy.copy(long_cfg); cfg_e3.stage2_min_spike_age_bars = 5
        print("  E3 Long+spike_age>=5...", flush=True)
        e = run_on_preloaded("E3 Long+age>=5", config=cfg_e3,
                             short_mode=False, news_mode="require",
                             regime_mode="off", enable_frd=False, frd_only=False, **common_long)
        all_results.append(e); _print(e)

        # E4: Long + spike age >= 10
        cfg_e4 = copy.copy(long_cfg); cfg_e4.stage2_min_spike_age_bars = 10
        print("  E4 Long+spike_age>=10...", flush=True)
        e = run_on_preloaded("E4 Long+age>=10", config=cfg_e4,
                             short_mode=False, news_mode="require",
                             regime_mode="off", enable_frd=False, frd_only=False, **common_long)
        all_results.append(e); _print(e)

        # E5: Short improved
        print("  E5 Short improved...", flush=True)
        e = run_on_preloaded("E5 Short improved", config=short_cfg,
                             short_mode=True, news_mode="exclude",
                             regime_mode="off", enable_frd=False, frd_only=False, **common_short)
        all_results.append(e); _print(e)

        # E6: Short baseline
        print("  E6 Short baseline...", flush=True)
        e = run_on_preloaded("E6 Short baseline", config=base_short_cfg,
                             short_mode=True, news_mode="exclude",
                             regime_mode="off", enable_frd=False, frd_only=False, **common_short)
        all_results.append(e); _print(e)

        # E7: FRD only
        print("  E7 FRD only...", flush=True)
        e = run_on_preloaded("E7 FRD only", config=short_cfg,
                             short_mode=False, news_mode="ignore",
                             regime_mode="off", enable_frd=True, frd_only=True, **common_long)
        all_results.append(e); _print(e)

        # E8: Long (regime-skip) + FRD
        print("  E8 Long+regime+FRD...", flush=True)
        e = run_on_preloaded("E8 Long+regime+FRD", config=long_cfg,
                             short_mode=False, news_mode="require",
                             regime_mode="skip", enable_frd=True, frd_only=False, **common_long)
        all_results.append(e); _print(e)

    # ── Results table ──────────────────────────────────────────────────────
    print(f"\n\n{'='*105}")
    print("EXPERIMENT RESULTS SUMMARY")
    print(f"{'='*105}")
    print(f"{'Experiment':<22} {'Period':<15} {'Trades':>7} {'WinRate':>8} {'TotalPnL':>11} {'AvgPnL':>9} {'MaxDD':>9} {'Hold':>6}")
    print("-" * 105)
    prev_period = None
    for r in all_results:
        if r.period != prev_period:
            if prev_period is not None:
                print()
            prev_period = r.period
        print(
            f"{r.label:<22} {r.period:<15} {r.trades:>7d} {r.win_rate:>7.1%} "
            f"${r.total_pnl:>10,.0f} ${r.avg_pnl:>8,.0f} ${r.max_dd:>8,.0f} {r.avg_hold:>5.1f}m"
        )

    # Save CSV
    out_path = Path("backtest_results/experiment_results.csv")
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "period", "trades", "win_rate", "total_pnl",
                    "avg_pnl", "max_drawdown", "avg_hold_minutes", "exit_reasons"])
        for r in all_results:
            w.writerow([r.label, r.period, r.trades, round(r.win_rate, 4),
                        r.total_pnl, r.avg_pnl, r.max_dd, r.avg_hold, str(r.exit_reasons)])
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
