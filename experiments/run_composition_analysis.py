"""
Composition analysis: H+ at $500k vs $10M liquidity filter.

Compares trade composition across:
  - DV bucket distribution (proxy for market cap)
  - Average hold time
  - MFE distribution (max favorable excursion)
  - Hard-stop rate
  - Profit concentration (top 5/10/20% of winners)

Runs over 2022 + 2023 (the periods where the filter matters most).
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

BREADTH_THRESHOLD = 35
PB_ATR, PB_TTL, PB_TARGET = 2.0, 60, 2.5
CHASE_TARGET = 4.0
DV_CAP = 0.20

ANALYSIS_PERIODS = [
    ("2022 bear",  date(2022, 1, 3), date(2022, 12, 30)),
    ("2023 mixed", date(2023, 1, 3), date(2023, 12, 29)),
]

CONFIGS = [
    ("$500k  baseline", 500_000),
    ("$10M   filtered", 10_000_000),
]

# DV buckets: (label, min_dv, max_dv)
DV_BUCKETS = [
    ("nano   <$1M",    0,          1_000_000),
    ("micro  $1-5M",   1_000_000,  5_000_000),
    ("small  $5-25M",  5_000_000, 25_000_000),
    ("mid+   >$25M",  25_000_000, float("inf")),
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


def avg_daily_dv(screener, sym, as_of: date) -> float:
    """20-day average daily dollar volume before as_of."""
    df = screener._daily_cache.get(sym)
    if df is None or df.empty:
        return 0.0
    past = df[df.index < pd.Timestamp(as_of)]
    if past.empty:
        return 0.0
    avg_vol = float(past["volume"].tail(20).mean())
    avg_close = float(past["close"].tail(20).mean())
    return avg_vol * avg_close


def baseline_vol(screener, sym, as_of):
    df = screener._daily_cache.get(sym)
    if df is None or df.empty:
        return 0.0
    past = df[df.index < pd.Timestamp(as_of)]
    return float(past["volume"].tail(20).mean()) / 390.0 if not past.empty else 0.0


def stocks_above_daily_ma(screener, candidates, as_of, ma_period) -> Set[str]:
    result = set()
    for sym in candidates:
        df = screener._daily_cache.get(sym)
        if df is None or df.empty:
            result.add(sym)
            continue
        past = df[df.index < pd.Timestamp(as_of)]
        if len(past) < ma_period:
            result.add(sym)
            continue
        if float(past["close"].iloc[-1]) >= float(past["close"].iloc[-ma_period:].mean()):
            result.add(sym)
    return result


def run_period(cfg, start, end, screener, fetcher, initial_equity, news_filter
               ) -> List[TradeRecord]:
    """Returns all closed trades for the period."""
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
        max_position_dv_pct=DV_CAP,
        pullback_entry_atr=PB_ATR,
        pullback_entry_ttl_bars=PB_TTL,
        pullback_target_atr=PB_TARGET,
        pullback_chase_tier=4,
    )

    for d in trading_days(start, end):
        if not spy_uptrend(d):
            continue
        cands = screener.candidates_for_date(d)
        eligible = cands
        if len(cands) < BREADTH_THRESHOLD:
            eligible = [c for c in cands
                        if c in stocks_above_daily_ma(screener, cands, d, 10)]

        cached = [s for s in eligible if (cache_dir / f"{s}_{d}.json").exists()]
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

    return all_trades


def dv_bucket(dv: float) -> str:
    for label, lo, hi in DV_BUCKETS:
        if lo <= dv < hi:
            return label
    return "mid+   >$25M"


def hold_minutes(t: TradeRecord) -> Optional[float]:
    if t.entry_time and t.exit_time:
        return (t.exit_time - t.entry_time).total_seconds() / 60
    return None


def print_section(title: str) -> None:
    print(f"\n  {'─'*60}")
    print(f"  {title}")
    print(f"  {'─'*60}")


def analyse(label: str, trades: List[TradeRecord], screener,
            start: date, initial_equity: float) -> None:

    if not trades:
        print(f"  {label}: no trades")
        return

    # Attach DV to each trade
    trade_dvs = {}
    for t in trades:
        # Use entry date to look up DV
        as_of = t.entry_time.date() if t.entry_time else start
        trade_dvs[id(t)] = avg_daily_dv(screener, t.ticker, as_of)

    wins   = [t for t in trades if t.pnl and t.pnl > 0]
    losses = [t for t in trades if t.pnl and t.pnl <= 0]
    total  = len(trades)
    net    = sum(t.pnl for t in trades if t.pnl)

    print(f"\n  ── {label} ── {total} trades | WR {len(wins)/total*100:.1f}% | "
          f"Net ${net:+,.0f} ({net/initial_equity*100:+.1f}%)")

    # DV bucket distribution
    print_section("DV bucket distribution (proxy for market cap)")
    bucket_counts: Dict[str, list] = {b[0]: [] for b in DV_BUCKETS}
    for t in trades:
        b = dv_bucket(trade_dvs[id(t)])
        bucket_counts[b].append(t)
    print(f"  {'Bucket':<22} {'#':>4} {'%tot':>6} {'WR':>6} {'Avg PnL':>9}")
    for b_label, _, _ in DV_BUCKETS:
        bts = bucket_counts[b_label]
        if not bts:
            continue
        bw = [t for t in bts if t.pnl and t.pnl > 0]
        avg_pnl = sum(t.pnl for t in bts if t.pnl) / len(bts)
        print(f"  {b_label:<22} {len(bts):>4} {len(bts)/total*100:>5.1f}% "
              f"{len(bw)/len(bts)*100:>5.1f}% {avg_pnl:>+9.0f}")

    # Hold time
    print_section("Hold time (minutes)")
    hold_times = [h for t in trades for h in [hold_minutes(t)] if h is not None]
    if hold_times:
        hold_series = pd.Series(hold_times)
        print(f"  Median: {hold_series.median():.0f} min | "
              f"Mean: {hold_series.mean():.0f} min | "
              f"p10: {hold_series.quantile(0.10):.0f} | "
              f"p90: {hold_series.quantile(0.90):.0f}")
        win_holds  = [hold_minutes(t) for t in wins if hold_minutes(t)]
        loss_holds = [hold_minutes(t) for t in losses if hold_minutes(t)]
        if win_holds:
            print(f"  Winners  median: {pd.Series(win_holds).median():.0f} min")
        if loss_holds:
            print(f"  Losers   median: {pd.Series(loss_holds).median():.0f} min")

    # MFE
    print_section("Max Favorable Excursion (% from entry)")
    mfe_vals = [t.mfe_pct for t in trades if t.mfe_pct is not None]
    if mfe_vals:
        mfe_series = pd.Series(mfe_vals)
        print(f"  Median: {mfe_series.median():.2f}% | "
              f"Mean: {mfe_series.mean():.2f}% | "
              f"p75: {mfe_series.quantile(0.75):.2f}% | "
              f"p90: {mfe_series.quantile(0.90):.2f}%")
        ever_pos = sum(1 for v in mfe_vals if v > 0)
        print(f"  Trades with MFE > 0: {ever_pos}/{len(mfe_vals)} "
              f"({ever_pos/len(mfe_vals)*100:.0f}%)")
        # MFE > 1 ATR proxy: use 2% as rough ATR stand-in
        over_2pct = sum(1 for v in mfe_vals if v > 2.0)
        print(f"  Trades with MFE > 2%: {over_2pct} ({over_2pct/len(mfe_vals)*100:.0f}%)")

        # MFE on losers specifically
        loser_mfe = [t.mfe_pct for t in losses if t.mfe_pct is not None]
        if loser_mfe:
            print(f"  Losers MFE median: {pd.Series(loser_mfe).median():.2f}% "
                  f"| mean: {pd.Series(loser_mfe).mean():.2f}%")

    # MAE
    print_section("Max Adverse Excursion (% from entry, negative = drawdown)")
    mae_vals = [t.mae_pct for t in trades if t.mae_pct is not None]
    if mae_vals:
        mae_series = pd.Series(mae_vals)
        print(f"  Median: {mae_series.median():.2f}% | "
              f"Mean: {mae_series.mean():.2f}% | "
              f"p25: {mae_series.quantile(0.25):.2f}% | "
              f"p10: {mae_series.quantile(0.10):.2f}%")

    # Exit reason breakdown
    print_section("Exit reasons")
    reasons: Dict[str, int] = {}
    for t in trades:
        r = t.exit_reason or "unknown"
        reasons[r] = reasons.get(r, 0) + 1
    for r, cnt in sorted(reasons.items(), key=lambda x: -x[1]):
        pnl_for_reason = [t.pnl for t in trades if t.exit_reason == r and t.pnl]
        avg = sum(pnl_for_reason) / len(pnl_for_reason) if pnl_for_reason else 0
        print(f"  {r:<20} {cnt:>4} ({cnt/total*100:>4.1f}%)  avg PnL {avg:>+8.0f}")

    # Profit concentration
    print_section("Profit concentration")
    pnls = sorted([t.pnl for t in trades if t.pnl and t.pnl > 0], reverse=True)
    total_profit = sum(pnls) if pnls else 0
    for pct in [0.05, 0.10, 0.20]:
        n = max(1, int(len(pnls) * pct))
        top_profit = sum(pnls[:n])
        share = top_profit / total_profit * 100 if total_profit > 0 else 0
        print(f"  Top {pct*100:.0f}% of winners ({n} trades): "
              f"${top_profit:,.0f} = {share:.0f}% of total profit")


def main():
    api_key    = os.environ["APCA_API_KEY_ID"]
    secret_key = os.environ["APCA_API_SECRET_KEY"]
    base_url   = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

    initial_equity = broker.get_account_info()["portfolio_value"]
    print(f"Starting equity: ${initial_equity:,.0f}")
    print(f"Composition analysis: H+ config, $500k vs $10M DV filter")
    print(f"Periods: 2022 bear + 2023 mixed\n")

    fetcher     = BarFetcher(api_key, secret_key)
    news_filter = NewsFilter(api_key, secret_key, cache_only=False)

    for period_label, start, end in ANALYSIS_PERIODS:
        print(f"\n{'='*70}")
        print(f"  {period_label}  ({start} → {end})")
        print(f"{'='*70}")

        screeners: Dict[str, CandidateScreener] = {}
        base_screener = None

        for cfg_label, dv in CONFIGS:
            base = make_gap_hold_config()
            base.stage2_min_vol_vs_prev_bar = 0.80
            base.target_atr_multiple = CHASE_TARGET
            base.min_avg_dollar_volume = dv

            if base_screener is None:
                screener = CandidateScreener(copy.copy(base), api_key, secret_key, base_url)
                print(f"  Loading daily data...", flush=True)
                screener.preload(start, end)
                base_screener = screener
            else:
                # Re-use cached daily data, rebuild candidates index with new DV threshold
                screener = CandidateScreener(copy.copy(base), api_key, secret_key, base_url)
                screener._universe = base_screener._universe
                screener._daily_cache = base_screener._daily_cache
                screener._build_candidates_index(start, end)

            screeners[cfg_label] = screener

        for cfg_label, dv in CONFIGS:
            base = make_gap_hold_config()
            base.stage2_min_vol_vs_prev_bar = 0.80
            base.target_atr_multiple = CHASE_TARGET
            base.min_avg_dollar_volume = dv
            trades = run_period(base, start, end, screeners[cfg_label],
                                fetcher, initial_equity, news_filter)
            analyse(cfg_label, trades, base_screener, start, initial_equity)

    print("\nDone.")


if __name__ == "__main__":
    main()
