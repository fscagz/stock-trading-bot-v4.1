"""
2023 trade forensics — find WHICH 2023 trades lose, not just how many.

Runs the base config (top-50, day_high_5pct, vol>=80%, target 4xATR, news tier-4
bypass, dynamic equity) over 2023, then reconstructs entry-time features for every
trade from the cached bars and slices PnL across each dimension:

  - time of day (entry minute bucket)
  - confidence tier (1-4, recomputed at the entry bar)
  - news catalyst vs tier-4 bypass entry
  - extension above VWAP at entry
  - day move at entry (close vs prior daily close)
  - opening gap vs intraday spike
  - daily candidate count (breadth) on the trade's day
  - R-multiple distribution by exit reason

Also writes all trades + features to backtest_results/forensics_2023.csv
so further slicing doesn't need a re-run.
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
from bot.intraday.indicators.vwap import VWAPIndicator
from bot.intraday.types import TradeRecord
from bot.momentum.validator import MomentumValidator
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")

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

def baseline_vol(screener: CandidateScreener, sym: str, as_of: date) -> float:
    df = screener._daily_cache.get(sym)
    if df is None or df.empty:
        return 0.0
    past = df[df.index < pd.Timestamp(as_of)]
    return float(past["volume"].tail(20).mean()) / 390.0 if not past.empty else 0.0

def prior_daily_close(screener: CandidateScreener, sym: str, as_of: date) -> float:
    df = screener._daily_cache.get(sym)
    if df is None or df.empty:
        return 0.0
    past = df[df.index < pd.Timestamp(as_of)]
    return float(past.iloc[-1]["close"]) if not past.empty else 0.0


def extract_features(
    trade: TradeRecord,
    bars,
    baseline: float,
    prev_close: float,
    had_news: bool,
    n_candidates: int,
    cfg,
) -> Optional[dict]:
    """Replay the day's bars up to the entry bar and recompute entry-time features."""
    vwap = VWAPIndicator()
    validator = MomentumValidator(cfg)
    entry_bar = None
    vwap_at_entry = None
    for b in bars:
        v = vwap.update(b)
        validator.update(b)
        if b.timestamp == trade.entry_time:
            entry_bar = b
            vwap_at_entry = v
            break
    if entry_bar is None:
        return None

    conf = validator.confidence_score(entry_bar, baseline)
    mult = cfg.confidence_multiplier(conf)
    risk = (trade.entry_price - trade.stop_price) * trade.shares
    pnl_r = trade.pnl / risk if risk > 0 else 0.0
    et = trade.entry_time.astimezone(_ET)
    first_open = bars[0].open if bars else 0.0

    return {
        "date": et.date().isoformat(),
        "ticker": trade.ticker,
        "entry_et": et.strftime("%H:%M"),
        "mins_after_open": (et.hour - 9) * 60 + et.minute - 30,
        "entry_price": trade.entry_price,
        "shares": trade.shares,
        "pnl": trade.pnl,
        "pnl_r": round(pnl_r, 2),
        "exit_reason": trade.exit_reason,
        "hold_mins": (trade.exit_time - trade.entry_time).total_seconds() / 60
            if trade.exit_time else 0.0,
        "tier": int(mult),
        "confidence": round(conf, 3),
        "had_news": had_news,
        "rel_vol": round(entry_bar.volume / baseline, 1) if baseline > 0 else 0.0,
        "pct_above_vwap": round((entry_bar.close - vwap_at_entry) / vwap_at_entry * 100, 2)
            if vwap_at_entry else 0.0,
        "day_move_pct": round((entry_bar.close - prev_close) / prev_close * 100, 1)
            if prev_close > 0 else 0.0,
        "gap_pct": round((first_open - prev_close) / prev_close * 100, 1)
            if prev_close > 0 else 0.0,
        "n_candidates": n_candidates,
    }


def slice_table(df: pd.DataFrame, col: str, bins=None, labels=None) -> None:
    """Print PnL stats grouped by a column (optionally bucketed)."""
    work = df.copy()
    if bins is not None:
        work[col] = pd.cut(work[col], bins=bins, labels=labels)
    g = work.groupby(col, observed=True)["pnl"]
    r = work.groupby(col, observed=True)["pnl_r"]
    out = pd.DataFrame({
        "trades": g.count(),
        "net_pnl": g.sum().round(0),
        "avg_R": r.mean().round(2),
        "win_rate": (work.groupby(col, observed=True)["pnl"]
                     .apply(lambda s: (s > 0).mean() * 100).round(1)),
    })
    print(out.to_string())
    print()


def main():
    api_key    = os.environ["APCA_API_KEY_ID"]
    secret_key = os.environ["APCA_API_SECRET_KEY"]
    base_url   = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

    initial_equity = broker.get_account_info()["portfolio_value"]

    cfg = make_gap_hold_config()
    cfg.target_atr_multiple = 4.0
    cfg.stage2_min_vol_vs_prev_bar = 0.80

    screener = CandidateScreener(copy.copy(cfg), api_key, secret_key, base_url)
    print("Loading 2023 screener data...", flush=True)
    screener.preload(START, END)
    print(f"  {len(screener._daily_cache):,} symbols loaded", flush=True)

    fetcher     = BarFetcher(api_key, secret_key)
    news_filter = NewsFilter(api_key, secret_key, cache_only=False)
    cache_dir   = fetcher._cache_dir

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

    rows: List[dict] = []
    running_eq = initial_equity

    for d in trading_days(START, END):
        if not spy_uptrend(d):
            continue
        cands = screener.candidates_for_date(d)
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
        result = sim.run_day(d, bars_by_sym, baselines)
        day_pnl = sum(t.pnl for t in result.trades if t.pnl is not None)
        running_eq = max(running_eq + day_pnl, 1.0)

        for t in result.trades:
            feats = extract_features(
                t, bars_by_sym.get(t.ticker, []), baselines.get(t.ticker, 0.0),
                prior_daily_close(screener, t.ticker, d),
                news_filter.has_catalyst(t.ticker, d),
                len(cands), cfg,
            )
            if feats:
                rows.append(feats)

    df = pd.DataFrame(rows)
    out_path = Path("backtest_results/forensics_2023.csv")
    df.to_csv(out_path, index=False)

    net = df["pnl"].sum()
    print(f"\n2023 baseline: {len(df)} trades, net ${net:,.0f} "
          f"({net / initial_equity * 100:+.1f}%), "
          f"WR {(df['pnl'] > 0).mean() * 100:.1f}%")
    print(f"Features written to {out_path}\n")

    print("=" * 70)
    print("BY EXIT REASON")
    print("=" * 70)
    slice_table(df, "exit_reason")

    print("=" * 70)
    print("BY ENTRY TIME (minutes after open)")
    print("=" * 70)
    slice_table(df, "mins_after_open",
                bins=[0, 15, 30, 60, 120, 240, 390],
                labels=["0-15m", "15-30m", "30-60m", "1-2h", "2-4h", ">4h"])

    print("=" * 70)
    print("BY CONFIDENCE TIER")
    print("=" * 70)
    slice_table(df, "tier")

    print("=" * 70)
    print("BY NEWS STATUS (False = tier-4 bypass entry)")
    print("=" * 70)
    slice_table(df, "had_news")

    print("=" * 70)
    print("BY EXTENSION ABOVE VWAP AT ENTRY (%)")
    print("=" * 70)
    slice_table(df, "pct_above_vwap",
                bins=[-100, 0, 2, 5, 10, 1000],
                labels=["below", "0-2%", "2-5%", "5-10%", ">10%"])

    print("=" * 70)
    print("BY DAY MOVE AT ENTRY (% vs prior close)")
    print("=" * 70)
    slice_table(df, "day_move_pct",
                bins=[-100, 20, 30, 50, 100, 10000],
                labels=["<20%", "20-30%", "30-50%", "50-100%", ">100%"])

    print("=" * 70)
    print("BY OPENING GAP (% — was the move a gap or intraday spike?)")
    print("=" * 70)
    slice_table(df, "gap_pct",
                bins=[-100, 0, 5, 15, 30, 10000],
                labels=["red open", "0-5%", "5-15%", "15-30%", ">30%"])

    print("=" * 70)
    print("BY BREADTH (candidate count on trade's day)")
    print("=" * 70)
    slice_table(df, "n_candidates",
                bins=[0, 15, 25, 35, 50],
                labels=["<15", "15-25", "25-35", "35-50"])


if __name__ == "__main__":
    main()
