"""
Bull-market-protection experiments — all run with news_mode="require" as the
baseline, then test four ways to recover bull-market gains without sacrificing
bear-market protection.

Options:
  A.  News always (baseline)
  1.  Rel-vol bypass at 40x  — exceptional volume implies institutional catalyst
  2.  Rel-vol bypass at 60x  — even stricter volume bar
  3.  Tier-3 bypass          — confidence mult ≥ 3x bypasses news gate
  4.  Tier-4 bypass          — confidence mult ≥ 4x bypasses news gate (highest bar)
  5.  IWM day >+1.5%         — small-cap ETF surge → market IS the catalyst
  6.  Expanded keywords      — broader keyword list captures more legitimate catalysts
  7.  Tier-4 + expanded kw   — combination of best quality signal + better detection
"""
from __future__ import annotations
import copy, os, warnings, logging
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from typing import Callable, Dict, List, Optional

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

import pandas as pd
import bot.broker_alpaca as broker
from bot.backtest.backtest_metrics import compute_metrics
from bot.backtest.bar_fetcher import BarFetcher
from bot.backtest.candidate_screener import CandidateScreener
from bot.backtest.news_filter import NewsFilter, _CATALYST_KEYWORDS
from bot.backtest.simulator import Simulator
from bot.config import make_gap_hold_config
from bot.data.daily_loader import get_daily
from bot.intraday.types import TradeRecord

PERIODS = {
    "2022 bear":         (date(2022, 1,  3),  date(2022, 12, 30)),
    "Jan–Apr 2025":      (date(2025, 1,  3),  date(2025, 4,  20)),
    "Jun 2025–May 2026": (date(2025, 6,  1),  date(2026, 5,  28)),
}

# Expanded keyword list for option 6/7
_EXPANDED_KEYWORDS = _CATALYST_KEYWORDS + [
    "launch", "initiat", "patent", "grant", "approval",
    "milestone", "strateg", "license", "exclusive",
    "buyback", "repurchas", "spinoff", "spin-off",
    "invest", "fund", "raise", "capital",
    "record", "high", "growth", "expansion",
    "ceo", "appoint", "restructur",
]

_spy: pd.DataFrame | None = None
_iwm: pd.DataFrame | None = None

def _load_spy() -> pd.DataFrame:
    global _spy
    if _spy is None:
        df = get_daily("SPY", start="2021-06-01", end="2026-06-01")
        df["ma20"] = df["close"].rolling(20).mean()
        _spy = df
    return _spy

def _load_iwm() -> pd.DataFrame:
    global _iwm
    if _iwm is None:
        _iwm = get_daily("IWM", start="2021-06-01", end="2026-06-01")
    return _iwm

def spy_uptrend_20(d: date) -> bool:
    df = _load_spy()
    past = df[df.index < pd.Timestamp(d)].dropna(subset=["ma20"])
    if past.empty:
        return True
    r = past.iloc[-1]
    return float(r["close"]) >= float(r["ma20"])

def iwm_day_return(d: date) -> float:
    """IWM % change on day d (open→close)."""
    df = _load_iwm()
    row = df[df.index == pd.Timestamp(d)]
    if row.empty:
        return 0.0
    r = row.iloc[0]
    return (float(r["close"]) - float(r["open"])) / float(r["open"])

def trading_days(start: date, end: date) -> List[date]:
    days, d = [], start
    while d <= end:
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

def extended_metrics(trades: List[TradeRecord], initial_equity: float) -> dict:
    m = compute_metrics(trades, initial_equity)
    m["total_return_pct"] = round(m["total_pnl"] / initial_equity * 100, 1)
    eq, peak = initial_equity, initial_equity
    for t in trades:
        if t.pnl is not None:
            eq += t.pnl
            peak = max(peak, eq)
    m["max_drawdown_pct"] = round(m["max_drawdown"] / peak * 100, 1) if peak > 0 else 0.0
    return m


class ExpandedNewsFilter(NewsFilter):
    def _classify(self, articles: list) -> bool:
        for article in articles:
            text = (
                (article.get("headline") or "") + " " +
                (article.get("summary") or "")
            ).lower()
            for kw in _EXPANDED_KEYWORDS:
                if kw in text:
                    return True
        return False


def run(
    cfg,
    days: List[date],
    screener: CandidateScreener,
    fetcher: BarFetcher,
    initial_equity: float,
    news_filter: Optional[NewsFilter],
    news_mode: str = "require",
    news_relvol_bypass: float = 0.0,
    news_tier_bypass: int = 0,
    iwm_bypass_pct: float = 0.0,      # if IWM day return > this → ignore news
) -> dict:
    trades: List[TradeRecord] = []
    running_eq = initial_equity
    cache_dir  = fetcher._cache_dir

    for d in days:
        if not spy_uptrend_20(d):
            continue

        # IWM bypass: if small-cap market is surging, momentum IS the catalyst
        if iwm_bypass_pct > 0 and iwm_day_return(d) > iwm_bypass_pct:
            active_nf   = None
            active_mode = "ignore"
        else:
            active_nf   = news_filter
            active_mode = news_mode

        sim = Simulator(
            cfg, running_eq,
            slippage_pct=0.001,
            overnight_holds=False,
            market_order_fill=True,
            news_filter=active_nf,
            news_mode=active_mode,
            news_relvol_bypass=news_relvol_bypass,
            news_tier_bypass=news_tier_bypass,
        )

        cands  = screener.candidates_for_date(d)
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

        result     = sim.run_day(d, bars_by_sym, baselines)
        trades.extend(result.trades)
        day_pnl    = sum(t.pnl for t in result.trades if t.pnl is not None)
        running_eq = max(running_eq + day_pnl, 1.0)

    return extended_metrics(trades, initial_equity)


def main():
    api_key    = os.environ["APCA_API_KEY_ID"]
    secret_key = os.environ["APCA_API_SECRET_KEY"]
    base_url   = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

    initial_equity = broker.get_account_info()["portfolio_value"]
    print(f"Starting equity: ${initial_equity:,.0f}\n")

    fetcher      = BarFetcher(api_key, secret_key)
    news_filter  = NewsFilter(api_key, secret_key, cache_only=False)
    exp_filter   = ExpandedNewsFilter(
        api_key, secret_key,
        cache_dir=news_filter._cache_dir,  # share same cache
        cache_only=False,
    )

    # (label, news_filter, relvol_bypass, tier_bypass, iwm_bypass_pct)
    configs = [
        ("A. News always (baseline)",    news_filter, 0.0,  0, 0.0),
        ("1. Rel-vol bypass  40x",       news_filter, 40.0, 0, 0.0),
        ("2. Rel-vol bypass  60x",       news_filter, 60.0, 0, 0.0),
        ("3. Tier-3 bypass",             news_filter, 0.0,  3, 0.0),
        ("4. Tier-4 bypass",             news_filter, 0.0,  4, 0.0),
        ("5. IWM day >+1.5% bypass",     news_filter, 0.0,  0, 0.015),
        ("6. Expanded keywords",         exp_filter,  0.0,  0, 0.0),
        ("7. Tier-4 + expanded kw",      exp_filter,  0.0,  4, 0.0),
    ]

    hdr = (f"  {'Config':<32} {'Trades':>7} {'WR':>7} {'Avg W':>8} {'Avg L':>8} "
           f"{'Net PnL':>10} {'Return':>8} {'MaxDD':>10} {'MaxDD%':>7}")
    sep = "  " + "-" * 106

    period_totals: Dict[str, List[float]] = {label: [] for label, *_ in configs}

    for period_label, (start, end) in PERIODS.items():
        cfg      = make_gap_hold_config()
        screener = CandidateScreener(copy.copy(cfg), api_key, secret_key, base_url)
        print(f"\n{'='*110}")
        print(f"  {period_label}  ({start} → {end})")
        print(f"{'='*110}")
        print(f"  Loading screener...", flush=True)
        screener.preload(start, end)
        print(f"  {len(screener._daily_cache):,} symbols loaded.\n")
        print(hdr)
        print(sep)

        days = trading_days(start, end)

        for label, nf, rvb, tb, iwm in configs:
            m = run(copy.copy(cfg), days, screener, fetcher, initial_equity,
                    nf, news_relvol_bypass=rvb, news_tier_bypass=tb, iwm_bypass_pct=iwm)
            period_totals[label].append(m["total_pnl"])
            print(
                f"  {label:<32} {m['total_trades']:>7,} {m['win_rate']:>6.1%} "
                f"${m['avg_winner']:>7,.0f} ${m['avg_loser']:>7,.0f} "
                f"${m['total_pnl']:>9,.0f} {m['total_return_pct']:>7.1f}% "
                f"${m['max_drawdown']:>9,.0f} {m['max_drawdown_pct']:>6.1f}%",
                flush=True,
            )

    print(f"\n{'='*110}")
    print("  COMBINED PnL ACROSS ALL THREE PERIODS")
    print(f"{'='*110}")
    print(f"  {'Config':<32} {'2022':>12} {'Jan-Apr 25':>12} {'Jun25-May26':>13} {'TOTAL':>12}")
    print("  " + "-" * 72)
    for label, *_ in configs:
        vals  = period_totals[label]
        total = sum(vals)
        print(f"  {label:<32} ${vals[0]:>10,.0f} ${vals[1]:>10,.0f} ${vals[2]:>11,.0f} ${total:>10,.0f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
