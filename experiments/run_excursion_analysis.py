"""
MFE/MAE excursion analysis — where do trades go after entry?

Re-runs the base-config backtest for 2023 and 2025-26, then for every trade
replays the day's bars from entry to exit-or-EOD and measures:

  - MFE: max favorable excursion (highest high after entry), in ATR units
  - MAE: max adverse excursion (lowest low after entry), in ATR units
  - time_to_green: minutes until bar.high first exceeds entry price
  - green_by_3/5/10: was the trade ever green within N minutes?
  - reach_kxATR: did price ever reach entry + k*ATR, for k = 0.5..4

Excursions are measured to the trade's actual exit time, because that is the
window the strategy actually had (post-exit price action is unactionable).
A second pass measures to EOD to show what a different exit could have had.

Goal: design exit rules from data instead of stacking more entry filters.
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
from bot.intraday.types import TradeRecord

PERIODS = [
    ("2023",    date(2023, 1, 3), date(2023, 12, 29)),
    ("2025-26", date(2025, 6, 1), date(2026, 5, 28)),
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

def baseline_vol(screener, sym, as_of):
    df = screener._daily_cache.get(sym)
    if df is None or df.empty:
        return 0.0
    past = df[df.index < pd.Timestamp(as_of)]
    return float(past["volume"].tail(20).mean()) / 390.0 if not past.empty else 0.0


def excursions(trade: TradeRecord, bars) -> Optional[dict]:
    """Measure price excursions from entry bar to actual exit, and to EOD."""
    # ATR at entry approximated from the recorded stop distance (1.5x ATR)
    stop_dist = trade.entry_price - trade.stop_price
    atr = stop_dist / 1.5
    if atr <= 0:
        return None

    after = [b for b in bars if b.timestamp > trade.entry_time]
    if not after:
        return None

    def measure(window):
        mfe = mae = 0.0
        time_to_green = None
        for b in window:
            mins = (b.timestamp - trade.entry_time).total_seconds() / 60
            mfe = max(mfe, b.high - trade.entry_price)
            mae = min(mae, b.low - trade.entry_price)
            if time_to_green is None and b.high > trade.entry_price:
                time_to_green = mins
        return mfe / atr, mae / atr, time_to_green

    to_exit = [b for b in after if trade.exit_time and b.timestamp <= trade.exit_time]
    mfe_x, mae_x, ttg = measure(to_exit if to_exit else after[:1])
    mfe_eod, mae_eod, _ = measure(after)

    return {
        "ticker": trade.ticker,
        "date": trade.entry_time.date().isoformat(),
        "pnl": trade.pnl,
        "win": trade.pnl > 0,
        "exit_reason": trade.exit_reason,
        "hold_mins": (trade.exit_time - trade.entry_time).total_seconds() / 60
            if trade.exit_time else 0.0,
        "mfe_atr": round(mfe_x, 2),       # to actual exit
        "mae_atr": round(mae_x, 2),
        "mfe_eod_atr": round(mfe_eod, 2), # to end of day (what was available)
        "time_to_green": ttg,             # None = never green before exit
    }


def report(rows: List[dict], label: str) -> None:
    df = pd.DataFrame(rows)
    w, l = df[df.win], df[~df.win]
    print(f"\n{'='*72}\n  {label}: {len(df)} trades  "
          f"({len(w)} winners / {len(l)} losers)\n{'='*72}")

    def stats(sub, name):
        if sub.empty:
            return
        never_green = sub["time_to_green"].isna().mean() * 100
        green5 = (sub["time_to_green"] <= 5).mean() * 100
        print(f"  {name:<8} MFE(exit) med={sub.mfe_atr.median():.2f}xATR  "
              f"MFE(EOD) med={sub.mfe_eod_atr.median():.2f}  "
              f"MAE med={sub.mae_atr.median():.2f}  "
              f"green<=5m {green5:.0f}%  never-green {never_green:.0f}%")

    stats(w, "WINNERS")
    stats(l, "LOSERS")

    print(f"\n  % of ALL trades whose price reached k x ATR before exit / by EOD:")
    print(f"  {'k':>6} {'by exit':>9} {'by EOD':>8}   (EOD = what a better exit could reach)")
    for k in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
        bx = (df.mfe_atr >= k).mean() * 100
        be = (df.mfe_eod_atr >= k).mean() * 100
        print(f"  {k:>6.1f} {bx:>8.0f}% {be:>7.0f}%")

    print(f"\n  LOSERS only — green-within-N-minutes (scratch-rule design):")
    for n in (3, 5, 10, 15):
        pct = (l["time_to_green"] <= n).mean() * 100
        print(f"    green within {n:>2}m: {pct:.0f}%")
    print(f"    never green before exit: {l['time_to_green'].isna().mean()*100:.0f}%")
    print(f"\n  WINNERS only — green-within-N-minutes:")
    for n in (3, 5, 10):
        pct = (w["time_to_green"] <= n).mean() * 100
        print(f"    green within {n:>2}m: {pct:.0f}%")


def main():
    api_key    = os.environ["APCA_API_KEY_ID"]
    secret_key = os.environ["APCA_API_SECRET_KEY"]
    base_url   = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

    initial_equity = broker.get_account_info()["portfolio_value"]
    fetcher     = BarFetcher(api_key, secret_key)
    news_filter = NewsFilter(api_key, secret_key, cache_only=False)
    cache_dir   = fetcher._cache_dir

    for label, start, end in PERIODS:
        cfg = make_gap_hold_config()
        cfg.target_atr_multiple = 4.0
        cfg.stage2_min_vol_vs_prev_bar = 0.80

        screener = CandidateScreener(copy.copy(cfg), api_key, secret_key, base_url)
        print(f"Loading {label}...", flush=True)
        screener.preload(start, end)

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
        for d in trading_days(start, end):
            if not spy_uptrend(d):
                continue
            cands = screener.candidates_for_date(d)
            cached = [s for s in cands if (cache_dir / f"{s}_{d}.json").exists()]
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
            day_pnl = sum(t.pnl for t in result.trades if t.pnl is not None)
            running_eq = max(running_eq + day_pnl, 1.0)
            for t in result.trades:
                e = excursions(t, bars_by_sym.get(t.ticker, []))
                if e:
                    rows.append(e)

        pd.DataFrame(rows).to_csv(f"backtest_results/excursions_{label}.csv", index=False)
        report(rows, label)

    print("\nDone.")


if __name__ == "__main__":
    main()
