"""
Fill-realism test: does the gap-hold strategy's backtest edge survive a
participation-capped fill model?

Live measurements (114 trades, 2026-06-17 → 2026-07-01) showed median
round-trip slippage of 0.51R, and forensics on the +$81k 2025-06→2026-05
backtest showed the top-10 winners (214% of total profit) required fills like
111k shares of a $0.45 microcap — impossible without moving the market.

Runs the same config as run_combined_backtest's "New config (rv=6 + gap ≥5%)"
over 2025-06-01 → 2026-04-30 (bar cache coverage) with three fill models:

  legacy:     any size fills at quote + 0.2% slippage  (the model that said +$81k)
  realistic:  fills capped at 10% of the fill bar's volume,
              slippage 0.2% + 5% × participation, applied on entry AND exit
  strict:     capped at 5% of bar volume, slippage 0.2% + 10% × participation

Uses only local caches (bars, screener pkl, news) — no API calls.
"""
from __future__ import annotations
import copy, os, sys, warnings, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from typing import Dict, List

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

import pandas as pd
from bot.backtest.bar_fetcher import BarFetcher
from bot.backtest.candidate_screener import CandidateScreener
from bot.backtest.news_filter import NewsFilter
from bot.backtest.simulator import Simulator
from bot.config import make_gap_hold_config
from bot.data.daily_loader import get_daily
from bot.intraday.types import TradeRecord

PB_ATR, PB_TTL, PB_TARGET = 2.0, 60, 2.5
CHASE_TARGET = 4.0
DV_CAP = 0.20
DV = 10_000_000
SLIPPAGE = 0.002
GAP_HOLD_BARS = 15
GAP_HOLD_TOLERANCE = 0.02
INITIAL_EQUITY = 75_000.0

START, END = date(2025, 6, 1), date(2026, 4, 30)

# (label, max_bar_participation, impact_slippage_coeff)
FILL_MODELS = [
    ("legacy (any size fills)",        0.00, 0.00),
    ("realistic (≤10% bar vol)",       0.10, 0.05),
    ("strict (≤5% bar vol)",           0.05, 0.10),
]

_spy = None

def spy_uptrend(d: date) -> bool:
    global _spy
    if _spy is None:
        _spy = get_daily("SPY", start="2024-06-01", end="2026-06-10")
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


def prev_close_for(screener, sym, as_of: date) -> float:
    df = screener._daily_cache.get(sym)
    if df is None or df.empty:
        return 0.0
    past = df[df.index < pd.Timestamp(as_of)]
    return float(past.iloc[-1]["close"]) if not past.empty else 0.0


def run_variant(label, participation, impact, screener, fetcher, news_filter) -> dict:
    cfg = copy.copy(screener._config) if hasattr(screener, "_config") else None
    base = make_gap_hold_config()
    base.min_avg_dollar_volume = DV
    base.stage2_min_vol_vs_prev_bar = 0.80
    base.target_atr_multiple = CHASE_TARGET
    base.stage2_min_relative_volume = 6.0

    sim = Simulator(
        base, INITIAL_EQUITY,
        slippage_pct=SLIPPAGE,
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
        gap_hold_entry=True,
        gap_hold_min_pct=0.05,
        gap_hold_bars=GAP_HOLD_BARS,
        gap_hold_tolerance=GAP_HOLD_TOLERANCE,
        max_bar_participation=participation,
        impact_slippage_coeff=impact,
    )

    cache_dir = fetcher._cache_dir
    running_eq = INITIAL_EQUITY
    peak = INITIAL_EQUITY
    max_dd_pct = 0.0
    all_trades: List[TradeRecord] = []

    for d in trading_days(START, END):
        if not spy_uptrend(d):
            continue
        cands = screener.candidates_for_date(d)
        cached = [s for s in cands if (cache_dir / f"{s}_{d}.json").exists()]
        if not cached:
            continue
        bars_by_sym: Dict = {}
        baselines: Dict = {}
        prev_closes: Dict = {}
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
        for sym in bars_by_sym:
            pc = prev_close_for(screener, sym, d)
            if pc > 0:
                prev_closes[sym] = pc
        sim._initial_equity = running_eq
        result = sim.run_day(d, bars_by_sym, baselines, prev_closes=prev_closes)
        all_trades.extend(result.trades)
        day_pnl = sum(t.pnl for t in result.trades if t.pnl is not None)
        running_eq = max(running_eq + day_pnl, 1.0)
        peak = max(peak, running_eq)
        if peak > 0:
            max_dd_pct = max(max_dd_pct, (peak - running_eq) / peak * 100)

    trades = all_trades
    wins = [t for t in trades if t.pnl and t.pnl > 0]
    losses = [t for t in trades if t.pnl is not None and t.pnl <= 0]
    total_gain = sum(t.pnl for t in wins)
    total_loss = abs(sum(t.pnl for t in losses)) if losses else 0.0
    pnl_sorted = sorted((t.pnl for t in trades if t.pnl is not None), reverse=True)
    top10 = sum(pnl_sorted[:10])
    net = running_eq - INITIAL_EQUITY
    return {
        "label": label, "trades": len(trades),
        "wr": len(wins) / len(trades) * 100 if trades else 0.0,
        "net": net, "ret_pct": net / INITIAL_EQUITY * 100,
        "max_dd": max_dd_pct,
        "pf": total_gain / total_loss if total_loss > 0 else float("inf"),
        "top10": top10,
        "trades_list": trades,
    }


def main():
    api_key = os.environ.get("APCA_API_KEY_ID", "")
    secret_key = os.environ.get("APCA_API_SECRET_KEY", "")
    base_url = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

    base = make_gap_hold_config()
    base.min_avg_dollar_volume = DV
    base.stage2_min_vol_vs_prev_bar = 0.80
    base.target_atr_multiple = CHASE_TARGET
    base.stage2_min_relative_volume = 6.0

    fetcher = BarFetcher(api_key, secret_key)
    news_filter = NewsFilter(api_key, secret_key, cache_only=True)
    screener = CandidateScreener(copy.copy(base), api_key, secret_key, base_url)
    print(f"Preloading screener {START} → {END}...", flush=True)
    screener.preload(START, END)

    print(f"\nFill-realism backtest  {START} → {END}  equity=${INITIAL_EQUITY:,.0f}")
    HDR = (f"  {'Fill model':<28} {'Trades':>6} {'WR':>6} {'Net P&L':>12} "
           f"{'Return':>8} {'MaxDD':>7} {'PF':>6} {'Top10 P&L':>12}")
    print(HDR)
    print("  " + "-" * (len(HDR) - 2))
    for label, part, impact in FILL_MODELS:
        m = run_variant(label, part, impact, screener, fetcher, news_filter)
        pf = f"{m['pf']:.2f}" if m["pf"] != float("inf") else "   inf"
        print(f"  {label:<28} {m['trades']:>6,} {m['wr']:>5.1f}% {m['net']:>+11,.0f} "
              f"{m['ret_pct']:>+7.1f}% {m['max_dd']:>6.1f}% {pf:>6} {m['top10']:>+11,.0f}",
              flush=True)
        # dump per-trade CSV for forensics
        out = Path("backtest_results") / f"fillrealism_{label.split()[0]}_{START}_{END}.csv"
        rows = []
        for t in m["trades_list"]:
            rows.append({
                "ticker": t.ticker, "entry_time": t.entry_time, "entry_price": t.entry_price,
                "shares": t.shares, "exit_time": t.exit_time, "exit_price": t.exit_price,
                "pnl": t.pnl, "exit_reason": t.exit_reason, "signals": ";".join(t.signals or []),
            })
        pd.DataFrame(rows).to_csv(out, index=False)
    print("\nDone. Per-trade CSVs in backtest_results/fillrealism_*.csv")


if __name__ == "__main__":
    main()
