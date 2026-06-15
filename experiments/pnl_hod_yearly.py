"""
Breaks out each calendar year individually (including 2025 and 2026 separately).
Uses the best regime filter (SPY<MA50) alongside no-filter baseline.
Params fixed: 75% min-run, 10-bar rejection, 2×ATR stop, 2×ATR target.
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd

_ET = ZoneInfo("America/New_York")
CACHE_DIR = Path("backtest_results/cache")
SLIPPAGE = 0.001

INITIAL_EQUITY   = 100_000.0
RISK_PER_TRADE   = 0.01
MAX_POSITION_PCT = 0.25
MIN_RUN_PCT      = 75
REJECTION_BARS   = 10
STOP_ATR_MULT    = 2.0
TARGET_ATR_MULT  = 2.0

_spy_df: Optional[pd.DataFrame] = None

def _spy() -> pd.DataFrame:
    global _spy_df
    if _spy_df is None:
        from bot.data.daily_loader import get_daily
        _spy_df = get_daily("SPY", start="2019-01-01", end="2026-06-15")
        _spy_df["ma50"] = _spy_df["close"].rolling(50).mean()
    return _spy_df

def spy_below_ma50(d: date) -> bool:
    spy = _spy()
    past = spy[spy.index < pd.Timestamp(d)]
    if len(past) < 50:
        return False
    row = past.iloc[-1]
    return pd.notna(row["ma50"]) and float(row["close"]) < float(row["ma50"])

def load_bars(path: Path) -> List[dict]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return []

def bar_et(b: dict) -> datetime:
    return datetime.fromisoformat(b["t"].replace("Z", "+00:00")).astimezone(_ET)

def compute_atr(bars: List[dict], period: int = 14) -> List[Optional[float]]:
    atrs: List[Optional[float]] = [None] * len(bars)
    trs = []
    for i, b in enumerate(bars):
        tr = b["h"] - b["l"]
        if i > 0:
            tr = max(tr, abs(b["h"] - bars[i-1]["c"]), abs(b["l"] - bars[i-1]["c"]))
        trs.append(tr)
    if len(trs) < period:
        return atrs
    smooth = sum(trs[:period]) / period
    for i in range(period - 1, len(bars)):
        if i == period - 1:
            smooth = sum(trs[:period]) / period
        else:
            smooth = (smooth * (period - 1) + trs[i]) / period
        atrs[i] = smooth
    return atrs

@dataclass
class Trade:
    pnl: float
    exit_reason: str

def simulate_stock_day(bars: List[dict], equity: float) -> Optional[Trade]:
    if len(bars) < 30:
        return None
    open_price = bars[0]["o"]
    if open_price <= 0:
        return None
    atrs = compute_atr(bars)
    times = [bar_et(b) for b in bars]

    def is_market(i: int) -> bool:
        t = times[i]
        return (t.hour > 9 or (t.hour == 9 and t.minute >= 30)) and t.hour < 16

    hod_close = open_price
    bars_since_hod = 0
    qualified = False
    entry_triggered = False

    for i, bar in enumerate(bars):
        if not is_market(i):
            continue
        if times[i].hour == 15 and times[i].minute >= 55:
            break
        close = bar["c"]
        atr = atrs[i]
        if close > hod_close:
            hod_close = close
            bars_since_hod = 0
        else:
            bars_since_hod += 1
        if (hod_close - open_price) / open_price * 100 >= MIN_RUN_PCT:
            qualified = True
        if not qualified or entry_triggered or atr is None or atr <= 0:
            continue
        if bars_since_hod < REJECTION_BARS:
            continue
        entry_price = round(close * (1 - SLIPPAGE), 4)
        stop_price  = round(hod_close + STOP_ATR_MULT * atr, 4)
        target_price = round(entry_price - TARGET_ATR_MULT * atr, 4)
        if target_price <= 0 or stop_price <= entry_price:
            continue
        stop_dist = stop_price - entry_price
        shares = min(
            (equity * RISK_PER_TRADE) / stop_dist,
            (equity * MAX_POSITION_PCT) / entry_price,
        )
        if shares < 1:
            continue
        entry_triggered = True
        exit_price = None
        exit_reason = None
        for j in range(i + 1, len(bars)):
            if not is_market(j):
                continue
            fb = bars[j]
            ft = times[j]
            if ft.hour == 15 and ft.minute >= 55:
                exit_price = fb["c"]; exit_reason = "eod"; break
            if fb["h"] >= stop_price:
                exit_price = max(stop_price, fb["o"]); exit_reason = "stop"; break
            if fb["l"] <= target_price:
                exit_price = target_price; exit_reason = "target"; break
        if exit_price is None:
            exit_price = bars[-1]["c"]; exit_reason = "eod_no_bar"
        return Trade(pnl=round((entry_price - exit_price) * shares, 2), exit_reason=exit_reason)
    return None

def run_year(year_str: str, use_regime: bool) -> dict:
    files = list(CACHE_DIR.glob(f"*_{year_str}-*.json"))
    by_date: Dict[date, List[Path]] = defaultdict(list)
    for f in files:
        parts = f.stem.rsplit("_", 1)
        if len(parts) == 2:
            try:
                by_date[date.fromisoformat(parts[1])].append(f)
            except ValueError:
                pass

    equity = INITIAL_EQUITY
    trades: List[Trade] = []
    curve: List[float] = [equity]

    for d in sorted(by_date):
        if use_regime and not spy_below_ma50(d):
            continue
        for f in by_date[d]:
            trade = simulate_stock_day(load_bars(f), equity)
            if trade:
                equity = max(equity + trade.pnl, 1.0)
                trades.append(trade)
                curve.append(equity)

    if not trades:
        return dict(n=0, wr=0, net_pnl=0, ret_pct=0, max_dd_pct=0, pf=0,
                    active_days=len(by_date))

    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    peak = curve[0]
    max_dd = 0.0
    for eq in curve:
        peak = max(peak, eq)
        max_dd = max(max_dd, peak - eq)

    return dict(
        n=len(trades),
        wr=len(wins) / len(pnls) * 100,
        net_pnl=sum(pnls),
        ret_pct=sum(pnls) / INITIAL_EQUITY * 100,
        max_dd_pct=max_dd / max(curve) * 100,
        pf=sum(wins) / abs(sum(losses)) if losses else float("inf"),
        active_days=len(by_date),
    )

YEARS = ["2022", "2023", "2024", "2025", "2026"]

def main() -> None:
    print("Loading SPY...", flush=True)
    _spy()

    print(f"\nParams: min_run={MIN_RUN_PCT}%  rej={REJECTION_BARS}bars  "
          f"stop={STOP_ATR_MULT}×ATR  target={TARGET_ATR_MULT}×ATR\n")

    hdr = f"  {'Year':<6} {'Filter':<14} {'Days':>5} {'N':>5} {'WR':>6} {'NetPnL':>11} {'Ret%':>6} {'MaxDD%':>7} {'PF':>6}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    totals_none   = dict(n=0, net_pnl=0.0, wins_gross=0.0, losses_gross=0.0)
    totals_regime = dict(n=0, net_pnl=0.0, wins_gross=0.0, losses_gross=0.0)

    for yr in YEARS:
        for use_regime, label in [(False, "no filter    "), (True,  "SPY<MA50     ")]:
            m = run_year(yr, use_regime)
            pnl_str = f"${m['net_pnl']:>+9,.0f}"
            print(f"  {yr:<6} {label:<14} {m['active_days']:>5} {m['n']:>5,} "
                  f"{m['wr']:>5.1f}% {pnl_str} {m['ret_pct']:>5.1f}% "
                  f"{m['max_dd_pct']:>6.1f}% {m['pf']:>5.2f}×")
        print()

if __name__ == "__main__":
    main()
