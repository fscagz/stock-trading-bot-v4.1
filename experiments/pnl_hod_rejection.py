"""
Dollar PnL simulation of the HOD-rejection short strategy across all four years,
with regime filter variants.

Regime filters tested (for shorts, we enter on DOWNTREND days, not uptrend):
  none          — always trade (baseline)
  spy_ma20      — SPY below 20-day MA
  spy_ma50      — SPY below 50-day MA
  spy_ma200     — SPY below 200-day MA
  spy_mom5      — SPY closed lower than 5 trading days ago
  spy_mom10     — SPY closed lower than 10 trading days ago

Best param combo (75%/10bars/2×/2×) fixed throughout.

Usage:
    python experiments/pnl_hod_rejection.py
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

_ET = ZoneInfo("America/New_York")
CACHE_DIR = Path("backtest_results/cache")
SLIPPAGE   = 0.001

INITIAL_EQUITY  = 100_000.0
RISK_PER_TRADE  = 0.01
MAX_POSITION_PCT = 0.25

# Fixed params — best all-year combo from the sweep
MIN_RUN_PCT    = 75
REJECTION_BARS = 10
STOP_ATR_MULT  = 2.0
TARGET_ATR_MULT = 2.0

PERIODS: Dict[str, Tuple[str, ...]] = {
    "2022": ("_2022-",),
    "2023": ("_2023-",),
    "2024": ("_2024-",),
    "2025": ("_2025-", "_2026-"),
}


# ── SPY daily data ────────────────────────────────────────────────────────────

_spy_df: Optional[pd.DataFrame] = None

def _spy() -> pd.DataFrame:
    global _spy_df
    if _spy_df is None:
        from bot.data.daily_loader import get_daily
        _spy_df = get_daily("SPY", start="2020-01-01", end="2026-06-01")
        _spy_df["ma20"]  = _spy_df["close"].rolling(20).mean()
        _spy_df["ma50"]  = _spy_df["close"].rolling(50).mean()
        _spy_df["ma200"] = _spy_df["close"].rolling(200).mean()
        _spy_df["close_5d_ago"]  = _spy_df["close"].shift(5)
        _spy_df["close_10d_ago"] = _spy_df["close"].shift(10)
    return _spy_df


def regime_flags(start: date, end: date) -> Dict[date, Dict[str, bool]]:
    """Pre-compute all regime conditions for every trading day in range.

    Returns dict[trade_date → dict[filter_name → bool (True = short allowed)]].
    'none' is always True.
    MA/momentum filters: True = SPY in DOWNTREND = shorts allowed.
    """
    spy = _spy()
    flags: Dict[date, Dict[str, bool]] = {}
    d = start
    while d <= end:
        if d.weekday() < 5:
            past = spy[spy.index < pd.Timestamp(d)].dropna(subset=["ma200"])
            if past.empty:
                flags[d] = {k: True for k in
                            ("none", "spy_ma20", "spy_ma50", "spy_ma200",
                             "spy_mom5", "spy_mom10")}
            else:
                row = past.iloc[-1]
                close = float(row["close"])
                flags[d] = {
                    "none":      True,
                    "spy_ma20":  close < float(row["ma20"])  if pd.notna(row["ma20"])  else True,
                    "spy_ma50":  close < float(row["ma50"])  if pd.notna(row["ma50"])  else True,
                    "spy_ma200": close < float(row["ma200"]) if pd.notna(row["ma200"]) else True,
                    "spy_mom5":  close < float(row["close_5d_ago"])  if pd.notna(row["close_5d_ago"])  else True,
                    "spy_mom10": close < float(row["close_10d_ago"]) if pd.notna(row["close_10d_ago"]) else True,
                }
        d += timedelta(days=1)
    return flags


# ── bar / ATR helpers ─────────────────────────────────────────────────────────

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


# ── per-stock-day simulation ──────────────────────────────────────────────────

@dataclass
class Trade:
    pnl: float
    exit_reason: str
    trade_date: date


def simulate_stock_day(
    bars: List[dict],
    equity: float,
    trade_date: date,
) -> Optional[Trade]:
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

    hod_close    = open_price
    bars_since_hod = 0
    qualified    = False
    entry_triggered = False

    for i, bar in enumerate(bars):
        if not is_market(i):
            continue
        if times[i].hour == 15 and times[i].minute >= 55:
            break

        close = bar["c"]
        atr   = atrs[i]

        if close > hod_close:
            hod_close = close
            bars_since_hod = 0
        else:
            bars_since_hod += 1

        gain_pct = (hod_close - open_price) / open_price * 100
        if gain_pct >= MIN_RUN_PCT:
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
                exit_price = fb["c"]
                exit_reason = "eod"
                break
            if fb["h"] >= stop_price:
                exit_price = max(stop_price, fb["o"])
                exit_reason = "stop"
                break
            if fb["l"] <= target_price:
                exit_price = target_price
                exit_reason = "target"
                break

        if exit_price is None:
            exit_price = bars[-1]["c"]
            exit_reason = "eod_no_bar"

        pnl = round((entry_price - exit_price) * shares, 2)
        return Trade(pnl=pnl, exit_reason=exit_reason, trade_date=trade_date)

    return None


# ── period runner ─────────────────────────────────────────────────────────────

def run_period(
    suffixes: Tuple[str, ...],
    filter_name: str,
    regime: Dict[date, Dict[str, bool]],
) -> Tuple[List[Trade], List[float]]:
    """Run all stock-days for the period, gated by the regime filter."""
    files = []
    for suffix in suffixes:
        files.extend(CACHE_DIR.glob(f"*{suffix}*.json"))

    # Group by date so we can apply regime filter once per day
    by_date: Dict[date, List[Path]] = defaultdict(list)
    for f in files:
        # stem: SYMBOL_YYYY-MM-DD
        parts = f.stem.rsplit("_", 1)
        if len(parts) == 2:
            try:
                d = date.fromisoformat(parts[1])
                by_date[d].append(f)
            except ValueError:
                pass

    equity = INITIAL_EQUITY
    trades: List[Trade] = []
    curve: List[float] = [equity]

    for d in sorted(by_date):
        day_regime = regime.get(d, {})
        if not day_regime.get(filter_name, True):
            continue   # regime says: don't short today

        for f in by_date[d]:
            bars = load_bars(f)
            trade = simulate_stock_day(bars, equity, d)
            if trade is not None:
                equity = max(equity + trade.pnl, 1.0)
                trades.append(trade)
                curve.append(equity)

    return trades, curve


# ── metrics ───────────────────────────────────────────────────────────────────

def metrics(trades: List[Trade], curve: List[float]) -> dict:
    if not trades:
        return dict(n=0, wr=0, net_pnl=0, ret_pct=0, max_dd_pct=0, pf=0)
    pnls  = [t.pnl for t in trades]
    wins  = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    net   = sum(pnls)
    peak  = curve[0]
    max_dd = 0.0
    for eq in curve:
        peak = max(peak, eq)
        max_dd = max(max_dd, peak - eq)
    gross_w = sum(wins)
    gross_l = abs(sum(losses))
    return dict(
        n=len(trades),
        wr=len(wins) / len(pnls) * 100,
        net_pnl=net,
        ret_pct=net / INITIAL_EQUITY * 100,
        max_dd_pct=max_dd / max(curve) * 100 if max(curve) > 0 else 0,
        pf=gross_w / gross_l if gross_l > 0 else float("inf"),
    )


# ── main ──────────────────────────────────────────────────────────────────────

FILTERS = ["none", "spy_ma20", "spy_ma50", "spy_ma200", "spy_mom5", "spy_mom10"]
FILTER_LABELS = {
    "none":      "no filter    ",
    "spy_ma20":  "SPY<MA20     ",
    "spy_ma50":  "SPY<MA50     ",
    "spy_ma200": "SPY<MA200    ",
    "spy_mom5":  "SPY<5d ago   ",
    "spy_mom10": "SPY<10d ago  ",
}

def main() -> None:
    print(f"Params: min_run={MIN_RUN_PCT}%  rej_bars={REJECTION_BARS}  "
          f"stop={STOP_ATR_MULT}×ATR  target={TARGET_ATR_MULT}×ATR  "
          f"equity=${INITIAL_EQUITY:,.0f}  risk={RISK_PER_TRADE*100:.0f}%\n")

    print("Loading SPY data...", flush=True)
    spy = _spy()   # warm cache

    # Pre-compute regime flags for all periods
    period_regimes: Dict[str, Dict[date, Dict[str, bool]]] = {}
    period_ranges = {
        "2022": (date(2022, 1, 1),  date(2022, 12, 31)),
        "2023": (date(2023, 1, 1),  date(2023, 12, 31)),
        "2024": (date(2024, 1, 1),  date(2024, 12, 31)),
        "2025": (date(2025, 6, 1),  date(2026, 5, 31)),
    }
    for yr, (s, e) in period_ranges.items():
        period_regimes[yr] = regime_flags(s, e)

    # Print SPY regime stats per year (how many days each filter is active)
    print("SPY regime coverage (trading days where filter ALLOWS shorts):")
    print(f"  {'Filter':<16}", end="")
    for yr in PERIODS:
        print(f"  {yr:>10}", end="")
    print()
    print("  " + "-" * (16 + len(PERIODS) * 12))
    for f in FILTERS:
        print(f"  {FILTER_LABELS[f]}", end="")
        for yr in PERIODS:
            reg = period_regimes[yr]
            allowed = sum(1 for v in reg.values() if v.get(f, True))
            total   = len(reg)
            print(f"  {allowed:>4}/{total:<4}", end="")
        print()

    print()
    yr_keys = list(PERIODS.keys())
    print(f"  {'Regime filter':<16}", end="")
    for yr in yr_keys:
        print(f"  {'── ' + yr + ' ──':^38}", end="")
    print()
    print(f"  {'':16}", end="")
    for _ in yr_keys:
        print(f"  {'N':>5} {'WR':>6} {'NetPnL':>10} {'Ret%':>6} {'MaxDD%':>7} {'PF':>5}", end="")
    print()
    print("  " + "-" * (16 + len(yr_keys) * 41))

    # Also collect totals across all 4 years per filter
    filter_totals: Dict[str, Dict] = {}

    for fname in FILTERS:
        print(f"  {FILTER_LABELS[fname]}", end="", flush=True)
        all_trades: List[Trade] = []
        all_curve:  List[float] = [INITIAL_EQUITY]
        for yr, suffixes in PERIODS.items():
            trades, curve = run_period(suffixes, fname, period_regimes[yr])
            m = metrics(trades, curve)
            pnl_str = f"${m['net_pnl']:>+9,.0f}"
            print(f"  {m['n']:>5,} {m['wr']:>5.1f}% {pnl_str} {m['ret_pct']:>5.1f}% "
                  f"{m['max_dd_pct']:>6.1f}% {m['pf']:>4.2f}×", end="")
            all_trades.extend(trades)
        print()
        # Rebuild combined equity curve (sequential across all years)
        eq = INITIAL_EQUITY
        combined_curve = [eq]
        for t in all_trades:
            eq = max(eq + t.pnl, 1.0)
            combined_curve.append(eq)
        filter_totals[fname] = metrics(all_trades, combined_curve)

    # Summary: all-year combined
    print()
    print("  4-year combined (2022+2023+2024+2025):")
    print(f"  {'':16}  {'N':>6}  {'WR':>6}  {'NetPnL':>10}  {'Ret%':>7}  {'MaxDD%':>8}  {'PF':>6}")
    print("  " + "-" * 66)
    for fname in FILTERS:
        m = filter_totals[fname]
        print(f"  {FILTER_LABELS[fname]}  {m['n']:>6,}  {m['wr']:>5.1f}%  "
              f"${m['net_pnl']:>+9,.0f}  {m['ret_pct']:>6.1f}%  "
              f"{m['max_dd_pct']:>7.1f}%  {m['pf']:>5.2f}×")


if __name__ == "__main__":
    main()
