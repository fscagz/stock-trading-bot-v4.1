"""
Simulate a real-time HOD-rejection short entry with stops.

Strategy logic (no look-ahead):
  1. Stock must have peaked at ≥ MIN_RUN_PCT from its open (scanner already finds these)
  2. After that peak, wait for N consecutive bars that fail to make a new HOD
     → "HOD rejection confirmed" signal
  3. Entry: short at the close of bar N (or next-bar open in "next open" mode)
  4. Stop: HOD + STOP_ATR × ATR (stop above the day's high)
  5. Target: entry − TARGET_ATR × ATR
  6. Hold until: target hit, stop hit, or EOD (15:55 ET)

Sweep several combinations of MIN_RUN_PCT, REJECTION_BARS, STOP_ATR, TARGET_ATR
and print a results table showing win rate, EV, profit factor.

Usage:
    python experiments/simulate_hod_rejection.py [--year 2022|2025]
"""
from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import List, Optional, Tuple
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_ET = ZoneInfo("America/New_York")
CACHE_DIR = Path("backtest_results/cache")
SLIPPAGE = 0.001   # 0.1% short sell slippage (worse fill)


def load_bars(path: Path) -> List[dict]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return []


def bar_et(b: dict) -> datetime:
    return datetime.fromisoformat(b["t"].replace("Z", "+00:00")).astimezone(_ET)


def compute_atr(bars: List[dict], period: int = 14) -> List[Optional[float]]:
    """Wilder ATR over the bar list."""
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


def simulate_stock_day(
    bars: List[dict],
    min_run_pct: float,
    rejection_bars: int,
    stop_atr_mult: float,
    target_atr_mult: float,
) -> Optional[dict]:
    """
    Run one stock-day through the HOD-rejection strategy.
    Returns trade dict or None if no entry triggered.
    """
    if len(bars) < 30:
        return None

    open_price = bars[0]["o"]
    if open_price <= 0:
        return None

    atrs = compute_atr(bars)
    times = [bar_et(b) for b in bars]

    # Only trade during market hours
    def is_market(i: int) -> bool:
        t = times[i]
        return (t.hour > 9 or (t.hour == 9 and t.minute >= 30)) and t.hour < 16

    # State machine
    hod_close = open_price       # highest close seen so far
    hod_bar_idx = 0
    bars_since_hod = 0
    qualified = False            # True once stock has run min_run_pct
    entry_triggered = False

    for i, bar in enumerate(bars):
        if not is_market(i):
            continue
        if times[i].hour == 15 and times[i].minute >= 55:
            break   # no new entries after 15:55

        close = bar["c"]
        atr = atrs[i]

        # Track HOD
        if close > hod_close:
            hod_close = close
            hod_bar_idx = i
            bars_since_hod = 0
        else:
            bars_since_hod += 1

        # Check qualification: stock ran min_run_pct from open
        gain_pct = (hod_close - open_price) / open_price * 100
        if gain_pct >= min_run_pct:
            qualified = True

        if not qualified:
            continue
        if entry_triggered:
            continue
        if atr is None or atr <= 0:
            continue

        # HOD rejection: N bars without a new high
        if bars_since_hod < rejection_bars:
            continue

        # Entry signal confirmed
        entry_price = round(close * (1 - SLIPPAGE), 4)   # sell-short: slight slippage worsens entry
        stop_price = round(hod_close + stop_atr_mult * atr, 4)
        target_price = round(entry_price - target_atr_mult * atr, 4)

        if target_price <= 0 or stop_price <= entry_price:
            continue

        entry_triggered = True
        entry_bar = i

        # Simulate forward
        max_price_seen = entry_price   # for MAE
        exit_price = None
        exit_reason = None
        exit_bar = None

        for j in range(i + 1, len(bars)):
            if not is_market(j):
                continue
            fb = bars[j]
            ft = times[j]

            max_price_seen = max(max_price_seen, fb["h"])

            # EOD force-close
            if ft.hour == 15 and ft.minute >= 55:
                exit_price = fb["c"]
                exit_reason = "eod"
                exit_bar = j
                break

            # Stop hit (price rose to stop — gap or intrabar)
            if fb["h"] >= stop_price:
                exit_price = max(stop_price, fb["o"])   # gap-through fill
                exit_reason = "stop"
                exit_bar = j
                break

            # Target hit
            if fb["l"] <= target_price:
                exit_price = target_price
                exit_reason = "target"
                exit_bar = j
                break

        if exit_price is None:
            # Never got a close bar after entry (halted?) — use last bar close
            exit_price = bars[-1]["c"]
            exit_reason = "eod_no_bar"
            exit_bar = len(bars) - 1

        pnl_pct = (entry_price - exit_price) / entry_price * 100
        mae_pct = (max_price_seen - entry_price) / entry_price * 100
        hold_bars = (exit_bar - entry_bar) if exit_bar else 0

        return {
            "pnl_pct": pnl_pct,
            "exit_reason": exit_reason,
            "mae_pct": mae_pct,
            "hold_bars": hold_bars,
            "run_pct": gain_pct,
            "bars_to_signal": i - hod_bar_idx,   # = rejection_bars (by definition)
        }

    return None   # no qualifying entry today


def run_sweep(year: str, verbose: bool = False) -> None:
    if year == "2022":
        pattern = "*_2022-*.json"
    elif year == "2025":
        pattern = "*_202[56]-*.json"
    else:
        pattern = "*.json"

    files = list(CACHE_DIR.glob(pattern))
    print(f"Loading {len(files):,} stock-day files ({year})...\n")

    # Pre-load bars once
    all_bars = {}
    for f in files:
        bars = load_bars(f)
        if len(bars) >= 30 and bars[0]["o"] > 0:
            all_bars[f.stem] = bars
    print(f"Loaded {len(all_bars):,} usable stock-days.\n")

    # Parameter sweep
    min_runs     = [20, 30, 50, 75]
    rej_bars_list = [3, 5, 10, 20]
    stop_mults   = [1.0, 2.0]
    target_mults = [2.0, 4.0]

    print(f"  {'MinRun':>7} {'RejBars':>8} {'Stop×ATR':>9} {'Tgt×ATR':>8} "
          f"{'Trades':>7} {'WR':>7} {'AvgPnL%':>8} {'AvgW%':>7} {'AvgL%':>7} "
          f"{'ProfFactor':>11} {'EV/trade':>10}")
    print("  " + "-" * 97)

    best_ev = -999
    best_params = None

    for min_run, rej_bars, stop_mult, tgt_mult in product(
        min_runs, rej_bars_list, stop_mults, target_mults
    ):
        trades = []
        for stem, bars in all_bars.items():
            result = simulate_stock_day(bars, min_run, rej_bars, stop_mult, tgt_mult)
            if result:
                trades.append(result)

        if len(trades) < 10:
            continue

        pnls = [t["pnl_pct"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        wr = len(wins) / len(pnls) * 100
        avg_pnl = sum(pnls) / len(pnls)
        avg_w = sum(wins) / len(wins) if wins else 0
        avg_l = sum(losses) / len(losses) if losses else 0
        gross_w = sum(wins)
        gross_l = abs(sum(losses))
        pf = gross_w / gross_l if gross_l > 0 else float("inf")
        ev = avg_pnl

        print(f"  {min_run:>6}%  {rej_bars:>8}  {stop_mult:>8.1f}×  {tgt_mult:>7.1f}×  "
              f"{len(trades):>7,}  {wr:>6.1f}%  {avg_pnl:>7.2f}%  "
              f"{avg_w:>6.2f}%  {avg_l:>6.2f}%  "
              f"{pf:>10.2f}×  {ev:>9.3f}%")

        if ev > best_ev:
            best_ev = ev
            best_params = (min_run, rej_bars, stop_mult, tgt_mult)

    if best_params and verbose:
        min_run, rej_bars, stop_mult, tgt_mult = best_params
        print(f"\nBest params: min_run={min_run}%, rej_bars={rej_bars}, "
              f"stop={stop_mult}×ATR, target={tgt_mult}×ATR  (EV={best_ev:.3f}%)")
        print("\nDetailed trades for best params:")
        trades = []
        for stem, bars in all_bars.items():
            result = simulate_stock_day(bars, min_run, rej_bars, stop_mult, tgt_mult)
            if result:
                result["sym_day"] = stem
                trades.append(result)
        by_reason = {}
        for t in trades:
            by_reason.setdefault(t["exit_reason"], []).append(t["pnl_pct"])
        for reason, pnls in sorted(by_reason.items()):
            print(f"  {reason:>12}: {len(pnls):>5} trades, "
                  f"avg={sum(pnls)/len(pnls):.2f}%, "
                  f"wr={sum(1 for p in pnls if p > 0)/len(pnls)*100:.0f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", choices=["2022", "2025", "all"], default="2025")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run_sweep(args.year, args.verbose)
