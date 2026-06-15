"""
Analyze intraday fade opportunities in the bar cache.

Questions answered:
  1. How often do big movers retrace significantly by EOD?
  2. When does the intraday peak happen (what minute)?
  3. What bar characteristics at/near the peak predict a subsequent retracement?
  4. If you entered short N bars after the peak, what's the expected outcome?
  5. Does requiring a minimum prior run (30%/50%/100%) improve edge?

Usage:
    python experiments/analyze_fade_opportunities.py [--year 2022|2025]
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_ET = ZoneInfo("America/New_York")
CACHE_DIR = Path("backtest_results/cache")

# ── bar helpers ─────────────────────────────────────────────────────────────

def load_bars(path: Path) -> List[dict]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return []

def bar_et(b: dict) -> datetime:
    return datetime.fromisoformat(b["t"].replace("Z", "+00:00")).astimezone(_ET)

# ── per-stock-day analysis ───────────────────────────────────────────────────

def analyze_day(bars: List[dict]) -> Optional[dict]:
    """Return summary stats for one stock-day, or None if too few bars."""
    if len(bars) < 30:
        return None

    open_price = bars[0]["o"]
    if open_price <= 0:
        return None

    # Compute per-bar gain from open
    closes = [b["c"] for b in bars]
    highs  = [b["h"] for b in bars]
    vols   = [b["v"] for b in bars]
    times  = [bar_et(b) for b in bars]

    # Use market hours only (9:30–16:00)
    mkt_bars = [(i, b) for i, b in enumerate(bars)
                if times[i].hour >= 9 and (times[i].hour < 16 or
                (times[i].hour == 16 and times[i].minute == 0))]
    if len(mkt_bars) < 20:
        return None

    mkt_indices = [i for i, _ in mkt_bars]

    # Peak: bar with highest close during market hours
    peak_idx = max(mkt_indices, key=lambda i: closes[i])
    peak_close = closes[peak_idx]
    peak_gain_pct = (peak_close - open_price) / open_price * 100

    # Final close (last market bar)
    final_idx = mkt_indices[-1]
    final_close = closes[final_idx]
    final_gain_pct = (final_close - open_price) / open_price * 100

    # Retracement from peak to close
    retracement_pct = (peak_close - final_close) / peak_close * 100

    # What minute of the day did the peak happen?
    peak_time = times[peak_idx]
    mins_since_open = (peak_time.hour - 9) * 60 + peak_time.minute - 30

    # Volume at peak bar vs avg bar volume
    avg_vol = sum(vols[i] for i in mkt_indices) / len(mkt_indices)
    peak_vol_ratio = vols[peak_idx] / avg_vol if avg_vol > 0 else 0

    # Bar characteristics at peak
    peak_bar = bars[peak_idx]
    bar_range = peak_bar["h"] - peak_bar["l"]
    upper_wick_pct = (peak_bar["h"] - peak_bar["c"]) / bar_range if bar_range > 0 else 0
    is_red = peak_bar["c"] < peak_bar["o"]

    # Volume trend: is volume at peak declining vs 3-bar avg before?
    prev_vol_avg = 0.0
    if peak_idx >= 3:
        prev_vol_avg = sum(vols[peak_idx - k] for k in range(1, 4)) / 3
    vol_declining_at_peak = (vols[peak_idx] < prev_vol_avg) if prev_vol_avg > 0 else False

    # Post-peak behavior: gains/losses N bars after peak
    post_peak = {}
    for lag in [1, 3, 5, 10, 20, 30]:
        future_idx = peak_idx + lag
        if future_idx < len(bars):
            post_pnl_pct = (peak_close - closes[future_idx]) / peak_close * 100
        else:
            post_pnl_pct = (peak_close - final_close) / peak_close * 100
        post_peak[lag] = post_pnl_pct

    # Max adverse excursion after peak (how far did it go ABOVE peak before falling?)
    max_high_after_peak = max((highs[i] for i in range(peak_idx + 1, len(bars))), default=peak_close)
    mae_pct = (max_high_after_peak - peak_close) / peak_close * 100

    return {
        "open": open_price,
        "peak_gain_pct": peak_gain_pct,
        "final_gain_pct": final_gain_pct,
        "retracement_pct": retracement_pct,       # % of peak close that was given back by EOD
        "retracement_abs_pct": peak_gain_pct - final_gain_pct,  # percentage points given back
        "mins_to_peak": mins_since_open,
        "peak_vol_ratio": peak_vol_ratio,
        "upper_wick_pct": upper_wick_pct,
        "is_red_at_peak": is_red,
        "vol_declining_at_peak": vol_declining_at_peak,
        "mae_pct": mae_pct,
        "post_peak": post_peak,
    }


# ── aggregation helpers ──────────────────────────────────────────────────────

def pct_positive(vals: List[float]) -> float:
    if not vals:
        return 0.0
    return sum(1 for v in vals if v > 0) / len(vals) * 100

def mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0

def median(vals: List[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    return (s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2)

def bucket(val: float, edges: List[float]) -> str:
    for i, e in enumerate(edges):
        if val < e:
            prev = edges[i - 1] if i > 0 else 0
            return f"{prev:.0f}-{e:.0f}%"
    return f"{edges[-1]:.0f}%+"


# ── main analysis ────────────────────────────────────────────────────────────

def run_analysis(year: str) -> None:
    if year == "2022":
        pattern = "*_2022-*.json"
    elif year == "2025":
        pattern = "*_202[56]-*.json"
    else:
        pattern = "*.json"

    files = list(CACHE_DIR.glob(pattern))
    print(f"Loading {len(files):,} stock-day files ({year})...")

    records = []
    for f in files:
        bars = load_bars(f)
        result = analyze_day(bars)
        if result and result["peak_gain_pct"] >= 5.0:  # only meaningful movers
            records.append(result)

    print(f"Usable stock-days (peak gain ≥5%): {len(records):,}\n")

    # ── Section 1: Retracement by peak move size ─────────────────────────────
    print("=" * 72)
    print("1. RETRACEMENT BY PEAK MOVE SIZE")
    print("   (what % of stocks that peaked at X% gave back ≥50% of the move by EOD)")
    print("=" * 72)
    move_buckets = [15, 30, 50, 75, 100]
    print(f"  {'Peak gain':>12}  {'N':>6}  {'Retrace≥25%':>12}  {'Retrace≥50%':>12}  "
          f"{'Retrace≥75%':>12}  {'MedRetrace':>11}  {'AvgMAE(stop)':>13}")
    print("  " + "-" * 80)
    for i, thresh in enumerate(move_buckets):
        lo = move_buckets[i - 1] if i > 0 else 5
        subset = [r for r in records if lo <= r["peak_gain_pct"] < thresh]
        if len(subset) < 10:
            continue
        retracements = [r["retracement_pct"] for r in subset]
        maes = [r["mae_pct"] for r in subset]
        print(f"  {lo:>5}-{thresh:<5}%   {len(subset):>6,}  "
              f"{pct_positive([v - 25 for v in retracements]):>11.0f}%  "
              f"{pct_positive([v - 50 for v in retracements]):>11.0f}%  "
              f"{pct_positive([v - 75 for v in retracements]):>11.0f}%  "
              f"{median(retracements):>10.1f}%  "
              f"{mean(maes):>12.1f}%")
    # 100%+
    subset = [r for r in records if r["peak_gain_pct"] >= 100]
    if len(subset) >= 5:
        retracements = [r["retracement_pct"] for r in subset]
        maes = [r["mae_pct"] for r in subset]
        print(f"  {'100%+':>10}   {len(subset):>6,}  "
              f"{pct_positive([v - 25 for v in retracements]):>11.0f}%  "
              f"{pct_positive([v - 50 for v in retracements]):>11.0f}%  "
              f"{pct_positive([v - 75 for v in retracements]):>11.0f}%  "
              f"{median(retracements):>10.1f}%  "
              f"{mean(maes):>12.1f}%")

    # ── Section 2: Timing — when does the peak happen? ───────────────────────
    print()
    print("=" * 72)
    print("2. TIMING — WHEN DOES THE PEAK HAPPEN? (for stocks peaking ≥30%)")
    print("=" * 72)
    big_movers = [r for r in records if r["peak_gain_pct"] >= 30]
    timing_buckets = [
        ("0-15min  (9:30-9:45)", 0, 15),
        ("15-30min (9:45-10:00)", 15, 30),
        ("30-60min (10:00-10:30)", 30, 60),
        ("60-120min(10:30-11:30)", 60, 120),
        ("120-180min(11:30-12:30)", 120, 180),
        ("180min+  (afternoon)",  180, 999),
    ]
    print(f"  {'Window':>26}  {'Count':>6}  {'%ofTotal':>9}  {'MedRetrace':>11}  {'Retrace≥50%':>12}")
    print("  " + "-" * 70)
    total_big = len(big_movers)
    for label, lo, hi in timing_buckets:
        sub = [r for r in big_movers if lo <= r["mins_to_peak"] < hi]
        if not sub:
            continue
        ret = [r["retracement_pct"] for r in sub]
        print(f"  {label:>26}  {len(sub):>6,}  {len(sub)/total_big*100:>8.1f}%  "
              f"{median(ret):>10.1f}%  {pct_positive([v-50 for v in ret]):>11.0f}%")

    # ── Section 3: Post-peak outcome by lag ──────────────────────────────────
    print()
    print("=" * 72)
    print("3. POST-PEAK SHORT PnL: enter short at peak bar, hold N bars")
    print("   (stocks peaked ≥30%; positive = short made money)")
    print("=" * 72)
    print(f"  {'Lag (bars)':>12}  {'WinRate':>8}  {'AvgGain%':>10}  "
          f"{'Avg if Win':>11}  {'Avg if Loss':>12}  {'ExpValue%':>11}")
    print("  " + "-" * 68)
    for lag in [1, 3, 5, 10, 20, 30]:
        gains = [r["post_peak"].get(lag, 0) for r in big_movers]
        wins = [g for g in gains if g > 0]
        losses = [g for g in gains if g <= 0]
        ev = mean(gains)
        print(f"  {lag:>12}  {pct_positive(gains):>7.1f}%  {mean(gains):>9.2f}%  "
              f"{mean(wins):>10.2f}%  {mean(losses):>11.2f}%  {ev:>10.3f}%")

    # ── Section 4: Does requiring a min run improve post-peak edge? ──────────
    print()
    print("=" * 72)
    print("4. MINIMUM RUN FILTER — post-peak edge by prior move size (lag=10 bars)")
    print("=" * 72)
    print(f"  {'Min peak gain':>14}  {'N':>6}  {'WinRate':>8}  {'AvgGain%':>10}  {'ExpValue%':>11}")
    print("  " + "-" * 56)
    for thresh in [10, 20, 30, 50, 75, 100]:
        sub = [r for r in records if r["peak_gain_pct"] >= thresh]
        if len(sub) < 10:
            continue
        gains = [r["post_peak"].get(10, 0) for r in sub]
        print(f"  {thresh:>13}%  {len(sub):>6,}  "
              f"{pct_positive(gains):>7.1f}%  {mean(gains):>9.2f}%  {mean(gains):>10.3f}%")

    # ── Section 5: Bar-level signals at peak ─────────────────────────────────
    print()
    print("=" * 72)
    print("5. BAR-LEVEL SIGNALS AT PEAK — do they predict the fade? (lag=10, peak≥30%)")
    print("=" * 72)
    sub = [r for r in records if r["peak_gain_pct"] >= 30]
    conditions = [
        ("all",               sub),
        ("upper wick >50%",   [r for r in sub if r["upper_wick_pct"] > 0.50]),
        ("upper wick >70%",   [r for r in sub if r["upper_wick_pct"] > 0.70]),
        ("red bar at peak",   [r for r in sub if r["is_red_at_peak"]]),
        ("vol declining",     [r for r in sub if r["vol_declining_at_peak"]]),
        ("wick>50%+red",      [r for r in sub if r["upper_wick_pct"] > 0.50 and r["is_red_at_peak"]]),
        ("vol decl+red",      [r for r in sub if r["vol_declining_at_peak"] and r["is_red_at_peak"]]),
        ("all 3",             [r for r in sub if r["upper_wick_pct"] > 0.50 and r["is_red_at_peak"] and r["vol_declining_at_peak"]]),
    ]
    print(f"  {'Condition':>22}  {'N':>6}  {'WinRate':>8}  {'AvgGain%':>10}  {'AvgMAE%':>9}")
    print("  " + "-" * 62)
    for label, s in conditions:
        if not s:
            continue
        gains = [r["post_peak"].get(10, 0) for r in s]
        maes = [r["mae_pct"] for r in s]
        print(f"  {label:>22}  {len(s):>6,}  {pct_positive(gains):>7.1f}%  "
              f"{mean(gains):>9.2f}%  {mean(maes):>8.2f}%")

    # ── Section 6: Simulated short entry after confirmed reversal ────────────
    print()
    print("=" * 72)
    print("6. SIMULATED ENTRY: wait N bars past peak before entering short")
    print("   (peak≥30%, measure PnL from entry to EOD close)")
    print("=" * 72)
    print(f"  {'Wait bars':>10}  {'N entries':>10}  {'WinRate':>8}  "
          f"{'AvgPnL%':>9}  {'AvgWin%':>9}  {'AvgLoss%':>10}  {'EV%':>8}")
    print("  " + "-" * 72)
    # entry price = close at (peak + wait), exit = final close
    for wait in [0, 1, 3, 5, 10, 15, 20, 30]:
        results = []
        for r in sub:
            entry_gain = r["post_peak"].get(wait, None)
            if entry_gain is None:
                continue
            # entry_price ≈ peak_close * (1 - entry_gain/100)
            # pnl from entry to EOD = entry_gain - retracement_abs from entry onward
            # Simpler: if entered at peak - wait, pnl = entry_close - final_close
            # We know: final_gain_pct from open, peak_gain_pct from open
            # entry_close / open = (1 + peak_gain_pct/100) * (1 - entry_gain_from_peak/100)
            # We want: (entry_close - final_close) / entry_close
            peak_mult = 1 + r["peak_gain_pct"] / 100
            entry_mult = peak_mult * (1 - entry_gain / 100)
            final_mult = 1 + r["final_gain_pct"] / 100
            if entry_mult <= 0:
                continue
            pnl_pct = (entry_mult - final_mult) / entry_mult * 100
            results.append(pnl_pct)
        if not results:
            continue
        wins = [v for v in results if v > 0]
        losses = [v for v in results if v <= 0]
        print(f"  {wait:>10}  {len(results):>10,}  {pct_positive(results):>7.1f}%  "
              f"{mean(results):>8.2f}%  {mean(wins) if wins else 0:>8.2f}%  "
              f"{mean(losses) if losses else 0:>9.2f}%  {mean(results):>7.2f}%")

    print()
    print("Notes:")
    print("  - 'Retrace' = (peak_close - final_close) / peak_close × 100")
    print("  - 'MAE%' = max adverse excursion above peak before EOD (cost of a stop above peak)")
    print("  - Section 6 entry is at the close of bar (peak + wait), exit at EOD close")
    print("  - No slippage, no ETB, no stop modeled — upper bound on fade edge")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", choices=["2022", "2025", "all"], default="2025")
    args = parser.parse_args()
    run_analysis(args.year)
