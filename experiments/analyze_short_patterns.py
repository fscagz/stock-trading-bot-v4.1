"""
Analyze pre-entry bar conditions for all short trades to find what separates
winners from losers. Loads cached 1-min bar data and computes features for
the bars leading up to each trade's entry.
"""
from __future__ import annotations
import csv
import json
from datetime import datetime, date, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
from collections import defaultdict

_ET = ZoneInfo("America/New_York")
CACHE_DIR = Path("backtest_results/cache")
TRADE_FILES = [
    "backtest_results/trades_short_2025-01-03_2025-04-04_etb_scale2.0.csv",
    "backtest_results/trades_short_2025-06-01_2026-05-28_etb_scale2.0.csv",
]
LOOKBACK = 10  # bars to examine before entry


def load_trades():
    trades = []
    for path in TRADE_FILES:
        p = Path(path)
        if not p.exists():
            continue
        with p.open() as f:
            for row in csv.DictReader(f):
                if row["direction"] != "short":
                    continue
                trades.append(row)
    return trades


def load_bars(symbol: str, trade_date: date):
    path = CACHE_DIR / f"{symbol}_{trade_date}.json"
    if not path.exists():
        return []
    raw = json.loads(path.read_text())
    bars = []
    for b in raw:
        ts = datetime.fromisoformat(b["t"].replace("Z", "+00:00"))
        bars.append({
            "ts": ts,
            "open": float(b["o"]),
            "high": float(b["h"]),
            "low": float(b["l"]),
            "close": float(b["c"]),
            "volume": int(b["v"]),
        })
    bars.sort(key=lambda b: b["ts"])
    return bars


def compute_features(trade: dict, bars: list) -> dict | None:
    entry_ts = datetime.fromisoformat(trade["entry_time"])
    entry_price = float(trade["entry_price"])
    pnl = float(trade["pnl"])

    # Find entry bar index
    entry_idx = None
    for i, b in enumerate(bars):
        if b["ts"] >= entry_ts:
            entry_idx = i
            break
    if entry_idx is None or entry_idx < 2:
        return None

    entry_bar = bars[entry_idx]
    prior_bars = bars[max(0, entry_idx - LOOKBACK):entry_idx]

    if not prior_bars:
        return None

    # Session open price
    session_open = bars[0]["open"]

    # Gain from session open to entry
    gain_from_open = (entry_price - session_open) / session_open if session_open > 0 else 0

    # Volume features
    volumes = [b["volume"] for b in prior_bars]
    peak_prior_volume = max(volumes) if volumes else 0
    entry_vol_ratio = entry_bar["volume"] / peak_prior_volume if peak_prior_volume > 0 else 1.0

    # How many of last 3 bars had declining volume (each bar < previous)
    last3 = prior_bars[-3:] if len(prior_bars) >= 3 else prior_bars
    vol_declining_count = sum(
        1 for i in range(1, len(last3)) if last3[i]["volume"] < last3[i-1]["volume"]
    )

    # How many of last 3 bars were red (close < open)
    pre_entry_red = sum(1 for b in last3 if b["close"] < b["open"])

    # Entry bar features
    entry_bar_red = entry_bar["close"] < entry_bar["open"]
    bar_range = entry_bar["high"] - entry_bar["low"]
    entry_close_pos = (entry_bar["close"] - entry_bar["low"]) / bar_range if bar_range > 0 else 0.5

    # Distance below the day's high up to entry
    day_high_to_entry = max(b["high"] for b in bars[:entry_idx + 1])
    dist_from_high = (day_high_to_entry - entry_price) / day_high_to_entry if day_high_to_entry > 0 else 0

    # Bars elapsed since stock first crossed +5% from session open
    spike_start_idx = None
    threshold = session_open * 1.05
    for i, b in enumerate(bars):
        if b["high"] >= threshold:
            spike_start_idx = i
            break
    spike_age = (entry_idx - spike_start_idx) if spike_start_idx is not None else 0

    # Volume trend over last 5 bars: is volume consistently declining?
    last5_vols = [b["volume"] for b in prior_bars[-5:]]
    if len(last5_vols) >= 3:
        # Linear regression slope (positive = growing, negative = declining)
        n = len(last5_vols)
        xs = list(range(n))
        mean_x = sum(xs) / n
        mean_y = sum(last5_vols) / n
        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, last5_vols))
        den = sum((x - mean_x) ** 2 for x in xs)
        vol_slope = num / den if den > 0 else 0
        vol_slope_pct = vol_slope / mean_y if mean_y > 0 else 0  # normalized
    else:
        vol_slope_pct = 0

    # Entry bar volume vs immediately prior bar
    prev_bar = prior_bars[-1] if prior_bars else None
    vol_vs_prev = entry_bar["volume"] / prev_bar["volume"] if prev_bar and prev_bar["volume"] > 0 else 1.0

    # Time from market open (9:30 ET) in minutes
    et_ts = entry_ts.astimezone(_ET)
    minutes_from_open = (et_ts.hour - 9) * 60 + et_ts.minute - 30

    return {
        "ticker": trade["ticker"],
        "entry_time": trade["entry_time"],
        "pnl": pnl,
        "winner": pnl > 0,
        "exit_reason": trade["exit_reason"],
        "gain_from_open_pct": round(gain_from_open * 100, 2),
        "entry_vol_vs_peak": round(entry_vol_ratio, 3),
        "vol_vs_prev_bar": round(vol_vs_prev, 3),
        "vol_declining_last3": vol_declining_count,
        "pre_entry_red_bars": pre_entry_red,
        "entry_bar_red": entry_bar_red,
        "entry_close_pos": round(entry_close_pos, 3),
        "dist_from_day_high_pct": round(dist_from_high * 100, 2),
        "spike_age_bars": spike_age,
        "vol_slope_pct": round(vol_slope_pct, 4),
        "minutes_from_open": minutes_from_open,
    }


def mean(vals):
    return sum(vals) / len(vals) if vals else 0


def pct_true(vals):
    return sum(1 for v in vals if v) / len(vals) * 100 if vals else 0


def bucket(vals, thresholds):
    """Count vals into buckets defined by thresholds."""
    counts = defaultdict(int)
    for v in vals:
        for t in thresholds:
            if v <= t:
                counts[f"<={t}"] += 1
                break
        else:
            counts[f">{thresholds[-1]}"] += 1
    return dict(counts)


def main():
    trades = load_trades()
    print(f"Loaded {len(trades)} short trades\n")

    features = []
    skipped = 0
    for t in trades:
        entry_ts = datetime.fromisoformat(t["entry_time"])
        trade_date = entry_ts.astimezone(_ET).date()
        bars = load_bars(t["ticker"], trade_date)
        if not bars:
            skipped += 1
            continue
        f = compute_features(t, bars)
        if f:
            features.append(f)
        else:
            skipped += 1

    print(f"Analyzed {len(features)} trades ({skipped} skipped — no cached bars)\n")

    winners = [f for f in features if f["winner"]]
    losers = [f for f in features if not f["winner"]]
    print(f"Winners: {len(winners)}  Losers: {len(losers)}  Win rate: {len(winners)/len(features)*100:.1f}%\n")

    numeric_features = [
        ("gain_from_open_pct",    "Gain from session open (%)"),
        ("entry_vol_vs_peak",     "Entry bar vol / prior 5-bar peak"),
        ("vol_vs_prev_bar",       "Entry bar vol / previous bar vol"),
        ("vol_declining_last3",   "# of last-3 bars with declining vol"),
        ("pre_entry_red_bars",    "# of last-3 pre-entry bars that were red"),
        ("entry_close_pos",       "Entry bar close position in range (0=low, 1=high)"),
        ("dist_from_day_high_pct","Distance below day high at entry (%)"),
        ("spike_age_bars",        "Bars elapsed since stock first crossed +5%"),
        ("vol_slope_pct",         "Volume slope last 5 bars (neg=declining)"),
        ("minutes_from_open",     "Minutes from 9:30 open to entry"),
    ]

    print("=" * 72)
    print(f"{'Feature':<42} {'Winners':>10} {'Losers':>10} {'Diff':>8}")
    print("=" * 72)
    for key, label in numeric_features:
        w_vals = [f[key] for f in winners]
        l_vals = [f[key] for f in losers]
        w_mean = mean(w_vals)
        l_mean = mean(l_vals)
        diff = w_mean - l_mean
        marker = " <--" if abs(diff) > 0.1 * max(abs(w_mean), abs(l_mean), 0.001) else ""
        print(f"{label:<42} {w_mean:>10.3f} {l_mean:>10.3f} {diff:>+8.3f}{marker}")

    print()
    bool_features = [
        ("entry_bar_red", "Entry bar was red (close < open)"),
    ]
    print(f"{'Boolean Feature':<42} {'Winners%':>10} {'Losers%':>10}")
    print("-" * 64)
    for key, label in bool_features:
        w_pct = pct_true([f[key] for f in winners])
        l_pct = pct_true([f[key] for f in losers])
        print(f"{label:<42} {w_pct:>9.1f}% {l_pct:>9.1f}%")

    print()
    print("── Exit reason breakdown ──────────────────────────────────")
    exit_counts = defaultdict(lambda: {"w": 0, "l": 0})
    for f in features:
        key = "w" if f["winner"] else "l"
        exit_counts[f["exit_reason"]][key] += 1
    print(f"{'Exit reason':<22} {'Wins':>6} {'Losses':>8} {'WinRate':>9}")
    for reason, counts in sorted(exit_counts.items()):
        total = counts["w"] + counts["l"]
        wr = counts["w"] / total * 100
        print(f"{reason:<22} {counts['w']:>6} {counts['l']:>8} {wr:>8.1f}%")

    print()
    print("── Spike age distribution (bars since +5% cross) ──────────")
    age_buckets = [0, 2, 5, 10, 20, 50]
    w_ages = [f["spike_age_bars"] for f in winners]
    l_ages = [f["spike_age_bars"] for f in losers]
    print(f"{'Age bucket':<15} {'Winner%':>9} {'Loser%':>9}")
    edges = [(0, 0), (1, 2), (3, 5), (6, 10), (11, 30), (31, 9999)]
    for lo, hi in edges:
        label = f"{lo}-{hi}" if hi < 9999 else f"{lo}+"
        w_in = sum(1 for a in w_ages if lo <= a <= hi)
        l_in = sum(1 for a in l_ages if lo <= a <= hi)
        w_pct = w_in / len(winners) * 100 if winners else 0
        l_pct = l_in / len(losers) * 100 if losers else 0
        print(f"{label:<15} {w_pct:>8.1f}% {l_pct:>8.1f}%")

    print()
    print("── Vol vs prior peak distribution ─────────────────────────")
    print(f"{'Vol/peak bucket':<15} {'Winner%':>9} {'Loser%':>9}")
    vp_edges = [(0, 0.5), (0.5, 0.75), (0.75, 1.0), (1.0, 1.5), (1.5, 99)]
    labels = ["<0.5", "0.5-0.75", "0.75-1.0", "1.0-1.5", ">1.5"]
    for (lo, hi), label in zip(vp_edges, labels):
        w_in = sum(1 for f in winners if lo <= f["entry_vol_vs_peak"] < hi)
        l_in = sum(1 for f in losers if lo <= f["entry_vol_vs_peak"] < hi)
        w_pct = w_in / len(winners) * 100 if winners else 0
        l_pct = l_in / len(losers) * 100 if losers else 0
        print(f"{label:<15} {w_pct:>8.1f}% {l_pct:>8.1f}%")

    print()
    print("── Time of day distribution ────────────────────────────────")
    print(f"{'Time bucket':<15} {'Winner%':>9} {'Loser%':>9} {'WinRate':>9}")
    time_edges = [(0, 30), (31, 60), (61, 120), (121, 210), (211, 390)]
    time_labels = ["9:30-10:00", "10:01-11:00", "11:01-12:00", "12:01-13:30", "13:31-16:00"]
    for (lo, hi), label in zip(time_edges, time_labels):
        w_in = sum(1 for f in winners if lo <= f["minutes_from_open"] <= hi)
        l_in = sum(1 for f in losers if lo <= f["minutes_from_open"] <= hi)
        total = w_in + l_in
        wr = w_in / total * 100 if total > 0 else 0
        w_pct = w_in / len(winners) * 100 if winners else 0
        l_pct = l_in / len(losers) * 100 if losers else 0
        print(f"{label:<15} {w_pct:>8.1f}% {l_pct:>8.1f}% {wr:>8.1f}%")

    print()
    print("── Gain from open at entry ─────────────────────────────────")
    print(f"{'Gain bucket':<15} {'Winner%':>9} {'Loser%':>9} {'WinRate':>9}")
    gain_edges = [(0, 5), (5, 10), (10, 20), (20, 40), (40, 999)]
    gain_labels = ["0-5%", "5-10%", "10-20%", "20-40%", ">40%"]
    for (lo, hi), label in zip(gain_edges, gain_labels):
        w_in = sum(1 for f in winners if lo <= f["gain_from_open_pct"] < hi)
        l_in = sum(1 for f in losers if lo <= f["gain_from_open_pct"] < hi)
        total = w_in + l_in
        wr = w_in / total * 100 if total > 0 else 0
        w_pct = w_in / len(winners) * 100 if winners else 0
        l_pct = l_in / len(losers) * 100 if losers else 0
        print(f"{label:<15} {w_pct:>8.1f}% {l_pct:>8.1f}% {wr:>8.1f}%")


if __name__ == "__main__":
    main()
