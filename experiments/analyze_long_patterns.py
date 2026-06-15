"""
Analyze pre-entry bar conditions for long trades to find what separates
winners (target/volume_collapse) from hard-stop losers.
"""
from __future__ import annotations
import csv
import json
from datetime import datetime, date
from pathlib import Path
from zoneinfo import ZoneInfo
from collections import defaultdict

_ET = ZoneInfo("America/New_York")
CACHE_DIR = Path("backtest_results/cache")
TRADE_FILES = [
    "backtest_results/trades_long_2022-01-03_2022-12-30_scale2.0.csv",
]
LOOKBACK = 10


def load_trades():
    seen = set()
    trades = []
    for path in TRADE_FILES:
        p = Path(path)
        if not p.exists():
            continue
        with p.open() as f:
            for row in csv.DictReader(f):
                if row["direction"] != "long":
                    continue
                key = (row["ticker"], row["entry_time"])
                if key in seen:
                    continue
                seen.add(key)
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
            "ts": ts, "open": float(b["o"]), "high": float(b["h"]),
            "low": float(b["l"]), "close": float(b["c"]), "volume": int(b["v"]),
        })
    bars.sort(key=lambda b: b["ts"])
    return bars


def compute_features(trade: dict, bars: list) -> dict | None:
    entry_ts = datetime.fromisoformat(trade["entry_time"])
    entry_price = float(trade["entry_price"])
    pnl = float(trade["pnl"])

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

    session_open = bars[0]["open"]

    # How extended is the stock at entry vs session open
    gain_from_open = (entry_price - session_open) / session_open if session_open > 0 else 0

    # Volume features
    volumes = [b["volume"] for b in prior_bars]
    peak_prior_volume = max(volumes) if volumes else 0
    entry_vol_ratio = entry_bar["volume"] / peak_prior_volume if peak_prior_volume > 0 else 1.0
    prev_bar = prior_bars[-1] if prior_bars else None
    vol_vs_prev = entry_bar["volume"] / prev_bar["volume"] if prev_bar and prev_bar["volume"] > 0 else 1.0

    # How many of last 3 bars had declining volume
    last3 = prior_bars[-3:] if len(prior_bars) >= 3 else prior_bars
    vol_declining_count = sum(1 for i in range(1, len(last3)) if last3[i]["volume"] < last3[i-1]["volume"])

    # Pre-entry green bars (close > open) — for longs, green = momentum in our direction
    pre_entry_green = sum(1 for b in last3 if b["close"] > b["open"])

    # Entry bar features
    entry_bar_green = entry_bar["close"] > entry_bar["open"]
    bar_range = entry_bar["high"] - entry_bar["low"]
    entry_close_pos = (entry_bar["close"] - entry_bar["low"]) / bar_range if bar_range > 0 else 0.5

    # How close to the day's high at entry
    day_high_to_entry = max(b["high"] for b in bars[:entry_idx + 1])
    dist_from_high = (day_high_to_entry - entry_price) / day_high_to_entry if day_high_to_entry > 0 else 0

    # Bars elapsed since stock first crossed +15% from session open (long threshold)
    spike_start_idx = None
    threshold = session_open * 1.15
    for i, b in enumerate(bars):
        if b["high"] >= threshold:
            spike_start_idx = i
            break
    spike_age = (entry_idx - spike_start_idx) if spike_start_idx is not None else 0

    # Volume trend slope over last 5 bars
    last5_vols = [b["volume"] for b in prior_bars[-5:]]
    if len(last5_vols) >= 3:
        n = len(last5_vols)
        xs = list(range(n))
        mean_x = sum(xs) / n
        mean_y = sum(last5_vols) / n
        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, last5_vols))
        den = sum((x - mean_x) ** 2 for x in xs)
        vol_slope_pct = (num / den) / mean_y if den > 0 and mean_y > 0 else 0
    else:
        vol_slope_pct = 0

    et_ts = entry_ts.astimezone(_ET)
    minutes_from_open = (et_ts.hour - 9) * 60 + et_ts.minute - 30

    # ATR-derived risk info
    stop_price = float(trade["stop_price"])
    target_price = float(trade["target_price"])
    stop_dist_pct = abs(entry_price - stop_price) / entry_price
    target_dist_pct = abs(target_price - entry_price) / entry_price
    rr_ratio = target_dist_pct / stop_dist_pct if stop_dist_pct > 0 else 0

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
        "pre_entry_green_bars": pre_entry_green,
        "entry_bar_green": entry_bar_green,
        "entry_close_pos": round(entry_close_pos, 3),
        "dist_from_day_high_pct": round(dist_from_high * 100, 2),
        "spike_age_bars": spike_age,
        "vol_slope_pct": round(vol_slope_pct, 4),
        "minutes_from_open": minutes_from_open,
        "stop_dist_pct": round(stop_dist_pct * 100, 2),
        "rr_ratio": round(rr_ratio, 2),
    }


def mean(vals):
    return sum(vals) / len(vals) if vals else 0

def pct_true(vals):
    return sum(1 for v in vals if v) / len(vals) * 100 if vals else 0


def main():
    trades = load_trades()
    print(f"Loaded {len(trades)} long trades\n")

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

    print(f"Analyzed {len(features)} trades ({skipped} skipped)\n")

    winners = [f for f in features if f["winner"]]
    losers  = [f for f in features if not f["winner"]]
    hard_stops = [f for f in features if f["exit_reason"] == "hard_stop"]
    print(f"Winners: {len(winners)}  Losers: {len(losers)}  Win rate: {len(winners)/len(features)*100:.1f}%")
    print(f"Hard stops: {len(hard_stops)} ({len(hard_stops)/len(features)*100:.1f}% of all trades)\n")

    numeric_features = [
        ("gain_from_open_pct",    "Gain from session open (%)"),
        ("entry_vol_vs_peak",     "Entry bar vol / prior 5-bar peak"),
        ("vol_vs_prev_bar",       "Entry bar vol / previous bar vol"),
        ("vol_declining_last3",   "# of last-3 bars with declining vol"),
        ("pre_entry_green_bars",  "# of last-3 pre-entry bars that were green"),
        ("entry_close_pos",       "Entry bar close position in range (0=low, 1=high)"),
        ("dist_from_day_high_pct","Distance below day high at entry (%)"),
        ("spike_age_bars",        "Bars elapsed since stock first crossed +15%"),
        ("vol_slope_pct",         "Volume slope last 5 bars (pos=growing)"),
        ("minutes_from_open",     "Minutes from 9:30 open to entry"),
        ("stop_dist_pct",         "Stop distance from entry (% ATR)"),
        ("rr_ratio",              "Reward:Risk ratio at entry"),
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
        threshold = 0.1 * max(abs(w_mean), abs(l_mean), 0.001)
        marker = " <--" if abs(diff) > threshold else ""
        print(f"{label:<42} {w_mean:>10.3f} {l_mean:>10.3f} {diff:>+8.3f}{marker}")

    print()
    print(f"{'Boolean Feature':<42} {'Winners%':>10} {'Losers%':>10}")
    print("-" * 64)
    print(f"{'Entry bar was green (close > open)':<42} {pct_true([f['entry_bar_green'] for f in winners]):>9.1f}% {pct_true([f['entry_bar_green'] for f in losers]):>9.1f}%")

    print()
    print("── Exit reason breakdown ──────────────────────────────────")
    exit_counts = defaultdict(lambda: {"w": 0, "l": 0})
    for f in features:
        exit_counts[f["exit_reason"]]["w" if f["winner"] else "l"] += 1
    print(f"{'Exit reason':<22} {'Wins':>6} {'Losses':>8} {'WinRate':>9} {'AvgPnL':>10}")
    for reason, counts in sorted(exit_counts.items()):
        total = counts["w"] + counts["l"]
        wr = counts["w"] / total * 100
        trades_for_reason = [f for f in features if f["exit_reason"] == reason]
        avg_pnl = mean([float(t["pnl"]) for t in load_trades() if t["exit_reason"] == reason])
        print(f"{reason:<22} {counts['w']:>6} {counts['l']:>8} {wr:>8.1f}% {avg_pnl:>10.0f}")

    print()
    print("── Gain from open at entry ─────────────────────────────────")
    print(f"{'Gain bucket':<15} {'Winner%':>9} {'Loser%':>9} {'WinRate':>9}")
    for lo, hi, label in [(0,20,"0-20%"),(20,50,"20-50%"),(50,100,"50-100%"),(100,200,"100-200%"),(200,999,">200%")]:
        w_in = sum(1 for f in winners if lo <= f["gain_from_open_pct"] < hi)
        l_in = sum(1 for f in losers if lo <= f["gain_from_open_pct"] < hi)
        total = w_in + l_in
        wr = w_in / total * 100 if total > 0 else 0
        print(f"{label:<15} {w_in/len(winners)*100:>8.1f}% {l_in/len(losers)*100:>8.1f}% {wr:>8.1f}%")

    print()
    print("── Time of day ─────────────────────────────────────────────")
    print(f"{'Time bucket':<15} {'Winner%':>9} {'Loser%':>9} {'WinRate':>9}")
    for (lo, hi), label in [((0,30),"9:30-10:00"),((31,60),"10:01-11:00"),((61,120),"11:01-12:00"),((121,210),"12:01-13:30"),((211,390),"13:31-16:00")]:
        w_in = sum(1 for f in winners if lo <= f["minutes_from_open"] <= hi)
        l_in = sum(1 for f in losers if lo <= f["minutes_from_open"] <= hi)
        total = w_in + l_in
        wr = w_in / total * 100 if total > 0 else 0
        print(f"{label:<15} {w_in/len(winners)*100:>8.1f}% {l_in/len(losers)*100:>8.1f}% {wr:>8.1f}%")

    print()
    print("── Hard stops deep-dive ────────────────────────────────────")
    hs_features = [f for f in features if f["exit_reason"] == "hard_stop"]
    non_hs = [f for f in features if f["exit_reason"] != "hard_stop"]
    print(f"Hard stops: {len(hs_features)}  Non-hard-stops: {len(non_hs)}\n")
    print(f"{'Feature':<42} {'HardStop':>10} {'Other':>10} {'Diff':>8}")
    print("-" * 72)
    for key, label in numeric_features[:8]:
        hs_mean  = mean([f[key] for f in hs_features])
        non_mean = mean([f[key] for f in non_hs])
        diff = hs_mean - non_mean
        threshold = 0.1 * max(abs(hs_mean), abs(non_mean), 0.001)
        marker = " <--" if abs(diff) > threshold else ""
        print(f"{label:<42} {hs_mean:>10.3f} {non_mean:>10.3f} {diff:>+8.3f}{marker}")
    print(f"{'Entry bar green':<42} {pct_true([f['entry_bar_green'] for f in hs_features]):>9.1f}% {pct_true([f['entry_bar_green'] for f in non_hs]):>9.1f}%")


if __name__ == "__main__":
    main()
