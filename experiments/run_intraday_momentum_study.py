"""
Does "buy the morning momentum and ride it" have an edge at INTRADAY timescale?

Everything tested so far in this repo (daily open->close, close->close) says
buying big gappers is negative. But the original product vision was explicitly
intraday: enter early in the day, ride the move. That has never been tested,
because the daily-bar studies cannot see it. Median open->high on these names is
+11-18%, so the upside exists -- the question is whether any rule stated IN
ADVANCE can capture it.

METHOD -- the thing that makes this honest:
Every screen is evaluated on information available AT THE DECISION MINUTE only
(gap, return since open, volume so far, VWAP position). We then measure forward
returns. Crucially the candidate set is ALL cached symbol-days (28k), not just
the days that ended up exploding -- so days that looked identical at 09:35 and
then fizzled are included and drag on the average, exactly as they would live.
Selecting on the intraday high (as the cohort study did) inflates entry stats
enormously and is not tradeable.

Costs: per-side slippage + commission applied to entry and exit. Defaults are
calibrated to what this repo measured on real fills (~0.55% round trip on thin
low-float names); a 2x-worse sensitivity case is also reported.

Usage:
    .venv/bin/python3.11 experiments/run_intraday_momentum_study.py
    .venv/bin/python3.11 experiments/run_intraday_momentum_study.py --slip-bps 50
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MINUTE_CACHE = ROOT / "backtest_results" / "cache"
DAILY_CACHES = [
    ROOT / "screener_cache" / "2017-06-01_2025-01-31.pkl",
    ROOT / "screener_cache" / "2024-11-28_2026-06-01.pkl",
    ROOT / "screener_cache" / "2021-11-29_2023-01-01.pkl",
]

# Decision minutes (ET) at which we screen and enter.
ENTRY_MINUTES = ["09:31", "09:35", "09:45", "10:00"]
# Fixed holding horizons in minutes, plus "eod".
EXIT_HORIZONS = [15, 30, 60, 120, "eod"]

MIN_SAMEDAY_DV = 5_000_000.0   # same-day $ volume by decision time, scaled to full day
MIN_RVOL_AT_ENTRY = 3.0


def log(msg):
    print(msg, flush=True)


def needed_keys():
    """(symbol, date) pairs we actually have minute bars for -- only these need
    a daily reference, which turns a ~12M-row build into ~28k lookups."""
    out = defaultdict(set)
    for p in MINUTE_CACHE.glob("*.json"):
        stem = p.stem
        if "_" not in stem:
            continue
        sym, date = stem.rsplit("_", 1)
        out[sym].add(date)
    return out


def load_daily_reference():
    """(symbol, date) -> (prev_close, vol20), restricted to symbols/dates that
    have minute bars. Split-adjusted sources preferred; the 2024-2026 pkl is NOT
    split-adjusted so rvol screening also guards it."""
    want = needed_keys()
    log(f"  need reference for {len(want)} symbols / "
        f"{sum(len(v) for v in want.values())} sym-days")
    prev_close, vol20 = {}, {}
    for path in DAILY_CACHES:
        if not path.exists():
            continue
        with open(path, "rb") as f:
            cache = pickle.load(f)
        for sym, dates in want.items():
            df = cache.get(sym)
            if df is None or len(df) < 25:
                continue
            df = df.sort_index()
            pc = df["close"].shift(1)
            v20 = df["volume"].rolling(20).mean().shift(1)
            idx = df.index.strftime("%Y-%m-%d")
            ref = pd.DataFrame({"d": idx, "pc": pc.values, "v20": v20.values})
            ref = ref[ref["d"].isin(dates)].dropna()
            for d, p, v in zip(ref["d"], ref["pc"], ref["v20"]):
                if p <= 0 or v <= 0:
                    continue
                prev_close.setdefault((sym, d), p)
                vol20.setdefault((sym, d), v)
        log(f"  loaded {path.name}: cumulative {len(prev_close)} sym-days")
    return prev_close, vol20


def load_minute(path: Path) -> pd.DataFrame | None:
    try:
        bars = json.loads(path.read_text())
    except Exception:
        return None
    if not bars or len(bars) < 30:
        return None
    df = pd.DataFrame(bars)
    if not {"t", "o", "h", "l", "c", "v"} <= set(df.columns):
        return None
    df["t"] = pd.to_datetime(df["t"], utc=True).dt.tz_convert("America/New_York")
    df = df.set_index("t").sort_index()
    # regular session only
    df = df.between_time("09:30", "15:59")
    if len(df) < 30:
        return None
    return df


def build_rows(prev_close, vol20, limit=None):
    files = sorted(MINUTE_CACHE.glob("*.json"))
    if limit:
        files = files[:limit]
    log(f"scanning {len(files)} cached minute files...")

    rows = []
    skipped_noref = skipped_bad = 0
    for n, path in enumerate(files, 1):
        if n % 5000 == 0:
            log(f"  {n}/{len(files)} ({len(rows)} candidate rows)")
        stem = path.stem
        if "_" not in stem:
            continue
        sym, date = stem.rsplit("_", 1)
        key = (sym, date)
        if key not in prev_close:
            skipped_noref += 1
            continue
        df = load_minute(path)
        if df is None:
            skipped_bad += 1
            continue

        pc = prev_close[key]
        v20 = vol20[key]
        day_open = float(df["o"].iloc[0])
        if day_open <= 0 or pc <= 0:
            continue

        gap = day_open / pc - 1.0
        cum_vol = df["v"].cumsum()
        cum_pv = (df["c"] * df["v"]).cumsum()
        session_minutes = 390

        for em in ENTRY_MINUTES:
            seg = df.between_time("09:30", em)
            if len(seg) < 1:
                continue
            i = len(seg) - 1
            t_entry = seg.index[-1]
            price = float(seg["c"].iloc[-1])
            if price <= 0:
                continue

            minutes_elapsed = max(1, i + 1)
            vol_so_far = float(cum_vol.iloc[i])
            # extrapolate to a full-day pace, then compare to prior 20d avg
            pace_vol = vol_so_far * (session_minutes / minutes_elapsed)
            rvol = pace_vol / v20 if v20 > 0 else 0.0
            dv_pace = pace_vol * price
            vwap = float(cum_pv.iloc[i] / cum_vol.iloc[i]) if cum_vol.iloc[i] > 0 else price

            fut = df.iloc[i + 1:]
            if len(fut) < 5:
                continue

            rec = {
                "symbol": sym, "date": date, "entry_min": em,
                "gap": gap, "entry_price": price,
                "ret_since_open": price / day_open - 1.0,
                "rvol_pace": rvol, "dv_pace": dv_pace,
                "above_vwap": price > vwap,
                "vwap_dist": price / vwap - 1.0,
                "day_high_so_far": float(seg["h"].max()),
            }
            rec["at_hod"] = price >= rec["day_high_so_far"] * 0.995

            for hz in EXIT_HORIZONS:
                if hz == "eod":
                    exit_px = float(fut["c"].iloc[-1])
                    hi = float(fut["h"].max())
                    lo = float(fut["l"].min())
                else:
                    w = fut.iloc[:hz]
                    if len(w) < min(5, hz):
                        rec[f"ret_{hz}"] = np.nan
                        continue
                    exit_px = float(w["c"].iloc[-1])
                    hi = float(w["h"].max())
                    lo = float(w["l"].min())
                rec[f"ret_{hz}"] = exit_px / price - 1.0
                rec[f"mfe_{hz}"] = hi / price - 1.0
                rec[f"mae_{hz}"] = lo / price - 1.0
            rows.append(rec)

    log(f"  done. rows={len(rows)}  skipped_noref={skipped_noref} skipped_bad={skipped_bad}")
    return pd.DataFrame(rows)


def evaluate(df: pd.DataFrame, slip_bps: float):
    """Apply costs and report expectancy per screen x entry x exit."""
    cost = 2 * (slip_bps / 10_000.0)  # entry + exit
    df = df.copy()
    df["year"] = pd.to_datetime(df["date"]).dt.year

    screens = {
        "gap>=20% & rvol>=3 & above VWAP": (
            (df.gap >= 0.20) & (df.rvol_pace >= MIN_RVOL_AT_ENTRY) & df.above_vwap),
        "gap>=30% & rvol>=3 & above VWAP": (
            (df.gap >= 0.30) & (df.rvol_pace >= MIN_RVOL_AT_ENTRY) & df.above_vwap),
        "gap>=20% & at HOD & above VWAP": (
            (df.gap >= 0.20) & df.at_hod & df.above_vwap),
        "gap>=50% & rvol>=3 & above VWAP": (
            (df.gap >= 0.50) & (df.rvol_pace >= MIN_RVOL_AT_ENTRY) & df.above_vwap),
        "up>=10% since open & rvol>=3": (
            (df.ret_since_open >= 0.10) & (df.rvol_pace >= MIN_RVOL_AT_ENTRY)),
        "ANY gap>=20% (no other filter)": (df.gap >= 0.20),
    }

    liquid = df.dv_pace >= MIN_SAMEDAY_DV
    out = []
    for sname, mask in screens.items():
        sub = df[mask & liquid]
        for em in ENTRY_MINUTES:
            s2 = sub[sub.entry_min == em]
            if len(s2) < 40:
                continue
            for hz in EXIT_HORIZONS:
                col = f"ret_{hz}"
                if col not in s2:
                    continue
                r = s2[col].dropna()
                if len(r) < 40:
                    continue
                net = r - cost
                out.append({
                    "screen": sname, "entry": em, "exit": str(hz), "n": len(net),
                    "mean_net_%": round(net.mean() * 100, 3),
                    "median_net_%": round(net.median() * 100, 3),
                    "win_%": round((net > 0).mean() * 100, 1),
                    "gross_mean_%": round(r.mean() * 100, 3),
                    "mfe_med_%": round(s2[f"mfe_{hz}"].median() * 100, 2) if f"mfe_{hz}" in s2 else np.nan,
                    "mae_med_%": round(s2[f"mae_{hz}"].median() * 100, 2) if f"mae_{hz}" in s2 else np.nan,
                })
    return pd.DataFrame(out), cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slip-bps", type=float, default=25.0,
                    help="per-side slippage+commission in bps (default 25 = 0.5%% round trip)")
    ap.add_argument("--limit", type=int, default=None, help="limit files (smoke test)")
    ap.add_argument("--rebuild", action="store_true", help="ignore cached rows parquet")
    args = ap.parse_args()

    cache_path = ROOT / "backtest_results" / "intraday_study_rows.parquet"
    if cache_path.exists() and not args.rebuild and not args.limit:
        log(f"loading cached rows from {cache_path.name}")
        rows = pd.read_parquet(cache_path)
    else:
        log("loading daily reference (prev_close, vol20)...")
        pc, v20 = load_daily_reference()
        rows = build_rows(pc, v20, limit=args.limit)
        if not args.limit and len(rows):
            rows.to_parquet(cache_path)
            log(f"saved {cache_path.name}")

    if rows.empty:
        log("no rows built")
        return

    log(f"\ncandidate rows: {len(rows)}  unique sym-days: {rows.groupby(['symbol','date']).ngroups}")

    for slip in (args.slip_bps, args.slip_bps * 2):
        res, cost = evaluate(rows, slip)
        log(f"\n{'='*100}\nCOSTS: {slip:.0f} bps/side  ({cost*100:.2f}% round trip)\n{'='*100}")
        if res.empty:
            log("no screen produced >=40 samples")
            continue
        for sname, g in res.groupby("screen"):
            log(f"\n--- {sname} ---")
            piv = g.pivot_table(index="entry", columns="exit",
                                values="mean_net_%", aggfunc="first")
            order = [c for c in ["15", "30", "60", "120", "eod"] if c in piv.columns]
            log("mean NET return % by entry (rows) x exit horizon (cols):")
            log(piv[order].round(3).to_string())
            best = g.sort_values("mean_net_%", ascending=False).head(3)
            log("best cells:")
            log(best[["entry", "exit", "n", "mean_net_%", "median_net_%",
                      "win_%", "mfe_med_%", "mae_med_%"]].to_string(index=False))

    # per-year stability of the single best configuration
    res, cost = evaluate(rows, args.slip_bps)
    if not res.empty:
        top = res.sort_values("mean_net_%", ascending=False).iloc[0]
        log(f"\n{'='*100}\nPER-YEAR STABILITY of best cell: {top['screen']} | entry {top['entry']} | exit {top['exit']}\n{'='*100}")
        df = rows.copy()
        df["year"] = pd.to_datetime(df["date"]).dt.year
        liquid = df.dv_pace >= MIN_SAMEDAY_DV
        screens = {
            "gap>=20% & rvol>=3 & above VWAP": ((df.gap >= 0.20) & (df.rvol_pace >= MIN_RVOL_AT_ENTRY) & df.above_vwap),
            "gap>=30% & rvol>=3 & above VWAP": ((df.gap >= 0.30) & (df.rvol_pace >= MIN_RVOL_AT_ENTRY) & df.above_vwap),
            "gap>=20% & at HOD & above VWAP": ((df.gap >= 0.20) & df.at_hod & df.above_vwap),
            "gap>=50% & rvol>=3 & above VWAP": ((df.gap >= 0.50) & (df.rvol_pace >= MIN_RVOL_AT_ENTRY) & df.above_vwap),
            "up>=10% since open & rvol>=3": ((df.ret_since_open >= 0.10) & (df.rvol_pace >= MIN_RVOL_AT_ENTRY)),
            "ANY gap>=20% (no other filter)": (df.gap >= 0.20),
        }
        sub = df[screens[top["screen"]] & liquid & (df.entry_min == top["entry"])]
        col = f"ret_{top['exit']}"
        sub = sub.dropna(subset=[col])
        sub["net"] = sub[col] - cost
        log(sub.groupby("year").agg(
            n=("net", "size"),
            mean_net_pct=("net", lambda s: round(s.mean() * 100, 3)),
            median_net_pct=("net", lambda s: round(s.median() * 100, 3)),
            win_pct=("net", lambda s: round((s > 0).mean() * 100, 1)),
        ).to_string())


if __name__ == "__main__":
    main()
