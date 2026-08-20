"""
Multi-regime test of the institutional-PEAD long, 2018-2024, vs the S&P 500.

Event (same rules that showed +3%/20d on 2024-26 data — frozen BEFORE looking
at this period, so this is out-of-sample in both directions):
  gap up ≥10% over prior close, prior close ≥ $20, event-day dollar volume
  ≥ $10M, 20-day avg dollar volume ≥ $50M, open above the prior 100-day high,
  and the day closes green (close > open).

Trade: buy the event-day close (+0.2% slip), exit at the first close 15%
below entry (disaster stop) or the close 20 trading days later (−0.2% slip).

Reported per year: n, mean/median raw return, % positive, mean SPY-excess
return over the identical window, t-stats. Portfolio sim (15% notional,
max 6 concurrent) vs SPY buy-and-hold.

Caveat: survivorship-biased universe (current active Alpaca assets) — missing
both delisted losers and acquired winners.
"""
from __future__ import annotations
import pickle, sys, warnings
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

PKL = Path(__file__).resolve().parent.parent / "screener_cache" / "2017-06-01_2025-01-31.pkl"
SLIP = 0.002
HOLD = 20
DISASTER = 0.85
GAP_MIN = 0.10
MIN_PREV_CLOSE = 20.0
MIN_DAY_DV = 10_000_000
MIN_AVG_DV = 50_000_000
NEW_HIGH_DAYS = 100
START_YEAR, END_YEAR = 2018, 2024


def main():
    with PKL.open("rb") as f:
        cache = pickle.load(f)
    cache = {s: df.sort_index() for s, df in cache.items() if df is not None and len(df) > 30}
    spy = cache.pop("SPY")
    spy_close = spy["close"]
    print(f"loaded {len(cache)} symbols; SPY {spy.index[0].date()} → {spy.index[-1].date()}")

    events = []  # (date, sym, ret, spy_ret)
    for sym, df in cache.items():
        pc = df["close"].shift(1)
        gap = df["open"] / pc - 1
        dv = df["close"] * df["volume"]
        avg_dv20 = dv.rolling(20).mean().shift(1)
        day_dv = df["open"] * df["volume"]
        prior_high = df["high"].rolling(NEW_HIGH_DAYS).max().shift(1)
        mask = (
            (gap >= GAP_MIN) & (pc >= MIN_PREV_CLOSE)
            & (day_dv >= MIN_DAY_DV) & (avg_dv20 >= MIN_AVG_DV)
            & (df["open"] > prior_high) & (df["close"] > df["open"])
        )
        closes = df["close"].values
        idx = df.index
        for i in np.where(mask.fillna(False).values)[0]:
            d = idx[i]
            if not (START_YEAR <= d.year <= END_YEAR):
                continue
            end = min(i + HOLD, len(closes) - 1)
            if end <= i:
                continue
            entry = closes[i] * (1 + SLIP)
            ret = None
            exit_i = end
            for k in range(i + 1, end + 1):
                if closes[k] < entry * DISASTER:
                    ret = closes[k] * (1 - SLIP) / entry - 1
                    exit_i = k
                    break
            if ret is None:
                ret = closes[end] * (1 - SLIP) / entry - 1
            # SPY over the identical calendar window
            s0 = spy_close.asof(d)
            s1 = spy_close.asof(idx[exit_i])
            spy_ret = float(s1 / s0 - 1) if s0 and s1 else 0.0
            events.append((d, sym, ret, spy_ret, idx[exit_i]))

    events.sort(key=lambda x: x[0])
    rets = np.array([e[2] for e in events])
    exc = np.array([e[2] - e[3] for e in events])
    print(f"\nevents 2018-2024: {len(events)}")

    def t(x):
        return x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 1 else float("nan")

    print(f"{'year':<6} {'n':>5} {'mean':>8} {'median':>8} {'pos%':>5} "
          f"{'SPYexc':>8} {'t(exc)':>7}")
    byy = defaultdict(list)
    for d, sym, r, sr, _ in events:
        byy[d.year].append((r, r - sr))
    for y in sorted(byy):
        v = np.array(byy[y])
        r, e = v[:, 0], v[:, 1]
        print(f"{y:<6} {len(r):>5} {r.mean()*100:>+7.2f}% {np.median(r)*100:>+7.2f}% "
              f"{(r>0).mean()*100:>4.0f}% {e.mean()*100:>+7.2f}% {t(e):>7.2f}")
    print(f"{'ALL':<6} {len(rets):>5} {rets.mean()*100:>+7.2f}% {np.median(rets)*100:>+7.2f}% "
          f"{(rets>0).mean()*100:>4.0f}% {exc.mean()*100:>+7.2f}% {t(exc):>7.2f}")

    # --- portfolio sim: 15% notional per position, max 6 concurrent ---
    eq = 100_000.0
    open_until = []
    curve = [(events[0][0], eq)] if events else []
    taken = 0
    for d, sym, r, sr, exit_d in events:
        open_until = [u for u in open_until if u > d]
        if len(open_until) >= 6:
            continue
        eq += eq * 0.15 * r
        open_until.append(exit_d)
        curve.append((exit_d, eq))
        taken += 1
    peak, dd = 100_000.0, 0.0
    for _, e in sorted(curve):
        peak = max(peak, e)
        dd = max(dd, (peak - e) / peak * 100)
    years = END_YEAR - START_YEAR + 1
    cagr = (eq / 100_000.0) ** (1 / years) - 1

    s0 = float(spy_close.asof(pd.Timestamp(f"{START_YEAR}-01-02")))
    s1 = float(spy_close.asof(pd.Timestamp(f"{END_YEAR}-12-31")))
    spy_total = s1 / s0
    spy_cagr = spy_total ** (1 / years) - 1
    speak, sdd = 0.0, 0.0
    w = spy_close[(spy_close.index >= f"{START_YEAR}-01-01") & (spy_close.index <= f"{END_YEAR}-12-31")]
    for v in w.values:
        speak = max(speak, v)
        sdd = max(sdd, (speak - v) / speak * 100)

    print(f"\nportfolio (15% notional, max 6): trades={taken} "
          f"final=${eq:,.0f} CAGR={cagr*100:+.1f}% maxDD={dd:.1f}%")
    print(f"SPY buy & hold same period:      CAGR={spy_cagr*100:+.1f}% maxDD={sdd:.1f}%")
    print("\nCAVEAT: survivorship-biased universe (missing delisted losers AND acquired winners).")


if __name__ == "__main__":
    main()
