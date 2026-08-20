"""
Episodic-pivot swing backtest (multi-day holds) over the screener daily cache.

Event: stock gaps up ≥ gap_min on real volume (day dollar-vol ≥ $10M), price ≥ $3,
with 20-day average dollar-volume ≥ $1M (excludes the untradeable microcap pumps
the fill-realism test disqualified). Optional news-catalyst requirement.

Entry modes (bot/backtest/swing_simulator.py):
  orb          break of the 30-min opening-range high (needs minute bars; ~71%
               of events have them cached)
  close_green  buy the event-day close if it closed above its open

Exits: stop at LOD, scale half at +2R, breakeven, trail on close < 10-day SMA,
20-day time stop. Slippage 0.3% per side on market fills.

Caveats printed with results: survivorship bias (current active universe —
delisted pumps excluded, which FLATTERS long results); news 'require' variants
also exclude events with no cached news lookup.
"""
from __future__ import annotations
import json, sys, warnings, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from collections import Counter, defaultdict
from datetime import date

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

import numpy as np
import pandas as pd
import pickle

from bot.backtest.news_filter import NewsFilter
from bot.backtest.swing_simulator import (
    SwingConfig, SwingEvent, run_portfolio, simulate_event,
)

ROOT = Path(__file__).resolve().parent.parent
BAR_CACHE = ROOT / "backtest_results" / "cache"
PKL = ROOT / "screener_cache" / "2024-11-28_2026-06-01.pkl"
INITIAL_EQUITY = 75_000.0

MIN_PRICE = 3.0
MIN_DAY_DV = 10_000_000
MIN_AVG_DV = 1_000_000

VARIANTS = [
    # (label, entry_mode, gap_min, require_news)
    ("orb  gap≥10%",          "orb",         0.10, False),
    ("orb  gap≥10% +news",    "orb",         0.10, True),
    ("orb  gap≥20%",          "orb",         0.20, False),
    ("orb  gap≥20% +news",    "orb",         0.20, True),
    ("close gap≥10%",         "close_green", 0.10, False),
    ("close gap≥10% +news",   "close_green", 0.10, True),
    ("close gap≥20%",         "close_green", 0.20, False),
    ("close gap≥20% +news",   "close_green", 0.20, True),
]


def build_events(cache, gap_min):
    events = []
    for sym, df in cache.items():
        if df is None or len(df) < 30:
            continue
        df = df.sort_index()
        pc = df["close"].shift(1)
        gap = df["open"] / pc - 1
        dv = df["close"] * df["volume"]
        avg_dv20 = dv.rolling(20).mean().shift(1)
        day_dv = df["open"] * df["volume"]
        mask = (
            (gap >= gap_min) & (pc >= MIN_PRICE)
            & (day_dv >= MIN_DAY_DV) & (avg_dv20 >= MIN_AVG_DV)
        )
        for i in np.where(mask.fillna(False))[0]:
            events.append(SwingEvent(symbol=sym, day=df.index[i].date(),
                                     gap_pct=float(gap.iloc[i])))
    return events


def load_minutes(sym, d):
    p = BAR_CACHE / f"{sym}_{d}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def main():
    print("Loading daily cache...", flush=True)
    with open(PKL, "rb") as f:
        cache = pickle.load(f)
    cache = {s: df.sort_index() for s, df in cache.items() if df is not None and len(df)}
    news = NewsFilter("", "", cache_only=True)
    cfg = SwingConfig()

    HDR = (f"  {'Variant':<22} {'Events':>6} {'Entered':>7} {'WR':>6} {'avgR':>7} "
           f"{'totR':>7} {'PF':>5} {'Top10R%':>8} {'Hold':>5} {'Port P&L':>10} {'MaxDD':>7}")
    print(f"\nEpisodic-pivot swing backtest 2024-12 → 2026-06  "
          f"(risk 1%/trade, max 4 positions, equity ${INITIAL_EQUITY:,.0f})")
    print(HDR)
    print("  " + "-" * (len(HDR) - 2))

    best = None
    for label, mode, gap_min, req_news in VARIANTS:
        events = build_events(cache, gap_min)
        results = []
        for ev in events:
            if req_news and not news.has_catalyst(ev.symbol, ev.day):
                continue
            minutes = load_minutes(ev.symbol, ev.day) if mode == "orb" else None
            res = simulate_event(ev, cache[ev.symbol], cfg,
                                 minute_bars=minutes, entry_mode=mode)
            results.append(res)

        entered = [r for r in results if r.entered and r.exits]
        # drop positions still open at data end — unknown outcome, count separately
        clean = [r for r in entered if r.exits[-1][3] != "data_end"]
        rs = [r.r_multiple for r in clean]
        if not rs:
            print(f"  {label:<22} {len(events):>6} {0:>7}")
            continue
        wins = [x for x in rs if x > 0]
        losses = [x for x in rs if x <= 0]
        pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float("inf")
        top10 = sum(sorted(rs, reverse=True)[:10])
        tot = sum(rs)
        top10_share = (top10 / tot * 100) if tot > 0 else float("nan")
        hold = np.mean([r.hold_days for r in clean])

        trades, curve = run_portfolio(clean, cfg, INITIAL_EQUITY)
        net = curve[-1][1] - INITIAL_EQUITY if curve else 0.0
        peak, dd = INITIAL_EQUITY, 0.0
        for _, eq in curve:
            peak = max(peak, eq)
            dd = max(dd, (peak - eq) / peak * 100)

        print(f"  {label:<22} {len(events):>6} {len(clean):>7} "
              f"{len(wins)/len(rs)*100:>5.1f}% {np.mean(rs):>+7.3f} {tot:>+7.1f} "
              f"{pf:>5.2f} {top10_share:>7.0f}% {hold:>5.1f} {net:>+10,.0f} {dd:>6.1f}%",
              flush=True)

        key = np.mean(rs)
        if best is None or key > best[0]:
            best = (key, label, clean, trades, curve)

    # Forensics on the best variant
    _, label, clean, trades, curve = best
    print(f"\n=== forensics: {label} ===")
    reasons = Counter(x[3] for r in clean for x in r.exits)
    print("exit reasons:", dict(reasons))
    byq = defaultdict(list)
    for r in clean:
        q = f"{r.entry_date.year}-Q{(r.entry_date.month - 1) // 3 + 1}"
        byq[q].append(r.r_multiple)
    print("by quarter (total R | n | WR):")
    for q in sorted(byq):
        v = byq[q]
        print(f"  {q}: {sum(v):+7.1f} | {len(v):4d} | {sum(1 for x in v if x > 0)/len(v)*100:.0f}%")
    rs_sorted = sorted((r.r_multiple, r.event.symbol, str(r.entry_date)) for r in clean)
    print("worst 5:", [(f"{x[0]:+.1f}R", x[1], x[2]) for x in rs_sorted[:5]])
    print("best 5: ", [(f"{x[0]:+.1f}R", x[1], x[2]) for x in rs_sorted[-5:]])

    out = ROOT / "backtest_results" / "ep_swing_trades.csv"
    pd.DataFrame([t.__dict__ for t in trades]).to_csv(out, index=False)
    print(f"\nportfolio trades for best variant → {out}")
    print("\nCAVEATS: survivorship-biased universe (flatters longs); "
          "0.3%/side slippage; 'news' variants exclude events missing cached lookups.")


if __name__ == "__main__":
    main()
