"""
Finalists — realism pass.

All configs run with the liquidity cap ON (max_position_dv_pct=0.20, position
<= 20% of 20-day avg daily volume) and report max drawdown of the running
equity curve. The +308% hybrid bull number from run_hybrid_validation.py was
produced WITHOUT the cap — this run shows what survives realistic sizing.

Finalists:
  B  pullback always (2xATR limit, ttl60, target 2.5x, stop 1.5x)
  B+ B with 10-day-MA filter on narrow (<35 candidate) days
  H  hybrid: tier-4 chase at market t4 / lower tiers pullback t2.5
  H+ H with the same narrow-day MA filter

Periods: 2022, 2023, OOS H1-2025, 2025-26.
"""
from __future__ import annotations
import copy, os, warnings, logging
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from typing import Dict, List, Optional, Set

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

import pandas as pd
import bot.broker_alpaca as broker
from bot.backtest.bar_fetcher import BarFetcher
from bot.backtest.candidate_screener import CandidateScreener
from bot.backtest.news_filter import NewsFilter
from bot.backtest.simulator import Simulator
from bot.config import make_gap_hold_config
from bot.data.daily_loader import get_daily
from bot.intraday.types import TradeRecord

BREADTH_THRESHOLD = 35
PB_ATR, PB_TTL, PB_TARGET = 2.0, 60, 2.5
CHASE_TARGET = 4.0
DV_CAP = 0.20

# Primary candidate vs challenger
DV_LEVELS = [
    ("$10M (primary)",    10_000_000),
    ("$20M (challenger)", 20_000_000),
]

PERIODS = [
    ("2022 bear",      date(2022, 1, 3), date(2022, 12, 30), -19.0),
    ("2023 mixed",     date(2023, 1, 3), date(2023, 12, 29), +26.0),
    ("2024 validation",date(2024, 1, 2), date(2024, 12, 31), +25.0),
    ("OOS H1-2025",    date(2025, 1, 2), date(2025, 5, 28),  +0.5),
    ("2025-26 bull",   date(2025, 6, 1), date(2026, 5, 28),  +30.0),
]

_spy: Optional[pd.DataFrame] = None
def spy_uptrend(d: date) -> bool:
    global _spy
    if _spy is None:
        _spy = get_daily("SPY", start="2021-06-01", end="2026-06-10")
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

def stocks_above_daily_ma(screener, candidates, as_of, ma_period) -> Set[str]:
    result = set()
    for sym in candidates:
        df = screener._daily_cache.get(sym)
        if df is None or df.empty:
            result.add(sym)
            continue
        past = df[df.index < pd.Timestamp(as_of)]
        if len(past) < ma_period:
            result.add(sym)
            continue
        if float(past["close"].iloc[-1]) >= float(past["close"].iloc[-ma_period:].mean()):
            result.add(sym)
    return result


def run_period(cfg, start, end, screener, fetcher, initial_equity, news_filter,
               sim_kwargs: dict, narrow_ma: int) -> dict:
    running_eq = initial_equity
    peak = initial_equity
    max_dd_pct = 0.0
    all_trades: List[TradeRecord] = []
    cache_dir = fetcher._cache_dir

    sim = Simulator(
        cfg, initial_equity,
        slippage_pct=0.001,
        overnight_holds=False,
        market_order_fill=True,
        news_filter=news_filter,
        news_mode="require",
        news_tier_bypass=4,
        stage2_min_dist_from_day_high_pct=0.05,
        max_position_dv_pct=DV_CAP,
        **sim_kwargs,
    )

    for d in trading_days(start, end):
        if not spy_uptrend(d):
            continue
        cands = screener.candidates_for_date(d)
        eligible = cands
        if narrow_ma > 0 and len(cands) < BREADTH_THRESHOLD:
            eligible = [c for c in cands
                        if c in stocks_above_daily_ma(screener, cands, d, narrow_ma)]

        cached = [s for s in eligible if (cache_dir / f"{s}_{d}.json").exists()]
        if not cached:
            continue
        bars_by_sym: Dict = {}
        baselines: Dict = {}
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
        sim._initial_equity = running_eq
        result = sim.run_day(d, bars_by_sym, baselines)
        all_trades.extend(result.trades)
        day_pnl = sum(t.pnl for t in result.trades if t.pnl is not None)
        running_eq = max(running_eq + day_pnl, 1.0)
        peak = max(peak, running_eq)
        if peak > 0:
            max_dd_pct = max(max_dd_pct, (peak - running_eq) / peak * 100)

    trades = all_trades
    if not trades:
        return {"trades": 0, "wr": 0.0, "ret_pct": 0.0, "max_dd": 0.0,
                "hard_stops": 0, "targets": 0, "chases": 0, "pullbacks": 0,
                "expectancy": 0.0, "profit_factor": 0.0, "top20_pct": 0.0}
    wins  = [t for t in trades if t.pnl and t.pnl > 0]
    losses = [t for t in trades if t.pnl and t.pnl <= 0]
    net = running_eq - initial_equity
    pbs = [t for t in trades if "momentum_pullback" in (t.signals or [])]

    total_gain = sum(t.pnl for t in wins)
    total_loss = abs(sum(t.pnl for t in losses)) if losses else 0.0
    profit_factor = total_gain / total_loss if total_loss > 0 else float("inf")
    expectancy = net / len(trades)  # avg dollar PnL per trade

    # Top-20% winner concentration
    win_pnls = sorted([t.pnl for t in wins], reverse=True)
    top20 = sum(win_pnls[:max(1, len(win_pnls) // 5)])
    top20_pct = top20 / total_gain * 100 if total_gain > 0 else 0.0

    return {
        "trades": len(trades),
        "wr": len(wins) / len(trades) * 100,
        "ret_pct": net / initial_equity * 100,
        "max_dd": max_dd_pct,
        "hard_stops": sum(1 for t in trades if t.exit_reason == "hard_stop"),
        "targets": sum(1 for t in trades if t.exit_reason == "target"),
        "chases": len(trades) - len(pbs),
        "pullbacks": len(pbs),
        "expectancy": expectancy,
        "profit_factor": profit_factor,
        "top20_pct": top20_pct,
    }


def _make_screener(base_cfg, base_screener, dv, api_key, secret_key, base_url,
                   start, end):
    """Reuse daily cache from base_screener; rebuild candidates index for new DV."""
    cfg = copy.copy(base_cfg)
    cfg.min_avg_dollar_volume = dv
    s = CandidateScreener(cfg, api_key, secret_key, base_url)
    s._universe = base_screener._universe
    s._daily_cache = base_screener._daily_cache
    s._build_candidates_index(start, end)
    return s


def main():
    api_key    = os.environ["APCA_API_KEY_ID"]
    secret_key = os.environ["APCA_API_SECRET_KEY"]
    base_url   = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

    initial_equity = broker.get_account_info()["portfolio_value"]
    print(f"Starting equity: ${initial_equity:,.0f}")
    print(f"Liquidity cap: position <= {DV_CAP:.0%} of 20-day avg daily volume")
    print(f"DV thresholds: {', '.join(l for l, _ in DV_LEVELS)}\n")

    fetcher     = BarFetcher(api_key, secret_key)
    news_filter = NewsFilter(api_key, secret_key, cache_only=False)

    pb_kwargs = dict(pullback_entry_atr=PB_ATR, pullback_entry_ttl_bars=PB_TTL,
                     pullback_target_atr=PB_TARGET)
    hy_kwargs = dict(pb_kwargs, pullback_chase_tier=4)
    t4_kwargs = dict(pullback_chase_tier=4, pullback_entry_atr=PB_ATR,
                     pullback_entry_ttl_bars=PB_TTL, pullback_target_atr=CHASE_TARGET,
                     tier4_only=True)
    configs = [
        ("B  pullback always",        pb_kwargs, 0),
        ("B+ pullback + MA10 narrow", pb_kwargs, 10),
        ("H  hybrid tier4-chase",     hy_kwargs, 0),
        ("H+ hybrid + MA10 narrow",   hy_kwargs, 10),
        ("T4 tier-4 market only",     t4_kwargs, 10),
    ]

    HDR = (f"  {'Config':<28} {'Trades':>6} {'WR':>6} {'Return':>8} "
           f"{'MaxDD':>7} {'vs S&P':>8} {'Expect$':>8} {'PF':>5} "
           f"{'Top20%':>7} {'HStop':>6} {'Tgts':>5} {'Chase':>6} {'PB':>5}")
    SEP = "  " + "-" * (len(HDR) - 2)

    for period_label, start, end, spy_ret in PERIODS:
        # Preload daily data once; share across DV levels.
        # make_gap_hold_config() now sets min_avg_dollar_volume=10M; DV sweep overrides per level.
        base = make_gap_hold_config()
        base.stage2_min_vol_vs_prev_bar = 0.80
        base.target_atr_multiple = CHASE_TARGET

        base_screener = CandidateScreener(copy.copy(base), api_key, secret_key, base_url)
        print(f"Loading {period_label}...", flush=True)
        base_screener.preload(start, end)

        print(f"\n{'='*(len(HDR)-2)}")
        print(f"  {period_label}  ({start}->{end})  S&P: {spy_ret:+.1f}%")

        for dv_label, dv in DV_LEVELS:
            screener = (base_screener if dv == DV_LEVELS[0][1]
                        else _make_screener(base, base_screener, dv,
                                            api_key, secret_key, base_url, start, end))

            print(f"\n  ── {dv_label} ──")
            print(HDR)
            print(SEP)
            for label, kw, ma in configs:
                cfg = copy.copy(base)
                cfg.min_avg_dollar_volume = dv
                m = run_period(cfg, start, end, screener, fetcher,
                               initial_equity, news_filter, kw, ma)
                vs   = m["ret_pct"] - spy_ret
                flag = " *" if m["ret_pct"] > 0 else "  "
                pf   = f"{m['profit_factor']:.2f}" if m["profit_factor"] != float("inf") else " inf"
                print(f"  {label:<28} {m['trades']:>6,} {m['wr']:>5.1f}% "
                      f"{m['ret_pct']:>+7.1f}%{flag} {m['max_dd']:>6.1f}% {vs:>+7.1f}% "
                      f"{m['expectancy']:>+8.0f} {pf:>5} "
                      f"{m['top20_pct']:>6.0f}% "
                      f"{m['hard_stops']:>6} {m['targets']:>5} "
                      f"{m['chases']:>6} {m['pullbacks']:>5}", flush=True)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
