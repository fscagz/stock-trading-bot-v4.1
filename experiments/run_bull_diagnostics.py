"""
Diagnostic run for 2025-26 bull period.

Instruments every entry rejection and exit reason to explain
why the live-equivalent config underperforms in a bull year.

Outputs:
  - Monthly trade/PnL breakdown
  - Exit reason breakdown (hard_stop / trailing_stop / target / eod)
  - Entry rejection breakdown (news / tier / validator / heat / regime)
  - Top winning and losing trades
"""
from __future__ import annotations
import copy, os, warnings, logging
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from typing import Dict, List, Optional

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
from bot.intraday.indicators.atr import ATRIndicator
from bot.intraday.indicators.vwap import VWAPIndicator
from bot.intraday.risk.portfolio import PortfolioState
from bot.intraday.risk.sizing import compute_position_size
from bot.intraday.types import Bar, Position, TradeRecord
from bot.momentum.validator import MomentumValidator
from bot.positions.manager import PositionManager
from zoneinfo import ZoneInfo

START       = date(2025, 6,  1)
END         = date(2026, 5, 28)
TIER_BYPASS = int(os.environ.get("TIER_BYPASS", "4"))
_ET   = ZoneInfo("America/New_York")
_EOD_HOUR, _EOD_MINUTE = 15, 55

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

def trading_days(start: date, end: date) -> List[date]:
    days, d = [], start
    while d <= end:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days

def baseline_vol(screener: CandidateScreener, sym: str, as_of: date) -> float:
    df = screener._daily_cache.get(sym)
    if df is None or df.empty:
        return 0.0
    past = df[df.index < pd.Timestamp(as_of)]
    return float(past["volume"].tail(20).mean()) / 390.0 if not past.empty else 0.0


def run_diagnostic(
    cfg,
    days: List[date],
    screener: CandidateScreener,
    fetcher: BarFetcher,
    initial_equity: float,
    news_filter: NewsFilter,
) -> None:
    """Run with full per-bar instrumentation, print detailed breakdown."""

    cache_dir    = fetcher._cache_dir
    running_eq   = initial_equity
    trades: List[TradeRecord] = []

    # Counters
    regime_blocked_days  = 0
    entry_rejections     = Counter()   # reason → count
    exit_reasons         = Counter()   # reason → count
    monthly_trades: Dict[str, list]  = defaultdict(list)
    monthly_regime_blocked: Dict[str, int] = defaultdict(int)

    validator   = MomentumValidator(cfg)

    for d in days:
        ym = d.strftime("%Y-%m")
        if not spy_uptrend(d):
            regime_blocked_days += 1
            monthly_regime_blocked[ym] += 1
            continue

        sim = Simulator(
            cfg, running_eq,
            slippage_pct=0.001,
            overnight_holds=False,
            market_order_fill=True,
            news_filter=news_filter,
            news_mode="require",
            news_tier_bypass=TIER_BYPASS,
        )

        cands  = screener.candidates_for_date(d)
        cached = [s for s in cands if (cache_dir / f"{s}_{d}.json").exists()]
        if not cached:
            entry_rejections["no_cached_bars_day"] += len(cands)
            continue

        bars_by_sym: Dict = {}
        baselines:   Dict = {}
        with ThreadPoolExecutor(max_workers=16) as pool:
            for sym, bars, bl in pool.map(
                lambda s: (s, fetcher.fetch(s, d), baseline_vol(screener, s, d)),
                cached,
            ):
                if bars and bl > 0:
                    bars_by_sym[sym] = bars
                    baselines[sym]   = bl

        if not bars_by_sym:
            continue

        # ── Per-bar diagnostic pass (replicate simulator logic) ──────────
        atr_ind    = {s: ATRIndicator(period=14) for s in bars_by_sym}
        vwap_ind   = {s: VWAPIndicator()         for s in bars_by_sym}
        portfolio  = PortfolioState(equity=running_eq, config=cfg)
        manager    = PositionManager(cfg)
        traded_today: set = set()
        pending_entries: Dict = {}

        # Pre-compute news
        has_cat: Dict[str, bool] = {}
        for sym in bars_by_sym:
            has_cat[sym] = news_filter.has_catalyst(sym, d)

        merged = sorted(
            (b for bars in bars_by_sym.values() for b in bars),
            key=lambda b: b.timestamp,
        )

        for bar in merged:
            sym      = bar.symbol
            baseline = baselines.get(sym, 0.0)
            bar_et   = bar.timestamp.astimezone(_ET)
            atr_val  = atr_ind[sym].update(bar)
            vwap_val = vwap_ind[sym].update(bar)

            # Fill pending entry
            if sym in pending_entries:
                pending_entries.pop(sym)
                continue

            # EOD
            if bar_et.hour == _EOD_HOUR and bar_et.minute == _EOD_MINUTE:
                if sym in portfolio.positions:
                    exit_reasons["eod"] += 1
                continue

            # Exit
            if sym in portfolio.positions:
                position = portfolio.positions[sym]
                if bar.low <= position.stop_price:
                    exit_reasons["hard_stop_gap"] += 1
                    portfolio.remove_position(sym)
                elif vwap_val is not None and baseline > 0:
                    instr = manager.on_bar(bar, position, vwap_val, baseline)
                    if instr:
                        exit_reasons[instr.reason] += 1
                        portfolio.remove_position(sym)
                continue

            # Entry eligibility
            if sym in traded_today or sym in pending_entries:
                continue
            if not (atr_val and baseline > 0):
                entry_rejections["no_atr_or_baseline"] += 1
                continue
            can_enter, reason = portfolio.can_enter(sector="Unknown", now=bar.timestamp)
            if not can_enter:
                entry_rejections[f"portfolio_{reason}"] += 1
                continue

            if not validator.validate(bar, baseline):
                continue  # didn't pass stage-2 — not a "rejection", just no signal

            # Passed stage-2: now check gates
            conf = validator.confidence_score(bar, baseline)
            mult = cfg.confidence_multiplier(conf)

            cat = has_cat.get(sym, False)
            if not cat:
                if mult >= TIER_BYPASS:
                    pass  # tier bypass
                else:
                    entry_rejections["news_no_cat_tier_low"] += 1
                    continue

            size = compute_position_size(portfolio.equity, atr_val, bar.close, cfg)
            if size.shares <= 0:
                entry_rejections["zero_shares"] += 1
                continue

            # Would enter — record for simulation pass
            traded_today.add(sym)

        # Run actual simulator for PnL
        result     = sim.run_day(d, bars_by_sym, baselines)
        day_trades = result.trades
        trades.extend(day_trades)
        for t in day_trades:
            if t.exit_reason:
                exit_reasons[f"sim_{t.exit_reason}"] += 1
            monthly_trades[ym].append(t)

        day_pnl    = sum(t.pnl for t in day_trades if t.pnl is not None)
        running_eq = max(running_eq + day_pnl, 1.0)

    # ── Print report ─────────────────────────────────────────────────────

    total_days = len(days)
    total_pnl  = sum(t.pnl for t in trades if t.pnl is not None)
    wins       = [t for t in trades if t.pnl and t.pnl > 0]
    losses     = [t for t in trades if t.pnl and t.pnl <= 0]

    print(f"\n{'='*70}")
    print(f"  DIAGNOSTIC: 2025-26 bull  ({START} → {END})  [tier_bypass={TIER_BYPASS}]")
    print(f"{'='*70}")
    print(f"  Starting equity : ${initial_equity:,.0f}")
    print(f"  Ending equity   : ${running_eq:,.0f}  ({(running_eq/initial_equity-1)*100:+.1f}%)")
    print(f"  Total trades    : {len(trades)}")
    print(f"  Win rate        : {len(wins)/len(trades)*100:.1f}%" if trades else "  Win rate: n/a")
    print(f"  Net PnL         : ${total_pnl:,.0f}")
    print(f"  Trading days    : {total_days - regime_blocked_days} / {total_days}  ({regime_blocked_days} regime-blocked)")

    print(f"\n  ── Exit reasons (from simulator) ──")
    sim_exits = {k.replace("sim_",""):v for k,v in exit_reasons.items() if k.startswith("sim_")}
    for reason, cnt in sorted(sim_exits.items(), key=lambda x: -x[1]):
        pct = cnt / len(trades) * 100 if trades else 0
        print(f"    {reason:<22} {cnt:>5}  ({pct:.0f}%)")

    # Avg PnL by exit reason
    by_reason: Dict[str, list] = defaultdict(list)
    for t in trades:
        if t.exit_reason and t.pnl is not None:
            by_reason[t.exit_reason].append(t.pnl)
    print(f"\n  ── Avg PnL by exit reason ──")
    for reason, pnls in sorted(by_reason.items(), key=lambda x: -len(x[1])):
        avg = sum(pnls) / len(pnls)
        print(f"    {reason:<22} n={len(pnls):>4}  avg=${avg:>8,.0f}  total=${sum(pnls):>10,.0f}")

    print(f"\n  ── Entry rejections (stage-2 pass but gate blocked) ──")
    for reason, cnt in sorted(entry_rejections.items(), key=lambda x: -x[1]):
        print(f"    {reason:<30} {cnt:>6}")

    print(f"\n  ── Monthly breakdown ──")
    print(f"  {'Month':>8}  {'Trades':>7} {'WR':>6} {'PnL':>11} {'Cum PnL':>11} {'Regime↓':>8}")
    cum = 0.0
    all_months = sorted(set(list(monthly_trades.keys()) + list(monthly_regime_blocked.keys())))
    for ym in all_months:
        month_trades = monthly_trades[ym]
        month_wins   = [t for t in month_trades if t.pnl and t.pnl > 0]
        month_pnl    = sum(t.pnl for t in month_trades if t.pnl is not None)
        cum         += month_pnl
        wr_str       = f"{len(month_wins)/len(month_trades)*100:.0f}%" if month_trades else "  -"
        blocked      = monthly_regime_blocked.get(ym, 0)
        print(f"  {ym:>8}  {len(month_trades):>7} {wr_str:>6} ${month_pnl:>10,.0f} ${cum:>10,.0f} {blocked:>8}")

    print(f"\n  ── Top 10 winners ──")
    top_wins = sorted(wins, key=lambda t: t.pnl, reverse=True)[:10]
    for t in top_wins:
        print(f"    {t.ticker:<6} {str(t.entry_time.date()):>12}  +${t.pnl:>8,.0f}  {t.exit_reason}")

    print(f"\n  ── Top 10 losers ──")
    top_losses = sorted(losses, key=lambda t: t.pnl)[:10]
    for t in top_losses:
        print(f"    {t.ticker:<6} {str(t.entry_time.date()):>12}  -${abs(t.pnl):>8,.0f}  {t.exit_reason}")

    print()


def main():
    api_key    = os.environ["APCA_API_KEY_ID"]
    secret_key = os.environ["APCA_API_SECRET_KEY"]
    base_url   = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

    initial_equity = broker.get_account_info()["portfolio_value"]
    cfg      = make_gap_hold_config()
    screener = CandidateScreener(copy.copy(cfg), api_key, secret_key, base_url)
    print("Loading screener...", flush=True)
    screener.preload(START, END)
    print(f"  {len(screener._daily_cache):,} symbols loaded.", flush=True)

    fetcher     = BarFetcher(api_key, secret_key)
    news_filter = NewsFilter(api_key, secret_key, cache_only=False)
    days        = trading_days(START, END)

    run_diagnostic(copy.copy(cfg), days, screener, fetcher, initial_equity, news_filter)


if __name__ == "__main__":
    main()
