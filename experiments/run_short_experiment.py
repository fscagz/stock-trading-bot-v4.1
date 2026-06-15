"""
Short (fade) strategy experiment — all three periods.

Uses the same screener universe as the long strategy (stocks that spiked 15%+
intraday). Adds a short entry when those same stocks reverse:
  - Negative ROC over 5 bars  (dropping, not climbing)
  - High relative volume       (same threshold as longs)
  - Selling pressure           (bar closes near its LOW, opposite of buying pressure)
  - Must be below VWAP         (confirming reversal, not just a brief dip)

Tests four configs per period:
  1. Long-only   (baseline at corrected top-50 cap)
  2. Short-only  (fade entries only)
  3. Long+Short  (both simultaneously, independent heat budgets)

All: dynamic equity, news tier-4 bypass, day_high_5pct on longs, slippage=0.001.
Short target_atr_multiple=4 (same as our best long config).
"""
from __future__ import annotations
import copy, os, warnings, logging
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from typing import Dict, List, Optional, Tuple

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
from bot.config import make_gap_hold_config, V4Config
from bot.data.daily_loader import get_daily
from bot.intraday.indicators.atr import ATRIndicator
from bot.intraday.indicators.vwap import VWAPIndicator
from bot.intraday.risk.kill_switch import KillSwitch
from bot.intraday.risk.portfolio import PortfolioState
from bot.intraday.risk.sizing import compute_position_size
from bot.intraday.types import Bar, Position, TradeRecord
from bot.momentum.validator import MomentumValidator
from bot.positions.manager import PositionManager
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")
_EOD_HOUR, _EOD_MINUTE = 15, 55

PERIODS = [
    ("2022 bear",    date(2022, 1,  3),  date(2022, 12, 30), -19.0),
    ("2023 mixed",   date(2023, 1,  3),  date(2023, 12, 29), +26.0),
    ("2025-26 bull", date(2025, 6,  1),  date(2026,  5, 28), +30.0),
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

def baseline_vol(screener: CandidateScreener, sym: str, as_of: date) -> float:
    df = screener._daily_cache.get(sym)
    if df is None or df.empty:
        return 0.0
    past = df[df.index < pd.Timestamp(as_of)]
    return float(past["volume"].tail(20).mean()) / 390.0 if not past.empty else 0.0


# ── Short validator: inverted momentum ──────────────────────────────────────

class ShortValidator:
    """Stage-2 validation for short (fade) entries.

    Fires when a stock that spiked is now reversing:
      - Negative ROC: price dropped >= roc_min over last 5 bars
      - High relative volume (same threshold as long)
      - Selling pressure: bar closes near its LOW
    """
    def __init__(self, cfg: V4Config) -> None:
        self._cfg = cfg
        from collections import deque
        from typing import Deque
        self._history: Dict[str, "deque[Bar]"] = {}

    def validate(self, bar: Bar, baseline: float) -> bool:
        from collections import deque
        sym = bar.symbol
        if sym not in self._history:
            lookback = self._cfg.stage2_roc_lookback_bars + 1
            self._history[sym] = deque(maxlen=lookback)
        self._history[sym].append(bar)

        history = list(self._history[sym])
        lookback = self._cfg.stage2_roc_lookback_bars
        if len(history) < lookback + 1:
            return False

        # Negative ROC (price is falling)
        past_close = history[-(lookback + 1)].close
        if past_close <= 0:
            return False
        roc = (bar.close - past_close) / past_close
        if roc > -self._cfg.stage2_roc_min_pct:
            return False

        # High relative volume
        if baseline <= 0 or bar.volume < baseline * self._cfg.stage2_min_relative_volume:
            return False

        # Selling pressure: bar closes near its low
        bar_range = bar.high - bar.low
        if bar_range <= 0:
            return False
        close_pos = (bar.close - bar.low) / bar_range
        selling_pressure_min = 1.0 - self._cfg.stage2_buying_pressure_min  # e.g. 0.15
        if close_pos > selling_pressure_min:
            return False

        return True


# ── Combined long+short day runner ──────────────────────────────────────────

def run_day_combined(
    trade_date: date,
    bars_by_sym: Dict,
    baselines: Dict,
    long_cfg: V4Config,
    short_cfg: V4Config,
    equity: float,
    news_filter: NewsFilter,
    run_longs: bool,
    run_shorts: bool,
) -> Tuple[List[TradeRecord], float]:
    """Run one trading day with independent long and short books."""
    merged: List[Bar] = sorted(
        (b for bars in bars_by_sym.values() for b in bars),
        key=lambda b: b.timestamp,
    )

    atr_inds  = {s: ATRIndicator(period=14) for s in bars_by_sym}
    vwap_inds = {s: VWAPIndicator()         for s in bars_by_sym}
    long_val  = MomentumValidator(long_cfg)
    short_val = ShortValidator(short_cfg)
    long_mgr  = PositionManager(long_cfg)
    short_mgr = PositionManager(short_cfg)
    long_ks   = KillSwitch(long_cfg)
    short_ks  = KillSwitch(short_cfg)

    long_port  = PortfolioState(equity=equity, config=long_cfg)
    short_port = PortfolioState(equity=equity, config=short_cfg)

    # Pre-compute news catalyst
    cat: Dict[str, bool] = {}
    for sym in bars_by_sym:
        cat[sym] = news_filter.has_catalyst(sym, trade_date)

    open_long:  Dict[str, TradeRecord] = {}
    open_short: Dict[str, TradeRecord] = {}
    closed:     List[TradeRecord]       = []
    long_today: set = set()
    short_today: set = set()
    day_highs:  Dict[str, float] = {}
    slippage = 0.001

    for bar in merged:
        sym      = bar.symbol
        baseline = baselines.get(sym, 0.0)
        bar_et   = bar.timestamp.astimezone(_ET)
        atr_val  = atr_inds[sym].update(bar)
        vwap_val = vwap_inds[sym].update(bar)
        day_highs[sym] = max(day_highs.get(sym, 0.0), bar.high)

        is_eod = bar_et.hour == _EOD_HOUR and bar_et.minute == _EOD_MINUTE

        # ── Existing position exits (one per symbol per bar) ─────────────────
        if sym in long_port.positions:
            pos = long_port.positions[sym]
            exited = False
            if is_eod:
                pnl = (bar.close - pos.entry_price) * pos.shares
                rec = open_long.pop(sym, None)
                if rec:
                    rec.exit_price = bar.close; rec.exit_time = bar.timestamp
                    rec.pnl = pnl; rec.exit_reason = "eod"; closed.append(rec)
                long_port.remove_position(sym); exited = True
            elif bar.low <= pos.stop_price:
                fill = min(pos.stop_price, bar.open)
                pnl  = (fill - pos.entry_price) * pos.shares
                rec  = open_long.pop(sym, None)
                if rec:
                    rec.exit_price = fill; rec.exit_time = bar.timestamp
                    rec.pnl = pnl; rec.exit_reason = "hard_stop"; closed.append(rec)
                long_port.remove_position(sym); exited = True
            elif bar.high >= pos.target_price:
                pnl = (pos.target_price - pos.entry_price) * pos.shares
                rec = open_long.pop(sym, None)
                if rec:
                    rec.exit_price = pos.target_price; rec.exit_time = bar.timestamp
                    rec.pnl = pnl; rec.exit_reason = "target"; closed.append(rec)
                long_port.remove_position(sym); exited = True
            elif vwap_val is not None and baseline > 0:
                instr = long_mgr.on_bar(bar, pos, vwap_val, baseline)
                if instr:
                    fill = bar.close
                    pnl  = (fill - pos.entry_price) * pos.shares
                    rec  = open_long.pop(sym, None)
                    if rec:
                        rec.exit_price = fill; rec.exit_time = bar.timestamp
                        rec.pnl = pnl; rec.exit_reason = instr.reason; closed.append(rec)
                    long_port.remove_position(sym); exited = True
            if not exited:
                continue  # still holding — skip entry logic

        elif sym in short_port.positions:
            pos = short_port.positions[sym]
            exited = False
            if is_eod:
                pnl = (pos.entry_price - bar.close) * pos.shares
                rec = open_short.pop(sym, None)
                if rec:
                    rec.exit_price = bar.close; rec.exit_time = bar.timestamp
                    rec.pnl = pnl; rec.exit_reason = "eod"; closed.append(rec)
                short_port.remove_position(sym); exited = True
            elif bar.high >= pos.stop_price:
                fill = max(pos.stop_price, bar.open)
                pnl  = (pos.entry_price - fill) * pos.shares
                rec  = open_short.pop(sym, None)
                if rec:
                    rec.exit_price = fill; rec.exit_time = bar.timestamp
                    rec.pnl = pnl; rec.exit_reason = "hard_stop"; closed.append(rec)
                short_port.remove_position(sym); exited = True
            elif bar.low <= pos.target_price:
                pnl = (pos.entry_price - pos.target_price) * pos.shares
                rec = open_short.pop(sym, None)
                if rec:
                    rec.exit_price = pos.target_price; rec.exit_time = bar.timestamp
                    rec.pnl = pnl; rec.exit_reason = "target"; closed.append(rec)
                short_port.remove_position(sym); exited = True
            elif vwap_val is not None and baseline > 0:
                instr = short_mgr.on_bar(bar, pos, vwap_val, baseline)
                if instr:
                    fill = bar.close
                    pnl  = (pos.entry_price - fill) * pos.shares
                    rec  = open_short.pop(sym, None)
                    if rec:
                        rec.exit_price = fill; rec.exit_time = bar.timestamp
                        rec.pnl = pnl; rec.exit_reason = instr.reason; closed.append(rec)
                    short_port.remove_position(sym); exited = True
            if not exited:
                continue  # still holding — skip entry logic

        if is_eod or not (atr_val and baseline > 0):
            continue

        # ── Long entry ───────────────────────────────────────────────────────
        if run_longs and sym not in long_today and sym not in long_port.positions:
            long_ks.check(long_port, bar.timestamp)
            can_enter, _ = long_port.can_enter(sector="Unknown", now=bar.timestamp)
            if can_enter and long_val.validate(bar, baseline):
                day_high = day_highs.get(sym, bar.high)
                min_dist = long_cfg.stage2_min_dist_from_day_high_pct
                too_close = (min_dist > 0 and day_high > 0
                             and (day_high - bar.close) / day_high < min_dist)
                if not too_close:
                    _try_long_entry(sym, bar, atr_val, baseline, long_cfg, long_port,
                                    open_long, long_today, cat, slippage, long_val)

        # ── Short entry ──────────────────────────────────────────────────────
        if run_shorts and sym not in short_today and sym not in short_port.positions:
            short_ks.check(short_port, bar.timestamp)
            can_enter, _ = short_port.can_enter(sector="Unknown", now=bar.timestamp)
            if can_enter and vwap_val is not None and bar.close < vwap_val:
                if short_val.validate(bar, baseline):
                    size = compute_position_size(short_port.equity, atr_val, bar.close, short_cfg)
                    if size.shares > 0:
                        fill  = round(bar.close * (1 - slippage), 2)
                        stop  = size.short_stop(fill)
                        tgt   = size.short_target(fill)
                        pos   = Position(ticker=sym, direction="short", shares=size.shares,
                                         entry_price=fill, stop_price=stop, target_price=tgt,
                                         entry_time=bar.timestamp, atr_at_entry=atr_val,
                                         signals=["fade"], sector="Unknown",
                                         highest_close=fill, entry_bar_volume=bar.volume)
                        short_port.add_position(pos)
                        open_short[sym] = TradeRecord(
                            ticker=sym, direction="short", entry_time=bar.timestamp,
                            entry_price=fill, shares=size.shares, stop_price=stop,
                            target_price=tgt, signals=["fade"], sector="Unknown",
                            regime="", portfolio_heat_at_entry=short_port.portfolio_heat_pct,
                            expected_slippage_pct=slippage,
                        )
                        short_today.add(sym)

    # Force-close anything still open at EOD (shouldn't normally happen)
    for sym, rec in list(open_long.items()):
        if sym in long_port.positions:
            pos = long_port.positions[sym]
            last_close = bars_by_sym[sym][-1].close if sym in bars_by_sym else pos.entry_price
            rec.exit_price = last_close; rec.pnl = (last_close - pos.entry_price) * pos.shares
            rec.exit_reason = "eod"; closed.append(rec)
    for sym, rec in list(open_short.items()):
        if sym in short_port.positions:
            pos = short_port.positions[sym]
            last_close = bars_by_sym[sym][-1].close if sym in bars_by_sym else pos.entry_price
            rec.exit_price = last_close; rec.pnl = (pos.entry_price - last_close) * pos.shares
            rec.exit_reason = "eod"; closed.append(rec)

    day_pnl = sum(t.pnl for t in closed if t.pnl is not None)
    return closed, day_pnl


def _try_long_entry(sym, bar, atr_val, baseline, cfg, port, open_long, long_today, cat, slippage,
                    validator, news_tier_bypass=4):
    has_cat = cat.get(sym, False)
    if not has_cat:
        conf = validator.confidence_score(bar, baseline)
        if cfg.confidence_multiplier(conf) < news_tier_bypass:
            return
    size = compute_position_size(port.equity, atr_val, bar.close, cfg)
    if size.shares <= 0:
        return
    mult = cfg.confidence_multiplier(validator.confidence_score(bar, baseline))
    if mult > 1.0:
        max_shares = int(port.equity * cfg.max_position_pct / bar.close)
        size.shares = min(int(size.shares * mult), max(0, max_shares))
    fill  = round(bar.close * (1 + slippage), 2)
    stop  = size.long_stop(fill)
    tgt   = size.long_target(fill)
    pos   = Position(ticker=sym, direction="long", shares=int(size.shares),
                     entry_price=fill, stop_price=stop, target_price=tgt,
                     entry_time=bar.timestamp, atr_at_entry=atr_val,
                     signals=["momentum"], sector="Unknown",
                     highest_close=fill, entry_bar_volume=bar.volume)
    port.add_position(pos)
    open_long[sym] = TradeRecord(
        ticker=sym, direction="long", entry_time=bar.timestamp,
        entry_price=fill, shares=int(size.shares), stop_price=stop,
        target_price=tgt, signals=["momentum"], sector="Unknown",
        regime="", portfolio_heat_at_entry=port.portfolio_heat_pct,
        expected_slippage_pct=slippage,
    )
    long_today.add(sym)


def run_period(
    label: str, start: date, end: date, spy_ret: float,
    screener: CandidateScreener, fetcher: BarFetcher,
    news_filter: NewsFilter, initial_equity: float,
    long_cfg: V4Config, short_cfg: V4Config,
) -> None:
    days      = trading_days(start, end)
    cache_dir = fetcher._cache_dir

    configs = [
        ("Long-only  (top-50 fix)", True,  False),
        ("Short-only (fade)",        False, True),
        ("Long+Short (combined)",    True,  True),
    ]

    print(f"\n{'='*72}")
    print(f"  {label}  ({start} → {end})  |  S&P: {spy_ret:+.0f}%")
    print(f"{'='*72}")
    hdr = f"  {'Config':<28} {'Trades':>7} {'WR':>6} {'Return':>8}  {'vs S&P':>8}  {'Long T':>7} {'Short T':>8}"
    print(hdr)
    print("  " + "-" * 75)

    for cfg_label, do_long, do_short in configs:
        running_eq   = initial_equity
        all_trades:  List[TradeRecord] = []

        for d in days:
            uptrend = spy_uptrend(d)
            if do_long and not uptrend and not do_short:
                continue
            # Shorts run regardless of regime; longs need uptrend
            if not uptrend and do_long and not do_short:
                continue

            cands  = screener.candidates_for_date(d)
            cached = [s for s in cands if (cache_dir / f"{s}_{d}.json").exists()]
            if not cached:
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

            # For long-only, use the existing Simulator (handles regime internally)
            if do_long and not do_short:
                sim = Simulator(
                    copy.copy(long_cfg), running_eq,
                    slippage_pct=0.001, overnight_holds=False,
                    market_order_fill=True, news_filter=news_filter,
                    news_mode="require", news_tier_bypass=4,
                    stage2_min_dist_from_day_high_pct=0.05,
                )
                if not uptrend:
                    continue
                result     = sim.run_day(d, bars_by_sym, baselines)
                day_trades = result.trades
            else:
                effective_long  = do_long and uptrend
                day_trades, _   = run_day_combined(
                    d, bars_by_sym, baselines,
                    copy.copy(long_cfg), copy.copy(short_cfg),
                    running_eq, news_filter,
                    run_longs=effective_long, run_shorts=do_short,
                )

            all_trades.extend(day_trades)
            day_pnl    = sum(t.pnl for t in day_trades if t.pnl is not None)
            running_eq = max(running_eq + day_pnl, 1.0)

        trades = all_trades
        if not trades:
            print(f"  {cfg_label:<28}      —      —       —         —")
            continue

        wins    = [t for t in trades if t.pnl and t.pnl > 0]
        net     = sum(t.pnl for t in trades if t.pnl is not None)
        ret_pct = net / initial_equity * 100
        vs_spy  = ret_pct - spy_ret
        longs   = [t for t in trades if t.direction == "long"]
        shorts  = [t for t in trades if t.direction == "short"]
        flag    = " ✓" if vs_spy >= 0 else "  "

        print(
            f"  {cfg_label:<28} {len(trades):>7,} {len(wins)/len(trades)*100:>5.1f}% "
            f"{ret_pct:>+7.1f}%  {vs_spy:>+7.1f}%{flag}  "
            f"{len(longs):>6}  {len(shorts):>7}",
            flush=True,
        )


def main():
    api_key    = os.environ["APCA_API_KEY_ID"]
    secret_key = os.environ["APCA_API_SECRET_KEY"]
    base_url   = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

    initial_equity = broker.get_account_info()["portfolio_value"]
    print(f"Starting equity: ${initial_equity:,.0f}")
    print(f"Screener cap: top-50 (live-accurate)  |  Long: day_high_5pct, target 4×  |  Short: fade below VWAP")

    long_cfg = make_gap_hold_config()
    long_cfg.target_atr_multiple = 4.0
    long_cfg.stage2_min_dist_from_day_high_pct = 0.05

    short_cfg = make_gap_hold_config()   # same base params
    short_cfg.target_atr_multiple = 4.0
    short_cfg.risk_per_trade = 0.01  # half the long risk — short strategies tend to be choppier

    fetcher     = BarFetcher(api_key, secret_key)
    news_filter = NewsFilter(api_key, secret_key, cache_only=False)

    for period_label, start, end, spy_ret in PERIODS:
        cfg_copy = copy.copy(long_cfg)
        screener = CandidateScreener(cfg_copy, api_key, secret_key, base_url)
        print(f"\nLoading screener for {period_label}...", flush=True)
        screener.preload(start, end)
        print(f"  {len(screener._daily_cache):,} symbols, top-{cfg_copy.scanner_top_n} cap", flush=True)

        run_period(
            period_label, start, end, spy_ret,
            screener, fetcher, news_filter, initial_equity,
            copy.copy(long_cfg), copy.copy(short_cfg),
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
