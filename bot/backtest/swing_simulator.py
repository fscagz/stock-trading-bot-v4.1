"""Multi-day swing backtest engine for episodic-pivot style strategies.

The existing Simulator resets its portfolio every day, so it cannot test
strategies that hold winners across days. This engine does two passes:

  1. simulate_event(): per-event trade simulation on that symbol's own daily
     series (plus optional minute bars for the entry day) — produces an
     outcome in R units, independent of account size. Valid because position
     sizes here are far below daily liquidity (no participation effects).
  2. run_portfolio(): chronological pass over event results, enforcing a
     concurrent-position cap and risk-based sizing with compounding equity.

Entry modes:
  "orb":         buy the break of the opening-range high on the event day
                 (requires minute bars); stop = low-of-day at entry time.
  "close_green": buy the event-day close if it closed above the open;
                 stop = event-day low.

Exit rules (Qullamaggie-style):
  - hard stop (gap-throughs fill at the open, not the stop)
  - scale out scale_out_frac at entry + scale_out_r × risk; stop → breakeven
  - trail remainder: exit on a close below the trail_ma_days SMA of closes
  - time stop at max_hold_days; exit at close
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime, timezone
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

_ET_OPEN = dtime(9, 30)


@dataclass
class SwingConfig:
    risk_per_trade: float = 0.01        # fraction of equity risked per trade
    max_positions: int = 4
    max_notional_pct: float = 0.25      # position value cap as fraction of equity
    entry_slippage_pct: float = 0.003
    exit_slippage_pct: float = 0.003
    scale_out_r: float = 2.0            # first scale-out at entry + N × risk
    scale_out_frac: float = 0.5
    trail_ma_days: int = 10
    max_hold_days: int = 20
    orb_minutes: int = 30
    entry_cutoff_minutes: int = 150     # no ORB entries later than N min after open


@dataclass
class SwingEvent:
    symbol: str
    day: date
    gap_pct: float


@dataclass
class EventResult:
    event: SwingEvent
    entered: bool
    skip_reason: str = ""
    entry_date: Optional[date] = None
    entry_price: float = 0.0
    stop_price: float = 0.0
    # (date, fill_price, fraction_of_position, reason)
    exits: List[Tuple[date, float, float, str]] = field(default_factory=list)
    r_multiple: float = 0.0
    hold_days: int = 0

    @property
    def exit_date(self) -> Optional[date]:
        return self.exits[-1][0] if self.exits else None


@dataclass
class SwingTrade:
    symbol: str
    entry_date: date
    exit_date: date
    entry_price: float
    stop_price: float
    shares: int
    risk_dollars: float
    pnl: float
    r_multiple: float
    exit_reasons: str
    hold_days: int


def _finalize(res: EventResult) -> EventResult:
    risk = res.entry_price - res.stop_price
    if risk <= 0:
        res.entered = False
        res.skip_reason = "degenerate_risk"
        return res
    pnl_per_share = sum((price - res.entry_price) * frac for _, price, frac, _ in res.exits)
    res.r_multiple = pnl_per_share / risk
    if res.exits:
        res.hold_days = max(0, (res.exits[-1][0] - res.entry_date).days)
    return res


def simulate_event(
    event: SwingEvent,
    daily: pd.DataFrame,
    cfg: SwingConfig,
    minute_bars: Optional[List[dict]] = None,
    entry_mode: str = "orb",
) -> EventResult:
    """Simulate one event on the symbol's daily series (index: DatetimeIndex)."""
    res = EventResult(event=event, entered=False)
    day_ts = pd.Timestamp(event.day)
    if day_ts not in daily.index:
        res.skip_reason = "no_daily_row"
        return res
    row = daily.loc[day_ts]

    # --- Entry ---
    if entry_mode == "close_green":
        if row["close"] <= row["open"]:
            res.skip_reason = "closed_red"
            return res
        entry_price = row["close"] * (1 + cfg.entry_slippage_pct)
        stop = row["low"]
        entry_date = event.day
    elif entry_mode == "orb":
        if not minute_bars:
            res.skip_reason = "no_minute_bars"
            return res
        or_high = None
        or_low = None
        lod = None
        entry_price = None
        stop = None
        stopped_same_day = False
        for b in minute_bars:
            ts = datetime.fromisoformat(b["t"].replace("Z", "+00:00"))
            # cache bars are UTC; 13:30 UTC == 09:30 ET during DST, 14:30 in winter.
            # Use minutes since the first bar instead of wall-clock conversion.
            if or_high is None:
                first_ts = ts
            minutes_in = (ts - first_ts).total_seconds() / 60
            lod = b["l"] if lod is None else min(lod, b["l"])
            if minutes_in < cfg.orb_minutes:
                or_high = b["h"] if or_high is None else max(or_high, b["h"])
                or_low = b["l"] if or_low is None else min(or_low, b["l"])
                continue
            if entry_price is None:
                if minutes_in > cfg.entry_cutoff_minutes:
                    break
                if b["h"] > or_high:
                    raw = max(or_high, b["o"])  # gap above OR-high fills at bar open
                    entry_price = raw * (1 + cfg.entry_slippage_pct)
                    stop = lod
                continue
            # position open — manage rest of entry day bar by bar
            if b["l"] <= stop:
                stopped_same_day = True
                break
        if entry_price is None:
            res.skip_reason = "no_orb_break"
            return res
        entry_date = event.day
        if stop is None or stop >= entry_price:
            res.skip_reason = "degenerate_risk"
            return res
        res.entered = True
        res.entry_date = entry_date
        res.entry_price = entry_price
        res.stop_price = stop
        if stopped_same_day:
            res.exits.append((event.day, stop * (1 - cfg.exit_slippage_pct), 1.0, "stop_same_day"))
            return _finalize(res)
    else:
        raise ValueError(f"unknown entry_mode {entry_mode!r}")

    if entry_mode == "close_green":
        if stop >= entry_price:
            res.skip_reason = "degenerate_risk"
            return res
        res.entered = True
        res.entry_date = entry_date
        res.entry_price = entry_price
        res.stop_price = stop

    # --- Multi-day management on the symbol's own subsequent rows ---
    risk = res.entry_price - res.stop_price
    scale_level = res.entry_price + cfg.scale_out_r * risk
    stop = res.stop_price
    remaining = 1.0
    scaled = False
    future = daily[daily.index > day_ts]
    closes_hist = list(daily.loc[:day_ts, "close"])

    for ts, r in future.iterrows():
        d = ts.date()
        held = (d - res.entry_date).days
        closes_hist.append(r["close"])

        # 1. stop (conservative: stop before scale-out when both hit)
        if r["open"] <= stop:
            res.exits.append((d, r["open"] * (1 - cfg.exit_slippage_pct), remaining, "gap_stop"))
            return _finalize(res)
        if r["low"] <= stop:
            res.exits.append((d, stop * (1 - cfg.exit_slippage_pct), remaining, "stop"))
            return _finalize(res)
        # 2. scale out into strength (limit order at level — no slippage)
        if not scaled and r["high"] >= scale_level:
            res.exits.append((d, scale_level, cfg.scale_out_frac, "scale_out"))
            remaining -= cfg.scale_out_frac
            scaled = True
            stop = max(stop, res.entry_price)  # breakeven
        # 3. trail: close below N-day SMA exits the remainder
        sma = sum(closes_hist[-cfg.trail_ma_days:]) / min(len(closes_hist), cfg.trail_ma_days)
        if scaled and r["close"] < sma:
            res.exits.append((d, r["close"] * (1 - cfg.exit_slippage_pct), remaining, "trail_ma"))
            return _finalize(res)
        # 4. time stop
        if held >= cfg.max_hold_days:
            res.exits.append((d, r["close"] * (1 - cfg.exit_slippage_pct), remaining, "time"))
            return _finalize(res)

    # data ended with position open — mark out at last close
    if len(future) > 0:
        last_ts, last_row = list(future.iterrows())[-1]
        res.exits.append((last_ts.date(), last_row["close"], remaining, "data_end"))
    else:
        res.exits.append((res.entry_date, res.entry_price, remaining, "data_end"))
    return _finalize(res)


def run_portfolio(
    results: List[EventResult],
    cfg: SwingConfig,
    initial_equity: float,
) -> Tuple[List[SwingTrade], List[Tuple[date, float]]]:
    """Chronological capital allocation over per-event results."""
    entered = sorted(
        (r for r in results if r.entered and r.exits),
        key=lambda r: (r.entry_date, r.event.symbol),
    )
    open_pos: List[Tuple[date, str]] = []   # (exit_date, symbol)
    trades: List[SwingTrade] = []
    curve: List[Tuple[date, float]] = []
    # P&L becomes available for sizing only once the trade has exited —
    # sizing off instantly-credited future profits would be look-ahead.
    realized: List[Tuple[date, float]] = []  # (exit_date, pnl)

    for r in entered:
        open_pos = [(xd, s) for xd, s in open_pos if xd > r.entry_date]
        if len(open_pos) >= cfg.max_positions:
            continue
        if any(s == r.event.symbol for _, s in open_pos):
            continue
        equity = initial_equity + sum(p for xd, p in realized if xd <= r.entry_date)
        risk_dollars = equity * cfg.risk_per_trade
        per_share_risk = r.entry_price - r.stop_price
        shares = int(risk_dollars / per_share_risk)
        max_shares = int(equity * cfg.max_notional_pct / r.entry_price)
        shares = min(shares, max_shares)
        if shares <= 0:
            continue
        actual_risk = shares * per_share_risk
        pnl = round(shares * per_share_risk * r.r_multiple, 2)
        exit_d = r.exits[-1][0]
        realized.append((exit_d, pnl))
        open_pos.append((exit_d, r.event.symbol))
        trades.append(SwingTrade(
            symbol=r.event.symbol,
            entry_date=r.entry_date,
            exit_date=exit_d,
            entry_price=round(r.entry_price, 4),
            stop_price=round(r.stop_price, 4),
            shares=shares,
            risk_dollars=round(actual_risk, 2),
            pnl=pnl,
            r_multiple=round(r.r_multiple, 3),
            exit_reasons=";".join(x[3] for x in r.exits),
            hold_days=r.hold_days,
        ))
    eq = initial_equity
    for xd, pnl in sorted(realized):
        eq += pnl
        curve.append((xd, round(eq, 2)))
    return trades, curve
