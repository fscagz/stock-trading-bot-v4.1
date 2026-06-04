"""
BacktestSimulator — offline replay of historical 1-min bars.

Runs the same indicator + signal stack as IntradayBot but simulates
order fills and exits using OHLC price boundaries instead of live orders.

Output: trade_log.csv in the same format as the live TradeLogger so the
file can be fed directly to ml/trainer.py.

Fill model:
  Entry  — bar i+1 open ± limit_offset_pct (long: open*(1+offset), short: open*(1-offset))
  Target — bar where high >= target_price (long) or low <= target_price (short)
  Stop   — bar where low <= stop_price (long) or high >= stop_price (short)
  EOD    — if no exit before eod_close, close at that bar's close
  Priority: if both stop and target would trigger on the same bar, stop wins.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")

from bot.intraday.config import IntradayConfig
from bot.intraday.execution.trade_log import TradeLogger
from bot.intraday.indicators.atr import ATRIndicator
from bot.intraday.indicators.ema import EMAIndicator
from bot.intraday.indicators.rsi import RSIIndicator
from bot.intraday.indicators.vwap import VWAPIndicator
from bot.intraday.risk.kill_switch import KillSwitch
from bot.intraday.risk.portfolio import PortfolioState
from bot.intraday.risk.regime import Regime
from bot.intraday.risk.sizing import compute_position_size
from bot.intraday.signals.aggregator import SignalAggregator
from bot.intraday.signals.technical import (
    check_breakout,
    check_ma_crossover,
    check_momentum_burst,
    check_rsi_extreme,
    check_vwap_continuation,
)
from bot.intraday.types import Bar, Position, TradeRecord

logger = logging.getLogger(__name__)


@dataclass
class _OpenPosition:
    ticker: str
    direction: str
    shares: int
    entry_price: float
    stop_price: float
    target_price: float
    entry_time: datetime
    atr_at_entry: float
    signals: List[str]
    open_risk: float


def _et_str(ts: datetime) -> str:
    """UTC timestamp → HH:MM Eastern (handles EDT/EST via ZoneInfo)."""
    et = ts.astimezone(_ET)
    return f"{et.hour:02d}:{et.minute:02d}"


class BacktestSimulator:
    """
    Replays a chronologically sorted list of Bar objects, generates signals
    using the live signal stack, simulates fills, and writes exit records.

    Usage:
        sim = BacktestSimulator(config, symbols, regime, equity, output_path)
        sim.run(all_bars)   # all_bars need not be pre-sorted
    """

    def __init__(
        self,
        config: IntradayConfig,
        symbols: List[str],
        regime: Regime,
        equity: float,
        output_path: str,
    ) -> None:
        self._cfg = config
        self._symbols = symbols
        self._regime = regime
        self._portfolio = PortfolioState(equity=equity, config=config)
        self._kill_switch = KillSwitch(config)
        self._trade_logger = TradeLogger(output_path)

        # Per-symbol indicators
        self._vwap: Dict[str, VWAPIndicator] = {s: VWAPIndicator() for s in symbols}
        self._atr: Dict[str, ATRIndicator] = {s: ATRIndicator(14) for s in symbols}
        self._rsi: Dict[str, RSIIndicator] = {s: RSIIndicator(14) for s in symbols}
        self._ema9: Dict[str, EMAIndicator] = {s: EMAIndicator(9) for s in symbols}
        self._ema21: Dict[str, EMAIndicator] = {s: EMAIndicator(21) for s in symbols}
        self._ema50: Dict[str, EMAIndicator] = {s: EMAIndicator(50) for s in symbols}

        self._aggregator = SignalAggregator(config)
        self._avg_volume: Dict[str, float] = {}
        self._bar_count: Dict[str, int] = {}
        self._prior_session_high: Dict[str, float] = {}
        self._prior_session_low: Dict[str, float] = {}
        self._open_positions: Dict[str, _OpenPosition] = {}

        # ORB state
        orb_total = 9 * 60 + 30 + config.opening_range_minutes
        self._orb_end_str = f"{orb_total // 60:02d}:{orb_total % 60:02d}"
        self._orb_high: Dict[str, float] = {}
        self._orb_low: Dict[str, float] = {}
        self._orb_confirmed: Dict[str, bool] = {}
        self._last_session_date: Optional[date] = None

    def run(self, all_bars: List[Bar]) -> None:
        sorted_bars = sorted(all_bars, key=lambda b: b.timestamp)
        for i, bar in enumerate(sorted_bars):
            self._check_exits(bar)
            self._process_bar(bar, i, sorted_bars)
        # EOD close any remaining positions using last bar's data
        if sorted_bars:
            for ticker, pos in list(self._open_positions.items()):
                self._exit_position(pos, sorted_bars[-1], "eod_close")

    def _check_exits(self, bar: Bar) -> None:
        """Check if bar triggers stop or target for any open position in this symbol."""
        pos = self._open_positions.get(bar.symbol)
        if pos is None:
            return

        et_str = _et_str(bar.timestamp)
        if et_str >= self._cfg.eod_close:
            self._exit_position(pos, bar, "eod_close")
            return

        if pos.direction == "long":
            stop_hit = bar.low <= pos.stop_price
            target_hit = bar.high >= pos.target_price
        else:
            stop_hit = bar.high >= pos.stop_price
            target_hit = bar.low <= pos.target_price

        if stop_hit:
            self._exit_position(pos, bar, "stop")
        elif target_hit:
            self._exit_position(pos, bar, "target")

    def _exit_position(self, pos: _OpenPosition, bar: Bar, reason: str) -> None:
        if reason == "stop":
            exit_price = pos.stop_price
        elif reason == "target":
            exit_price = pos.target_price
        else:
            exit_price = bar.close

        pnl = (
            (exit_price - pos.entry_price) * pos.shares
            if pos.direction == "long"
            else (pos.entry_price - exit_price) * pos.shares
        )

        slippage = abs(exit_price - bar.close) / bar.close if bar.close > 0 else 0.0

        self._trade_logger.log_exit(
            ticker=pos.ticker,
            entry_time=pos.entry_time,
            exit_price=exit_price,
            exit_time=bar.timestamp,
            exit_reason=reason,
            actual_slippage_pct=slippage,
        )

        self._portfolio.remove_position(pos.ticker)
        self._portfolio.equity += pnl
        if pnl < 0:
            self._portfolio.consecutive_losses += 1
        else:
            self._portfolio.consecutive_losses = 0

        del self._open_positions[pos.ticker]
        logger.debug("EXIT %s %s@%.2f pnl=%.2f reason=%s", pos.ticker, pos.direction,
                     exit_price, pnl, reason)

    def _process_bar(self, bar: Bar, idx: int, all_bars: List[Bar]) -> None:
        sym = bar.symbol
        et_str = _et_str(bar.timestamp)

        if et_str >= self._cfg.eod_close:
            return

        # Daily reset
        bar_date = bar.timestamp.date()
        if bar_date != self._last_session_date:
            self._last_session_date = bar_date
            self._orb_high.clear()
            self._orb_low.clear()
            self._orb_confirmed.clear()
            self._bar_count.clear()
            self._avg_volume.clear()

        # ORB tracking
        if "09:30" <= et_str < self._orb_end_str:
            self._orb_high[sym] = max(self._orb_high.get(sym, 0.0), bar.high)
            self._orb_low[sym] = min(self._orb_low.get(sym, float("inf")), bar.low)
        elif et_str >= self._orb_end_str and not self._orb_confirmed.get(sym, False):
            self._orb_confirmed[sym] = True

        # Update indicators
        vwap = self._vwap[sym].update(bar)
        atr = self._atr[sym].update(bar)
        rsi = self._rsi[sym].update(bar)
        ema9 = self._ema9[sym].update(bar)
        ema21 = self._ema21[sym].update(bar)
        ema50 = self._ema50[sym].update(bar)

        bc = self._bar_count.get(sym, 0) + 1
        self._bar_count[sym] = bc
        prev_avg = self._avg_volume.get(sym, float(bar.volume))
        self._avg_volume[sym] = prev_avg + (bar.volume - prev_avg) / min(bc, 20)

        if et_str < self._cfg.session_start:
            return
        if atr is None or rsi is None or ema9 is None or ema21 is None or ema50 is None:
            return
        if et_str >= self._cfg.session_end:
            return

        # Skip if already in a position for this symbol
        if sym in self._open_positions:
            return

        trend = "up" if ema9 > ema50 else "down"
        avg_vol = self._avg_volume[sym]
        prior_high = self._prior_session_high.get(sym, bar.high)
        prior_low = self._prior_session_low.get(sym, bar.low)

        # Generate signals
        for check_fn, kwargs in [
            (check_vwap_continuation, {"vwap": vwap, "avg_volume_20": avg_vol, "session_trend": trend}),
            (check_momentum_burst, {"avg_volume_20": avg_vol, "session_trend": trend}),
            (check_rsi_extreme, {"rsi": rsi, "session_trend": trend}),
            (check_breakout, {"prior_session_high": prior_high, "prior_session_low": prior_low,
                              "avg_volume_20": avg_vol}),
        ]:
            sig = check_fn(bar=bar, config=self._cfg, **kwargs)
            if sig:
                self._aggregator.add(sig)

        ema50_slope = "up" if trend == "up" else "down"
        ma_sig = check_ma_crossover(bar, ema9, ema21, ema50_slope, avg_vol, self._cfg)
        if ma_sig:
            self._aggregator.add(ma_sig)

        signals = self._aggregator.get_signals(sym)
        if not signals or self._aggregator.has_conflict(sym):
            self._aggregator.clear(sym)
            return

        # Portfolio/risk checks
        if self._regime == Regime.CRASH:
            self._aggregator.clear(sym)
            return
        if self._regime == Regime.HIGH_VOL and len(self._open_positions) >= 2:
            self._aggregator.clear(sym)
            return

        allowed, _ = self._portfolio.can_enter("Unknown", bar.timestamp)
        if not allowed:
            self._aggregator.clear(sym)
            return

        triggered, _ = self._kill_switch.check(self._portfolio, bar.timestamp)
        if triggered:
            self._aggregator.clear(sym)
            return

        direction = signals[0].direction
        size = compute_position_size(
            equity=self._portfolio.equity, atr=atr,
            entry_price=bar.close, config=self._cfg,
        )
        if size.shares == 0:
            self._aggregator.clear(sym)
            return

        # Simulate entry at next bar's open for this symbol
        next_sym_bar: Optional[Bar] = None
        for j in range(idx + 1, len(all_bars)):
            if all_bars[j].symbol == sym:
                next_sym_bar = all_bars[j]
                break

        if next_sym_bar is None:
            self._aggregator.clear(sym)
            return

        offset = self._cfg.limit_offset_pct
        if direction == "long":
            entry_price = next_sym_bar.open * (1 + offset)
            stop_price = size.long_stop(entry_price)
            target_price = size.long_target(entry_price)
        else:
            entry_price = next_sym_bar.open * (1 - offset)
            stop_price = size.short_stop(entry_price)
            target_price = size.short_target(entry_price)

        open_risk = size.shares * size.stop_distance
        position = Position(
            ticker=sym, direction=direction,
            shares=size.shares, entry_price=entry_price,
            stop_price=stop_price, target_price=target_price,
            entry_time=next_sym_bar.timestamp, atr_at_entry=atr,
            signals=[s.signal_type for s in signals],
            sector="Unknown", open_risk=open_risk,
        )
        self._portfolio.add_position(position)

        open_pos = _OpenPosition(
            ticker=sym, direction=direction, shares=size.shares,
            entry_price=entry_price, stop_price=stop_price, target_price=target_price,
            entry_time=next_sym_bar.timestamp, atr_at_entry=atr,
            signals=[s.signal_type for s in signals], open_risk=open_risk,
        )
        self._open_positions[sym] = open_pos

        record = TradeRecord(
            ticker=sym, direction=direction,
            entry_time=next_sym_bar.timestamp, entry_price=entry_price,
            shares=size.shares, stop_price=stop_price, target_price=target_price,
            signals=[s.signal_type for s in signals],
            sector="Unknown", regime=self._regime.value,
            portfolio_heat_at_entry=self._portfolio.portfolio_heat_pct,
            expected_slippage_pct=self._cfg.expected_entry_slippage_pct,
        )
        self._trade_logger.log_entry(record)
        logger.debug("ENTRY %s %s %d@%.2f stop=%.2f target=%.2f",
                     direction.upper(), sym, size.shares, entry_price, stop_price, target_price)

        self._aggregator.clear(sym)
