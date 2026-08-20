from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional

from bot.config import V4Config
from bot.intraday.types import Bar, Position

logger = logging.getLogger(__name__)


@dataclass
class ExitInstruction:
    reason: str    # "hard_stop" | "trailing_stop" | "vwap_break" | "volume_collapse"
                   # | "structure_break" | "eod"
    action: str    # "market_exit" | "limit_exit"
    limit_price: Optional[float] = None


class PositionManager:
    """Monitors open positions on each bar and returns exit instructions when conditions are met.

    Does not call the broker directly — the main loop acts on returned ExitInstructions.
    """

    def __init__(self, config: V4Config) -> None:
        self._cfg = config
        self._structure_break_count: dict = {}
        self._prev_bar: dict = {}

    def on_bar(
        self,
        bar: Bar,
        position: Position,
        vwap: float,
        baseline_volume_per_min: float,
    ) -> Optional[ExitInstruction]:
        sym = bar.symbol

        # Layer 1: hard stop
        if bar.close <= position.stop_price:
            return ExitInstruction(reason="hard_stop", action="market_exit")

        # Layer 2: update trailing stop
        if bar.close > position.highest_close:
            position.highest_close = bar.close
            new_stop = round(bar.close - self._cfg.trailing_stop_atr_multiple * position.atr_at_entry, 2)
            if new_stop > position.stop_price:
                position.stop_price = new_stop

        # Break-even stop: once profit >= trigger multiple of ATR, floor stop at entry
        be_trigger = getattr(self._cfg, "breakeven_trigger_atr_multiple", 0.0)
        if be_trigger > 0 and position.atr_at_entry > 0:
            if bar.close >= position.entry_price + be_trigger * position.atr_at_entry:
                if position.stop_price < position.entry_price:
                    position.stop_price = position.entry_price

        if bar.close <= position.stop_price:
            return ExitInstruction(reason="trailing_stop", action="market_exit")

        # Change #2 (2026-06-30): once the stop is at or above entry, the position is
        # past breakeven — the trailing stop already caps risk at ~$0, so the protective
        # micro-exits below (vwap_break / volume_collapse / structure_break) only serve
        # to amputate winners before the 4R target (the 6 target hits over 110 trades
        # were the only positive bucket). Suppress them while green; KEEP them while the
        # trade is still at risk (there they cut a loser before the full hard stop).
        past_breakeven = position.stop_price >= position.entry_price

        # Layer 3a: VWAP break on elevated volume
        vwap_volume_threshold = baseline_volume_per_min * self._cfg.vwap_break_volume_ratio
        if bar.close < vwap and bar.volume >= vwap_volume_threshold:
            if past_breakeven:
                logger.info("EXIT-SUPPRESSED %s vwap_break — past breakeven, letting trailing stop run", sym)
            else:
                return ExitInstruction(reason="vwap_break", action="limit_exit", limit_price=bar.close)

        # Layer 3b: volume collapse (skip first 2 bars post-entry to avoid instant exits)
        bars_since_entry = int((bar.timestamp - position.entry_time).total_seconds() / 60)
        if bars_since_entry > 2 and position.entry_bar_volume > 0:
            collapse_threshold = position.entry_bar_volume * self._cfg.volume_collapse_ratio
            if bar.volume < collapse_threshold:
                if past_breakeven:
                    logger.info("EXIT-SUPPRESSED %s volume_collapse — past breakeven, letting trailing stop run", sym)
                else:
                    return ExitInstruction(reason="volume_collapse", action="market_exit")

        # Layer 3c: structure break (consecutive lower high + lower low)
        prev = self._prev_bar.get(sym)
        if prev is not None:
            if bar.high < prev.high and bar.low < prev.low:
                self._structure_break_count[sym] = self._structure_break_count.get(sym, 0) + 1
            else:
                self._structure_break_count[sym] = 0
            if self._structure_break_count.get(sym, 0) >= self._cfg.structure_break_bars:
                self._structure_break_count[sym] = 0
                if past_breakeven:
                    logger.info("EXIT-SUPPRESSED %s structure_break — past breakeven, letting trailing stop run", sym)
                else:
                    return ExitInstruction(reason="structure_break", action="limit_exit", limit_price=bar.close)

        self._prev_bar[sym] = bar
        return None

    def should_hold_overnight(
        self,
        bar: Bar,
        position: Position,
        vwap: float,
        baseline_volume_per_min: float,
    ) -> bool:
        if bar.close < vwap:
            return False
        volume_threshold = baseline_volume_per_min * self._cfg.overnight_min_volume_ratio
        if bar.volume < volume_threshold:
            return False
        return True
