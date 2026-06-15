from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from bot.config import V4Config
from bot.intraday.types import Bar, Position


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

        # Layer 3a: VWAP break on elevated volume
        vwap_volume_threshold = baseline_volume_per_min * self._cfg.vwap_break_volume_ratio
        if bar.close < vwap and bar.volume >= vwap_volume_threshold:
            return ExitInstruction(reason="vwap_break", action="limit_exit", limit_price=bar.close)

        # Layer 3b: volume collapse
        if position.entry_bar_volume > 0:
            collapse_threshold = position.entry_bar_volume * self._cfg.volume_collapse_ratio
            if bar.volume < collapse_threshold:
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
