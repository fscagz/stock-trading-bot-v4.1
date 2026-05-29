from __future__ import annotations
from collections import deque
from typing import Deque, Dict

from bot.config import V4Config
from bot.intraday.types import Bar


class MomentumValidator:
    """Stage 2 momentum validation: rate-of-change, relative volume, buying pressure.

    Call update(bar) to feed bars into the history buffer, then validate(bar, baseline)
    to check whether the bar meets all three entry conditions.
    """

    def __init__(self, config: V4Config) -> None:
        self._cfg = config
        self._history: Dict[str, Deque[Bar]] = {}

    def update(self, bar: Bar) -> None:
        sym = bar.symbol
        if sym not in self._history:
            lookback = self._cfg.stage2_roc_lookback_bars + 1
            self._history[sym] = deque(maxlen=lookback)
        self._history[sym].append(bar)

    def validate(self, bar: Bar, baseline_volume_per_min: float) -> bool:
        self.update(bar)
        history = list(self._history.get(bar.symbol, []))
        lookback = self._cfg.stage2_roc_lookback_bars

        if len(history) < lookback + 1:
            return False

        return (
            self._check_roc(bar, history, lookback)
            and self._check_relative_volume(bar, baseline_volume_per_min)
            and self._check_buying_pressure(bar)
        )

    def _check_roc(self, bar: Bar, history: list, lookback: int) -> bool:
        past_close = history[-(lookback + 1)].close
        if past_close <= 0:
            return False
        roc = (bar.close - past_close) / past_close
        return roc >= self._cfg.stage2_roc_min_pct

    def _check_relative_volume(self, bar: Bar, baseline_volume_per_min: float) -> bool:
        if baseline_volume_per_min <= 0:
            return False
        return bar.volume >= baseline_volume_per_min * self._cfg.stage2_min_relative_volume

    def _check_buying_pressure(self, bar: Bar) -> bool:
        bar_range = bar.high - bar.low
        if bar_range <= 0:
            return False
        close_position = (bar.close - bar.low) / bar_range
        return close_position >= self._cfg.stage2_buying_pressure_min
