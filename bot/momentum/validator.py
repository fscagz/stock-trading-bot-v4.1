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

    def confidence_score(self, bar: Bar, baseline_volume_per_min: float) -> float:
        """Return 0–1 score based on how strongly each signal exceeds its minimum.

        Only meaningful after validate() returns True. Used by the live bot and
        backtest to scale position size via confidence tiers (1×/2×/4×/8×).
        """
        history = list(self._history.get(bar.symbol, []))
        lookback = self._cfg.stage2_roc_lookback_bars
        if len(history) < lookback + 1 or baseline_volume_per_min <= 0:
            return 0.5

        past_close = history[-(lookback + 1)].close
        roc = (bar.close - past_close) / past_close if past_close > 0 else 0
        roc_min = self._cfg.stage2_roc_min_pct
        roc_range_mult = getattr(self._cfg, "confidence_score_roc_range_mult", 3.0)
        roc_score = min(1.0, max(0.0, (roc - roc_min) / (roc_range_mult * roc_min))) if roc_min > 0 else 0.5

        rel_vol = bar.volume / baseline_volume_per_min
        vol_min = self._cfg.stage2_min_relative_volume
        vol_range_mult = getattr(self._cfg, "confidence_score_vol_range_mult", 3.0)
        vol_score = min(1.0, max(0.0, (rel_vol - vol_min) / (vol_range_mult * vol_min))) if vol_min > 0 else 0.5

        bar_range = bar.high - bar.low
        close_pos = (bar.close - bar.low) / bar_range if bar_range > 0 else 0.5
        p_min = self._cfg.stage2_buying_pressure_min
        pressure_score = min(1.0, max(0.0, (close_pos - p_min) / max(1.0 - p_min, 1e-9)))

        return (roc_score + vol_score + pressure_score) / 3

    def _check_buying_pressure(self, bar: Bar) -> bool:
        bar_range = bar.high - bar.low
        if bar_range <= 0:
            return False
        close_position = (bar.close - bar.low) / bar_range
        return close_position >= self._cfg.stage2_buying_pressure_min
