from __future__ import annotations
from collections import deque
from typing import Deque, Dict, Optional

from bot.intraday.types import Bar


class ATRIndicator:
    """ATR(period) computed on streaming bars using simple average of true ranges."""

    def __init__(self, period: int = 14) -> None:
        self.period = period
        self._bars: Dict[str, Deque[Bar]] = {}

    def update(self, bar: Bar) -> Optional[float]:
        sym = bar.symbol
        if sym not in self._bars:
            self._bars[sym] = deque(maxlen=self.period + 1)
        self._bars[sym].append(bar)
        return self._compute(sym)

    def get(self, symbol: str) -> Optional[float]:
        return self._compute(symbol)

    def _compute(self, sym: str) -> Optional[float]:
        bars = list(self._bars.get(sym, []))
        if len(bars) < 2:
            return None
        true_ranges = []
        for i in range(1, len(bars)):
            prev_close = bars[i - 1].close
            tr = max(
                bars[i].high - bars[i].low,
                abs(bars[i].high - prev_close),
                abs(bars[i].low - prev_close),
            )
            true_ranges.append(tr)
        return sum(true_ranges) / len(true_ranges)
