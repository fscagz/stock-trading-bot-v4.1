from __future__ import annotations
from collections import deque
from dataclasses import replace
from typing import Deque, Dict, Optional

from bot.intraday.types import Bar


class ATRIndicator:
    """ATR(period) computed on streaming bars using simple average of true ranges.

    aggregate_seconds > 0 buckets incoming bars into windows of that length
    (e.g. 60 → 1-minute ATR from a 15-second stream). Without it, ATR(14) on
    sub-minute bars measures seconds of range and any N×ATR stop sits inside
    ordinary noise. 0 = no aggregation (each bar is its own period), which is
    correct when the feed already delivers bars at the intended resolution.
    """

    def __init__(self, period: int = 14, aggregate_seconds: int = 0) -> None:
        self.period = period
        self.aggregate_seconds = aggregate_seconds
        self._bars: Dict[str, Deque[Bar]] = {}
        self._partial: Dict[str, Bar] = {}
        self._partial_key: Dict[str, int] = {}

    def update(self, bar: Bar) -> Optional[float]:
        sym = bar.symbol
        if sym not in self._bars:
            self._bars[sym] = deque(maxlen=self.period + 1)

        if self.aggregate_seconds <= 0:
            self._bars[sym].append(bar)
            return self._compute(sym)

        epoch = int(bar.timestamp.timestamp())
        bucket = epoch - epoch % self.aggregate_seconds
        cur = self._partial.get(sym)
        if cur is not None and self._partial_key.get(sym) == bucket:
            cur.high = max(cur.high, bar.high)
            cur.low = min(cur.low, bar.low)
            cur.close = bar.close
            cur.volume += bar.volume
        else:
            if cur is not None:
                self._bars[sym].append(cur)
            self._partial[sym] = replace(bar)
            self._partial_key[sym] = bucket
        return self._compute(sym)

    def get(self, symbol: str) -> Optional[float]:
        return self._compute(symbol)

    def _compute(self, sym: str) -> Optional[float]:
        bars = list(self._bars.get(sym, []))
        partial = self._partial.get(sym)
        if partial is not None:
            bars.append(partial)
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
        true_ranges = true_ranges[-self.period:]
        return sum(true_ranges) / len(true_ranges)
